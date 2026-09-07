'''

TopK / BatchTopK: Auxiliary loss weight 0.05 with 1
JumpReLU: STE bandwidth ε = 0.001. Target L0 set to match k of other architectures.
Matryoshka: Nested feature groups with geometrically spaced sizes (dsae/8, dsae/8, dsae/4, remainder). Otherwise same as BatchTopK.
Standard (ℓ1): Sparsity weight λ ∈ 0.03, 0.04, 0.1.

'''

from matryoshka_sae.sae import GlobalBatchTopKMatryoshkaSAE, JumpReLUSAE, BatchTopKSAE, VanillaSAE
import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.argprint import print_args
from diffusers import UNet2DConditionModel
from transformers import AutoProcessor, CLIPVisionModel,CLIPVisionModelWithProjection
from diffusers.image_processor import VaeImageProcessor
from diffusers import DiffusionPipeline, AutoencoderKL
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader,random_split
from collections import defaultdict
import time
from tqdm import tqdm
from datasets import load_dataset

from experiment_helpers.init_helpers import default_parser,repo_api_init
from experiment_helpers.saving_helpers import save_and_load_functions
from sklearn.neighbors import NearestNeighbors
from regularization import graph_laplacian,contractive,cosine_constrastive

TOPK="topk"
VANILLA="vanilla"
MATRYOSHKA="matryoshka"
JUMP="jumprelu"

parser=default_parser()
parser.add_argument("--dataset_path", type=str, default="jlbaker361/model")
parser.add_argument("--sae",type=str,default=VANILLA, help=f"one of {TOPK, VANILLA,MATRYOSHKA,JUMP}")
parser.add_argument("--dict_size",type=int,default=None,help="sae dictionary size, defaults to 8x the embedding size")
parser.add_argument("--top_k",type=int,default=32,help="for topk/batchtopk/matryoshka sae")
parser.add_argument("--l1_coeff",type=float,default=0.04,help="sparsity weight for vanilla/jumprelu sae")
parser.add_argument("--aux_penalty",type=float,default=0.05,help="dead feature auxiliary loss weight for topk/batchtopk/matryoshka sae")
parser.add_argument("--top_k_aux",type=int,default=512,help="for topk/batchtopk/matryoshka sae")
parser.add_argument("--bandwidth",type=float,default=0.001,help="STE bandwidth for jumprelu sae")
parser.add_argument("--n_batches_to_dead",type=int,default=20)
parser.add_argument("--group_sizes",nargs="*",type=int,default=None,help="matryoshka nested group sizes, defaults to dict_size/8, dict_size/8, dict_size/4, remainder")
parser.add_argument("--max_grad_norm",type=float,default=100000.0)

parser.add_argument("--use_graph_laplacian",action="store_true",help="pull each sample's sae activations towards its k nearest neighbors' activations")
parser.add_argument("--graph_laplacian_weight",type=float,default=0.01)
parser.add_argument("--use_contractive",action="store_true",help="penalize the jacobian of the sae encoder, https://icml.cc/2011/papers/455_icmlpaper.pdf")
parser.add_argument("--contractive_weight",type=float,default=0.01)
parser.add_argument("--use_cosine_contrastive",action="store_true",help="like graph_laplacian but weighted by cosine similarity of the raw embeddings to their neighbors")
parser.add_argument("--cosine_contrastive_weight",type=float,default=0.01)
parser.add_argument("--k_neighbors",type=int,default=5,help="number of nearest neighbors (in raw embedding space) precomputed for graph_laplacian/cosine_contrastive")

SAE_CLASSES={
    TOPK:BatchTopKSAE,
    VANILLA:VanillaSAE,
    MATRYOSHKA:GlobalBatchTopKMatryoshkaSAE,
    JUMP:JumpReLUSAE,
}

class EmbeddingDataset(Dataset):
    def __init__(self,dataset_path:str,k_neighbors:int=0):
        super().__init__()
        self.hf_data=load_dataset(dataset_path,split="train")
        self.neighbor_indices=None
        if k_neighbors>0:
            features=np.array(self.hf_data["features"])
            n_neighbors=min(k_neighbors+1,len(features))
            nn=NearestNeighbors(n_neighbors=n_neighbors).fit(features)
            _,indices=nn.kneighbors(features)
            self.neighbor_indices=indices[:,1:]

    def __len__(self):
        return len(self.hf_data)

    def __getitem__(self, index):
        item = self.hf_data[index]
        result={
            "image":item["image"],
            "text":item["text"],
            "embedding":torch.tensor(item["features"])
        }
        if self.neighbor_indices is not None:
            neighbor_rows=self.hf_data[self.neighbor_indices[index].tolist()]
            result["neighbor_embedding"]=torch.tensor(np.array(neighbor_rows["features"]))
        return result

def collate_embeddings(batch):
    result={
        "image":[item["image"] for item in batch],
        "text":[item["text"] for item in batch],
        "embedding":torch.stack([item["embedding"] for item in batch]),
    }
    if "neighbor_embedding" in batch[0]:
        result["neighbor_embedding"]=torch.stack([item["neighbor_embedding"] for item in batch])
    return result


def main(args):
    api,accelerator,device=repo_api_init(args)
    mixed_precision : str = args.mixed_precision
    project_name : str = args.project_name
    gradient_accumulation_steps : int = args.gradient_accumulation_steps
    repo_id : str = args.repo_id
    lr : float = args.lr
    epochs : int = args.epochs
    limit : int = args.limit
    save_dir : str = args.save_dir
    batch_size : int = args.batch_size
    val_interval : int = args.val_interval
    load_hf  = args.load_hf
    dataset_path:str=args.dataset_path
    use_neighbors=args.use_graph_laplacian or args.use_cosine_contrastive

    embedding_dataset=EmbeddingDataset(dataset_path,k_neighbors=args.k_neighbors if use_neighbors else 0)

    train_split,val_split,test_split=random_split(embedding_dataset,[0.9,0.05,0.05])
    train_loader=DataLoader(train_split,batch_size=batch_size,shuffle=True,collate_fn=collate_embeddings)
    val_loader=DataLoader(val_split,batch_size=batch_size,shuffle=False,collate_fn=collate_embeddings)
    test_loader=DataLoader(test_split,batch_size=batch_size,shuffle=False,collate_fn=collate_embeddings)

    act_size=embedding_dataset[0]["embedding"].shape[-1]
    dict_size=args.dict_size if args.dict_size is not None else act_size*8

    group_sizes=args.group_sizes
    if group_sizes is None:
        eighth=dict_size//8
        quarter=dict_size//4
        group_sizes=[eighth,eighth,quarter,dict_size-2*eighth-quarter]

    cfg={
        "seed":123,
        "act_size":act_size,
        "dict_size":dict_size,
        "device":device,
        "dtype":torch.float32,
        "input_unit_norm":True,
        "l1_coeff":args.l1_coeff,
        "top_k":args.top_k,
        "top_k_aux":args.top_k_aux,
        "aux_penalty":args.aux_penalty,
        "bandwidth":args.bandwidth,
        "n_batches_to_dead":args.n_batches_to_dead,
        "group_sizes":group_sizes,
    }

    sae=SAE_CLASSES[args.sae](cfg)
    optimizer=torch.optim.Adam(sae.parameters(),lr=lr)

    save,load=save_and_load_functions({"sae.pt":sae},save_dir,api,repo_id)
    start_epoch=load(load_hf)

    sae,optimizer,train_loader,val_loader,test_loader=accelerator.prepare(
        sae,optimizer,train_loader,val_loader,test_loader
    )

    def run_batch(batch,train:bool):
        embedding=batch["embedding"].to(device).float()
        sae.train(train)
        with torch.set_grad_enabled(train):
            output=sae(embedding)
            total_loss=output["loss"]

            if train and args.use_contractive:
                dictionary=accelerator.unwrap_model(sae).W_dec
                reg=contractive(output["feature_acts"],dictionary)
                output["contractive_loss"]=reg
                total_loss=total_loss+args.contractive_weight*reg

            if train and (args.use_graph_laplacian or args.use_cosine_contrastive):
                neighbor_embedding=batch["neighbor_embedding"].to(device).float()
                nb,nk,nd=neighbor_embedding.shape
                neighbor_acts=sae(neighbor_embedding.reshape(nb*nk,nd))["feature_acts"].reshape(nb,nk,-1)

                if args.use_graph_laplacian:
                    reg=graph_laplacian(output["feature_acts"],neighbor_acts)
                    output["graph_laplacian_loss"]=reg
                    total_loss=total_loss+args.graph_laplacian_weight*reg

                if args.use_cosine_contrastive:
                    reg=cosine_constrastive(output["feature_acts"],neighbor_acts,embedding,neighbor_embedding)
                    output["cosine_contrastive_loss"]=reg
                    total_loss=total_loss+args.cosine_contrastive_weight*reg

            output["total_loss"]=total_loss

        if train:
            with accelerator.accumulate(sae):
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(sae.parameters(),args.max_grad_norm)
                    accelerator.unwrap_model(sae).make_decoder_weights_and_grad_unit_norm()
                optimizer.step()
                optimizer.zero_grad()
        return output

    metric_keys=["loss","total_loss","contractive_loss","graph_laplacian_loss","cosine_contrastive_loss"]

    def epoch_pass(loader,train:bool,desc:str):
        metrics=defaultdict(list)
        for b,batch in enumerate(loader):
            if limit>0 and b>=limit:
                break
            output=run_batch(batch,train)
            for key in metric_keys:
                if key in output:
                    metrics[key].append(output[key].item())
        return {key:float(np.mean(values)) for key,values in metrics.items()}

    for epoch in range(start_epoch,epochs+1):
        train_metrics=epoch_pass(train_loader,True,f"epoch {epoch} train")
        accelerator.log({f"train_{key}":value for key,value in train_metrics.items()},step=epoch)

        if epoch%val_interval==0:
            val_metrics=epoch_pass(val_loader,False,f"epoch {epoch} val")
            accelerator.log({f"val_{key}":value for key,value in val_metrics.items()},step=epoch)

        if accelerator.is_main_process:
            save(epoch)

    test_metrics=epoch_pass(test_loader,False,"test")
    accelerator.log({f"test_{key}":value for key,value in test_metrics.items()})

    accelerator.end_training()

if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print_args(parser)
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")