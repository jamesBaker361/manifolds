# trains a saev (saev_repo submodule) sparse autoencoder on precomputed activation shards,
# with the option to add regularization.py's regularizers and metrics.py's subspace-capture (R2) metric

import os
import sys
import pathlib
import time
from collections import defaultdict

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from experiment_helpers.gpu_details import print_details
from experiment_helpers.argprint import print_args
from experiment_helpers.init_helpers import default_parser, repo_api_init
from huggingface_hub import snapshot_download

from regularization import graph_laplacian, contractive, cosine_constrastive
import metrics

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "saev_repo", "src"))
import saev.data
import saev.nn.modeling as modeling
import saev.nn.objectives as objectives

RELU = "relu"
TOPK = "topk"
BATCH_TOPK = "batchtopk"
ACTIVATIONS = [RELU, TOPK, BATCH_TOPK]

FAMILIES = ["bird-mae", "clip", "dinov2", "dinov3", "fake-clip", "pe-core", "pe-spatial", "siglip"]

parser = default_parser()
parser.add_argument("--train_shards", type=str, default="training_shards", help="directory with saev activation shards for training; if missing/unset, shards are generated from --dataset_name/--dataset_split")
parser.add_argument("--train_layer", type=int, default=13, help="which ViT layer to read from the training shards (also the layer captured when generating shards)")
parser.add_argument("--val_shards", type=str, default=None, help="directory with saev activation shards for validation; if set but missing, shards are generated from --val_dataset_name/--val_dataset_split")
parser.add_argument("--val_layer", type=int, default=13, help="which ViT layer to read from the validation shards (also the layer captured when generating shards)")

parser.add_argument("--dataset_name", type=str, default=None, help="HF dataset repo (with image + label ClassLabel columns) to compute training shards from, if --train_shards doesn't already exist")
parser.add_argument("--dataset_split", type=str, default="train", help="split of --dataset_name to use for training shards")
parser.add_argument("--val_dataset_name", type=str, default=None, help="HF dataset repo to compute validation shards from, if --val_shards is set but doesn't already exist")
parser.add_argument("--val_dataset_split", type=str, default="validation", help="split of --val_dataset_name to use for validation shards")
parser.add_argument("--family", type=str, default="dinov3", help=f"ViT family used when generating shards, one of {FAMILIES}")
parser.add_argument("--checkpoint", type=str, default="dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth", help="ViT checkpoint used when generating shards; for family=dinov3 this must be a local path to Meta's original .pth checkpoint, not a transformers-format hub id")
parser.add_argument("--vit_d_model", type=int, default=1280, help="ViT activation dimension used when generating shards (1280 for dinov3 vith16plus)")
parser.add_argument("--content_tokens_per_example", type=int, default=196, help="number of content (non-CLS) tokens per example used when generating shards (14x14 for a 224px/16px-patch ViT)")
parser.add_argument("--shards_root", type=str, default="saev_shards", help="root directory that generated train/val shards are written under")
parser.add_argument("--vit_batch_size", type=int, default=256, help="batch size for ViT inference when generating shards")
parser.add_argument("--n_shard_workers", type=int, default=8, help="number of dataloader workers when generating shards")
parser.add_argument("--max_tokens_per_shard", type=int, default=2_400_000, help="maximum activations per shard file when generating shards")

parser.add_argument("--activation", type=str, default=TOPK, help=f"one of {ACTIVATIONS}")
parser.add_argument("--d_sae", type=int, default=None, help="sae dictionary size, defaults to 8x the activation size")
parser.add_argument("--top_k", type=int, default=32, help="for topk/batchtopk activations")
parser.add_argument("--batch_top_k_momentum", type=float, default=0.1, help="running-threshold momentum for batchtopk")
parser.add_argument("--l1_coeff", type=float, default=4e-4, help="l1 sparsity coefficient for relu activation")
parser.add_argument("--k_aux", type=int, default=512, help="number of dead latents used by the auxk loss")
parser.add_argument("--aux_alpha", type=float, default=1 / 32, help="auxk loss weight")
parser.add_argument("--n_prefixes", type=int, default=10, help="number of matryoshka prefixes")
parser.add_argument("--dead_threshold_tokens", type=int, default=1_000_000, help="tokens without activation before a latent is considered dead")
parser.add_argument("--max_grad_norm", type=float, default=1.0)

parser.add_argument("--use_graph_laplacian", action="store_true", help="pull each sample's sae activations towards its k nearest neighbors' activations")
parser.add_argument("--graph_laplacian_weight", type=float, default=0.01)
parser.add_argument("--use_contractive", action="store_true", help="penalize the jacobian of the sae encoder, https://icml.cc/2011/papers/455_icmlpaper.pdf")
parser.add_argument("--contractive_weight", type=float, default=0.01)
parser.add_argument("--use_cosine_contrastive", action="store_true", help="like graph_laplacian but weighted by cosine similarity of the raw activations to their neighbors")
parser.add_argument("--cosine_contrastive_weight", type=float, default=0.01)
parser.add_argument("--k_neighbors", type=int, default=5, help="number of nearest neighbors precomputed for graph_laplacian/cosine_contrastive")
parser.add_argument("--knn_pool_size", type=int, default=2048, help="number of activations randomly sampled once (via random access into the training shards) to build the neighbor pool for graph_laplacian/cosine_contrastive")
parser.add_argument("--knn_cache_dir", type=str, default=None, help="where to cache the sampled neighbor pool + neighbor indices on disk, so repeated runs skip resampling/refitting; defaults to <save_dir>/knn_cache")

parser.add_argument("--eval_r2", action="store_true", help="compute metrics.get_R2 subspace-capture score on validation data")
parser.add_argument("--r2_n_samples", type=int, default=2000, help="number of validation activations used for the r2 metric")
parser.add_argument("--r2_max_k", type=int, default=100)
parser.add_argument("--r2_var_threshold", type=float, default=0.95)


def make_activation_cfg(args):
    if args.activation == RELU:
        return modeling.Relu(sparsity=modeling.L1Sparsity(coeff=args.l1_coeff))
    if args.activation == TOPK:
        return modeling.TopK(top_k=args.top_k, aux=modeling.AuxK(k_aux=args.k_aux, alpha=args.aux_alpha))
    if args.activation == BATCH_TOPK:
        return modeling.BatchTopK(top_k=args.top_k, momentum=args.batch_top_k_momentum, aux=modeling.AuxK(k_aux=args.k_aux, alpha=args.aux_alpha))
    raise ValueError(f"unknown activation {args.activation}, must be one of {ACTIVATIONS}")


def generate_shards(args, dataset_name, dataset_split, layer, shards_subdir, device):
    """Computes and saves ViT activation shards, returning the resulting shard directory.

    saev_repo's dinov3 loader (saev.data.dinov3.Vit) expects a local file in Meta's
    original DINOv3 release format (e.g. 'dinov3_vitl16_pretrain_lvd1689m-<hash>.pth').
    A transformers-format hub checkpoint like facebook/dinov3-vitl16-pretrain-lvd1689m
    has a different state_dict layout (HF's own reimplementation) and can't be loaded
    by it directly, so we fail fast here instead of deep inside saev's own loader.
    """
    if args.family == "dinov3" and not os.path.isfile(args.checkpoint):
        raise ValueError(
            f"--family dinov3 needs --checkpoint to be a local path to Meta's original "
            f"DINOv3 checkpoint (e.g. 'dinov3_vitl16_pretrain_lvd1689m-<hash>.pth'), got "
            f"'{args.checkpoint}'. The Hugging Face repo facebook/dinov3-vitl16-pretrain-lvd1689m "
            "is a transformers-format safetensors checkpoint with a different state_dict layout "
            "than saev_repo's from-scratch DINOv3 implementation, so it can't be loaded as-is. "
            "Download the original weights from Meta's DINOv3 release "
            "(https://ai.meta.com/resources/models-and-libraries/dinov3-license) and point "
            "--checkpoint at that local .pth file instead."
        )

    assert dataset_name is not None, "need --dataset_name (or --val_dataset_name) to generate shards"

    from saev.data import shards as saev_shards

    shards_root = pathlib.Path(args.shards_root) / shards_subdir / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)

    return saev_shards.worker_fn(
        data=saev.data.datasets.Imagenet(name=dataset_name, split=dataset_split),
        family=args.family,
        ckpt=args.checkpoint,
        d_model=args.vit_d_model,
        layers=[layer],
        content_tokens_per_example=args.content_tokens_per_example,
        cls_token=True,
        max_tokens_per_shard=args.max_tokens_per_shard,
        batch_size=args.vit_batch_size,
        n_workers=args.n_shard_workers,
        device=str(device),
        shards_root=shards_root,
    )


def resolve_shards(shards_arg, dataset_name, dataset_split, layer, shards_subdir, args, device):
    if shards_arg and os.path.isdir(shards_arg):
        return pathlib.Path(shards_arg)
    print(f"no shards at '{shards_arg}', generating from {dataset_name} ({dataset_split})...")
    return generate_shards(args, dataset_name, dataset_split, layer, shards_subdir, device)


def _knn_cache_path(cache_dir, shards_dir, layer, pool_size, k_neighbors):
    # saev shard directories are already named by a content hash of their config,
    # so that name plus our own sampling knobs fully determines the cached pool.
    shard_hash = pathlib.Path(shards_dir).name
    fname = f"{shard_hash}_layer{layer}_pool{pool_size}_k{k_neighbors}.pt"
    return pathlib.Path(cache_dir) / fname


def build_knn_pool(shards_dir, layer, pool_size, k_neighbors, device, cache_dir=None):
    cache_path = _knn_cache_path(cache_dir, shards_dir, layer, pool_size, k_neighbors) if cache_dir else None
    if cache_path is not None and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        print(f"loaded knn pool from cache: {cache_path}")
        return cached["acts"].to(device), cached["neighbor_indices"].to(device)

    cfg = saev.data.IndexedConfig(shards=pathlib.Path(shards_dir), layer=layer)
    dataset = saev.data.IndexedDataset(cfg)
    n = min(pool_size, len(dataset))
    indices = torch.randperm(len(dataset))[:n].tolist()
    acts = torch.stack([dataset[i]["act"] for i in indices]).float()

    n_neighbors = min(k_neighbors + 1, len(acts))
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(acts.numpy())
    _, neighbor_indices_np = nn.kneighbors(acts.numpy())
    neighbor_indices = torch.tensor(neighbor_indices_np[:, 1:], dtype=torch.long)

    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"acts": acts, "neighbor_indices": neighbor_indices}, cache_path)
            print(f"cached knn pool to: {cache_path}")
        except OSError as e:
            print(f"failed to cache knn pool: {e}")

    return acts.to(device), neighbor_indices.to(device)


def main(args):
    api, accelerator, device = repo_api_init(args)
    repo_id: str = args.repo_id
    lr: float = args.lr
    epochs: int = args.epochs
    limit: int = args.limit
    save_dir: str = args.save_dir
    batch_size: int = args.batch_size
    val_interval: int = args.val_interval
    load_hf = args.load_hf
    use_neighbors = args.use_graph_laplacian or args.use_cosine_contrastive

    train_shards_dir = resolve_shards(args.train_shards, args.dataset_name, args.dataset_split, args.train_layer, "train", args, device)
    train_cfg = saev.data.ShuffledConfig(shards=train_shards_dir, layer=args.train_layer, batch_size=batch_size)
    train_loader = saev.data.ShuffledDataLoader(train_cfg)
    d_model = train_loader.metadata.d_model

    val_loader = None
    val_shards_dir = None
    if args.val_shards or args.val_dataset_name:
        val_shards_dir = resolve_shards(args.val_shards, args.val_dataset_name, args.val_dataset_split, args.val_layer, "val", args, device)
        val_cfg = saev.data.ShuffledConfig(shards=val_shards_dir, layer=args.val_layer, batch_size=batch_size)
        val_loader = saev.data.ShuffledDataLoader(val_cfg)

    d_sae = args.d_sae if args.d_sae is not None else d_model * 8
    activation_cfg = make_activation_cfg(args)
    sae_cfg = modeling.SparseAutoencoderConfig(d_model=d_model, d_sae=d_sae, activation=activation_cfg)
    sae = modeling.SparseAutoencoder(sae_cfg)
    objective = objectives.MatryoshkaObjective(
        objectives.Matryoshka(n_prefixes=args.n_prefixes, dead_threshold_tokens=args.dead_threshold_tokens)
    ).to(device)

    knn_pool = None
    if use_neighbors:
        knn_cache_dir = args.knn_cache_dir or os.path.join(save_dir, "knn_cache")
        knn_pool = build_knn_pool(train_shards_dir, args.train_layer, args.knn_pool_size, args.k_neighbors, device, cache_dir=knn_cache_dir)

    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    class EpochState:
        def __init__(self):
            self.epoch = 1

        def state_dict(self):
            return {"epoch": self.epoch}

        def load_state_dict(self, state_dict):
            self.epoch = state_dict["epoch"]

    epoch_state = EpochState()
    checkpoint_dir = os.path.join(save_dir, "checkpoint")

    if load_hf:
        try:
            snapshot_download(repo_id, allow_patterns="checkpoint/*", local_dir=save_dir)
        except Exception as e:
            print(f"failed to download checkpoint: {e}")

    sae, optimizer = accelerator.prepare(sae, optimizer)
    accelerator.register_for_checkpointing(epoch_state)

    start_epoch = 1
    if os.path.isdir(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
        try:
            accelerator.load_state(checkpoint_dir)
            start_epoch = epoch_state.epoch + 1
            print(f"resumed from checkpoint at epoch {start_epoch}")
        except Exception as e:
            print(f"failed to load checkpoint: {e}")

    def run_batch(x, train: bool):
        unwrapped_sae = accelerator.unwrap_model(sae)
        unwrapped_sae.train(train)
        objective.train(train)
        with torch.set_grad_enabled(train):
            unwrapped_sae.normalize_w_dec()
            loss, fwd = objective(unwrapped_sae, x)
            output = dict(loss.metrics())
            total_loss = loss.loss

            dictionary = unwrapped_sae.W_dec

            if train and args.use_contractive:
                reg = contractive(fwd.f_x, dictionary)
                output["contractive_loss"] = reg
                total_loss = total_loss + args.contractive_weight * reg

            if train and knn_pool is not None:
                pool_acts, neighbor_indices = knn_pool
                anchor_idx = torch.randint(0, pool_acts.shape[0], (min(batch_size, pool_acts.shape[0]),), device=device)
                anchor_acts = pool_acts[anchor_idx]
                neighbor_acts = pool_acts[neighbor_indices[anchor_idx]]
                nb, nk, nd = neighbor_acts.shape
                anchor_codes = unwrapped_sae.encode(anchor_acts).f_x
                neighbor_codes = unwrapped_sae.encode(neighbor_acts.reshape(nb * nk, nd)).f_x.reshape(nb, nk, -1)

                if args.use_graph_laplacian:
                    reg = graph_laplacian(anchor_codes, neighbor_codes)
                    output["graph_laplacian_loss"] = reg
                    total_loss = total_loss + args.graph_laplacian_weight * reg

                if args.use_cosine_contrastive:
                    reg = cosine_constrastive(anchor_codes, neighbor_codes, anchor_acts, neighbor_acts)
                    output["cosine_contrastive_loss"] = reg
                    total_loss = total_loss + args.cosine_contrastive_weight * reg

            output["total_loss"] = total_loss

        if train:
            accelerator.backward(total_loss)
            unwrapped_sae.remove_parallel_grads()
            if accelerator.sync_gradients:
                torch.nn.utils.clip_grad_norm_(sae.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        return output

    metric_keys = ["loss", "mse", "l0", "l1", "sparsity", "aux", "n_dead", "total_loss", "contractive_loss", "graph_laplacian_loss", "cosine_contrastive_loss"]

    def epoch_pass(loader, train: bool):
        metrics_accum = defaultdict(list)
        for b, batch in enumerate(loader):
            if limit > 0 and b >= limit:
                break
            x = batch["act"].to(device).float()
            output = run_batch(x, train)
            for key in metric_keys:
                if key in output:
                    value = output[key]
                    metrics_accum[key].append(value.item() if torch.is_tensor(value) else float(value))
        return {key: float(np.mean(values)) for key, values in metrics_accum.items() if values}

    def eval_r2(loader):
        collected = []
        n = 0
        for batch in loader:
            x = batch["act"].to(device).float()
            collected.append(x)
            n += x.shape[0]
            if n >= args.r2_n_samples:
                break
        data = torch.cat(collected, dim=0)[: args.r2_n_samples]
        unwrapped_sae = accelerator.unwrap_model(sae)
        unwrapped_sae.train(False)
        with torch.no_grad():
            r2 = metrics.get_R2(data, unwrapped_sae, max_k=args.r2_max_k, var_threshold=args.r2_var_threshold)
        return float(r2.item())

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = epoch_pass(train_loader, True)
        accelerator.log({f"train_{key}": value for key, value in train_metrics.items()}, step=epoch)

        if val_loader is not None and epoch % val_interval == 0:
            val_metrics = epoch_pass(val_loader, False)
            accelerator.log({f"val_{key}": value for key, value in val_metrics.items()}, step=epoch)

            if args.eval_r2:
                r2 = eval_r2(val_loader)
                accelerator.log({"val_r2": r2}, step=epoch)

        epoch_state.epoch = epoch
        accelerator.save_state(checkpoint_dir)
        if accelerator.is_main_process:
            try:
                api.upload_folder(repo_id=repo_id, folder_path=checkpoint_dir, path_in_repo="checkpoint")
            except Exception as e:
                print(f"failed to upload checkpoint: {e}")

    if val_loader is not None:
        test_metrics = epoch_pass(val_loader, False)
        accelerator.log({f"test_{key}": value for key, value in test_metrics.items()})

    accelerator.end_training()

    train_loader.shutdown()
    if val_loader is not None:
        val_loader.shutdown()


if __name__ == '__main__':
    print_args(parser)
    print_details()
    start = time.time()
    args = parser.parse_args()
    print_args(parser)
    print(args)
    main(args)
    end = time.time()
    seconds = end - start
    hours = seconds / (60 * 60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")
