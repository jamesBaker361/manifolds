# gets all of our data into a nice hf dataset

# this trains the sae and then does SAEURON stuyle removal I think?

import os
import argparse
import itertools
from experiment_helpers.gpu_details import print_details
from experiment_helpers.saving_helpers import save_and_load_functions
from experiment_helpers.argprint import print_args
from diffusers import DiffusionPipeline,UNet2DConditionModel,AutoencoderKL,Krea2Pipeline
from diffusers.image_processor import VaeImageProcessor
from transformers import pipeline
import torch
import numpy as np
import csv
import sys

import time
import torch.nn.functional as F
from datasets import load_dataset,Features,Value,Image as HFImage,Sequence
import json
from PIL import Image

from experiment_helpers.loop_decorator import optimization_loop
from experiment_helpers.data_helpers import split_data
from experiment_helpers.init_helpers import default_parser,repo_api_init
from transformers import CLIPVisionModelWithProjection,CLIPImageProcessor,CLIPProcessor,CLIPModel
from peft import LoraConfig
from accelerate import Accelerator
from datasets import Dataset


parser=default_parser()

GRAYSCALE="grayscale"
BRIGHTEN="brighten"
DARKEN="darken"

available_effects=[GRAYSCALE,BRIGHTEN,DARKEN]

DINO_V3="dino_v3"

embedding_model_checkpoints={
    DINO_V3:"facebook/dinov3-vits16-pretrain-lvd1689m"
}

parser.add_argument("--prompt_files",nargs="*",help="each of these files becomes part of the promps")
parser.add_argument("--conjunction",type=str,default=",")
parser.add_argument("--checkpoint",type=str,default="SimianLuo/LCM_Dreamshaper_v7")
parser.add_argument("--num_inference_steps",type=int,default=16)
parser.add_argument("--effects",nargs="*",help=f"effects that linearly effect images like {available_effects}")
parser.add_argument("--embedding_model",type=str,default=DINO_V3)
parser.add_argument("--images_per_prompt",type=int,default=5)
parser.add_argument("--size",type=int,default=512)

def main(args):
    api,accelerator,device=repo_api_init(args)
    mixed_precision : str = args.mixed_precision
    project_name : str = args.project_name
    gradient_accumulation_steps : int = args.gradient_accumulation_steps
    repo_id : str = args.repo_id
    prompt_files=args.prompt_files
    checkpoint=args.checkpoint
    
    prompt_categories=[]
    for p_file in prompt_files:
        with open(p_file,"r") as pf:
            prompt_categories.append([line.strip() for line in pf.readlines() if line.strip()])

    prompts=[args.conjunction.join(combination) for combination in itertools.product(*prompt_categories)]
    
    pipe=Krea2Pipeline.from_pretrained("krea/Krea-2-Turbo", torch_dtype=torch.bfloat16).to(device)

    feature_extractor=pipeline(
        task="image-feature-extraction",
        model=embedding_model_checkpoints[args.embedding_model],
        device=device
    )

    images=[]
    num_inference_steps=4
    text_list=[]
    feature_list=[]
    for p,text in enumerate(prompts):
        for n in range(args.images_per_prompt):
            generator=torch.Generator(device=device).manual_seed(n)
            img=pipe(
                text,
                generator=generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=8.0,
                height=args.size,
                width=args.size,
            ).images[0]

            embedding=np.mean(feature_extractor(img)[0],axis=0).tolist()

            images.append(img)
            text_list.append(text)
            feature_list.append(embedding)

    if accelerator.is_main_process:
        features=Features({
            "image":HFImage(),
            "text":Value("string"),
            "features":Sequence(Value("float32"))
        })

        dataset=Dataset.from_dict({
            "image":images,
            "text":text_list,
            "features":feature_list,
        },features=features)

        dataset.push_to_hub(repo_id)

if __name__=='__main__':
    print_args(parser)
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