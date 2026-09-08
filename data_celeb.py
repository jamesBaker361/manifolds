# turns celeba-hq into a hf dataset with image, text, embedding columns

import time
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict, Features, Value, Image as HFImage, Sequence
from transformers import pipeline

from experiment_helpers.gpu_details import print_details
from experiment_helpers.argprint import print_args
from experiment_helpers.init_helpers import default_parser, repo_api_init

parser = default_parser(
    {"repo_id":"jlbaker361/celeb"}
)

parser.add_argument("--embedding_model", type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m")

def build_split(src_split, feature_extractor):
    label_names = src_split.features["label"].names

    images = []
    text_list = []
    embedding_list = []
    for row in src_split:
        img = row["image"]
        text = label_names[row["label"]]
        embedding = np.mean(feature_extractor(img)[0], axis=0).tolist()

        images.append(img)
        text_list.append(text)
        embedding_list.append(embedding)

    features = Features({
        "image": HFImage(),
        "text": Value("string"),
        "embedding": Sequence(Value("float32")),
    })

    return Dataset.from_dict({
        "image": images,
        "text": text_list,
        "embedding": embedding_list,
    }, features=features)

def main(args):
    api, accelerator, device = repo_api_init(args)
    repo_id: str = args.repo_id

    src_dataset = load_dataset("mattymchen/celeba-hq")

    feature_extractor = pipeline(
        task="image-feature-extraction",
        model=args.embedding_model,
        device=device
    )

    train_dataset = build_split(src_dataset["train"], feature_extractor)
    validation_dataset = build_split(src_dataset["validation"], feature_extractor)

    if accelerator.is_main_process:
        dataset = DatasetDict({
            "train": train_dataset,
            "validation": validation_dataset,
        })

        dataset.push_to_hub(repo_id)

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
