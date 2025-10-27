import numpy as np
import time
import functools
import torch
import os
import json
import argparse, inspect

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from datasets import load_dataset, Dataset, load_from_disk, Array2D
from torch.nn import functional as F
from datasets import Dataset, Features, Value, Sequence, Array2D, Array3D, Image
from pathlib import Path
from PIL import Image as PILImage
from pycocotools import mask as maskUtils


def timed_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"Time: {elapsed:.6f}s\n")
        return result

    return wrapper


def resize_pad_mask(mask, target_len=256):
    mask = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, h, w]
    resize = ResizeLongestSide(target_length=1024)
    resized_mask = resize.apply_image_torch(mask)
    h, w = resized_mask.shape[-2:]
    padh, padw = 1024 - h, 1024 - w
    padded_mask = F.pad(resized_mask, (0, padw, 0, padh))
    output_mask = F.interpolate(padded_mask, size=(target_len, target_len), mode="bilinear", align_corners=False)

    return output_mask.squeeze(0).squeeze(0).clamp_(0, 1).to(torch.float16)


@timed_function
def gen_sam_info_from_coco_anno(
        sam_ckpt,
        anno_path,
        save_anno_dir,
        save_sam_embed_dir,
        image_dir=None,
        input_saved_by_datasets=False,
        data_files=None,
        max_num_items=2000,
        split=None, st=None, ed=None
):
    """
    Turn the polygonal segmentation result to mask.
    Generate SAM embedding, unpadded_size and
    mask_float_256 which is resized & padded result of GT mask.
    sam_embed is saved for sharing between datasets, in save_sam_embed_dir.
    """
    features = Features({
        "image": Image(decode=True),
        "image_filename": Value("string"),
        "text": Sequence(Value("string")),
        "mask": Sequence(Sequence(Value("bool"))),
        "image_id": Value("string"),
        "ann_id": Value("string"),
        "bbox": Sequence(Value("float32"), length=4),
        "original_hw": Sequence(Value("int32"), length=2),
        "unpadded_hw": Sequence(Value("int32"), length=2),
        "mask_float_256": Array2D(shape=(256, 256), dtype="float16"),
        "embed_path": Value("string"),
    })
    data_items = []
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt)
    sam.to("cuda:0").eval()
    predictor = SamPredictor(sam)
    if input_saved_by_datasets:
        ds = load_from_disk(anno_path)
    else:
        ds = load_dataset(anno_path, data_files=data_files) if data_files is not None else load_dataset(anno_path)
    if split is not None:
        ds = ds[split]
    start = 0 if st is None else st
    end = len(ds) if ed is None else ed

    existing_embeds = set()
    for b in range(10):
        for p in (Path(save_sam_embed_dir) / f"b{b}").glob("*.pt"):
            existing_embeds.add(p.name)
    with torch.no_grad():
        i_start = start
        for i in range(start, end):
            anno = ds[i]
            image_filename = Path(anno["image_path"]).name
            image = PILImage.open(os.path.join(image_dir, image_filename)).convert("RGB")
            image_np = np.array(image)

            embed_name = image_filename.rsplit('.', 1)[0]+".pt"
            embed_path = os.path.join("b" + str(anno["image_id"])[-1], embed_name)
            if embed_name not in existing_embeds:
                predictor.set_image(image_np)
                image_embed = predictor.get_image_embedding().squeeze(0).to(torch.float16).cpu()
                torch.save(image_embed, os.path.join(save_sam_embed_dir, embed_path))
                orig_size, input_size = predictor.original_size, predictor.input_size
                if not (orig_size[0] == image_np.shape[0] and orig_size[1] == image_np.shape[1]):
                    print("Warning! original size not matched")
            else:
                resize = ResizeLongestSide(target_length=1024)
                orig_size = (image_np.shape[0], image_np.shape[1])
                input_size = resize.get_preprocess_shape(orig_size[0], orig_size[1], resize.target_length)

            raw_anno = json.loads(anno["raw_anns"])
            rles = maskUtils.frPyObjects(raw_anno["segmentation"], orig_size[0], orig_size[1])
            mask = maskUtils.decode(maskUtils.merge(rles))
            if mask.ndim == 3:  # 多 RLE 通道时做并集
                mask = np.any(mask, axis=2)
            mask = mask.tolist()

            mask_float_256 = resize_pad_mask(mask, target_len=256).cpu().numpy().astype(np.float16)

            if i % 100 == 0:
                print_split = split if split else "whole"
                print(f"[{i}/{len(ds)}] [{print_split}] Examples processed...Start: {start}...End: {end}", flush=True)

            data_items.append({
                "image": image,
                "image_filename": image_filename,
                "text": anno["captions"],
                "mask": mask,
                "image_id": str(anno["image_id"]),
                "ann_id": str(anno["ann_id"]),
                "bbox": anno["bbox"],
                "original_hw": [int(x) for x in orig_size],
                "unpadded_hw": [int(x) for x in input_size],
                "mask_float_256": mask_float_256,
                "embed_path": embed_path
            })
            if len(data_items) >= max_num_items and end-start != max_num_items:
                save_dir = os.path.join(save_anno_dir, f"{i_start:05d}_{i+1:05d}")
                i_start = i + 1
                dataset = Dataset.from_list(data_items, features=features)
                dataset.save_to_disk(save_dir)
                print(f"saved: {save_dir}")
                data_items = []
                del dataset

        save_dir = save_anno_dir if end-start <= max_num_items else os.path.join(save_anno_dir, f"{i_start:05d}_{end:05d}")
        if data_items:
            dataset = Dataset.from_list(data_items, features=features)
            dataset.save_to_disk(save_dir)
            print(f"saved: {save_dir}")


@timed_function
def gen_sam_info_from_reasonseg(
        sam_ckpt,
        anno_path,
        save_anno_dir,
        input_saved_by_datasets=False,
        data_files=None,
        split=None, st=None, ed=None,
):
    """
    Turn the polygonal segmentation result to mask.
    Generate SAM embedding, unpadded_size and
    mask_float_256 which is resized & padded result of GT mask.
    sam_embed is saved for sharing between datasets, in save_sam_embed_dir.
    """
    features = Features({
        "image": Image(decode=True),
        "image_filename": Value("string"),
        "text": Value("string"),
        "mask": Sequence(Sequence(Value("bool"))),
        "image_id": Value("int32"),
        "ann_id": Value("int32"),
        "original_hw": Sequence(Value("int32"), length=2),
        "unpadded_hw": Sequence(Value("int32"), length=2),
        "mask_float_256": Array2D(shape=(256, 256), dtype="float16"),
        "sam_embed": Array3D(shape=(256, 64, 64), dtype="float16"),
    })
    data_items = []
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt)
    sam.to("cuda:0").eval()
    predictor = SamPredictor(sam)
    if input_saved_by_datasets:
        ds = load_from_disk(anno_path)
    else:
        ds = load_dataset(anno_path, data_files=data_files) if data_files is not None else load_dataset(anno_path)
    if split is not None:
        ds = ds[split]
    start = 0 if st is None else st
    end = len(ds) if ed is None else ed

    with torch.no_grad():
        for idx in range(start, end):
            anno = ds[idx]
            image = anno["image"].convert("RGB")
            image_np = np.array(image)
            image_filename = str(anno["image_id"])

            predictor.set_image(image_np)
            sam_embed = predictor.get_image_embedding().squeeze(0).cpu().numpy().astype(np.float16)
            orig_size, input_size = predictor.original_size, predictor.input_size
            if not (orig_size[0] == image_np.shape[0] and orig_size[1] == image_np.shape[1]):
                print("Warning! original size not matched")

            mask = anno["mask"]
            if not isinstance(mask, list):
                print("Warning! Mask in annotation is not list")
            mask_float_256 = resize_pad_mask(mask, target_len=256).cpu().numpy().astype(np.float16)

            if idx % 100 == 0:
                print_split = split if split else "whole"
                print(f"[{idx}/{len(ds)}] [{print_split}] Examples processed...Start: {start}...End: {end}", flush=True)

            data_items.append({
                "image": image,
                "image_filename": image_filename,
                "text": str(anno["text"]),
                "mask": mask,
                "image_id": idx,
                "ann_id": idx,
                "original_hw": [int(x) for x in orig_size],
                "unpadded_hw": [int(x) for x in input_size],
                "mask_float_256": mask_float_256,
                "sam_embed": sam_embed,
            })

    save_dir = save_anno_dir if (st is None and ed is None) else (
        os.path.join(save_anno_dir, f"{start:05d}_{end:05d}"))
    if data_items:
        dataset = Dataset.from_list(data_items, features=features)
        dataset.save_to_disk(save_dir)
        print(f"saved: {save_dir}")

if __name__ == "__main__":
    anno_path = "data/reasonseg/test"

    # data_files = {
    #     "train": sorted(str(p) for p in root.glob("train-*.parquet")),
    #     "validation": sorted(str(p) for p in root.glob("validation-*.parquet")),
    #     "test": sorted(str(p) for p in root.glob("test-*.parquet")),
    #     # "testB": sorted(str(p) for p in root.glob("testB-*.parquet")),
    # }
    gen_sam_info_from_reasonseg(
        sam_ckpt="pretrained_models/sam_vit_h_4b8939.pth",
        anno_path=anno_path,
        save_anno_dir="data/reasonseg/reasonseg_test",
        input_saved_by_datasets=False,
        split="test", st=400,
    )
