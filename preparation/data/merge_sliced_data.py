import shutil

from pathlib import Path
from datasets import load_from_disk, concatenate_datasets, DatasetDict


def merge_sliced_data(dataset_dir: str, out_dir: str, overwrite: bool = False) -> str:
    """
    Merge a directory composed of HF dataset slices,
    where data is preprocessed, e.g. by offline generating sam information.
    """
    root = Path(dataset_dir)

    parts = [p for p in root.iterdir()
             if p.is_dir() and p.name != "sam_embed"
             and (p / "dataset_info.json").exists()]
    if not parts:
        raise FileNotFoundError(f"No dataset slices found under {root}")
    parts.sort()
    out_dir = Path(out_dir)

    merged = merge_slice_list(parts)

    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} already exists (set overwrite=True to replace).")
        shutil.rmtree(out_dir)

    if isinstance(merged, DatasetDict):
        for split in merged.keys():
            if getattr(merged[split], "info", None) is not None:
                merged[split].info.splits = None  # avoid the chaos after merging
                merged[split].info.download_checksums = None  # avoid leaking personal information
    else:
        if getattr(merged, "info", None) is not None:
            merged.info.splits = None
            merged.info.download_checksums = None

    merged.save_to_disk(str(out_dir))
    return str(out_dir)


def merge_slice_list(slices):
    """
    Merge a list of HF dataset slices where data is preprocessed.
    """
    merged = load_from_disk(str(slices[0]))
    for p in slices[1:]:
        d = load_from_disk(str(p))
        if isinstance(merged, DatasetDict):
            for split in merged.keys():
                merged[split] = concatenate_datasets([merged[split], d[split]])
        else:
            merged = concatenate_datasets([merged, d])
    return merged


if __name__ == "__main__":
    merge_sliced_data("data/refcoco/refcoco_sliced_train", "data/refcoco/refcoco_train")