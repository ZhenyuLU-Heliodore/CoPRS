import os
import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor


def register_special_tokens(
    model_path: str,
    save_path: str,
    special_tokens: dict,
    dtype=torch.bfloat16,
):
    os.makedirs(save_path, exist_ok=True)

    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    old_vocab_size = len(tokenizer)
    print(f"Original tokenizer size: {old_vocab_size}")

    num_added = 0
    for k, v in special_tokens.items():
        added = tokenizer.add_tokens(v, special_tokens=True)
        num_added += added
    tokenizer.add_special_tokens(special_tokens_dict=special_tokens, replace_additional_special_tokens=False)

    new_vocab_size = len(tokenizer)
    print(f"New tokenizer size: {new_vocab_size} (added {num_added} tokens)")

    print(f"Loading model from: {model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    if num_added > 0:
        num_embeddings = model.get_input_embeddings().num_embeddings
        print(f"Previous model embeddings {num_embeddings}")
        model.resize_token_embeddings(num_embeddings + num_added)

    for _, ref_token in special_tokens.items():
        ref_token_id = tokenizer.convert_tokens_to_ids(ref_token)
        print(f"{ref_token} → ID: {ref_token_id}")

    # processor handling
    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(proc, "tokenizer"):
        proc.tokenizer = tokenizer
    elif hasattr(proc, "text_tokenizer"):
        proc.text_tokenizer = tokenizer

    model.save_pretrained(save_path, safe_serialization=True)
    proc.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Token addition complete.")


if __name__ == "__main__":
    register_special_tokens(
        model_path="Qwen2.5-VL-7B-Instruct",
        save_path="pretrained_models/CoPRS-7B-init",
        special_tokens={"additional_special_tokens": ["<REF_POS>"]},
    )
