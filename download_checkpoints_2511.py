import torch

from diffusers import QwenImageEditPlusPipeline
from huggingface_hub import snapshot_download, hf_hub_download


def fetch_model():

    # Download base model to standard HF cache
    snapshot_download(repo_id="Qwen/Qwen-Image-Edit-2511")

    # Download Lightning LoRA weights to standard HF cache
    hf_hub_download(
        repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
        filename="Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
    )

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        torch_dtype=torch.bfloat16
    )
    pipe.load_lora_weights(
        "lightx2v/Qwen-Image-Edit-2511-Lightning",
        weight_name="Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
        adapter_name="lightning",
        weight_dtype=torch.bfloat16,
    )


if __name__ == "__main__":
    fetch_model()
