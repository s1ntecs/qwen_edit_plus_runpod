import torch

from diffusers import QwenImageEditPlusPipeline


def fetch_model():

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe.load_lora_weights(
        "lightx2v/Qwen-Image-Edit-2511-Lightning",
        weight_name="Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
        adapter_name="lightning",
        weight_dtype=torch.bfloat16,
    )


if __name__ == "__main__":
    fetch_model()
