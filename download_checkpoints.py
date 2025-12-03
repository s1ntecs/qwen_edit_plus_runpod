import torch

from diffusers import QwenImageEditPlusPipeline

from huggingface_hub import hf_hub_download, snapshot_download


def fetch_lora():
    """
    Скачивает LoRA
    """
    hf_hub_download(
        repo_id='flymy-ai/qwen-image-edit-inscene-lora',
        filename='flymy_qwen_image_edit_inscene_lora.safetensors',
        local_dir='./',
        local_dir_use_symlinks=False
    )


# def fetch_model():
#     snapshot_download(
#         repo_id="Qwen/Qwen-Image-Edit",
#         # local_dir="checkpoints/Qwen-Image-Edit"
#         # ничего не загружаем в память; просто кладём файлы
#     )


def fetch_model():
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16
    )
    pipe.load_lora_weights("models/material-transfer_000004769.safetensors",
                           weight_dtype=torch.bfloat16)


if __name__ == "__main__":
    # fetch_lora()
    fetch_model()
    # get_pipeline()
