import torch
import math

from diffusers import (QwenImageEditPlusPipeline,
                       FlowMatchEulerDiscreteScheduler)
from diffusers.models import QwenImageTransformer2DModel

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
    model = QwenImageTransformer2DModel.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )
    # Configure scheduler for Lightning (shift=3 for distillation)
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        transformer=model,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16
    )
    pipe.load_lora_weights("models/material-transfer_000004769.safetensors",
                           adapter_name="material",
                           weight_dtype=torch.bfloat16)
    pipe.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning",
        weight_name="Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors",
        adapter_name="lightning",
        weight_dtype=torch.bfloat16,
    )


if __name__ == "__main__":
    # fetch_lora()
    fetch_model()
    # get_pipeline()
