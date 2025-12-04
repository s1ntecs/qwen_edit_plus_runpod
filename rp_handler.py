import time
import math
import base64

from io import BytesIO
from PIL import Image

from diffusers import (QwenImageEditPlusPipeline,
                       FlowMatchEulerDiscreteScheduler)
from diffusers.models import QwenImageTransformer2DModel
from diffusers.utils import load_image
import torch
import runpod
from runpod.serverless.modules.rp_logger import RunPodLogger

logger = RunPodLogger()

# Load model on startup
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
pipe.load_lora_weights(
    "models/material-transfer_000004769.safetensors",
    adapter_name="material",
    weight_dtype=torch.bfloat16,
)

# --- 3. Lightning-LoRA для скорости ---
# Нужен конкретный weight_name из репо lightx2v/Qwen-Image-Lightning,
# под edit-модель, например 8-step вариант.
pipe.load_lora_weights(
    "lightx2v/Qwen-Image-Lightning",
    weight_name="Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors",
    adapter_name="lightning",
    weight_dtype=torch.bfloat16,
)


pipe.set_adapters(["material", "lightning"],
                  adapter_weights=[1.0, 1.0])

# 3) Фьюзим их в базовые веса (одна матрица вместо базовой+LoRA)
pipe.fuse_lora(adapter_names=["material", "lightning"],
               lora_scale=1.0)

pipe = pipe.to("cuda")


def pil_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def handler(job):
    """
    Runpod handler function. Receives job input and returns output.
    """
    try:
        input_data = job["input"]
        prompt = input_data.get("prompt", "Enhance the image")
        negative_prompt = input_data.get("negative_prompt", "")
        image_urls = input_data.get("images")
        steps = input_data.get("steps", 8)
        cfg_scale = float(input_data.get("cfg_scale", 1.0))

        if not image_urls:
            return {"error": "Missing 'images' parameter."}

        start_time = time.time()
        input_images = []
        for image_url in image_urls:
            input_images.append(load_image(image_url))
        logger.info(f"IMAGES DOWNLOADED FOR: {(time.time()) - start_time}")
        with torch.inference_mode(), torch.autocast("cuda",
                                                    dtype=torch.bfloat16):
            output_image = pipe(image=input_images,
                                num_inference_steps=steps,
                                true_cfg_scale=cfg_scale,
                                negative_prompt=negative_prompt,
                                prompt=prompt).images[0]
            b_64_img = pil_to_b64(output_image)

        return {
            "images_base64": [b_64_img],
            "time": round(time.time() - job["created"],
                          2) if "created" in job else None,
            "steps": steps,
            "seed": "N/A"
        }
    except Exception as e:
        return {"error": str(e)}


# ------------------------- RUN WORKER ------------------------------------ #
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
