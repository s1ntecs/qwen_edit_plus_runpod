import time
import runpod
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image
import torch
from io import BytesIO
import base64
from PIL import Image

# Load model on startup
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16
).to("cuda")
pipe.load_lora_weights(
    "models/material-transfer_000004769.safetensors",
    adapter_name="material",
    weight_dtype=torch.bfloat16,
)

# (если вдруг не сработает с прямым путём, можно так:)
# pipe.load_lora_weights(
#     "models",
#     weight_name="material-transfer_000004769.safetensors",
#     adapter_name="material",
#     weight_dtype=torch.bfloat16,
# )

# --- 3. Lightning-LoRA для скорости ---
# Нужен конкретный weight_name из репо lightx2v/Qwen-Image-Lightning,
# под edit-модель, например 8-step вариант.
pipe.load_lora_weights(
    "lightx2v/Qwen-Image-Lightning",
    weight_name="Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors",
    adapter_name="lightning",
    weight_dtype=torch.bfloat16,
)



# По умолчанию включим только твою LoRA (HQ-режим)
pipe.set_adapters(["material"], adapter_weights=[1.0])
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
        steps = input_data.get("steps", 20)
        cfg_scale = input_data.get("cfg_scale", 4)

        if not image_urls:
            return {"error": "Missing 'images' parameter."}

        input_images = []
        for image_url in image_urls:
            input_images.append(load_image(image_url))
        with torch.inference_mode():
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
