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
pipe.load_lora_weights("models/material-transfer_000004769.safetensors",
                       weight_dtype=torch.bfloat16)
# pipe.lora_scale = 0.5

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
