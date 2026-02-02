# Use a modern Runpod PyTorch base image
FROM runpod/pytorch:1.0.2-cu1281-torch271-ubuntu2204

# Install dependencies
RUN pip install --no-cache-dir diffusers transformers accelerate safetensors pillow runpod hf_transfer bitsandbytes peft git+https://github.com/huggingface/diffusers

# Install sage-attention for optimized attention (required by LightX2V)
RUN pip install --no-cache-dir sageattention>=1.0.0 || echo "sage-attention not available"

# Copy handler file
WORKDIR /app
COPY rp_handler.py .
# COPY download_checkpoints.py .
COPY download_checkpoints_2511.py .
COPY models ./models

# Hugging Face token for authenticated downloads during build
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# RUN python download_checkpoints.py
RUN python download_checkpoints_2511.py

# Set entrypoint
# CMD ["python", "rp_handler.py"]
CMD ["python", "rp_handler_2511.py"]
