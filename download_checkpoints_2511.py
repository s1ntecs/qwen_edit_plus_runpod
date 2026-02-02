from huggingface_hub import snapshot_download, hf_hub_download


def fetch_model():
    # Download base model to standard HF cache
    print("Downloading Qwen/Qwen-Image-Edit-2511...")
    snapshot_download(repo_id="Qwen/Qwen-Image-Edit-2511")

    # Download Lightning LoRA weights to standard HF cache
    print("Downloading Lightning LoRA weights...")
    hf_hub_download(
        repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
        filename="Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
    )
    print("Download complete!")


if __name__ == "__main__":
    fetch_model()
