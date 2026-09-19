"""Download local models expected by config.py into data/models."""
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parent.parent
SAFETENSORS = ROOT / "data" / "models" / "safetensors"
GGUF = ROOT / "data" / "models" / "gguf"


def main():
    resnet_dir = SAFETENSORS / "microsoft" / "resnet-50"
    print("Downloading microsoft/resnet-50 ...")
    snapshot_download(
        repo_id="microsoft/resnet-50",
        local_dir=str(resnet_dir),
        allow_patterns=["model.safetensors", "config.json", "*.json"],
    )

    embed_dir = SAFETENSORS / "sentence-transformers" / "all-MiniLM-L6-v2"
    print("Downloading sentence-transformers/all-MiniLM-L6-v2 ...")
    snapshot_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        local_dir=str(embed_dir),
    )

    gguf_dir = GGUF / "LoveSeaW" / "Qwen2.5-1.5b-instruct-gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading Qwen2.5-1.5B GGUF ...")
    hf_hub_download(
        repo_id="LoveSeaW/Qwen2.5-1.5b-instruct-gguf",
        filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
        local_dir=str(gguf_dir),
    )
    print("All models downloaded.")


if __name__ == "__main__":
    main()
