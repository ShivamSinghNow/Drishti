from __future__ import annotations

import sys

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from tbx11k_utils import DATA_DIR, load_records, random_records


MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
PROMPT = "Analyze this chest X-ray image and describe what you observe about the lungs."


def torchvision_is_available() -> bool:
    try:
        import torchvision  # noqa: F401
    except Exception as exc:
        print("ERROR: torchvision is not installed or cannot be imported, but Qwen2-VL processing requires it.")
        print("Install torch, torchvision, and torchaudio together from the ROCm index URL for the target host.")
        print(f"Details: {exc}")
        return False
    return True


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm is not available. Model verification must run on the MI300X ROCm host.")
        print(f"torch version: {torch.__version__}")
        print(f"ROCm version reported by torch: {getattr(torch.version, 'hip', None)}")
        return 1
    if not torchvision_is_available():
        return 1

    records, _, _ = load_records(DATA_DIR)
    train_records = random_records(records, "train", 1)
    if not train_records:
        print(f"ERROR: No training image found under {DATA_DIR}. Run download_dataset.py first.")
        return 1
    image_path = train_records[0].path

    print(f"Loading {MODEL_NAME} ...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": str(image_path.resolve())},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=160)

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"Image: {image_path}")
    print("\nModel output:")
    print(output.strip())
    print(f"\nPeak GPU memory usage: {peak_gb:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
