from __future__ import annotations

import sys

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from tbx11k_utils import DATA_DIR, load_records, random_records


MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
PROMPT = "Analyze this chest X-ray image and describe what you observe about the lungs."


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm is not available. Model verification must run on the MI300X ROCm host.")
        print(f"torch version: {torch.__version__}")
        print(f"ROCm version reported by torch: {getattr(torch.version, 'hip', None)}")
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
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=160)

    input_length = inputs["input_ids"].shape[1]
    generated_ids = generated_ids[:, input_length:]
    output = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"Image: {image_path}")
    print("\nModel output:")
    print(output.strip())
    print(f"\nPeak GPU memory usage: {peak_gb:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
