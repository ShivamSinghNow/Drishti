window.DRISHTI_API = {
  analyzeUrl: "/api/analyze",
};

window.DRISHTI_MODEL = {
  name: "Drishti run4 vision-LoRA",
  checkpointStep: 4950,
  baseModel: "Qwen/Qwen2-VL-7B-Instruct",
  adapterRepo: "ShivSingh123/drishti-qlora-run4-vision-lora-ablation",
  adapterPath: "checkpoints/checkpoint-4950",
  ggufRepo: "ShivSingh123/drishti-qwen2vl-run4-gguf",
  ggufFile: "drishti-qwen2vl-run4-quantized-q4_k_m.gguf",
  mmprojFile: "mmproj-drishti-qwen2vl-run4-quantized-f16.gguf",
  labels: ["active_tb", "healthy", "sick_but_non_tb"],
  outputFormat: "Classification: <label>",
  validation: {
    split: "TBX11K val",
    samples: 1800,
    accuracy: 0.987222,
    macroF1: 0.978787,
    activeTbF1: 0.953317,
    healthyF1: 0.996881,
    sickNonTbF1: 0.986164,
  },
  gguf: {
    quant: "Q4_K_M",
    size: "4.68 GB",
    mmprojSize: "1.26 GiB",
    architecture: "qwen2vl",
    params: "8B",
  },
  hosting: {
    gpu: "NVIDIA A10G",
    quant: "4-bit NF4 LoRA",
    imageSize: "512×512",
    maxUpload: "12 MB",
    coldStartNote: "First request after idle may take 2–3 min while the model loads.",
  },
  site: {
    vercel: "https://drishti-demo.vercel.app",
    github: "https://github.com/ShivamSinghNow/Drishti",
    hfGguf: "https://huggingface.co/ShivSingh123/drishti-qwen2vl-run4-gguf",
    hfAdapter: "https://huggingface.co/ShivSingh123/drishti-qlora-run4-vision-lora-ablation",
  },
};
