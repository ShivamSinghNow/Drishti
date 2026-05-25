"""Modal GPU service for the Drishti chest X-ray demo."""

import modal

APP_NAME = "drishti-demo"
MODEL_VOLUME = modal.Volume.from_name("drishti-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "transformers==5.7.0",
        "peft==0.19.1",
        "bitsandbytes>=0.45.0",
        "accelerate",
        "Pillow",
        "fastapi[standard]==0.115.12",
        "python-multipart",
        "huggingface_hub",
    )
    .add_local_python_source("demo")
)

app = modal.App(APP_NAME)


@app.cls(
    gpu="A10G",
    image=image,
    timeout=900,
    scaledown_window=300,
    volumes={"/root/.cache/huggingface": MODEL_VOLUME},
)
class DrishtiClassifier:
    @modal.enter()
    def load(self) -> None:
        from demo.inference import load_model_and_processor

        self.model, self.processor = load_model_and_processor()

    @modal.method()
    def analyze(self, image_bytes: bytes) -> dict:
        from demo.inference import analyze_image_bytes

        return analyze_image_bytes(self.model, self.processor, image_bytes)


@app.function(image=image)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware

    api = FastAPI(title="Drishti Demo API", version="1.0.0")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    classifier = DrishtiClassifier()

    @api.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": APP_NAME}

    @api.post("/analyze")
    async def analyze(image: UploadFile = File(...)) -> dict:
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Upload must be an image file.")

        payload = await image.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Empty image upload.")
        if len(payload) > 12 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image must be smaller than 12 MB.")

        try:
            return await classifier.analyze.remote.aio(payload)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - surfaced to client as 500
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return api
