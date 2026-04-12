from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
import io
import uvicorn
import torch
from services.captioner import captioner

app = FastAPI(title="Image Captioning API")

@app.post("/caption")
async def generate_caption(image: UploadFile = File(...), mode: str = Form("fast")):
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

    try:
        print(f"Generating caption in {mode} mode...")
        print(f"Using device: {captioner.device}")
        if mode == "fast":
            caption = await run_in_threadpool(captioner.generate_caption, pil_image, max_size=224, max_new_tokens=512)
        else:
            caption = await run_in_threadpool(captioner.generate_caption, pil_image, max_size=720, max_new_tokens=1024)
        
        return {"caption": caption}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate caption: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)