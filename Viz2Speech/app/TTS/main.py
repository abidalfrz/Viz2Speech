from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from services.generator import generator
import soundfile as sf
import io
import tempfile
import uvicorn
import os

app = FastAPI(title="TTS API")

@app.post("/generate")
async def generate_audio(text: str = Form(...), ref_audio: UploadFile = File(None)):
    ref_audio_path = None
    temp_path = ""
    
    try:
        if ref_audio is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(await ref_audio.read())
                temp_path = temp_file.name
                ref_audio_path = temp_path

        sr, audio_numpy = await run_in_threadpool(
            generator.generate_speech, 
            text, 
            ref_audio_path
        )
        
        wav_io = io.BytesIO()
        sf.write(wav_io, audio_numpy, sr, format='WAV')

        audio_bytes = wav_io.getvalue()

        return Response(
            content=audio_bytes,
            media_type="audio/wav"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


    
    
