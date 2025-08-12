from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from transformers import AutoProcessor, BarkModel
import torch
import soundfile as sf
from pydub import AudioSegment
import io
import uvicorn

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark").to(device)

class TTSRequest(BaseModel):
    text: str

@app.post("/tts")
async def tts(request: TTSRequest):
    voice_preset = "v2/en_speaker_6"
    
    inputs = processor(request.text, voice_preset=voice_preset, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()

    wav_io = io.BytesIO()
    sf.write(wav_io, audio_array, samplerate=24000, subtype='PCM_16')
    wav_io.seek(0)

    audio = AudioSegment.from_wav(wav_io)
    mp3_io = io.BytesIO()
    audio.export(mp3_io, format="mp3")
    mp3_io.seek(0)

    return Response(content=mp3_io.read(), media_type="audio/mpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2025)
