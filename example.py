from fastapi import FastAPI, Response
from pydantic import BaseModel
import io
from bark import SAMPLE_RATE, generate_audio, preload_models
import torch
from pydub import AudioSegment
import soundfile as sf

app = FastAPI()

# 预加载模型（启动时）
preload_models()

class TTSRequest(BaseModel):
    text: str

@app.post("/tts")
async def tts(request: TTSRequest):
    # 将长文本分段处理
    segment_length = 1000  # 每段的字符长度
    segments = [request.text[i:i+segment_length] for i in range(0, len(request.text), segment_length)]
    
    audio_pieces = []
    for segment in segments:
        audio_array = generate_audio(segment)
        audio_pieces.append(audio_array)
    
    # 合并音频片段
    full_audio = np.concatenate(audio_pieces)
    
    # 转wav写入内存
    wav_io = io.BytesIO()
    sf.write(wav_io, full_audio, samplerate=SAMPLE_RATE, subtype='PCM_16', format='WAV')
    wav_io.seek(0)

    # wav -> mp3
    audio = AudioSegment.from_wav(wav_io)
    mp3_io = io.BytesIO()
    audio.export(mp3_io, format="mp3")
    mp3_io.seek(0)

    return Response(content=mp3_io.read(), media_type="audio/mpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2025)
