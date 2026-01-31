import argparse
import os
import uuid
import wave
from io import BytesIO

import torch
import soundfile as sf
import numpy as np
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from qwen_tts import Qwen3TTSModel


def _resolve_torch_dtype(dtype_name: str, device_map: str) -> torch.dtype:
    if ("cpu" in device_map.lower()) or (not torch.cuda.is_available()):
        return torch.float32
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(dtype_name.lower(), torch.bfloat16)


class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    language: str | None = Field(default=None, description="Language name or Auto")
    speaker: str | None = Field(default=None, description="CustomVoice speaker")
    instruct: str | None = Field(default=None, description="Natural-language instruction")
    ref_audio: str | None = Field(default=None, description="Voice clone reference audio path or URL")
    ref_text: str | None = Field(default=None, description="Transcript for reference audio")
    media_type: str | None = Field(default=None, description="wav or raw")
    streaming_mode: bool | None = Field(default=None, description="Enable streaming response")


class GPTSoVITSRequest(BaseModel):
    text: str | None = None
    text_lang: str | None = None
    ref_audio_path: str | None = None
    prompt_lang: str | None = None
    prompt_text: str | None = None
    media_type: str | None = None
    streaming_mode: bool | None = None
    speaker: str | None = None
    instruct: str | None = None


def _write_wav_bytes(wav, sr: int) -> bytes:
    buf = BytesIO()
    sf.write(buf, wav, sr, format="wav")
    buf.seek(0)
    return buf.read()

def _infer_channels(wav: np.ndarray) -> int:
    if wav.ndim == 1:
        return 1
    if wav.ndim == 2:
        return wav.shape[1]
    raise ValueError("Unsupported audio shape")


def _to_pcm16_bytes(wav: np.ndarray) -> bytes:
    if wav.dtype == np.int16:
        pcm = wav
    else:
        wav = np.clip(wav, -1.0, 1.0)
        pcm = (wav * 32767.0).astype(np.int16)
    return pcm.tobytes()


def _wave_header(channels: int, sample_width: int, sample_rate: int) -> bytes:
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(b"")
    wav_buf.seek(0)
    return wav_buf.read()


def _stream_bytes(data: bytes, chunk_size: int = 4096):
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


def _build_model(model_id: str, device_map: str, dtype_name: str, attn_impl: str | None):
    dtype = _resolve_torch_dtype(dtype_name, device_map)
    kwargs = {
        "device_map": device_map,
        "dtype": dtype,
    }
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    return Qwen3TTSModel.from_pretrained(model_id, **kwargs)


def create_app(
    model_id: str,
    mode: str,
    device_map: str,
    dtype_name: str,
    attn_impl: str | None,
    default_language: str | None,
    default_speaker: str | None,
) -> FastAPI:
    app = FastAPI()
    model = _build_model(model_id, device_map, dtype_name, attn_impl)

    def _run_tts(req: TTSRequest):
        language = req.language or default_language or "Auto"
        if mode == "custom_voice":
            speaker = req.speaker or default_speaker
            if not speaker:
                raise HTTPException(status_code=400, detail="speaker is required for custom_voice")
            wavs, sr = model.generate_custom_voice(
                text=req.text,
                language=language,
                speaker=speaker,
                instruct=req.instruct or "",
            )
        elif mode == "voice_design":
            wavs, sr = model.generate_voice_design(
                text=req.text,
                language=language,
                instruct=req.instruct or "",
            )
        elif mode == "voice_clone":
            if not req.ref_audio:
                raise HTTPException(status_code=400, detail="ref_audio is required for voice_clone")
            wavs, sr = model.generate_voice_clone(
                text=req.text,
                language=language,
                ref_audio=req.ref_audio,
                ref_text=req.ref_text or "",
            )
        else:
            raise HTTPException(status_code=400, detail="unsupported mode")

        media_type = (req.media_type or "wav").lower()
        streaming_mode = bool(req.streaming_mode)
        if media_type not in ["wav", "raw"]:
            raise HTTPException(status_code=400, detail="media_type must be wav or raw")

        wav = wavs[0]
        if streaming_mode:
            channels = _infer_channels(wav)
            pcm_bytes = _to_pcm16_bytes(wav)

            def streaming_generator():
                if media_type == "wav":
                    yield _wave_header(channels, 2, sr)
                for chunk in _stream_bytes(pcm_bytes):
                    yield chunk

            return StreamingResponse(
                streaming_generator(),
                media_type=f"audio/{media_type}",
            )

        if media_type == "raw":
            pcm_bytes = _to_pcm16_bytes(wav)
            return Response(pcm_bytes, media_type="audio/raw")

        audio_bytes = _write_wav_bytes(wav, sr)
        return Response(audio_bytes, media_type="audio/wav")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "model_id": model_id, "mode": mode}

    @app.post("/tts/predict")
    def tts_predict(req: TTSRequest):
        return _run_tts(req)

    @app.post("/tts/stream-predict")
    def tts_stream_predict(req: TTSRequest):
        # Streaming endpoint for compatibility; uses the same handler.
        req.streaming_mode = True if req.streaming_mode is None else req.streaming_mode
        return _run_tts(req)

    @app.get("/tts")
    def gpt_sovits_tts_get(
        text: str | None = None,
        text_lang: str | None = None,
        ref_audio_path: str | None = None,
        prompt_lang: str | None = None,
        prompt_text: str | None = None,
        speaker: str | None = None,
        instruct: str | None = None,
        media_type: str | None = None,
        streaming_mode: bool | None = None,
    ):
        req = GPTSoVITSRequest(
            text=text,
            text_lang=text_lang,
            ref_audio_path=ref_audio_path,
            prompt_lang=prompt_lang,
            prompt_text=prompt_text,
            speaker=speaker,
            instruct=instruct,
            media_type=media_type,
            streaming_mode=streaming_mode,
        )
        return _handle_gpt_sovits_request(req)

    @app.post("/tts")
    def gpt_sovits_tts_post(req: GPTSoVITSRequest):
        return _handle_gpt_sovits_request(req)

    @app.get("/control")
    def control(command: str | None = None):
        # Compatibility endpoint: GPT-SoVITS supports restart/exit. This server is stateless.
        if not command:
            raise HTTPException(status_code=400, detail="command is required")
        return {"message": "ignored", "command": command}

    @app.get("/set_gpt_weights")
    def set_gpt_weights(weights_path: str | None = None):
        # Compatibility endpoint: no-op for Qwen3-TTS
        return {"message": "ignored", "weights_path": weights_path}

    @app.get("/set_sovits_weights")
    def set_sovits_weights(weights_path: str | None = None):
        # Compatibility endpoint: no-op for Qwen3-TTS
        return {"message": "ignored", "weights_path": weights_path}

    @app.post("/set_refer_audio")
    async def set_refer_audio(audio_file: UploadFile = File(...)):
        if not audio_file.content_type or not audio_file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="file type is not supported")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        upload_dir = os.path.join(base_dir, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        suffix = os.path.splitext(audio_file.filename or "")[1] or ".wav"
        file_name = f"ref_{uuid.uuid4().hex}{suffix}"
        save_path = os.path.join(upload_dir, file_name)
        content = await audio_file.read()
        with open(save_path, "wb") as f:
            f.write(content)
        return JSONResponse(status_code=200, content={"path": save_path})

    def _handle_gpt_sovits_request(req: GPTSoVITSRequest):
        if not req.text:
            raise HTTPException(status_code=400, detail="text is required")
        language = req.text_lang or req.prompt_lang or default_language or "Auto"
        gpt_req = TTSRequest(
            text=req.text,
            language=language,
            speaker=req.speaker,
            instruct=req.instruct,
            ref_audio=req.ref_audio_path,
            ref_text=req.prompt_text,
            media_type=req.media_type,
            streaming_mode=req.streaming_mode,
        )
        return _run_tts(gpt_req)

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-TTS minimal API server")
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["custom_voice", "voice_design", "voice_clone"], required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9888)
    parser.add_argument("--device-map", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--default-language", type=str, default="Auto")
    parser.add_argument("--default-speaker", type=str, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    app = create_app(
        model_id=args.model_id,
        mode=args.mode,
        device_map=args.device_map,
        dtype_name=args.dtype,
        attn_impl=args.attn_implementation,
        default_language=args.default_language,
        default_speaker=args.default_speaker,
    )
    uvicorn.run(app=app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
