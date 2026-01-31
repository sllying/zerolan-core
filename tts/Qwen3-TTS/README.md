# Qwen3-TTS Standalone Service (GPT-SoVITS Compatible)

This folder provides a **standalone** Qwen3‑TTS service so you can replace GPT‑SoVITS **without changing the main `zerolan-core` code**. It mimics the GPT‑SoVITS HTTP API (`/tts`, `/control`, `/set_gpt_weights`, `/set_sovits_weights`) while running Qwen3‑TTS under the hood.

> Goal: "seamless replacement" for clients that already call GPT‑SoVITS.

---

## 1. Prerequisites

- Python 3.10+ recommended (3.12 works)
- A CUDA GPU is recommended for real-time or large models
- A working C++ build toolchain is recommended for best performance (optional)

Install dependencies (Windows example):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U qwen-tts torch soundfile fastapi uvicorn
```

Optional (if your GPU supports it):

```powershell
pip install -U flash-attn --no-build-isolation
```

---

## 2. Choose a Model and Mode

You must choose **one** model and **one** mode for the server instance:

### Recommended mapping

| Mode | Model ID |
|---|---|
| `custom_voice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` or `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` |
| `voice_design` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |
| `voice_clone` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` or `Qwen/Qwen3-TTS-12Hz-0.6B-Base` |

**Notes**
- `custom_voice` supports fixed speakers (like "Vivian", "Ryan", etc).
- `voice_clone` needs a reference audio path or URL.
- One server instance = one model + one mode.

---

## 3. Start the Server

From repository root:

```powershell
python tts/Qwen3-TTS/qwen3_tts_api.py ^
  --model-id Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice ^
  --mode custom_voice ^
  --default-speaker Vivian ^
  --device-map cuda:0 ^
  --host 0.0.0.0 ^
  --port 9888
```

CPU mode:

```powershell
python tts/Qwen3-TTS/qwen3_tts_api.py ^
  --model-id Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice ^
  --mode custom_voice ^
  --device-map cpu ^
  --dtype float32 ^
  --port 9888
```

---

## 4. GPT‑SoVITS Compatibility

This server exposes **GPT‑SoVITS compatible endpoints**:

- `GET /tts`
- `POST /tts`
- `POST /set_refer_audio` (upload ref audio, returns server path)
- `GET /control` (no-op)
- `GET /set_gpt_weights` (no-op)
- `GET /set_sovits_weights` (no-op)

### Field mapping

| GPT‑SoVITS field | Qwen3‑TTS field |
|---|---|
| `text` | `text` |
| `text_lang` | `language` |
| `ref_audio_path` | `ref_audio` |
| `prompt_text` | `ref_text` |
| `prompt_lang` | (ignored, fallback to `text_lang`) |
| `speaker` | `speaker` (for `custom_voice` mode only) |
| `instruct` | `instruct` |

### Known differences

1. **`media_type` is always WAV**  
   The compatibility layer accepts `media_type`. This server supports `wav` and `raw`.
2. **Streaming**  
   When `streaming_mode=true`, `/tts` returns **Transfer-Encoding: chunked** and streams **WAV header + raw PCM**.
3. **Weights switching is not supported**  
   `/set_gpt_weights` and `/set_sovits_weights` exist but are no-ops.

---

## 5. API Usage Examples

### GPT‑SoVITS style GET

```bash
curl "http://127.0.0.1:9888/tts?text=你好&text_lang=Chinese&ref_audio_path=C:\path\ref.wav&prompt_text=参考文本&prompt_lang=Chinese" --output out.wav
```

### GPT‑SoVITS style POST

```bash
curl -X POST http://127.0.0.1:9888/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好",
    "text_lang": "Chinese",
    "ref_audio_path": "C:\\\\path\\\\ref.wav",
    "prompt_text": "参考文本",
    "prompt_lang": "Chinese",
    "media_type": "wav",
    "streaming_mode": false
  }' \
  --output out.wav
```

### Upload ref_audio then call /tts (recommended)

```bash
# 1) Upload audio to server
curl -X POST http://127.0.0.1:9888/set_refer_audio \
  -F "audio_file=@ref.wav" 

# Response: {"path":".../uploads/ref_xxx.wav"}

# 2) Use returned path in /tts
curl -X POST http://127.0.0.1:9888/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好",
    "text_lang": "Chinese",
    "ref_audio_path": "E:\\\\Work\\\\AI\\\\FORK\\\\zerolan-core\\\\tts\\\\Qwen3-TTS\\\\uploads\\\\ref_xxx.wav",
    "prompt_text": "参考文本",
    "prompt_lang": "Chinese"
  }' \
  --output out.wav
```

### Streaming WAV (chunked)

```bash
curl "http://127.0.0.1:9888/tts?text=你好&text_lang=Chinese&ref_audio_path=C:\path\ref.wav&prompt_text=参考文本&prompt_lang=Chinese&media_type=wav&streaming_mode=true" --output out.wav
```

### CustomVoice (recommended for quick setup)

```bash
curl -X POST http://127.0.0.1:9888/tts/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello!",
    "language": "English",
    "speaker": "Ryan",
    "instruct": "Very happy"
  }' \
  --output out.wav
```

### VoiceDesign

```bash
curl -X POST http://127.0.0.1:9888/tts/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Welcome back!",
    "language": "English",
    "instruct": "Young male, upbeat, playful"
  }' \
  --output out.wav
```

### VoiceClone

```bash
curl -X POST http://127.0.0.1:9888/tts/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a new line.",
    "language": "English",
    "ref_audio": "C:\\\\path\\\\ref.wav",
    "ref_text": "This is the reference text."
  }' \
  --output out.wav
```

---

## 6. Drop‑in Replacement Checklist

1. **Stop GPT‑SoVITS service**
2. **Start Qwen3‑TTS service** (same port as before if needed)
3. **Keep your client requests unchanged**
   - If they call `/tts` with `text_lang`, `ref_audio_path`, `prompt_text`, it will still work.
4. **Validate output**
   - Listen to `out.wav` and verify quality and latency.

---

## 7. Troubleshooting

**Q: It starts, but first request is slow.**  
A: Initial model load can take minutes and big memory. Try smaller model (0.6B) or warm up once.

**Q: I get out-of-memory (OOM).**  
A: Use 0.6B models, reduce concurrency, or use CPU mode.

**Q: My ref_audio is ignored.**  
A: Ensure `--mode voice_clone` is used and `ref_audio_path`/`ref_audio` is a valid local path or URL.

---

## 8. Optional: Run on the same port as GPT‑SoVITS

If your old GPT‑SoVITS ran on `9880`, just set:

```powershell
--port 9880
```

Then your clients can continue calling the same base URL.
