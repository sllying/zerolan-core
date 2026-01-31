# Qwen3‑TTS 独立服务（兼容 GPT‑SoVITS）

该目录提供一个 **独立** 的 Qwen3‑TTS 服务，让你可以在 **不修改主项目 `zerolan-core` 代码** 的情况下替换 GPT‑SoVITS。它在底层运行 Qwen3‑TTS，但对外 **模拟 GPT‑SoVITS 的 HTTP API**（`/tts`, `/control`, `/set_gpt_weights`, `/set_sovits_weights`）。

> 目标：对已经在调用 GPT‑SoVITS 的客户端实现“无缝替换”。

---

## 1. 前置条件

- 推荐 Python 3.10+（3.12 可用）
- 推荐使用 CUDA GPU（实时或大模型更合适）
- 推荐具备可用的 C++ 构建工具链以获得更佳性能（可选）

安装依赖（Windows 示例）：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U qwen-tts torch soundfile fastapi uvicorn
```

可选（如果你的 GPU 支持）：

```powershell
pip install -U flash-attn --no-build-isolation
```

---

## 2. 选择模型与模式

你必须为该服务实例选择 **一个** 模型与 **一种** 模式：

### 推荐映射

| 模式 | 模型 ID |
|---|---|
| `custom_voice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` 或 `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` |
| `voice_design` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |
| `voice_clone` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` 或 `Qwen/Qwen3-TTS-12Hz-0.6B-Base` |

**说明**
- `custom_voice` 支持固定说话人（例如 “Vivian”“Ryan” 等）。
- `voice_clone` 需要参考音频的本地路径或 URL。
- 一个服务实例 = 一个模型 + 一个模式。

---

## 3. 启动服务

在仓库根目录执行：

```powershell
python tts/Qwen3-TTS/qwen3_tts_api.py ^
  --model-id Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice ^
  --mode custom_voice ^
  --default-speaker Vivian ^
  --device-map cuda:0 ^
  --host 0.0.0.0 ^
  --port 9888
```

CPU 模式：

```powershell
python tts/Qwen3-TTS/qwen3_tts_api.py ^
  --model-id Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice ^
  --mode custom_voice ^
  --device-map cpu ^
  --dtype float32 ^
  --port 9888
```

---

## 4. 与 GPT‑SoVITS 的兼容性

该服务对外提供 **GPT‑SoVITS 兼容的接口**：

- `GET /tts`
- `POST /tts`
- `POST /set_refer_audio`（上传参考音频，返回服务端保存路径）
- `GET /control`（空操作 no-op）
- `GET /set_gpt_weights`（空操作 no-op）
- `GET /set_sovits_weights`（空操作 no-op）

### 字段映射

| GPT‑SoVITS 字段 | Qwen3‑TTS 字段 |
|---|---|
| `text` | `text` |
| `text_lang` | `language` |
| `ref_audio_path` | `ref_audio` |
| `prompt_text` | `ref_text` |
| `prompt_lang` |（忽略，回退到 `text_lang`）|
| `speaker` | `speaker`（仅 `custom_voice` 模式有效）|
| `instruct` | `instruct` |

### 已知差异

1. **`media_type` 固定为 WAV**  
   兼容层会接收 `media_type`。该服务支持 `wav` 和 `raw`。
2. **流式输出**  
   当 `streaming_mode=true` 时，`/tts` 会以 **Transfer-Encoding: chunked** 方式返回，并流式输出 **WAV 头 + raw PCM**。
3. **不支持切换权重**  
   `/set_gpt_weights` 与 `/set_sovits_weights` 接口存在但为 no-op。

---

## 5. API 使用示例

### GPT‑SoVITS 风格 GET

```bash
curl "http://127.0.0.1:9888/tts?text=你好&text_lang=Chinese&ref_audio_path=C:\path\ref.wav&prompt_text=参考文本&prompt_lang=Chinese" --output out.wav
```

### GPT‑SoVITS 风格 POST

```bash
curl -X POST http://127.0.0.1:9888/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好",
    "text_lang": "Chinese",
    "ref_audio_path": "C:\\path\\ref.wav",
    "prompt_text": "参考文本",
    "prompt_lang": "Chinese",
    "media_type": "wav",
    "streaming_mode": false
  }' \
  --output out.wav
```

### 先上传 ref_audio 再调用 /tts（推荐）

```bash
# 1) 上传音频到服务端
curl -X POST http://127.0.0.1:9888/set_refer_audio \
  -F "audio_file=@ref.wav"

# 返回示例：{"path":".../uploads/ref_xxx.wav"}

# 2) 在 /tts 中使用返回的路径
curl -X POST http://127.0.0.1:9888/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好",
    "text_lang": "Chinese",
    "ref_audio_path": "E:\\Work\\AI\\FORK\\zerolan-core\\tts\\Qwen3-TTS\\uploads\\ref_xxx.wav",
    "prompt_text": "参考文本",
    "prompt_lang": "Chinese"
  }' \
  --output out.wav
```

### 流式 WAV（chunked）

```bash
curl "http://127.0.0.1:9888/tts?text=你好&text_lang=Chinese&ref_audio_path=C:\path\ref.wav&prompt_text=参考文本&prompt_lang=Chinese&media_type=wav&streaming_mode=true" --output out.wav
```

### CustomVoice（推荐用于快速上手）

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
    "ref_audio": "C:\\path\\ref.wav",
    "ref_text": "This is the reference text."
  }' \
  --output out.wav
```

---

## 6. 无缝替换检查清单

1. **停止 GPT‑SoVITS 服务**
2. **启动 Qwen3‑TTS 服务**（如有需要可与原服务使用相同端口）
3. **保持客户端请求不变**
   - 如果客户端仍然用 `/tts` 并携带 `text_lang`、`ref_audio_path`、`prompt_text` 等字段，仍然可以工作。
4. **验证输出**
   - 播放 `out.wav`，检查音质与延迟。

---

## 7. 故障排查

**Q：能启动，但第一次请求很慢。**  
A：首次加载模型可能需要数分钟并占用大量内存。可尝试更小的 0.6B 模型，或先“热身”请求一次。

**Q：显存/内存不足（OOM）。**  
A：使用 0.6B 模型、降低并发，或切换到 CPU 模式。

**Q：我的 ref_audio 被忽略了。**  
A：请确保使用 `--mode voice_clone`，并且 `ref_audio_path`/`ref_audio` 是有效的本地路径或 URL。

---

## 8. 可选：使用与 GPT‑SoVITS 相同端口运行

如果你之前的 GPT‑SoVITS 运行在 `9880`，只需要设置：

```powershell
--port 9880
```

这样客户端就可以继续调用同一个 base URL。
