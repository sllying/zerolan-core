# Qwen3-ASR（Transformers 后端）

## 适用场景
- 想快速在本项目里替换 ASR 模型
- 不需要真正的流式转写（返回一次完整文本即可）
- 机器资源一般，或希望先用最稳妥的方式跑通

## 目录说明
- `config.py`：模型配置结构
- `model.py`：模型封装（实现统一的 ASR 接口）
- `pyproject.toml` / `requirements.txt`：依赖清单

---

## 一、Conda 路线（推荐小白）

### 1. 创建并进入环境

```bash
cd asr/qwen3_asr_transformers
conda create -n asr_qwen3_tf python=3.12 -y
conda activate asr_qwen3_tf
```

### 2. 安装依赖

```bash
pip install -e .
```

> 可选：如果你有 NVIDIA GPU，并且显存紧张或想提速，可以安装 FlashAttention 2：
>
> ```bash
> pip install -U flash-attn --no-build-isolation
> ```

### 3. 修改配置文件

回到项目根目录，打开 `config.yaml`，把 ASR 的 `id` 改成下面这个：

```
Qwen/Qwen3-ASR-1.7B-transformers
```

然后在 `ASR.config` 里填入对应配置（可直接复制，按需改）：

```yaml
ASR:
  id: "Qwen/Qwen3-ASR-1.7B-transformers"
  host: "0.0.0.0"
  port: 11001
  config:
    Qwen/Qwen3-ASR-1.7B-transformers:
      model_path: "Qwen/Qwen3-ASR-1.7B"
      device_map: "cuda:0"
      dtype: "bfloat16"
      max_inference_batch_size: 8
      max_new_tokens: 256
      attn_implementation: null
      forced_aligner: null
      forced_aligner_device_map: "cuda:0"
      forced_aligner_dtype: "bfloat16"
```

### 4. 启动服务

回到项目根目录启动：

```bash
cd ../../
python starter.py asr
```

---

## 二、uv 路线（更快、更干净）

### 1. 创建并进入 uv 虚拟环境

```bash
cd asr/qwen3_asr_transformers
uv venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux / macOS：

```bash
source .venv/bin/activate
```

### 2. 安装依赖

```bash
uv pip install -e .
```

### 3. 修改配置文件 + 启动

同上面的 Conda 路线，改 `config.yaml` 后回到项目根目录运行：

```bash
cd ../../
uv run starter.py asr
```

---

## 三、测试接口

从项目根目录执行：

```bash
curl -X POST http://localhost:11001/asr/predict \
  -F "audio=@tests/resources/tts-test.wav;type=audio/wav" \
  -F "json={\"audio_path\": \"\", \"channels\": 2};type=application/json"
```

正常会返回类似：

```json
{"id":"...","transcript":"..."}
```

---

## 常见问题

1. **下载模型很慢**：
   你可以在系统环境变量里设置 Hugging Face 镜像（如果你在国内网络环境），或提前用 ModelScope 下载模型到本地，然后把 `model_path` 改成你的本地路径。

2. **显存不够 / OOM**：
   - 把 `max_inference_batch_size` 调小（如 1、2）
   - 把 `max_new_tokens` 调小
   - 换用 0.6B 模型（把 `model_path` 改成 `Qwen/Qwen3-ASR-0.6B`）

3. **CPU 很慢**：
   这是正常现象，ASR 模型对 GPU 依赖很高。建议使用 NVIDIA GPU。
