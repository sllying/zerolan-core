# Qwen3-ASR（vLLM 后端）

## 适用场景
- 需要更快的推理速度或更高并发
- 计划使用 vLLM 后端
- 仍然使用本项目统一的 `/asr/predict` 接口

> 注意：当前项目的 `/asr/stream-predict` 依旧是“整段返回”，不是真正的流式增量输出。
> 如果你要做浏览器级别的实时流式，请参考官方的 streaming demo 或单独部署 vLLM 服务。

## 目录说明
- `config.py`：模型配置结构
- `model.py`：vLLM 后端封装（实现统一的 ASR 接口）
- `pyproject.toml` / `requirements.txt`：依赖清单

---

## 一、Conda 路线（推荐小白）

### 1. 创建并进入环境

```bash
cd asr/qwen3_asr_vllm
conda create -n asr_qwen3_vllm python=3.12 -y
conda activate asr_qwen3_vllm
```

### 2. 安装依赖

```bash
pip install -e .
```

> 可选：如果你要输出时间戳，建议安装 FlashAttention 2：
>
> ```bash
> pip install -U flash-attn --no-build-isolation
> ```

### 3. 修改配置文件

回到项目根目录，打开 `config.yaml`，把 ASR 的 `id` 改成下面这个：

```
Qwen/Qwen3-ASR-1.7B-vllm
```

然后在 `ASR.config` 里填入对应配置（可直接复制，按需改）：

```yaml
ASR:
  id: "Qwen/Qwen3-ASR-1.7B-vllm"
  host: "0.0.0.0"
  port: 11001
  config:
    Qwen/Qwen3-ASR-1.7B-vllm:
      model_path: "Qwen/Qwen3-ASR-1.7B"
      gpu_memory_utilization: 0.7
      max_inference_batch_size: 128
      max_new_tokens: 4096
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
cd asr/qwen3_asr_vllm
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

1. **必须有 GPU 吗？**
   vLLM 强依赖 GPU，CPU 几乎不可用。请确保有 NVIDIA GPU。

2. **显存不够 / OOM**：
   - 降低 `gpu_memory_utilization`（如 0.6）
   - 降低 `max_inference_batch_size`
   - 降低 `max_new_tokens`
   - 换用 0.6B 模型（把 `model_path` 改成 `Qwen/Qwen3-ASR-0.6B`）

3. **想要真正的流式转写**：
   当前接口是一次性返回。如果你需要实时流式，请考虑另起一个 vLLM 服务或使用官方 streaming demo。
