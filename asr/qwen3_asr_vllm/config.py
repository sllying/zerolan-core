from dataclasses import dataclass


@dataclass
class Qwen3ASRVLLMConfig:
    model_path: str = "Qwen/Qwen3-ASR-1.7B"
    gpu_memory_utilization: float = 0.7
    max_inference_batch_size: int = 128
    max_new_tokens: int = 4096
    forced_aligner: str | None = None
    forced_aligner_device_map: str = "cuda:0"
    forced_aligner_dtype: str = "bfloat16"
