from dataclasses import dataclass


@dataclass
class Qwen3ASRTransformersConfig:
    model_path: str = "Qwen/Qwen3-ASR-1.7B"
    device_map: str = "cuda:0"
    dtype: str = "bfloat16"
    max_inference_batch_size: int = 8
    max_new_tokens: int = 256
    attn_implementation: str | None = None
    forced_aligner: str | None = None
    forced_aligner_device_map: str = "cuda:0"
    forced_aligner_dtype: str = "bfloat16"
