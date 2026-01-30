"""
Qwen3-ASR (Transformers backend)
"""

import numpy as np
import torch
from loguru import logger
from qwen_asr import Qwen3ASRModel

from common.decorator import log_model_loading
from utils import audio_util
from zerolan.data.pipeline.asr import ASRPrediction, ASRQuery, ASRStreamQuery

from asr.qwen3_asr_transformers.config import Qwen3ASRTransformersConfig


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


def _normalize_language(language) -> str | None:
    if not language:
        return None
    if isinstance(language, str):
        return language.strip() or None
    return None


class Qwen3ASRTransformers:
    def __init__(self, config: Qwen3ASRTransformersConfig):
        self._config = config
        self._model: Qwen3ASRModel | None = None

    @log_model_loading("Qwen/Qwen3-ASR-1.7B")
    def load_model(self):
        dtype = _resolve_torch_dtype(self._config.dtype, self._config.device_map)
        kwargs = {
            "dtype": dtype,
            "device_map": self._config.device_map,
            "max_inference_batch_size": self._config.max_inference_batch_size,
            "max_new_tokens": self._config.max_new_tokens,
        }
        if self._config.attn_implementation:
            kwargs["attn_implementation"] = self._config.attn_implementation
        if self._config.forced_aligner:
            kwargs["forced_aligner"] = self._config.forced_aligner
            kwargs["forced_aligner_kwargs"] = {
                "dtype": _resolve_torch_dtype(
                    self._config.forced_aligner_dtype,
                    self._config.forced_aligner_device_map,
                ),
                "device_map": self._config.forced_aligner_device_map,
            }
        self._model = Qwen3ASRModel.from_pretrained(self._config.model_path, **kwargs)

    def predict(self, query: ASRQuery) -> ASRPrediction | None:
        language = _normalize_language(getattr(query, "language", None))
        results = self._model.transcribe(audio=query.audio_path, language=language)
        transcript = results[0].text if results else ""
        logger.info("ASR: " + transcript)
        return ASRPrediction(transcript=transcript)

    def stream_predict(self, query: ASRStreamQuery) -> ASRPrediction | None:
        if query.media_type and query.media_type.lower() == "raw":
            wave_nparray = np.frombuffer(query.audio_data, dtype=np.float32)
            sample_rate = query.sample_rate
        else:
            wave_nparray, sample_rate = audio_util.from_bytes_to_np_ndarray(
                query.audio_data, "float32"
            )
        language = _normalize_language(getattr(query, "language", None))
        results = self._model.transcribe(
            audio=(wave_nparray, sample_rate),
            language=language,
        )
        transcript = results[0].text if results else ""
        logger.info("ASR: " + transcript)
        return ASRPrediction(transcript=transcript)
