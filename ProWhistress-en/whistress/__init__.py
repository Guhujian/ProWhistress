# 轻量初始化，避免在训练入口导入推理客户端导致额外依赖问题
from .inference_client.whistress_client import WhiStressInferenceClient

__all__ = ["WhiStressInferenceClient"]