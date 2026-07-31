"""DGX Spark launcher for NeMo AutoModel 26.06.

Patches one transformers-5.x incompatibility: AutoModel's initialize_weights()
calls LlamaRMSNorm.reset_parameters(), which transformers 5.x removed. We re-add
it (RMSNorm init = weights set to 1), then run the stock entrypoint unchanged.
"""
import torch, runpy
from transformers.models.llama.modeling_llama import LlamaRMSNorm

def _reset(self):
    with torch.no_grad():
        self.weight.fill_(1.0)

if not hasattr(LlamaRMSNorm, "reset_parameters"):
    LlamaRMSNorm.reset_parameters = _reset
    print("[spark-patch] added LlamaRMSNorm.reset_parameters")

runpy.run_path("/opt/Automodel/examples/llm_finetune/finetune.py", run_name="__main__")
