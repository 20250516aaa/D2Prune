import gc
import inspect
import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
import time
from loguru import logger

def cleanup_memory(verbose=True) -> None:
    """Clear GPU memory by running garbage collection and emptying cache."""
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass
    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))
    memory_before = total_reserved_mem()
    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbose:
            logger.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )


def distribute_model(model, device_map=None) -> None:
    """Distribute the model across available GPUs. NB: only implemented for Llama-2/3/Qwen-2."""
    no_split_module_classes = ['LlamaDecoderLayer']
    max_memory = get_balanced_memory(model, no_split_module_classes=no_split_module_classes)
    if not device_map:
        device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_module_classes)
    logger.info(f"Using device_map: {device_map} for llm evaluation")
    start = time.time()
    dispatch_model(model, device_map=device_map, offload_buffers=True, offload_dir="offload", state_dict=model.state_dict())
    cleanup_memory()
    logger.info(f"distribute finish, runing time:{time.time() - start}")