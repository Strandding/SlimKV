from .utils import FileLogger, DefaultDataCollator, makedirs, split_file_dir_name_ext, clear_dir, get_max_length_in_nested_lists, pad_nested_lists, mask_nested_lists, normalize_text, wrap_text, load_json, save_json, load_pickle, save_pickle, add_eos, remove_eos, format_numel_str
from .chat import apply_chat_template
from .args import ModelArgs
from .data import Data
from .modeling_utils import evaluate_perplexity, evaluate_generation, evaluate_nll, move_to_device, get_shifted_labels

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)


def get_model_and_tokenizer(model_args, device="cpu", evaluation_mode=True, return_tokenizer_only=False, **kwargs):    
    import torch
    import transformers
    from dataclasses import asdict
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from transformers.configuration_utils import PretrainedConfig
    from transformers.utils import logging
    from transformers.integrations import is_deepspeed_zero3_enabled
    from packaging import version

    from .args import ModelArgs

    logger = logging.get_logger(__name__)

    model_args: ModelArgs

    model_args_dict = asdict(model_args)
    model_args_dict.update(**kwargs)

    model_name_or_path = model_args_dict["model_name_or_path"]
    cache_dir = model_args_dict["model_cache_dir"]
    access_token = model_args_dict["access_token"]

    logger.info(f"Loading model and tokenizer from {model_name_or_path}...")

    tokenizer_kwargs = {}
    if model_args_dict["no_use_fast"]:
        tokenizer_kwargs = {"use_fast": False}

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, 
        cache_dir=cache_dir, 
        padding_side=model_args_dict["padding_side"], 
        token=access_token, 
        trust_remote_code=True,
        **tokenizer_kwargs
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if return_tokenizer_only:
        return tokenizer

    dtype = model_args_dict["dtype"]
    if dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
        
    device_map = model_args_dict["device_map"]
    if device_map is None and not is_deepspeed_zero3_enabled():
        device_map = {"": device}
    
    rope_kwargs = {}
    rope_theta = model_args_dict["rope_theta"]
    if rope_theta is not None:
        rope_kwargs["rope_theta"] = rope_theta
    rope_method = model_args_dict["rope_method"]
    if rope_method is not None:
        rope_factor = model_args_dict["rope_factor"]
        rope_scaling = {
            "type": rope_method,
            "factor": rope_factor
        }
        # NOTE: do not destroy the default rope_scaling of the model
        rope_kwargs["rope_scaling"] = rope_scaling

    attn_kwargs = {}
    attn_impl = model_args_dict["attn_impl"]
    if attn_impl is not None:
        if version.parse(transformers.__version__) <= version.parse("4.36"):
            if attn_impl == "flash_attention_2":
                attn_kwargs["use_flash_attention_2"] = True
        else:
            attn_kwargs["attn_implementation"] = attn_impl

    # from_pretrained_kwargs = {}
    # if attn_impl == "flash_attention_2" and version.parse(transformers.__version__) <= version.parse("4.36"):
    #     from_pretrained_kwargs["use_flash_attention_2"] = True

    # Normalize the shorthand before constructing config kwargs.
    # NOTE: in from_pretrained -> from_dict flow, kwargs are applied via setattr
    # after __init__, so relying on config __init__ convenience logic alone is
    # not sufficient for beacon_kv_compression.
    beacon_kv_compression = model_args_dict.get("beacon_kv_compression")
    if beacon_kv_compression is not None:
        if model_args_dict.get("beacon_k_compression") is None:
            model_args_dict["beacon_k_compression"] = beacon_kv_compression
        if model_args_dict.get("beacon_v_compression") is None:
            model_args_dict["beacon_v_compression"] = beacon_kv_compression

    beacon_kwargs = {}
    for k, v in model_args_dict.items():
        if k.startswith("beacon") and v is not None:
            beacon_kwargs[k] = v

    probe_kwargs = {}
    for k, v in model_args_dict.items():
        if k.startswith("beacon_probe") and v is not None:
            probe_kwargs[k] = v

    # read raw config dict first to avoid strict AutoConfig validation on older transformers
    probe_config_dict, _ = PretrainedConfig.get_config_dict(
        model_name_or_path,
        cache_dir=cache_dir,
        token=access_token,
        trust_remote_code=True,
    )
    architectures = probe_config_dict.get("architectures", None)
    if not architectures:
        raise ValueError(f"Cannot determine model architecture from config: {model_name_or_path}")
    architecture = architectures[0]
    rope_scaling_probe = probe_config_dict.get("rope_scaling", None)
    rope_type_probe = None
    if isinstance(rope_scaling_probe, dict):
        rope_type_probe = rope_scaling_probe.get("rope_type", rope_scaling_probe.get("type", None))
    is_llama3_model = architecture == "LlamaForCausalLM" and rope_type_probe == "llama3"

    # Check if any lowrank beacon compression is requested
    lowrank_requested = False
    for _flag in ("beacon_kv_compression", "beacon_k_compression", "beacon_v_compression"):
        val = model_args_dict.get(_flag)
        if val not in (None, "none"):
            lowrank_requested = True
            break
    if lowrank_requested:
        if not model_args_dict["enable_beacon"]:
            raise ValueError("Beacon KV compression requires `enable_beacon=True`.")
        supports_lowrank = (
            architecture == "Qwen2ForCausalLM"
            or (architecture == "LlamaForCausalLM" and is_llama3_model)
        )
        if not supports_lowrank:
            raise NotImplementedError("Low-rank beacon KV is currently only supported for Qwen2 and Llama3 beacon models.")

    extra_kwargs = {}
    if model_args_dict["max_position_embeddings"] is not None:
        extra_kwargs["max_position_embeddings"] = model_args_dict["max_position_embeddings"]
    if model_args_dict["load_in_4_bit"]:
        extra_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        device_map = None

    raw_nokrope_kwargs = {}
    if model_args_dict.get("raw_nokrope", False):
        if architecture != "LlamaForCausalLM":
            logger.warning("raw_nokrope=True is ignored because architecture is not LlamaForCausalLM.")
        elif not is_llama3_model:
            logger.warning("raw_nokrope=True is ignored because model is not Llama3.")
        elif model_args_dict["enable_beacon"]:
            logger.warning("raw_nokrope=True is ignored when enable_beacon=True.")
        elif model_args_dict["enable_vllm"]:
            logger.warning("raw_nokrope=True is ignored when enable_vllm=True.")
        else:
            raw_nokrope_kwargs["raw_nokrope"] = True

    llama_config_class = None
    llama_model_class = None
    if architecture == "LlamaForCausalLM":
        from .llama3 import LlamaForCausalLM as LlamaModelClass, LlamaConfig as LlamaConfigClass
        llama_config_class = LlamaConfigClass
        llama_model_class = LlamaModelClass

    if model_args_dict["enable_beacon"]:
        from .qwen2 import Qwen2ForCausalLM, Qwen2Config
        ARCHITECTURE_TO_CLASS = {
            'LlamaForCausalLM': (llama_config_class, llama_model_class),
            'Qwen2ForCausalLM': (Qwen2Config, Qwen2ForCausalLM),
        }

        if ARCHITECTURE_TO_CLASS.get(architecture, (None, None))[0] is None:
            raise ValueError(f"Architecture {architecture} is not supported.")
        config_class, model_class = ARCHITECTURE_TO_CLASS[architecture]

        config = config_class.from_pretrained(
            model_name_or_path, 
            cache_dir=cache_dir,
            token=access_token,
            # NOTE: keep the torch_dtype in config consistent with that in model
            torch_dtype=dtype,
            **beacon_kwargs,
            **rope_kwargs,
            **attn_kwargs,
            **extra_kwargs,
        )
        model = model_class.from_pretrained(
            model_name_or_path, 
            config=config,
            cache_dir=cache_dir, 
            torch_dtype=dtype,
            device_map=device_map, 
            token=access_token,
        )

    else:
        if model_args_dict["enable_vllm"]:
            from .vllm_utils import HFStyleVllmModel
            if model_args_dict["dtype"] == "fp32":
                vllm_dtype = "float32"
            elif model_args_dict["dtype"] == "fp16":
                vllm_dtype = "float16"
            elif model_args_dict["dtype"] == "bf16":
                vllm_dtype = "bfloat16"

            vllm_kwargs = {}
            if model_args_dict["vllm_len"] is not None:
                vllm_kwargs["max_model_len"] = model_args_dict["vllm_len"]

            model = HFStyleVllmModel(
                model=model_name_or_path,
                dtype=vllm_dtype,
                gpu_memory_utilization=model_args_dict["vllm_mem"],
                tensor_parallel_size=model_args_dict["vllm_tp"],
                disable_custom_all_reduce=model_args_dict["vllm_disable_ar"],
                enforce_eager=False,
                trust_remote_code=True,
                **rope_kwargs,
                **vllm_kwargs,
            )

        else:
            if architecture == "LlamaForCausalLM":
                config = llama_config_class.from_pretrained(
                    model_name_or_path,
                    cache_dir=cache_dir,
                    token=access_token,
                    torch_dtype=dtype,
                    **probe_kwargs,
                    **rope_kwargs,
                    **raw_nokrope_kwargs,
                    **attn_kwargs,
                    **extra_kwargs,
                )
                model = llama_model_class.from_pretrained(
                    model_name_or_path,
                    config=config,
                    cache_dir=cache_dir,
                    torch_dtype=dtype,
                    device_map=device_map,
                    token=access_token,
                )
                model._enable_beacon = False
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path, 
                    cache_dir=cache_dir, 
                    torch_dtype=dtype,
                    device_map=device_map,
                    token=access_token,
                    trust_remote_code=True,

                    # NOTE: do not destroy the default rope_scaling of the model
                    **rope_kwargs,
                    **attn_kwargs,
                    **extra_kwargs,
                )

    # load lora
    if model_args_dict["lora"] is not None:
        logger.info(f"loading lora from {model_args_dict['lora']}...")

        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model, 
            model_args_dict["lora"],
            torch_dtype=dtype,
            device_map=device_map,
        )
        if model_args_dict["lora_unload"]:
            model = model.merge_and_unload()

    if model_args_dict["enable_tp"]:
        import tensor_parallel as tp
        logger.info("enabling tensor parallelism...")
        
        # model = tp.tensor_parallel(model, device_ids=list(range(8)), distributed=False, sharded=False)
        model = tp.tensor_parallel(model, sharded=True)

        if model.generation_config.eos_token_id == 128001:
            model.generation_config.eos_token_id = [128001, 128009]

    if isinstance(model, transformers.modeling_utils.PreTrainedModel):
        model = model.eval()
        if evaluation_mode:
            # NOTE: essential to disable all gradient in-place, so that when calling accelerator.prepare, the forward function will not be wrapped that may consume extra GPU memory
            model.requires_grad_(False)
        logger.info(model.config)

    # override the default generation config
    generation_config = model_args.get_generation_config()
    if len(generation_config):
        model.generation_config.update(**generation_config)
    logger.info(f"Specified generation config: {generation_config}")

    return model, tokenizer
