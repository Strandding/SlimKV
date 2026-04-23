import torch
from vllm import LLM, SamplingParams
from transformers import GenerationConfig
from dataclasses import dataclass
from typing import List, Union


@dataclass
class BeaconModelOutput:
    # Keep the output contract used by vLLM eval path without depending on modeling_utils.
    batch_loss: List[float]
    valid_token_num: List[int]

HF_KEY_TO_VLLM_KEY = {
    "top_k": "top_k", 
    "top_p": "top_p", 
    "temperature": "temperature",
    "max_new_tokens": "max_tokens",
    "eos_token_id": "stop_token_ids",
}


class HFStyleVllmModel:
    def __init__(
        self,
        generation_config={},
        **kwargs,
    ):
        self.model = LLM(**kwargs)
        # vLLM changed internal API across versions; keep compatibility here.
        generation_fields = getattr(self.model.llm_engine, "generation_config_fields", None)
        if isinstance(generation_fields, dict):
            self.generation_config = GenerationConfig(**generation_fields)
        else:
            self.generation_config = GenerationConfig()

    @property
    def device(self):
        return self.model.llm_engine.device_config.device

    def parse_generation_config(self, generation_config:Union[dict,GenerationConfig]):
        """Rename hf generation config to vllm generation config."""
        vllm_config = {}

        if isinstance(generation_config, GenerationConfig):
            generation_config = generation_config.to_dict()

        for k, v in generation_config.items():
            if k in HF_KEY_TO_VLLM_KEY:
                if v is None:
                    continue
                k = HF_KEY_TO_VLLM_KEY[k]
                if k == "stop_token_ids" and isinstance(v, int):
                    v = [v]
                vllm_config[k] = v

        if generation_config.get("do_sample", None) == False:
            vllm_config["temperature"] = 0
        return vllm_config

    def generate(
        self, 
        prompts:Union[List[str],str]=None,
        input_ids:torch.Tensor=None, 
        attention_mask:torch.Tensor=None, 
        use_tqdm:bool=False,
        **kwargs
    ):
        # override the default sampling params
        sampling_params_dict = self.parse_generation_config(self.generation_config)
        sampling_params_dict.update(self.parse_generation_config(kwargs))
        sampling_params = SamplingParams(**sampling_params_dict) 

        if input_ids is not None:
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()
            outputs = self._generate_with_token_ids(
                input_ids=input_ids,
                sampling_params=sampling_params,
                use_tqdm=use_tqdm,
            )
        else:
            outputs = self.model.generate(
                prompts=prompts,
                sampling_params=sampling_params,
                use_tqdm=use_tqdm,
            )
        outputs = [output.outputs[0].text for output in outputs]
        return outputs

    def _generate_with_token_ids(self, input_ids, sampling_params, use_tqdm=False):
        # Compatibility across vLLM versions:
        # - old: model.generate(prompt_token_ids=...)
        # - new: model.generate(prompts=[{"prompt_token_ids": ...}], ...)
        try:
            return self.model.generate(
                prompt_token_ids=input_ids,
                sampling_params=sampling_params,
                use_tqdm=use_tqdm,
            )
        except TypeError:
            token_prompts = [{"prompt_token_ids": ids} for ids in input_ids]
            try:
                return self.model.generate(
                    prompts=token_prompts,
                    sampling_params=sampling_params,
                    use_tqdm=use_tqdm,
                )
            except TypeError:
                return self.model.generate(
                    token_prompts,
                    sampling_params=sampling_params,
                    use_tqdm=use_tqdm,
                )

    def __call__(self, input_ids, attention_mask, labels, **kwargs):
        if isinstance(input_ids, torch.Tensor):
            device = input_ids.device
            input_ids = input_ids.tolist()
            labels = labels.tolist()

        sampling_params = SamplingParams(prompt_logprobs=True, max_tokens=1, temperature=0)
        outputs = self._generate_with_token_ids(
            input_ids=input_ids,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        batch_losses = []
        valid_token_nums = []

        for i, output in enumerate(outputs):
            log_probs = output.prompt_logprobs
            batch_loss = 0
            valid_token_num = 0
            for j, log_prob in enumerate(log_probs):
                if j == 0:
                    assert log_prob is None
                    continue
                # NOTE: we do not need to rotate here because
                label = labels[i][j]
                if label == -100:
                    continue
                assert label in log_prob
                batch_loss -= log_prob[label].logprob
                valid_token_num += 1
            batch_losses.append(batch_loss / valid_token_num)
            valid_token_nums.append(valid_token_num)

        return BeaconModelOutput(
            batch_loss=batch_losses,
            valid_token_num=valid_token_nums
        )
