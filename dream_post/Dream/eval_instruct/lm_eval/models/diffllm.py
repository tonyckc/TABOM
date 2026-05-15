import atexit
import json
import logging
import gc
import os
import random
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Type, TypeVar, Union

import torch
import torch.distributions as dists
import torch.nn.functional as F
import transformers
from transformers.utils import is_torchdynamo_compiling
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
    find_executable_batch_size,
)
from datasets import Dataset
from packaging import version
from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator, get_dtype

from lm_eval.models.llada_block_generate import (
    llada_block_diffusion_generate,
    resolve_mask_id as llada_resolve_mask_id,
)

from lm_eval.models.generation_utils import DreamGenerationConfig,DreamModelOutput
from lm_eval.models.modeling_dream import DreamModel  # Adjust import based on actual file

# Process-wide accum: decode_token_slots = total decode slots in batch (B * max_new_tokens),
# forward_count = diffusion decode forwards; TPF = decode_token_slots / forward_count.
_DIFFLLM_DECODE_ACC: dict = {"forward_count": 0, "decode_token_slots": 0}
_DIFFLLM_DECODE_META: dict = {"rank": 0, "world_size": 1}
_DIFFLLM_DECODE_ATEXIT_REGISTERED = False


def _diffllm_flush_decode_stats() -> None:
    path = os.environ.get("DIFFLLM_DECODE_STATS_PATH", "").strip()
    if not path:
        return
    fc = int(_DIFFLLM_DECODE_ACC.get("forward_count", 0))
    slots = int(_DIFFLLM_DECODE_ACC.get("decode_token_slots", 0))
    if fc <= 0:
        return
    rank = int(_DIFFLLM_DECODE_META.get("rank", 0))
    ws = int(_DIFFLLM_DECODE_META.get("world_size", 1))
    out_path = path
    if ws > 1:
        out_path = f"{path}.rank_{rank}"
    tpf = float(slots) / float(fc)
    payload = {
        "forward_count": fc,
        "decode_token_slots": slots,
        "TPF": tpf,
        "rank": rank,
        "world_size": ws,
    }
    try:
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _diffllm_register_decode_stats_atexit() -> None:
    global _DIFFLLM_DECODE_ATEXIT_REGISTERED
    if _DIFFLLM_DECODE_ATEXIT_REGISTERED:
        return
    atexit.register(_diffllm_flush_decode_stats)
    _DIFFLLM_DECODE_ATEXIT_REGISTERED = True




eval_logger = logging.getLogger(__name__)
T = TypeVar("T", bound="LM")


def _boolish(v, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", ""):
        return False
    return default


def empty_cache_by_memory(threshold_gb=70):
    """
    Empty CUDA cache if allocated memory exceeds threshold
    Args:
        threshold_gb: Memory threshold in GB
    """
    if torch.cuda.is_available():
        # Get current memory allocated
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB

        if allocated > threshold_gb:
            # Clear cache
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Cache cleared. Memory freed: {allocated:.2f} GB")



def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits

def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            #x0 = dists.Categorical(probs=probs).sample()
            #confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
            confidence, x0 = probs.max(dim=-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0


class ExtendedDreamModel(DreamModel):
    @torch.no_grad()
    def diffusion_generate_inference(
            self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
        generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
                hasattr(generation_config, "pad_token_id") and
                torch.any(input_ids == generation_config.pad_token_id) and
                attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        self._diffllm_dream_step_acc = {"reveals": 0, "steps": 0}
        result = self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            generation_tokens_hook_func=generation_tokens_hook_func,
            generation_logits_hook_func=generation_logits_hook_func,
        )
        return result
    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        

        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        # this allows user-defined token control of the intermediate steps
        #x = generation_tokens_hook_func(None, x, None)
        for i in range(steps):
            mask_index = (x == mask_token_id)

            if not isinstance(mask_index, torch.Tensor):
                mask_index = torch.full(x.shape, bool(mask_index), device=x.device, dtype=torch.bool)
            elif mask_index.dim() == 0:
                mask_index = mask_index.unsqueeze(0).expand(x.shape)

            if not mask_index.any():
                break

            x_step_start = x.clone()

            logits = self(x, attention_mask, tok_idx).logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

            # this allows user-defined logits control of the intermediate steps
            logits = generation_logits_hook_func(i, x, logits)

            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]
        
            if alg == 'origin':
                p_transfer = 1 - s / t if i < steps - 1 else 1
                x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                _, x0[transfer_index_t_s]= sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
                x[mask_index] = x0.clone()
            else:
                if alg == 'maskgit_plus':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                elif alg == 'topk_margin':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
                elif alg == 'entropy':
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                else:
                    raise RuntimeError(f"Unknown alg: {alg}")

                _fixed = getattr(generation_config, "dream_fixed_tokens_per_step", None)
                _thr_raw = getattr(generation_config, "dream_confidence_threshold", None)
                _fixed_k: Optional[int] = None
                if _fixed is not None:
                    try:
                        _ki = int(_fixed)
                        _fixed_k = _ki if _ki > 0 else None
                    except (TypeError, ValueError):
                        _fixed_k = None
                _thr: Optional[float] = None
                if _fixed_k is None and _thr_raw is not None:
                    try:
                        _thr = float(_thr_raw)
                    except (TypeError, ValueError):
                        _thr = None

                if _thr is not None:
                    # Align with LLaDA confidence_threshold + default: softmax top-1 on raw logits; multi unmask/step
                    p_mask = F.softmax(mask_logits.float(), dim=-1)
                    maxprob, x0_mp = p_mask.max(dim=-1)
                    maxprob = maxprob.to(dtype=logits.dtype)
                    x0_mp = x0_mp.to(dtype=torch.long)
                    full_maxprob = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                    full_maxprob[mask_index] = maxprob
                    x0_full = torch.full_like(x, mask_token_id, device=self.device, dtype=torch.long)
                    x0_full[mask_index] = x0_mp
                    transfer_sel = torch.zeros_like(x, dtype=torch.bool, device=x.device)
                    neg_inf = torch.tensor(-float("inf"), device=x.device, dtype=logits.dtype)
                    b_sz = x.size(0)
                    for j in range(b_sz):
                        sub_mask = mask_index[j]
                        if not sub_mask.any():
                            continue
                        row_fc = full_maxprob[j]
                        hit = sub_mask & (row_fc >= _thr)
                        if hit.any():
                            transfer_sel[j] = hit
                        else:
                            masked_fc = torch.where(sub_mask, row_fc, neg_inf)
                            pos = int(torch.argmax(masked_fc).item())
                            transfer_sel[j, pos] = True
                    x[transfer_sel] = x0_full[transfer_sel]
                else:
                    if _fixed_k is not None:
                        mpr = mask_index.sum(dim=1)
                        active = mpr[mpr > 0]
                        number_transfer_tokens = (
                            min(_fixed_k, int(active.min().item())) if active.numel() > 0 else 0
                        )
                    else:
                        num_mask_token = mask_index.sum() / mask_index.shape[0]
                        number_transfer_tokens = (
                            int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
                        )
                        _cap = getattr(generation_config, "per_step_transfer_cap", None)
                        if _cap is not None:
                            try:
                                _ci = int(_cap)
                                if _ci > 0:
                                    number_transfer_tokens = min(number_transfer_tokens, _ci)
                            except (TypeError, ValueError):
                                pass
                    full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                    full_confidence[mask_index] = confidence
                    if number_transfer_tokens > 0:
                        if alg_temp is None or alg_temp == 0:
                            _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                        else:
                            full_confidence = full_confidence / alg_temp
                            full_confidence = F.softmax(full_confidence, dim=-1)
                            transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                        x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                        x_[mask_index] = x0.clone()
                        row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                        x[row_indices, transfer_index] = x_[row_indices, transfer_index]

            # this allows user-defined token control of the intermediate steps
            x = generation_tokens_hook_func(i, x, logits)

            acc = getattr(self, "_diffllm_dream_step_acc", None)
            if isinstance(acc, dict):
                acc["steps"] = acc.get("steps", 0) + 1
                acc["reveals"] = acc.get("reveals", 0) + int(
                    ((x_step_start == mask_token_id) & (x != mask_token_id)).sum().item()
                )

            if histories is not None:
                histories.append(x.clone())
        
        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
            )
        else:
            return x




@register_model("diffllm")
class DiffLLM(LM):
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        max_prompt_len: Optional[int] = 1024,
        max_new_tokens: Optional[int] = 128,
        nll_type: Optional[str] = "mc",
        log_type: Optional[str] = "ftb",
        classifier_free_guidance: Optional[float] = 1.0,
        pad_to_max_len: Optional[bool] = False,
        sampling_eps: Optional[float] = 1e-3,
        diffusion_steps: Optional[int] = 32,
        trust_remote_code: Optional[bool] = True,
        parallelize: Optional[bool] = False,
        autogptq: Optional[Union[bool, str]] = False,
        lora_path: Optional[str] = None,
        first_lora_path: Optional[str] = None,
        backend: str = "dream",
        block_length: Optional[Union[int, str]] = None,
        logits_eos_inf: Union[bool, str, int] = False,
        confidence_eos_eot_inf: Union[bool, str, int] = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.student_backend = str(backend or "dream").strip().lower()
        try:
            self.block_length = (
                int(block_length)
                if block_length is not None and str(block_length).strip() != ""
                else None
            )
        except (TypeError, ValueError):
            self.block_length = None
        self.logits_eos_inf = _boolish(logits_eos_inf, False)
        self.confidence_eos_eot_inf = _boolish(confidence_eos_eot_inf, False)

        # prepare for parallelism
        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))

        gpus = torch.cuda.device_count()
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self.accelerator = accelerator

        if "npu" in accelerator.device.type:
            gpus = torch.npu.device_count()

        # using one process with no model parallelism
        if not (parallelize or accelerator.num_processes > 1):
            # use user-passed device
            device_list = set(
                ["cuda", "cpu"]
                + [f"cuda:{i}" for i in range(gpus)]
                + ["mps", "mps:0"]
                + [f"npu:{i}" for i in range(gpus)]
            )
            if device and device in device_list:
                self._device = torch.device(device)
                eval_logger.info(f"Using device '{device}'")
                if device in ("mps", "mps:0") and version.parse(
                    torch.__version__
                ) < version.parse("2.1"):
                    raise RuntimeError(
                        f"mps requires torch >= 2.1. You have {torch.__version__}"
                    )
            else:
                eval_logger.info("Device not specified")
                eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                self._device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
        else:  # Parallelism managed by accelerate
            if device != "cuda":
                eval_logger.info(
                    f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model."
                )
            # TODO: include in warning that `load_in_8bit` etc. affect this too
            self._device = (
                self.accelerator.device
                if hasattr(self, "accelerator")
                else torch.device(device)
            )

        self.batch_size_per_gpu = batch_size
        if isinstance(batch_size, str):
            self.batch_size_per_gpu = int(batch_size)
        self._create_model_and_tokenizer(
            pretrained,
            dtype,
            trust_remote_code,
            lora_path=lora_path,
            first_lora_path=first_lora_path,
        )

        if isinstance(pretrained, str):
            if gpus >= 1 or str(self.device) == "mps":
                # TODO: can remove this whole snippet except in the mps case, perhaps?
                if not (parallelize or autogptq or hasattr(self, "accelerator")):
                    # place model onto device requested manually,
                    # if not using HF Accelerate or device_map
                    # or any other option that preloads model onto device
                    try:
                        self.model.to(self.device)
                    except ValueError:
                        eval_logger.debug(
                            "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
                        )
            # multigpu data-parallel support when launched with accelerate
            if gpus > 1:
                if accelerator.num_processes > 1:
                    if parallelize:
                        eval_logger.warning(
                            "You are both using a HF Accelerate `device_map` (`--model_args parallelize=True`) and launching via `accelerate launch`. This will attempt to do model and data parallelism depending on the resources available."
                        )
                    elif gpus > accelerator.num_processes:
                        eval_logger.warning(
                            "WARNING: The number of total system GPUs does not match the number of spawned processes. "
                            "If you would like to use data parallelism, please launch the script "
                            "with 'accelerate launch *script*'. "
                            f"Current run will proceed with {accelerator.num_processes} devices."
                        )
                        if self.accelerator.is_local_main_process:
                            eval_logger.info(
                                f"Using {gpus} devices with data parallelism"
                            )

                    self._device = torch.device(f"{accelerator.device}")
                    self.accelerator = accelerator

                    self._rank = self.accelerator.local_process_index
                    self._world_size = self.accelerator.num_processes
                else:
                    # if we aren't launching via accelerate, ditch
                    self._rank = 0
                    self._world_size = 1
        else:
            # if a PreTrainedModel was passed into HFLM, we forgo distributed setup.
            eval_logger.warning(
                "Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration"
            )
            self._rank = 0
            self._world_size = 1

        # generation params
        self.max_prompt_len = max_prompt_len
        self.max_new_tokens = max_new_tokens
        self.diffusion_steps = diffusion_steps
        self.temperature = kwargs.get("temperature", 0.1)
        self.top_p = kwargs.get("top_p", 0.9)
        self.alg = kwargs.get("alg", "entropy")
        self.alg_temp = kwargs.get("alg_temp", 0.0)
        self.top_k = kwargs.get("top_k", None)

        _fk = kwargs.get("llada_fixed_confidence_topk", None)
        if _fk is not None and str(_fk).strip() != "":
            try:
                self.llada_fixed_confidence_topk = int(_fk)
                if self.llada_fixed_confidence_topk <= 0:
                    self.llada_fixed_confidence_topk = None
            except (TypeError, ValueError):
                self.llada_fixed_confidence_topk = None
        else:
            self.llada_fixed_confidence_topk = None

        _ct = kwargs.get("confidence_threshold", None)
        if self.llada_fixed_confidence_topk is not None:
            self.confidence_threshold = None
        elif _ct is not None and str(_ct).strip() != "":
            self.confidence_threshold = float(_ct)
        else:
            self.confidence_threshold = None

        _psc = kwargs.get("per_step_transfer_cap", None)
        if _psc is not None and str(_psc).strip() != "":
            try:
                self.per_step_transfer_cap = int(_psc)
                if self.per_step_transfer_cap <= 0:
                    self.per_step_transfer_cap = None
            except (TypeError, ValueError):
                self.per_step_transfer_cap = None
        else:
            self.per_step_transfer_cap = None

        _df = kwargs.get("dream_fixed_tokens_per_step", None)
        if _df is not None and str(_df).strip() != "":
            try:
                self.dream_fixed_tokens_per_step = int(_df)
                if self.dream_fixed_tokens_per_step <= 0:
                    self.dream_fixed_tokens_per_step = None
            except (TypeError, ValueError):
                self.dream_fixed_tokens_per_step = None
        else:
            self.dream_fixed_tokens_per_step = None

        _dct = kwargs.get("dream_confidence_threshold", None)
        if self.dream_fixed_tokens_per_step is not None:
            self.dream_confidence_threshold = None
        elif _dct is not None and str(_dct).strip() != "":
            try:
                self.dream_confidence_threshold = float(_dct)
            except (TypeError, ValueError):
                self.dream_confidence_threshold = None
        else:
            self.dream_confidence_threshold = None

        _diffllm_register_decode_stats_atexit()
        _DIFFLLM_DECODE_META["rank"] = int(self._rank)
        _DIFFLLM_DECODE_META["world_size"] = int(self._world_size)

        # loglikelihood params
        self.nll_type = nll_type
        self.log_type = log_type
        self.classifier_free_guidance = classifier_free_guidance
        self.pad_to_max_len = pad_to_max_len
        self.sampling_eps = sampling_eps

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _create_model_and_tokenizer(
        self,
        pretrained,
        dtype,
        trust_remote_code,
        lora_path=None,
        first_lora_path=None,
    ):
        from peft import PeftModel

        if self.student_backend == "llada":
            eval_logger.info("DiffLLM backend=llada: loading AutoModelForCausalLM + optional LoRA")
            base = (
                transformers.AutoModelForCausalLM.from_pretrained(
                    pretrained,
                    torch_dtype=get_dtype(dtype),
                    trust_remote_code=trust_remote_code,
                )
                .eval()
            ).to(self.device)

            if first_lora_path is not None:
                eval_logger.info(
                    f"Merging first LoRA into LLaDA base (merge_and_unload): {first_lora_path}"
                )
                base = PeftModel.from_pretrained(base, first_lora_path)
                base = base.merge_and_unload()
                base = base.eval().to(self.device)

            if lora_path is not None:
                eval_logger.info(f"Loading LoRA adapter from: {lora_path}")
                self.model = PeftModel.from_pretrained(base, lora_path).eval()
            else:
                self.model = base

            if lora_path is not None:
                tok_source = lora_path
            elif first_lora_path is not None:
                tok_source = first_lora_path
            else:
                tok_source = pretrained
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                tok_source, trust_remote_code=trust_remote_code, padding_side="left"
            )
            return

        base = (
            ExtendedDreamModel.from_pretrained(
                pretrained,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=trust_remote_code,
            )
            .eval()
        ).to(self.device)

        if first_lora_path is not None:
            eval_logger.info(
                f"Merging first LoRA into base (merge_and_unload): {first_lora_path}"
            )
            base = PeftModel.from_pretrained(base, first_lora_path)
            base = base.merge_and_unload()
            base = base.eval().to(self.device)

        if lora_path is not None:
            eval_logger.info(f"Loading LoRA adapter from: {lora_path}")
            
            self.model = PeftModel.from_pretrained(base, lora_path).eval()
        else:
            self.model = base

        # Prefer stage-2 adapter tokenizer, then stage-1, then base (matches train_tabom second_train).
        if lora_path is not None:
            tok_source = lora_path
        elif first_lora_path is not None:
            tok_source = first_lora_path
        else:
            tok_source = pretrained
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tok_source, trust_remote_code=trust_remote_code
        )

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def tok_encode(self, text, add_special_tokens=True):
        return self.tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids
    @classmethod
    def create_from_arg_string(
        cls: Type[T], arg_string: str, additional_config: Optional[dict] = None
    ) -> T:
        """
        Creates an instance of the LM class using the given argument string and additional config.

        Parameters:
        - arg_string: A string containing arguments in the format key1=value1,key2=value2.
        - additional_config: Optional dictionary containing additional configuration parameters.

        Returns:
        - Instance of the LM class.
        """
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    def apply_chat_template(
        self, chat_history, add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

        return chat_templated

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        # tokenize
        
        prompt_ids = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").input_ids
        prompt_ids = prompt_ids[:, -self.max_prompt_len:]
        attn_mask = prompt_ids.ne(self.tokenizer.pad_token_id)
        prompt_ids = prompt_ids.to(device=self.device)
        attn_mask = attn_mask.to(device=self.device)

        if self.student_backend == "llada":
            block_len = self.block_length if self.block_length is not None else self.max_new_tokens
            if self.max_new_tokens % block_len != 0:
                raise ValueError(
                    f"backend=llada requires max_new_tokens % block_length == 0; "
                    f"got max_new_tokens={self.max_new_tokens}, block_length={block_len}"
                )
            if self.diffusion_steps % (self.max_new_tokens // block_len) != 0:
                raise ValueError(
                    f"backend=llada requires diffusion_steps % num_blocks == 0 "
                    f"(num_blocks = max_new_tokens // block_length); "
                    f"got diffusion_steps={self.diffusion_steps}, max_new_tokens={self.max_new_tokens}, "
                    f"block_length={block_len}"
                )
            mask_id = llada_resolve_mask_id(self.tokenizer, self.model)
            eos_id = self.tokenizer.eos_token_id
            if eos_id is None:
                eos_id = getattr(self.model.config, "eos_token_id", None)
            if eos_id is None:
                eos_id = 126081

            with torch.inference_mode():
                full, fc = llada_block_diffusion_generate(
                    self.model,
                    prompt_ids,
                    mask_id=mask_id,
                    alg=self.alg,
                    steps=int(self.diffusion_steps),
                    gen_length=int(self.max_new_tokens),
                    block_length=int(block_len),
                    temperature=float(self.temperature),
                    logits_eos_inf=self.logits_eos_inf,
                    confidence_eos_eot_inf=self.confidence_eos_eot_inf,
                    eos_token_id=int(eos_id),
                    confidence_threshold=self.confidence_threshold,
                    fixed_confidence_topk=self.llada_fixed_confidence_topk,
                )
            _DIFFLLM_DECODE_ACC["forward_count"] = _DIFFLLM_DECODE_ACC.get("forward_count", 0) + int(fc)
            _DIFFLLM_DECODE_ACC["decode_token_slots"] = _DIFFLLM_DECODE_ACC.get(
                "decode_token_slots", 0
            ) + int(prompt_ids.shape[0]) * int(self.max_new_tokens)

            class _SeqOut:
                __slots__ = ("sequences",)

                def __init__(self, sequences: torch.Tensor):
                    self.sequences = sequences

            generation_ids = _SeqOut(full)
        else:
            # generate: use ExtendedDreamModel.diffusion_generate_inference with DreamGenerationConfig
            mask_token_id = getattr(self.tokenizer, "mask_token_id", None)
            gen_config = DreamGenerationConfig(
                max_new_tokens=self.max_new_tokens,
                steps=self.diffusion_steps,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                alg=self.alg,
                alg_temp=self.alg_temp,
                return_dict_in_generate=True,
                output_history=False,
                mask_token_id=mask_token_id,
                per_step_transfer_cap=self.per_step_transfer_cap,
                dream_fixed_tokens_per_step=self.dream_fixed_tokens_per_step,
                dream_confidence_threshold=self.dream_confidence_threshold,
            )
            generation_ids = self.model.diffusion_generate_inference(
                inputs=prompt_ids,
                attention_mask=attn_mask,
                generation_config=gen_config,
            )
            acc = getattr(self.model, "_diffllm_dream_step_acc", None)
            if isinstance(acc, dict) and int(acc.get("steps", 0)) > 0:
                _DIFFLLM_DECODE_ACC["forward_count"] = _DIFFLLM_DECODE_ACC.get("forward_count", 0) + int(
                    acc["steps"]
                )
                _DIFFLLM_DECODE_ACC["decode_token_slots"] = _DIFFLLM_DECODE_ACC.get(
                    "decode_token_slots", 0
                ) + int(acc["reveals"])
            else:
                _DIFFLLM_DECODE_ACC["forward_count"] = _DIFFLLM_DECODE_ACC.get("forward_count", 0) + int(
                    self.diffusion_steps
                )
                _DIFFLLM_DECODE_ACC["decode_token_slots"] = _DIFFLLM_DECODE_ACC.get(
                    "decode_token_slots", 0
                ) + int(prompt_ids.shape[0]) * int(self.max_new_tokens)

        # decode
        responses = [
            self.tokenizer.decode(g[len(p) :].tolist()).split(self.tokenizer.eos_token)[0]
            for p, g in zip(prompt_ids, generation_ids.sequences)
        ]

        return responses

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):
        res = []

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        doc_id = [4]
        for batch_idx in range(0, len(requests), self.batch_size):
            #global_idx = batch_idx + self.rank * self.batch_size
            #if global_idx in doc_id:
            #global_id = batch_idx * 4 + self.rank
            #if global_id in doc_id:
            #    print("ckc")
                
            batch_requests = requests[batch_idx : batch_idx + self.batch_size]
            contexts, gen_args = zip(*[req.arguments for req in batch_requests])
            responses = self._generate_batch(contexts)

            for i, r in enumerate(responses):
                for s in gen_args[0]['until']:
                    r = r.split(s)[0]
                responses[i] = r

            if self.rank == 0:
                print(f"Context:\n{contexts[0]}\nResponse:\n{responses[0]}\n")

            res.extend(responses)
            pbar.update(len(contexts))
            #else:
            #    responses = [""] * self.batch_size
            #    res.extend(responses)
            #    contexts = [""] * self.batch_size
            #    pbar.update(len(contexts))


        return res

    def _forward_process(self, batch):
        b, l = batch.shape
        # sample from U[0, 1] following https://arxiv.org/pdf/2107.00630 I.1
        u0 = torch.rand(1, device=batch.device, dtype=torch.float32)
        indices = torch.arange(b, device=batch.device).float()
        t = (u0 + indices / b) % 1

        p_mask = (1 - self.sampling_eps) * t + self.sampling_eps

        p_mask = p_mask[:, None].repeat(1, l)

        mask_indices = torch.rand((b, l), device=batch.device) < p_mask
        # always unmask bos and eos
        mask_indices[:, 0] = False
        mask_indices[:, -1] = False

        noisy_batch = torch.where(mask_indices, self.tokenizer.mask_token_id, batch)
        return noisy_batch, p_mask

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        '''
        prompt_index : 1D bool tensor, length=batch.shape[1]
        '''
        if self.student_backend == "llada":
            raise NotImplementedError("DiffLLM.get_logits / NLL paths are not implemented for backend=llada")

        if self.classifier_free_guidance > 1.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.tokenizer.mask_token_id
            batch = torch.cat([batch, un_batch])

        if self.pad_to_max_len:
            raise NotImplementedError
        else:
            input = batch

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = self.model(input, 'full').logits
            # since bos always unmask, the first logits will not be used
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

        if self.classifier_free_guidance > 1.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + self.cfg * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def _eval_target_nll_mc(self, prefix, target):
        if prefix is None:
            seq = target[None, :]
        else:
            seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        if self.log_type == 'ftb':
            prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        else:
            prompt_index = torch.arange(seq.shape[1], device=self.device) >= len(prefix)

        loss_acc = []
        mc_num = self.diffusion_steps
        for _ in range(max(mc_num // self.batch_size, 1)):
            perturbed_seq = seq.clone()
            perturbed_seq_, p_mask = self._forward_process(seq)
            if self.log_type == 'ftb':
                perturbed_seq[:, -len(target):] = perturbed_seq_[:, -len(target):]
            elif self.log_type == 'btf':
                perturbed_seq[:, :len(prefix)] = perturbed_seq_[:, :len(prefix)]
            elif self.log_type == 'union':
                perturbed_seq = perturbed_seq_
            else:
                raise NotImplementedError(self.log_type)

            mask_indices = perturbed_seq == self.tokenizer.mask_token_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())
            del logits, loss, perturbed_seq, perturbed_seq_, p_mask, mask_indices
            empty_cache_by_memory(threshold_gb=70)

        return sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def _eval_target_nll_ar(self, prefix, target):
        prefix, target = prefix.unsqueeze(0), target.unsqueeze(0) # 1*l1, 1*l2
        assert self.log_type in ['ftb', 'btf']
        assert self.nll_type in ['ar_ftb', 'ar_btf']

        if self.log_type == 'ftb':
            prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) < prefix.shape[1]
        else:
            prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) >= prefix.shape[1]

        if self.log_type == 'ftb':
            perturbed_ = target.repeat(target.shape[1], 1).clone().contiguous() # l2*l2
        else:
            perturbed_ = prefix.repeat(prefix.shape[1], 1).clone().contiguous() # l1*l1

        mask_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        if self.nll_type == 'ar_ftb':
            mask_index = torch.triu(mask_index)
        else:
            mask_index = torch.tril(mask_index)
        perturbed_[mask_index] = self.tokenizer.mask_token_id
        if self.log_type == 'ftb':
            perturbed_seq = torch.cat([prefix.repeat(perturbed_.shape[0], 1), perturbed_], dim=-1)
        else:
            perturbed_seq = torch.cat([perturbed_, target.repeat(perturbed_.shape[0], 1)], dim=-1)

        logits_ = []
        num = len(perturbed_seq) // self.batch_size if len(perturbed_seq) % self.batch_size == 0 else len(perturbed_seq) // self.batch_size + 1
        for i in range(num):
            end = (i + 1) * self.batch_size if (i + 1) * self.batch_size < len(perturbed_seq) else len(perturbed_seq)
            perturbed_seq_ = perturbed_seq[i * self.batch_size: end]
            perturbed_seq_ = perturbed_seq_.to(self.device)
            if len(perturbed_seq_.shape) == 1:
                perturbed_seq_ = perturbed_seq_.unsqueeze(0)
            logits = self.get_logits(perturbed_seq_, prompt_index)
            logits_.append(logits.cpu())
        logits = torch.cat(logits_, dim=0)

        temp_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        if self.nll_type == 'ar_ftb':
            temp_index = torch.triu(temp_index, diagonal=1)
        else:
            temp_index = torch.tril(temp_index, diagonal=-1)
        mask_index[temp_index] = False
        if self.log_type == 'ftb':
            logits_index = torch.cat([torch.zeros((perturbed_.shape[1], prefix.shape[1]), dtype=torch.bool), mask_index], dim=-1)
        else:
            logits_index = torch.cat([mask_index, torch.zeros((perturbed_.shape[1], target.shape[1]), dtype=torch.bool)], dim=-1)

        if self.log_type == 'ftb':
            loss = F.cross_entropy(logits[logits_index], target[0], reduction='sum').cpu().item()
        else:
            loss = F.cross_entropy(logits[logits_index], prefix[0], reduction='sum').cpu().item()
        return loss

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer.encode(context + continuation) + [
            self.tokenizer.eos_token_id
        ]
        context_enc = self.tokenizer.encode(context)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        if self.student_backend == "llada":
            raise NotImplementedError("DiffLLM.loglikelihood is not implemented for backend=llada")

        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                if self.nll_type == 'mc':
                    ll = -self._eval_target_nll_mc(prefix, target)
                    if self.log_type == 'union':
                        ll = ll / (len(target) + len(prefix))
                elif self.nll_type == 'ar_ftb' or self.nll_type == 'ar_btf':
                    ll = -self._eval_target_nll_ar(prefix, target)
                else:
                    raise NotImplementedError(self.nll_type)

                # TODO: greedy decoding
                is_target_greedy_dec = False

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        return out

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError