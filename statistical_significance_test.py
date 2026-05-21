"""
Run statistical significance tests comparing ZCP metrics on two datasets using
open-weight models served via vLLM.

Statistical tests:
1. McNemar's Test        — for binary metrics (accuracy, consistency)
2. Bootstrap Resampling  — non-parametric test (10,000 resamples) for continuous
                           metrics (first/all token confidence); makes no
                           distributional assumptions.

Supported model types (auto-detected or set via --model-type):
  - Qwen Math series    (--model-type qwen)
  - DeepSeek Math series (--model-type deepseek)

--truncate-ratio:
  0.0 — Zero-CoT Probe: model generates the answer with no CoT (the ZCP setting).
  1.0 — Full-CoT baseline: full solution provided; forced-answer generation skipped.

Usage examples:

# Evaluate DeepSeek-Math on GSM8K with the Zero-CoT Probe:
CUDA_VISIBLE_DEVICES=0 python statistical_significance_test.py \
    --model deepseek-ai/deepseek-math-7b-rl \
    --dataset-path-a <path/to/paraphrased_dataset.jsonl> \
    --dataset-path-b <path/to/modified_dataset.jsonl> \
    --data-type-a original \
    --data-type-b modified \
    --truncate-ratio 0 \
    --output-dir <path/to/output_dir> \
    --batch-size 512 \
    --gpu-memory-utilization 0.8

# Evaluate a fine-tuned Qwen3-8B with a LoRA adapter:
CUDA_VISIBLE_DEVICES=0 python statistical_significance_test.py \
    --model Qwen/Qwen3-8B \
    --lora-path <path/to/lora_adapter> \
    --dataset-path-a <path/to/positive_dataset.jsonl> \
    --dataset-path-b <path/to/negative_dataset.jsonl> \
    --data-type-a original \
    --data-type-b modified \
    --truncate-ratio 0 \
    --output-dir <path/to/output_dir> \
    --batch-size 512 \
    --gpu-memory-utilization 0.8

# Using a locally merged model (two clean-format datasets):
CUDA_VISIBLE_DEVICES=0 python statistical_significance_test.py \
    --model <path/to/merged_model> \
    --dataset-path-a <path/to/dataset_c.jsonl> \
    --dataset-path-b <path/to/dataset_u.jsonl> \
    --data-type-a clean \
    --data-type-b clean \
    --truncate-ratio 0 \
    --output-dir <path/to/output_dir>

# Manually specify model type (if auto-detection fails):
CUDA_VISIBLE_DEVICES=0 python statistical_significance_test.py \
    --model /path/to/custom/model \
    --model-type deepseek \
    --dataset-path-a dataset_a.jsonl \
    --dataset-path-b dataset_b.jsonl \
    --data-type-a clean \
    --data-type-b clean \
    --truncate-ratio 0 \
    --output-dir ./results
"""

import os
# Fix MKL threading library conflict
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"

import json
import argparse
import re
import random
import gc
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
import seaborn as sns

from math_grade import grade_answer


def compute_ccs(cohens_d: float, p_value: float, pi: float = 0.5) -> float:
    """
    Compute the Composite Confidence Score (CCS).
    
    Uses a p-value-based Bayes factor calibration method to compute posterior confidence.
    
    Method:
    1. Convert p-value to a calibrated Bayes factor upper bound:
       - BF_10 = 1/(-e*p*ln(p)) when p < 1/e
       - BF_10 = 1 when p >= 1/e
    2. Convert BF to posterior probability using prior π:
       Conf_B = (BF*π) / (BF*π + (1-π))
    3. Apply direction gate D (positive effect D=1, negative effect D=0):
       CCS = D * Conf_B
    
    Args:
        cohens_d: Cohen's d effect size (can be positive or negative)
        p_value: p-value from the statistical test
        pi: prior probability P(H_1), default 0.5 (uninformative prior)
    
    Returns:
        CCS score in [0, 1]; higher means stronger and more significant effect
    """
    if p_value is None or cohens_d is None:
        return None
    
    import math
    
    # 1. Compute the calibrated Bayes factor
    if p_value <= 0 or p_value >= 1:
        return None
    
    if p_value < 1/math.e:
        bf_10 = 1.0 / (-math.e * p_value * math.log(p_value))
    else:
        bf_10 = 1.0
    
    # 2. Convert Bayes factor to posterior confidence
    conf_b = (bf_10 * pi) / (bf_10 * pi + (1 - pi))
    
    # 3. Direction gate (only positive effects score)
    D = 1.0 if cohens_d > 0 else 0.0
    
    # 4. Composite score
    ccs = D * conf_b
    
    return float(ccs)


def filter_by_index_range(results: List[Dict], index_start: int = None, index_end: int = None) -> List[Dict]:
    """
    Filter results by index range (based on the dataset's own index, not original_index)
    
    Args:
        results: list of results
        index_start: start index (inclusive); None means from the beginning
        index_end: end index (inclusive); None means to the end
        
    Returns:
        filtered list of results
    """
    if index_start is None and index_end is None:
        return results
    
    filtered = []
    for item in results:
        idx = item.get('index', -1)
        
        # If there is no index field, keep the item
        if idx == -1:
            continue
        
        # Check whether the index is within the requested range
        in_range = True
        if index_start is not None and idx < index_start:
            in_range = False
        if index_end is not None and idx > index_end:
            in_range = False
        
        if in_range:
            filtered.append(item)
    
    print(f"  Filtered by index range: {len(results)} -> {len(filtered)} samples")
    if index_start is not None or index_end is not None:
        range_str = f"[{index_start if index_start is not None else '0'}:{index_end if index_end is not None else 'end'}]"
        print(f"  Index range: {range_str}")
    
    return filtered


def match_datasets_by_original_index(
    results_a: List[Dict],
    results_b: List[Dict]
) -> Tuple[List[Dict], List[Dict]]:
    """
    Match results from two datasets by original_index field and validate the matching.
    
    Args:
        results_a: Results from dataset A (prefer original_index field; fall back to index)
        results_b: Results from dataset B (must have original_index field)
    
    Returns:
        Two matched result lists in corresponding order
        
    Raises:
        ValueError: If dataset B is missing original_index field, or if matching validation fails
    """
    # Build index map for dataset A
    # Prefer original_index field; fall back to index if absent
    a_dict = {}
    a_has_original_index = False
    for item in results_a:
        if 'original_index' in item:
            orig_idx = str(item['original_index'])  # convert to string
            a_has_original_index = True
        else:
            orig_idx = str(item['index'])  # convert to string
        a_dict[orig_idx] = item
    
    if a_has_original_index:
        print("  Dataset A: using original_index field")
    else:
        print("  Dataset A: using index field as original_index")
    
    # Build index map for dataset B
    b_dict = {}
    for item in results_b:
        if 'original_index' not in item:
            raise ValueError("Dataset B must contain an original_index field")
        orig_idx = str(item['original_index'])  # convert to string
        b_dict[orig_idx] = item
    
    print("  Dataset B: using original_index field")
    
    # Find the intersection of original_index values
    common_indices = sorted(set(a_dict.keys()) & set(b_dict.keys()))
    
    if not common_indices:
        raise ValueError("The two datasets have no common original_index values; cannot match")
    
    # Re-order results according to common_indices
    matched_a = [a_dict[idx] for idx in common_indices]
    matched_b = [b_dict[idx] for idx in common_indices]
    
    # Validate matching by comparing question text
    print(f"\nValidating dataset matching...")
    mismatches = []
    for idx, (item_a, item_b) in enumerate(zip(matched_a, matched_b)):
        orig_idx = common_indices[idx]
        
        # Extract question text from dataset A (prefer 'question', fall back to 'original_problem')
        problem_a = item_a.get('question', item_a.get('original_problem', ''))
        
        # Extract question text from dataset B (using 'original_problem')
        problem_b = item_b.get('original_problem', '')
        
        # Simple text matching: strip extra whitespace before comparing
        problem_a_normalized = ' '.join(problem_a.split())
        problem_b_normalized = ' '.join(problem_b.split())
        
        if problem_a_normalized != problem_b_normalized:
            mismatches.append({
                'original_index': orig_idx,
                'problem_a': problem_a[:100] + '...' if len(problem_a) > 100 else problem_a,
                'problem_b': problem_b[:100] + '...' if len(problem_b) > 100 else problem_b
            })
    
    if mismatches:
        print(f"\n❌ Found {len(mismatches)} mismatched sample pair(s)!")
        print("First 5 mismatches:")
        for i, mismatch in enumerate(mismatches[:5]):
            print(f"\n  [{i+1}] original_index = {mismatch['original_index']}")
            print(f"      Dataset A question: {mismatch['problem_a']}")
            print(f"      Dataset B question: {mismatch['problem_b']}")
        
        raise ValueError(
            f"Dataset matching validation failed: found {len(mismatches)} sample pair(s) with mismatched question text. "
            f"Please verify that dataset A and dataset B use the correct original_index correspondence."
        )
    
    print(f"✓ Validation passed: all {len(matched_a)} sample pairs have matching question text")
    
    print(f"\nDataset matching complete:")
    print(f"  Dataset A original size: {len(results_a)}")
    print(f"  Dataset B original size: {len(results_b)}")
    print(f"  Matched sample pairs: {len(matched_a)}")
    
    if len(matched_a) < len(results_a) or len(matched_b) < len(results_b):
        a_only = set(a_dict.keys()) - set(b_dict.keys())
        b_only = set(b_dict.keys()) - set(a_dict.keys())
        if a_only:
            print(f"  Warning: {len(a_only)} sample(s) in A have no match in B: {sorted(list(a_only))[:10]}...")
        if b_only:
            print(f"  Warning: {len(b_only)} sample(s) in B have no match in A: {sorted(list(b_only))[:10]}...")
    
    return matched_a, matched_b


def make_absolute_path(path):
    """Ensure the path is absolute (but do not convert HuggingFace model names)"""
    if not path:
        return path
    
    # Check if the path looks like a HuggingFace model name (e.g., 'Qwen/Qwen2.5-Math-7B-Instruct')
    # HuggingFace model names contain '/' but do not start with '/' and lack other path indicators
    if '/' in path and not path.startswith('/') and not path.startswith('.') and not os.path.exists(path):
        # Simple check: if it matches 'org/model-name' format and is not an existing path, keep as-is
        parts = path.split('/')
        if len(parts) == 2 and not any(c in path for c in ['..', '~']):
            return path  # Keep HuggingFace model name unchanged
    
    # For real relative paths, convert to absolute
    if not os.path.isabs(path):
        return os.path.abspath(path)
    
    return path


def set_random_seed(seed: int = 42):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")


class StatisticalSignificanceTester:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        gpu_id: Optional[int] = None,
        seed: int = 42,
        lora_path: Optional[str] = None,
        model_type: Optional[str] = None
    ):
        """
        Initialize the tester
        
        Args:
            model_name: model name or path
            tensor_parallel_size: tensor parallel size
            gpu_memory_utilization: GPU memory utilization
            max_model_len: maximum model length
            gpu_id: GPU ID to use
            seed: random seed
            lora_path: LoRA adapter path (if applicable)
            model_type: model type ('qwen'/'deepseek'/'auto'); defaults to auto-detect
        """
        if gpu_id is not None:
            print(f"Hint: ensure CUDA_VISIBLE_DEVICES={gpu_id} is set before running")
        
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            print(f"Current CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        else:
            print("Warning: CUDA_VISIBLE_DEVICES environment variable is not set")
        
        # Convert to absolute path
        self.model_name = make_absolute_path(model_name)
        self.lora_path = make_absolute_path(lora_path) if lora_path else None
        
        # Detect model type
        self.model_type = self._detect_model_type(model_name, model_type)
        print(f"Detected model type: {self.model_type}")

        # The Qwen3 chat template supports an enable_thinking flag.
        # For Qwen/Qwen3-8B, disable thinking by default (per HF docs) to prevent <think> tokens from interfering with parsing.
        model_name_lower_raw = str(model_name).lower() if model_name else ""
        model_name_lower_abs = str(self.model_name).lower() if self.model_name else ""
        self._is_qwen3_8b = (
            self.model_type == "qwen"
            and ("qwen3-8b" in model_name_lower_raw or "qwen3-8b" in model_name_lower_abs)
        )
        self._disable_qwen3_thinking = self._is_qwen3_8b
        if self._disable_qwen3_thinking:
            print("Detected Qwen3-8B: setting enable_thinking=False")
        
        print(f"Loading vLLM model: {self.model_name}")
        if self.lora_path:
            print(f"LoRA adapter path: {self.lora_path}")
        
        set_random_seed(seed)
        
        llm_kwargs = {
            "model": self.model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
            "seed": seed
        }
        
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        
        # If a LoRA path is provided, configure vLLM to use LoRA
        if self.lora_path:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = 64
        
        self.llm = LLM(**llm_kwargs)
        
        # Load tokenizer
        tokenizer_path = self.lora_path if self.lora_path and os.path.exists(os.path.join(self.lora_path, "tokenizer_config.json")) else self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=False,
            trust_remote_code=True
        )
        
        self.gpu_id = gpu_id
        self.seed = seed
        
        # HuggingFace model (lazy-loaded)
        self.hf_model = None
        self.hf_tokenizer = None
        
        print(f"vLLM model loaded successfully")
    
    def _detect_model_type(self, model_name: str, model_type: Optional[str] = None) -> str:
        """
        Detect the model type
        
        Args:
            model_name: model name or path
            model_type: manually specified model type
            
        Returns:
            model type: 'qwen' or 'deepseek'
        """
        if model_type and model_type.lower() != 'auto':
            return model_type.lower()
        
        # Auto-detect
        model_name_lower = model_name.lower()
        if 'deepseek' in model_name_lower:
            return 'deepseek'
        elif 'qwen' in model_name_lower:
            return 'qwen'

        # Default to 'qwen' (consistent with the script description) to avoid returning None.
        print(f"Warning: cannot auto-detect model type ({model_name}); defaulting to qwen format")
        return 'qwen'

    def _apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool
    ) -> str:
        """Unified wrapper for tokenizer.apply_chat_template, compatible with Qwen3's enable_thinking parameter."""
        kwargs = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        }

        if getattr(self, "_disable_qwen3_thinking", False):
            kwargs["enable_thinking"] = False

        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            # Some tokenizers do not support enable_thinking; fall back to the default signature
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _make_vllm_sampling_params(self, *, max_tokens: int) -> SamplingParams:
        """Create vLLM SamplingParams with Qwen3-8B non-thinking best practices.

        For Qwen3-8B when thinking is disabled, use:
        Temperature=0.7, TopP=0.8, TopK=20, MinP=0.
        Falls back gracefully if the local vLLM version doesn't support some fields.
        """

        if getattr(self, "_is_qwen3_8b", False) and getattr(self, "_disable_qwen3_thinking", False):
            params_kwargs = {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
                "max_tokens": max_tokens,
                "stop": None,
            }
        else:
            # Default: deterministic generation for evaluation.
            params_kwargs = {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": max_tokens,
                "stop": None,
            }

        # vLLM version compatibility: some versions may not support min_p/top_k.
        for key_to_drop in (None, "min_p", "top_k"):
            try:
                if key_to_drop is not None:
                    params_kwargs.pop(key_to_drop, None)
                return SamplingParams(**params_kwargs)
            except TypeError:
                continue

        # As a last resort, only keep universally supported fields.
        return SamplingParams(
            temperature=params_kwargs.get("temperature", 0.0),
            top_p=params_kwargs.get("top_p", 1.0),
            max_tokens=max_tokens,
            stop=None,
        )
    
    def format_chat_prompt(self, problem: str, solution_prefix: str = "") -> str:
        """Format a chat prompt"""
        system_content = "Please reason step by step, and put your final answer within \\boxed{}."
        
        if self.model_type == 'deepseek':
            # DeepSeek does not use the system role; append system content to the user message
            messages = [
                {"role": "user", "content": f"{problem}\n{system_content}"}
            ]
        else:  # qwen
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": problem}
            ]
        
        if solution_prefix:
            messages.append({"role": "assistant", "content": solution_prefix})
            text = self._apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            text = text.rstrip()
            # Remove end token
            if self.model_type == 'deepseek':
                if text.endswith("<｜end▁of▁sentence｜>"):
                    text = text[:-len("<｜end▁of▁sentence｜>")]
            else:  # qwen
                if text.endswith("<|im_end|>"):
                    text = text[:-len("<|im_end|>")]
        else:
            text = self._apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        return text
    
    def format_continued_prompt(
        self,
        problem: str,
        truncated_response: str,
        force_answer_prompt: str = "\n\nThe final answer is:\n\\[\n\\boxed"
    ) -> str:
        """Format the continuation prompt after truncation"""
        system_content = "Please reason step by step, and put your final answer within \\boxed{}."
        
        if self.model_type == 'deepseek':
            # DeepSeek does not use the system role
            messages_continued = [
                {"role": "user", "content": f"{problem}\n{system_content}"},
                {"role": "assistant", "content": truncated_response + force_answer_prompt}
            ]
        else:  # qwen
            messages_continued = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": truncated_response + force_answer_prompt}
            ]
        
        text_continued = self._apply_chat_template(messages_continued, tokenize=False, add_generation_prompt=False)
        
        text_continued = text_continued.rstrip()
        # Remove end token
        if self.model_type == 'deepseek':
            if text_continued.endswith("<｜end▁of▁sentence｜>"):
                text_continued = text_continued[:-len("<｜end▁of▁sentence｜>")]
        else:  # qwen
            if text_continued.endswith("<|im_end|>"):
                text_continued = text_continued[:-len("<|im_end|>")]
        
        return text_continued
    
    def extract_boxed_answer(self, text: str) -> Optional[str]:
        """
        Extract the answer from text
        Prefer extracting from \\boxed{} (extract the last \\boxed{})
        If that fails, try extracting after "The answer is:"
        """
        # Method 1: extract from \\boxed{}
        start_pattern = r'\\boxed\{'
        matches = list(re.finditer(start_pattern, text))
        
        if matches:
            # Extract starting from the last \boxed{
            match = matches[-1]
            start_pos = match.end()
            brace_count = 1
            i = start_pos
            
            while i < len(text) and brace_count > 0:
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                i += 1
            
            if brace_count == 0:
                return text[start_pos:i-1].strip()
        
        # Method 2: extract after "The answer is:" (fallback for DeepSeek and similar models)
        # Match pattern: The answer is: $...$  or  The answer is: ...
        answer_patterns = [
            r'[Tt]he answer is:\s*\$([^$]+)\$',  # The answer is: $17$
            r'[Tt]he answer is:\s*\\\[([^\]]+)\\\]',  # The answer is: \[17\]
            r'[Tt]he answer is:\s*([^\n\.]+)',  # The answer is: 17
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, text)
            if match:
                answer = match.group(1).strip()
                # Remove possible LaTeX formatting wrappers
                answer = re.sub(r'^\\\(|\\\)$', '', answer)
                answer = re.sub(r'^\\\[|\\\]$', '', answer)
                return answer
        
        return None
    
    def release_vllm_model(self):
        """Release GPU memory occupied by the vLLM model."""
        print("\nReleasing vLLM model GPU memory...")
        if hasattr(self, 'llm') and self.llm is not None:
            del self.llm
            self.llm = None
        
        gc.collect()
        torch.cuda.empty_cache()
        
        import torch.distributed as dist
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as e:
                print(f"Error cleaning up distributed process group: {e}")
        
        # Multiple cleanup passes to ensure GPU memory is fully released
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        
        import time
        time.sleep(5)  # extra wait time
        
        print("vLLM model GPU memory released")
    
    def load_hf_model(self):
        """Load HuggingFace model for confidence computation"""
        if self.hf_model is not None:
            print("HuggingFace model already loaded")
            return
        
        print(f"\nLoading HuggingFace model: {self.model_name}")
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"HuggingFace model target device: {device}")
        
        # Load tokenizer
        tokenizer_path = self.lora_path if self.lora_path and os.path.exists(os.path.join(self.lora_path, "tokenizer_config.json")) else self.model_name
        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=False
        )
        
        # Load base model
        print(f"Loading base model: {self.model_name}")
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        
        # If a LoRA path is provided, load the LoRA adapter
        if self.lora_path:
            print(f"Loading LoRA adapter: {self.lora_path}")
            self.hf_model = PeftModel.from_pretrained(
                self.hf_model,
                self.lora_path,
                device_map=device
            )
        
        self.hf_model.eval()
        print(f"HuggingFace model loaded")
    
    def release_hf_model(self):
        """Release GPU memory used by the HuggingFace model"""
        print("\nReleasing HuggingFace model GPU memory...")
        if hasattr(self, 'hf_model') and self.hf_model is not None:
            del self.hf_model
            del self.hf_tokenizer
            self.hf_model = None
            self.hf_tokenizer = None
        
        # Multiple cleanup passes to ensure GPU memory is fully released
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        
        import time
        time.sleep(3)
        
        print("HuggingFace model GPU memory released")
    
    def compute_token_confidences(
        self,
        prompt: str,
        answer: str,
        compute_all_tokens: bool = True
    ) -> Dict:
        """
        Compute token confidence for an answer (teacher-forcing style)
        
        Args:
            prompt: input prompt (problem + truncated solution)
            answer: ground truth answer
            compute_all_tokens: whether to compute the average confidence over all tokens
            
        Returns:
            dict containing confidence information
        """
        # Build the full answer format
        answer_prefix = "\n\nThe final answer is:\n\\[\n\\boxed{"
        answer_suffix = "}\n\\]"
        formatted_answer = f"{answer_prefix}{answer}{answer_suffix}"
        
        # Tokenization
        prompt_ids = self.hf_tokenizer.encode(prompt, add_special_tokens=False)
        prefix_ids = self.hf_tokenizer.encode(answer_prefix, add_special_tokens=False)
        answer_ids = self.hf_tokenizer.encode(answer, add_special_tokens=False)
        suffix_ids = self.hf_tokenizer.encode(answer_suffix, add_special_tokens=False)
        
        # Combine the full input
        full_ids = prompt_ids + prefix_ids + answer_ids + suffix_ids
        
        # Convert to tensor
        device = self.hf_model.device
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        
        # Forward pass to get logits
        with torch.no_grad():
            outputs = self.hf_model(input_ids)
            logits = outputs.logits
        
        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Extract token log-probs for the answer portion
        answer_start_idx = len(prompt_ids) + len(prefix_ids)
        
        if len(answer_ids) == 0:
            return {
                "first_token_confidence": 0.0,
                "all_token_confidence": 0.0,
                "num_answer_tokens": 0
            }
        
        answer_token_logprobs = []
        answer_tokens = []
        
        for i in range(len(answer_ids)):
            position = answer_start_idx + i
            if position > 0 and position - 1 < logits.shape[1]:
                token_id = answer_ids[i]
                logprob = log_probs[0, position - 1, token_id].item()
                answer_token_logprobs.append(logprob)
                answer_tokens.append(self.hf_tokenizer.decode([token_id]))
        
        # Compute confidence (probability = exp(logprob))
        # Use geometric mean: exp(mean(logprobs)); more standard in sequence modeling
        confidences = [np.exp(lp) for lp in answer_token_logprobs]
        
        return {
            "first_token_confidence": float(confidences[0]) if confidences else 0.0,
            "all_token_confidence": float(np.exp(np.mean(answer_token_logprobs))) if answer_token_logprobs else 0.0,
            "num_answer_tokens": len(answer_ids),
            "tokens": answer_tokens,
            "confidences": confidences
        }
    
    def compute_metrics_on_dataset(
        self,
        dataset_path: str,
        truncate_ratio: float,
        max_samples: Optional[int] = None,
        batch_size: int = 32,
        data_type: str = "clean",
        pregenerated_solutions: Optional[List[str]] = None,
        use_original_index: bool = False
    ) -> Tuple[List[Dict], List[str]]:
        """
        Compute metrics on a dataset
        
        Args:
            dataset_path: path to the dataset file (JSONL)
            truncate_ratio: truncation ratio (0.0–1.0)
                - 0.0: fully truncated; generate from scratch
                - 0.5: truncate 50% of the solution
                - 1.0: use the full solution; no forced answer generated (uses full solution result directly)
            max_samples: Maximum number of test samples
            batch_size: batch size
            data_type: data type ("clean", "original", "paraphrased", "modified")
            pregenerated_solutions: pre-generated solutions
            use_original_index: whether to use the original_index field (if False, use file order as index)
            
        Returns:
            list of results with metrics for all samples, along with generated solutions
            
        Note:
            When truncate_ratio=1.0, the forced-answer generation step is skipped;
            the full solution result is used directly, and forced_answer equals full_answer.
        """
        # Loading dataset
        print(f"Loading dataset: {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]
        
        if max_samples:
            dataset = dataset[:max_samples]
        
        print(f"Dataset size: {len(dataset)}")
        
        # Prepare sample data
        samples = []
        for idx, item in enumerate(dataset):
            if data_type == "clean":
                problem = item.get('question')
                answer = item.get('answer')
            elif data_type == "paraphrased":
                problem = item.get('paraphrased_problem', '')
                answer = item.get('answer', item.get('paraphrased_answer', ''))
            elif data_type == "modified":
                problem = item.get('modified_problem', '')
                answer = item.get('modified_answer', '')
            else:  # original
                problem = item.get('original_problem', item.get('problem', ''))
                answer = item.get('original_answer', item.get('answer', ''))
            
            if problem and answer:
                # Use original_index field if present and use_original_index=True; otherwise use file order
                original_index = item.get('original_index', idx) if use_original_index else idx
                samples.append({
                    'index': idx,
                    'original_index': original_index,
                    'problem': problem,
                    'answer': answer
                })
        
        print(f"Valid sample count: {len(samples)}")
        
        problems = [s['problem'] for s in samples]
        
        # Step 1: Generate full solutions (unless pre-generated solutions are provided)
        if pregenerated_solutions is None:
            print(f"\nStep 1: generating full solutions...")
            prompts = [self.format_chat_prompt(p) for p in problems]

            sampling_params = self._make_vllm_sampling_params(max_tokens=2048)
            
            all_solutions = []
            for batch_start in tqdm(range(0, len(prompts), batch_size), desc="Generating solutions"):
                batch_end = min(batch_start + batch_size, len(prompts))
                batch_prompts = prompts[batch_start:batch_end]
                
                # Specify the LoRA adapter if LoRA is being used
                if self.lora_path:
                    from vllm.lora.request import LoRARequest
                    lora_request = LoRARequest("adapter", 1, self.lora_path)
                    outputs = self.llm.generate(batch_prompts, sampling_params, lora_request=lora_request)
                else:
                    outputs = self.llm.generate(batch_prompts, sampling_params)
                
                for output in outputs:
                    all_solutions.append(output.outputs[0].text)
            
            print(f"Successfully generated {len(all_solutions)} solutions")
        else:
            print(f"\nStep 1: using pre-generated solutions (skipping generation step)")
            all_solutions = pregenerated_solutions
            print(f"Reusing {len(all_solutions)} solutions")
        
        # Step 2: Truncate and generate forced answers
        # Special case: when truncate_ratio == 1.0, use the full solution directly; no forced answers needed
        if truncate_ratio == 1.0:
            print(f"\nStep 2: truncate_ratio=1.0, using full solutions directly (skipping forced-answer generation)")
            all_truncated_solutions = all_solutions.copy()
            all_forced_answers_text = ["" for _ in all_solutions]  # empty string = not generated
            all_forced_answers_full = all_solutions.copy()  # use full solution
            print(f"Reusing {len(all_solutions)} full solutions as forced answers")
        else:
            print(f"\nStep 2: generating forced answers after truncation...")
            
            all_truncated_prompts = []
            all_truncated_solutions = []
            
            for problem, solution in zip(problems, all_solutions):
                truncate_position = int(len(solution) * truncate_ratio)
                truncated_solution = solution[:truncate_position]
                all_truncated_solutions.append(truncated_solution)
                
                continued_prompt = self.format_continued_prompt(problem, truncated_solution)
                all_truncated_prompts.append(continued_prompt)
            
            # Batch generate forced answers
            forced_sampling_params = self._make_vllm_sampling_params(max_tokens=32)
            
            all_forced_answers_text = []
            all_forced_answers_full = []
            
            for batch_start in tqdm(range(0, len(all_truncated_prompts), batch_size), desc="Generating forced answers"):
                batch_end = min(batch_start + batch_size, len(all_truncated_prompts))
                batch_prompts = all_truncated_prompts[batch_start:batch_end]
                
                if self.lora_path:
                    from vllm.lora.request import LoRARequest
                    lora_request = LoRARequest("adapter", 1, self.lora_path)
                    outputs = self.llm.generate(batch_prompts, forced_sampling_params, lora_request=lora_request)
                else:
                    outputs = self.llm.generate(batch_prompts, forced_sampling_params)
                
                for idx, output in enumerate(outputs):
                    generated_text = output.outputs[0].text
                    all_forced_answers_text.append(generated_text)
                    full_forced_text = batch_prompts[idx] + generated_text
                    all_forced_answers_full.append(full_forced_text)
            
            print(f"Successfully generated {len(all_forced_answers_text)} forced answers")
        
        # Release vLLM and load HF model
        self.release_vllm_model()
        self.load_hf_model()
        
        # Step 3: Compute all metrics
        print(f"\nStep 3: computing metrics...")
        
        results = []
        for idx, sample in enumerate(tqdm(samples, desc="Computing metrics")):
            problem = problems[idx]
            gt_answer = sample['answer']
            solution = all_solutions[idx]
            truncated_sol = all_truncated_solutions[idx]
            forced_full = all_forced_answers_full[idx]
            
            # Extract answers
            full_answer = self.extract_boxed_answer(solution)
            forced_answer = self.extract_boxed_answer(forced_full)
            
            # Compute accuracy and consistency
            full_correct = grade_answer(full_answer, gt_answer) if full_answer else False
            forced_correct = grade_answer(forced_answer, gt_answer) if forced_answer else False
            consistent = grade_answer(full_answer, forced_answer) if (full_answer and forced_answer) else False
            
            # Compute confidence
            prompt_for_confidence = self.format_chat_prompt(problem, truncated_sol)
            confidence_result = self.compute_token_confidences(
                prompt_for_confidence, gt_answer.strip(), compute_all_tokens=True
            )
            
            result = {
                'index': sample['index'],
                'original_index': sample['original_index'],
                'problem': problem,
                'ground_truth_answer': gt_answer,
                'full_solution': solution,
                'truncated_solution': truncated_sol,
                # Full model output for forced answer (including prompt + generated text), and the pure generated portion
                'forced_response_full': forced_full,
                'forced_response_generated': all_forced_answers_text[idx],
                'full_answer': full_answer,
                'forced_answer': forced_answer,
                'full_accuracy': 1 if full_correct else 0,
                'accuracy': 1 if forced_correct else 0,
                'consistency': 1 if consistent else 0,
                'first_token_confidence': confidence_result['first_token_confidence'],
                'all_token_confidence': confidence_result['all_token_confidence'],
                'num_answer_tokens': confidence_result['num_answer_tokens']
            }
            results.append(result)
        
        return results, all_solutions


def perform_mcnemar_test(results_a: List[Dict], results_b: List[Dict], metric: str, alternative: str = 'greater') -> Dict:
    """
    Perform McNemar's Test on two sets of results
    
    Args:
        results_a: results from model/dataset A
        results_b: results from model/dataset B
        metric: metric name to test ("accuracy" or "consistency")
        alternative: test type — 'two-sided', 'greater' (A>B one-sided), 'less' (A<B one-sided)
        
    Returns:
        dict containing test results
    """
    assert len(results_a) == len(results_b), "Both result sets must have the same number of samples"
    
    # Extract metric values
    values_a = [r[metric] for r in results_a]
    values_b = [r[metric] for r in results_b]
    
    # Build contingency table
    # McNemar's test requires a 2x2 contingency table
    # Row: Model A (0=incorrect, 1=correct)
    # Column: Model B (0=incorrect, 1=correct)
    n_00 = sum(1 for a, b in zip(values_a, values_b) if a == 0 and b == 0)
    n_01 = sum(1 for a, b in zip(values_a, values_b) if a == 0 and b == 1)
    n_10 = sum(1 for a, b in zip(values_a, values_b) if a == 1 and b == 0)
    n_11 = sum(1 for a, b in zip(values_a, values_b) if a == 1 and b == 1)
    
    contingency_table = np.array([[n_00, n_01], [n_10, n_11]])
    
    # Perform McNemar's test (one-sided or two-sided)
    try:
        result = mcnemar(contingency_table, exact=True, correction=True)
        statistic = result.statistic
        
        # Compute p-value based on alternative
        if alternative == 'two-sided':
            p_value = result.pvalue
        elif alternative == 'greater':
            # One-sided test: H1: n_10 > n_01 (A > B)
            p_value = result.pvalue / 2 if n_10 > n_01 else 1 - result.pvalue / 2
        elif alternative == 'less':
            # One-sided test: H1: n_10 < n_01 (A < B)
            p_value = result.pvalue / 2 if n_10 < n_01 else 1 - result.pvalue / 2
        else:
            raise ValueError(f"Invalid alternative: {alternative}")
    except Exception as e:
        print(f"McNemar's test failed: {e}")
        statistic = None
        p_value = None
    
    # Compute effect size
    n_discordant = n_01 + n_10
    if n_discordant > 0:
        effect_size = abs(n_10 - n_01) / n_discordant
    else:
        effect_size = 0.0
    
    # Compute CCS (Composite Confidence Score)
    # McNemar's test uses effect_size as the effect measure (analogous to Cohen's d)
    ccs = compute_ccs(effect_size, p_value) if p_value is not None else None
    
    return {
        "metric": metric,
        "test": f"McNemar's Test ({alternative})",
        "alternative": alternative,
        "contingency_table": {
            "both_wrong": n_00,
            "a_wrong_b_correct": n_01,
            "a_correct_b_wrong": n_10,
            "both_correct": n_11
        },
        "statistic": float(statistic) if statistic is not None else None,
        "p_value": f"{p_value:.3e}" if p_value is not None else None,
        "significant_at_0.05": bool(p_value < 0.05) if p_value is not None else None,
        "significant_at_0.01": bool(p_value < 0.01) if p_value is not None else None,
        "effect_size": float(effect_size),
        "ccs": float(ccs) if ccs is not None else None,
        "a_performance": sum(values_a) / len(values_a),
        "b_performance": sum(values_b) / len(values_b),
        "performance_difference": sum(values_b) / len(values_b) - sum(values_a) / len(values_a)
    }


def perform_bootstrap_test(
    results_a: List[Dict], 
    results_b: List[Dict], 
    metric: str, 
    n_bootstrap: int = 10000,
    alternative: str = 'greater',
    confidence_level: float = 0.95,
    seed: int = 42
) -> Dict:
    """
    Perform Bootstrap Resampling Test on two sets of results (using scipy.stats.bootstrap)
    
    Bootstrap resampling estimates the statistic distribution via resampling; it is a nonparametric method
    that does not require distributional assumptions, especially suited for small or non-normally distributed samples.
    
    Args:
        results_a: results from model/dataset A
        results_b: results from model/dataset B
        metric: metric name to test ("first_token_confidence" or "all_token_confidence")
        n_bootstrap: number of bootstrap resamples (default 10000)
        alternative: test type — 'two-sided', 'greater' (A>B one-sided), 'less' (A<B one-sided)
        confidence_level: confidence level (default 0.95)
        seed: random seed
        
    Returns:
        dict containing test results
    """
    assert len(results_a) == len(results_b), "Both result sets must have the same number of samples"
    
    # Extract metric values, handling None
    values_a = [r.get(metric, 0.0) if r.get(metric) is not None else 0.0 for r in results_a]
    values_b = [r.get(metric, 0.0) if r.get(metric) is not None else 0.0 for r in results_b]
    
    # Filter out sample pairs where both confidence values are 0
    valid_pairs = [(a, b) for a, b in zip(values_a, values_b) if not (a == 0 and b == 0)]
    
    if len(valid_pairs) < 2:
        return {
            "metric": metric,
            "test": f"Bootstrap Test ({alternative})",
            "alternative": alternative,
            "p_value": None,
            "error": "Insufficient valid samples"
        }
    
    valid_a = np.array([a for a, b in valid_pairs])
    valid_b = np.array([b for a, b in valid_pairs])
    
    # === Key correction: pre-compute paired differences ===
    # Transform the paired problem into a single-sample mean problem (H0: mean(diffs) = 0)
    diffs = valid_a - valid_b
    observed_diff = np.mean(diffs)
    
    # Define statistic function: compute mean of differences
    def statistic_diff(d, axis):
        return np.mean(d, axis=axis)
    
    # Use scipy.stats.bootstrap for the bootstrap test
    try:
        from scipy.stats import bootstrap
        
        # Bootstrap the paired differences (not independently resampling two arrays)
        rng = np.random.default_rng(seed)
        result = bootstrap(
            (diffs,),  # Note: pass the differences array to preserve pairing
            statistic_diff,
            n_resamples=n_bootstrap,
            confidence_level=confidence_level,
            alternative=alternative,
            method='percentile',
            random_state=rng
        )
        
        # Extract confidence interval and standard error
        ci_lower = result.confidence_interval.low
        ci_upper = result.confidence_interval.high
        bootstrap_se = result.standard_error
        
        # Try to reuse bootstrap distribution (scipy >= 1.10.0)
        # If unavailable, generate manually (older scipy)
        if hasattr(result, 'bootstrap_distribution'):
            # Optimization: reuse the already-generated bootstrap distribution
            bootstrap_means = result.bootstrap_distribution
            print(f"  Optimization: reusing {len(bootstrap_means)} samples from scipy.stats.bootstrap")
        else:
            # Fallback: manually generate bootstrap distribution for p-value (null shift)
            print(f"  Note: older scipy version; generating bootstrap distribution manually")
            rng = np.random.default_rng(seed)
            n_samples = len(diffs)
            bootstrap_means = np.array([
                np.mean(rng.choice(diffs, size=n_samples, replace=True))
                for _ in range(n_bootstrap)
            ])
        
        # Shift bootstrap distribution to null-hypothesis center (H0: μ_diff = 0)
        # This is the key step in bootstrap hypothesis testing
        distribution_mean = np.mean(bootstrap_means)
        null_distribution = bootstrap_means - distribution_mean  # shift to 0 center
        
        # Compute p-value: how extreme the observed value is under the null distribution
        if alternative == 'two-sided':
            # Two-sided: proportion of |bootstrap_mean| >= |observed_diff|
            p_value = np.mean(np.abs(null_distribution) >= np.abs(observed_diff))
        elif alternative == 'greater':
            # One-sided H1: A > B (observed_diff > 0)
            # Proportion of null distribution >= observed_diff
            p_value = np.mean(null_distribution >= observed_diff)
        elif alternative == 'less':
            # One-sided H1: A < B (observed_diff < 0)
            # Proportion of null distribution <= observed_diff
            p_value = np.mean(null_distribution <= observed_diff)
        else:
            raise ValueError(f"Invalid alternative: {alternative}")
        
        # Compute Cohen's d_z as effect size (standard formula for paired data)
        # Use the standard deviation of differences, not the pooled standard deviation
        std_diff = np.std(diffs, ddof=1)
        cohens_d = observed_diff / std_diff if std_diff > 0 else 0.0
        
        # Compute CCS (Composite Confidence Score)
        ccs = compute_ccs(cohens_d, p_value)
        
        return {
            "metric": metric,
            "test": f"Paired Bootstrap Resampling Test ({alternative})",
            "method": "scipy.stats.bootstrap on paired differences + centered null hypothesis",
            "alternative": alternative,
            "n_bootstrap": n_bootstrap,
            "observed_difference": float(observed_diff),
            "p_value": f"{p_value:.3e}",
            "p_value_note": "Computed by centering bootstrap distribution at H0: μ_diff = 0",
            "significant_at_0.05": bool(p_value < 0.05),
            "significant_at_0.01": bool(p_value < 0.01),
            "confidence_level": confidence_level,
            "confidence_interval": {
                "lower": float(ci_lower) if ci_lower is not None else None,
                "upper": float(ci_upper) if ci_upper is not None else None
            },
            "bootstrap_se": float(bootstrap_se) if bootstrap_se is not None else None,
            "bootstrap_mean": float(distribution_mean),
            "bootstrap_median": float(np.median(bootstrap_means)),
            "cohens_d": float(cohens_d),
            "cohens_d_type": "paired (Cohen's d_z based on difference std)",
            "ccs": float(ccs) if ccs is not None else None,
            "a_mean": float(np.mean(valid_a)),
            "b_mean": float(np.mean(valid_b)),
            "num_valid_pairs": len(valid_a),
            "num_total_pairs": len(values_a)
        }
    
    except ImportError:
        # If scipy is too old, fall back to manual implementation
        print("Warning: scipy.stats.bootstrap not available; using manual implementation")
        return perform_bootstrap_test_manual(results_a, results_b, metric, n_bootstrap, 
                                            alternative, confidence_level, seed)
    except Exception as e:
        print(f"Bootstrap test failed: {e}")
        return {
            "metric": metric,
            "test": f"Bootstrap Test ({alternative})",
            "error": str(e)
        }


def perform_bootstrap_test_manual(
    results_a: List[Dict], 
    results_b: List[Dict], 
    metric: str, 
    n_bootstrap: int = 10000,
    alternative: str = 'greater',
    confidence_level: float = 0.95,
    seed: int = 42
) -> Dict:
    """
    Manually implemented Bootstrap Resampling Test (fallback)
    
    Use this function when scipy.stats.bootstrap is not available.
    """
    # Set random seed
    np.random.seed(seed)
    
    # Extract metric values, handling None
    values_a = [r.get(metric, 0.0) if r.get(metric) is not None else 0.0 for r in results_a]
    values_b = [r.get(metric, 0.0) if r.get(metric) is not None else 0.0 for r in results_b]
    
    # Filter out pairs where both confidence values are 0
    valid_pairs = [(a, b) for a, b in zip(values_a, values_b) if not (a == 0 and b == 0)]
    
    if len(valid_pairs) < 2:
        return {
            "metric": metric,
            "test": f"Bootstrap Test ({alternative})",
            "alternative": alternative,
            "p_value": None,
            "error": "Insufficient valid samples"
        }
    
    valid_a = np.array([a for a, b in valid_pairs])
    valid_b = np.array([b for a, b in valid_pairs])
    
    # === Key correction: pre-compute paired differences ===
    diffs = valid_a - valid_b
    observed_diff = np.mean(diffs)
    
    # Bootstrap resampling: resample directly on differences (preserving pairing)
    n_samples = len(diffs)
    bootstrap_means = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Directly resample the differences array
        resampled_diffs = np.random.choice(diffs, size=n_samples, replace=True)
        bootstrap_means[i] = np.mean(resampled_diffs)
    
    # Shift bootstrap distribution to null-hypothesis center (H0: μ_diff = 0)
    distribution_mean = np.mean(bootstrap_means)
    null_distribution = bootstrap_means - distribution_mean
    
    # Compute p-value: how extreme the observed value is under the null distribution
    if alternative == 'two-sided':
        # Two-sided test
        p_value = np.mean(np.abs(null_distribution) >= np.abs(observed_diff))
    elif alternative == 'greater':
        # One-sided test H1: A > B
        p_value = np.mean(null_distribution >= observed_diff)
    elif alternative == 'less':
        # One-sided test H1: A < B
        p_value = np.mean(null_distribution <= observed_diff)
    else:
        raise ValueError(f"Invalid alternative: {alternative}")
    
    # Compute confidence interval
    alpha = 1 - confidence_level
    if alternative == 'two-sided':
        ci_lower = np.percentile(bootstrap_means, (alpha / 2) * 100)
        ci_upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    elif alternative == 'greater':
        ci_lower = np.percentile(bootstrap_means, alpha * 100)
        ci_upper = np.inf
    else:  # less
        ci_lower = -np.inf
        ci_upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    
    # Compute standard error
    bootstrap_se = np.std(bootstrap_means)
    
    # Compute Cohen's d_z as effect size (standard formula for paired data)
    std_diff = np.std(diffs, ddof=1)
    cohens_d = observed_diff / std_diff if std_diff > 0 else 0.0
    
    # Compute CCS (Composite Confidence Score)
    ccs = compute_ccs(cohens_d, p_value)
    
    return {
        "metric": metric,
        "test": f"Paired Bootstrap Resampling Test ({alternative})",
        "method": "manual paired bootstrap + centered null hypothesis",
        "alternative": alternative,
        "n_bootstrap": n_bootstrap,
        "observed_difference": float(observed_diff),
        "p_value": f"{p_value:.3e}",
        "p_value_note": "Computed by centering bootstrap distribution at H0: μ_diff = 0",
        "significant_at_0.05": bool(p_value < 0.05),
        "significant_at_0.01": bool(p_value < 0.01),
        "confidence_level": confidence_level,
        "confidence_interval": {
            "lower": float(ci_lower) if ci_lower != -np.inf else "-inf",
            "upper": float(ci_upper) if ci_upper != np.inf else "inf"
        },
        "bootstrap_se": float(bootstrap_se),
        "bootstrap_mean": float(distribution_mean),
        "bootstrap_median": float(np.median(bootstrap_means)),
        "cohens_d": float(cohens_d),
        "cohens_d_type": "paired (Cohen's d_z based on difference std)",
        "ccs": float(ccs) if ccs is not None else None,
        "a_mean": float(np.mean(valid_a)),
        "b_mean": float(np.mean(valid_b)),
        "num_valid_pairs": len(valid_a),
        "num_total_pairs": len(values_a)
    }


def plot_comparison(
    results_a: List[Dict],
    results_b: List[Dict],
    output_dir: str,
    label_a: str = "Model A",
    label_b: str = "Model B"
):
    """
    Plot comparison charts for two sets of results
    
    Args:
        results_a: results from model/dataset A
        results_b: results from model/dataset B
        output_dir: output directory
        label_a: label for A
        label_b: label for B
    """
    print("\nGenerating comparison chart...")
    
    # Create 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Accuracy comparison (Full and Truncated)
    ax = axes[0, 0]
    full_acc_a = [r.get('full_accuracy', 0) for r in results_a]
    full_acc_b = [r.get('full_accuracy', 0) for r in results_b]
    acc_a = [r['accuracy'] for r in results_a]
    acc_b = [r['accuracy'] for r in results_b]
    
    x_pos = np.arange(2)
    width = 0.35
    
    # Bar chart for full accuracy
    full_means = [np.mean(full_acc_a), np.mean(full_acc_b)]
    full_stds = [np.std(full_acc_a), np.std(full_acc_b)]
    bars1 = ax.bar(x_pos - width/2, full_means, width, yerr=full_stds, 
                   capsize=5, alpha=0.7, label='Full Solution',
                   color=['steelblue', 'coral'], edgecolor='black')
    
    # Bar chart for truncated accuracy
    trunc_means = [np.mean(acc_a), np.mean(acc_b)]
    trunc_stds = [np.std(acc_a), np.std(acc_b)]
    bars2 = ax.bar(x_pos + width/2, trunc_means, width, yerr=trunc_stds,
                   capsize=5, alpha=0.7, label='Truncated Solution',
                   color=['darkblue', 'darkred'], edgecolor='black')
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Comparison (Full vs Truncated)', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([label_a, label_b])
    ax.set_ylim([0, 1.1])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars, means, stds in [(bars1, full_means, full_stds), (bars2, trunc_means, trunc_stds)]:
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                    f'{mean:.3f}',
                    ha='center', va='bottom', fontsize=9)
    
    # 2. Consistency comparison
    ax = axes[0, 1]
    cons_a = [r['consistency'] for r in results_a]
    cons_b = [r['consistency'] for r in results_b]
    
    means = [np.mean(cons_a), np.mean(cons_b)]
    stds = [np.std(cons_a), np.std(cons_b)]
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7,
                   color=['steelblue', 'coral'], edgecolor='black')
    ax.set_ylabel('Consistency', fontsize=12)
    ax.set_title('Consistency Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([label_a, label_b])
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.3f}±{std:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    # 3. First Token Confidence comparison
    ax = axes[1, 0]
    ftc_a = [r.get('first_token_confidence', 0.0) if r.get('first_token_confidence') is not None else 0.0 for r in results_a]
    ftc_b = [r.get('first_token_confidence', 0.0) if r.get('first_token_confidence') is not None else 0.0 for r in results_b]
    
    # Filter out zero values
    ftc_a_valid = [v for v in ftc_a if v > 0]
    ftc_b_valid = [v for v in ftc_b if v > 0]
    
    if ftc_a_valid or ftc_b_valid:
        data_to_plot = []
        labels_to_plot = []
        colors = []
        
        if ftc_a_valid:
            data_to_plot.append(ftc_a_valid)
            labels_to_plot.append(f'{label_a}\n(n={len(ftc_a_valid)})')
            colors.append('steelblue')
        
        if ftc_b_valid:
            data_to_plot.append(ftc_b_valid)
            labels_to_plot.append(f'{label_b}\n(n={len(ftc_b_valid)})')
            colors.append('coral')
        
        bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True,
                        showmeans=True, meanline=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax.set_ylabel('First Token Confidence', fontsize=12)
    ax.set_title('First Token Confidence Comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. All Token Confidence comparison
    ax = axes[1, 1]
    atc_a = [r.get('all_token_confidence', 0.0) if r.get('all_token_confidence') is not None else 0.0 for r in results_a]
    atc_b = [r.get('all_token_confidence', 0.0) if r.get('all_token_confidence') is not None else 0.0 for r in results_b]
    
    # Filter out zero values
    atc_a_valid = [v for v in atc_a if v > 0]
    atc_b_valid = [v for v in atc_b if v > 0]
    
    if atc_a_valid or atc_b_valid:
        data_to_plot = []
        labels_to_plot = []
        colors = []
        
        if atc_a_valid:
            data_to_plot.append(atc_a_valid)
            labels_to_plot.append(f'{label_a}\n(n={len(atc_a_valid)})')
            colors.append('steelblue')
        
        if atc_b_valid:
            data_to_plot.append(atc_b_valid)
            labels_to_plot.append(f'{label_b}\n(n={len(atc_b_valid)})')
            colors.append('coral')
        
        bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True,
                        showmeans=True, meanline=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax.set_ylabel('All Token Confidence', fontsize=12)
    ax.set_title('All Token Confidence Comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Statistical Comparison: {label_a} vs {label_b}',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "comparison_plots.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison chart saved to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Statistical Significance Test")
    
    # Model parameters
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path (supports HuggingFace models or local merged models)")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="LoRA adapter path (optional)")
    parser.add_argument("--model-type", type=str, default="auto", choices=["qwen", "deepseek", "auto"],
                        help="Model type: qwen/deepseek/auto (default: auto-detect)")
    
    # Dataset parameters
    parser.add_argument("--dataset-path-a", type=str, required=True,
                        help="Path to dataset A (JSONL format)")
    parser.add_argument("--dataset-path-b", type=str, required=True,
                        help="Path to dataset B (JSONL format)")
    parser.add_argument("--data-type-a", type=str, default="clean",
                        choices=["clean", "original", "paraphrased", "modified"],
                        help="Data type for dataset A")
    parser.add_argument("--data-type-b", type=str, default="clean",
                        choices=["clean", "original", "paraphrased", "modified"],
                        help="Data type for dataset B")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of test samples")
    parser.add_argument("--index-range", type=str, default=None,
                        help="Specify data index range, e.g. '1-10' or '0-99' (based on dataset's own index, starting from 0)")
    
    # Other parameters
    # NOTE: accept as str to support both space-separated and comma-separated values (e.g. --truncate-ratio 0 1 or --truncate-ratio 0,1)
    parser.add_argument(
        "--truncate-ratio",
        type=str,
        nargs='+',
        default=["0.5"],
        help="Truncation ratio (multiple values; space- or comma-separated, e.g. 0 0.5 1 or 0,0.5,1)"
    )
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--gpu-id", type=int, default=None,
                        help="Specify GPU ID")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="GPU memory utilization (recommended: 0.8–0.85 to avoid OOM)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert paths to absolute paths
    model_name = make_absolute_path(args.model)
    lora_path = make_absolute_path(args.lora_path) if args.lora_path else None
    dataset_a = make_absolute_path(args.dataset_path_a)
    dataset_b = make_absolute_path(args.dataset_path_b)
    
    def _parse_truncate_ratios(raw_values) -> List[float]:
        """Parse truncate ratios from argparse.

        Accepts both:
        - space separated: --truncate-ratio 0 0.5 1
        - comma separated: --truncate-ratio 0,0.5,1
        - mixed: --truncate-ratio 0,0.5 1
        """
        if raw_values is None:
            return [0.5]

        # argparse with nargs='+' gives list; be defensive.
        if isinstance(raw_values, (float, int)):
            values_list = [str(raw_values)]
        elif isinstance(raw_values, str):
            values_list = [raw_values]
        else:
            values_list = [str(v) for v in raw_values]

        parsed: List[float] = []
        for token in values_list:
            for part in token.split(','):
                part = part.strip()
                if not part:
                    continue
                parsed.append(float(part))

        if not parsed:
            return [0.5]

        for r in parsed:
            if not (0.0 <= r <= 1.0):
                raise ValueError(f"truncate-ratio must be in [0,1], but got: {r}")

        return parsed

    # Process truncate_ratio parameter (compatible with comma-separated notation like 0,1)
    try:
        truncate_ratios = _parse_truncate_ratios(args.truncate_ratio)
    except ValueError as e:
        print(f"Error: invalid --truncate-ratio: {e}")
        print("Example: --truncate-ratio 0 1   or   --truncate-ratio 0,1")
        return
    
    # Parse the index_range argument
    index_start, index_end = None, None
    if args.index_range:
        try:
            parts = args.index_range.split('-')
            if len(parts) == 2:
                index_start = int(parts[0])
                index_end = int(parts[1])
                if index_start < 0 or index_end < index_start:
                    raise ValueError("Invalid range")
                print(f"Will only evaluate index range: {index_start}-{index_end}")
            else:
                raise ValueError("Range format should be 'start-end'")
        except ValueError as e:
            print(f"Error: invalid index-range format '{args.index_range}'; expected 'start-end' format (e.g. '1-10')")
            return
    
    # Generate labels
    model_basename = os.path.basename(model_name)
    dataset_a_basename = os.path.basename(dataset_a)
    dataset_b_basename = os.path.basename(dataset_b)
    
    if lora_path:
        lora_basename = os.path.basename(lora_path)
        model_label = f"{model_basename} + {lora_basename}"
    else:
        model_label = model_basename
    
    label_a = f"{model_label} on {dataset_a_basename} ({args.data_type_a})"
    label_b = f"{model_label} on {dataset_b_basename} ({args.data_type_b})"
    
    print(f"\n{'='*80}")
    print(f"Statistical Significance Test")
    print(f"{'='*80}")
    print(f"Model: {model_label}")
    print(f"Dataset A: {dataset_a_basename} (type: {args.data_type_a})")
    print(f"Dataset B: {dataset_b_basename} (type: {args.data_type_b})")
    print(f"Truncation ratios: {truncate_ratios}")
    if index_start is not None or index_end is not None:
        range_str = f"[{index_start if index_start is not None else '0'}:{index_end if index_end is not None else 'end'}]"
        print(f"Index range: {range_str}")
    print(f"{'='*80}\n")
    
    # Initialize tester (only once)
    print(f"\n{'='*80}")
    print(f"Initializing model...")
    print(f"{'='*80}\n")
    
    tester = StatisticalSignificanceTester(
        model_name=model_name,
        gpu_id=args.gpu_id,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        lora_path=lora_path,
        model_type=args.model_type
    )
    
    # Variables for caching full solutions
    cached_solutions_a = None
    cached_solutions_b = None
    
    # Test each truncate_ratio
    for ratio_idx, truncate_ratio in enumerate(truncate_ratios):
        print(f"\n{'='*80}")
        print(f"Processing truncation ratio {ratio_idx+1}/{len(truncate_ratios)}: {truncate_ratio}")
        print(f"{'='*80}\n")
        
        # Ensure vLLM model is loaded (may need reloading when processing a new ratio)
        if tester.llm is None:
            print(f"\nReloading vLLM model...")
            llm_kwargs = {
                "model": tester.model_name,
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "trust_remote_code": True,
                "seed": args.seed
            }
            if tester.lora_path:
                llm_kwargs["enable_lora"] = True
                llm_kwargs["max_lora_rank"] = 64
            tester.llm = LLM(**llm_kwargs)
            print("vLLM model reloaded")
        
        # Compute metrics for dataset A
        print(f"\n{'='*80}")
        print(f"Computing metrics on dataset A: {dataset_a_basename} ({args.data_type_a})")
        print(f"{'='*80}\n")
        
        results_a, solutions_a = tester.compute_metrics_on_dataset(
            dataset_path=dataset_a,
            truncate_ratio=truncate_ratio,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            data_type=args.data_type_a,
            pregenerated_solutions=cached_solutions_a,
            use_original_index=True  # dataset A uses the index or original_index field from the file
        )
        
        # Cache solutions after the first generation
        if cached_solutions_a is None:
            cached_solutions_a = solutions_a
            print(f"Cached {len(cached_solutions_a)} solutions from dataset A for reuse in subsequent ratios")
        
        # Free GPU memory before processing the next dataset
        tester.release_hf_model()
        
        # Reload vLLM model (if it was released earlier)
        if tester.llm is None:
            print(f"\nReloading vLLM model...")
            llm_kwargs = {
                "model": tester.model_name,
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "trust_remote_code": True,
                "seed": args.seed
            }
            if tester.lora_path:
                llm_kwargs["enable_lora"] = True
                llm_kwargs["max_lora_rank"] = 64
            tester.llm = LLM(**llm_kwargs)
            print("vLLM model reloaded")
        
        # Compute metrics for dataset B
        print(f"\n{'='*80}")
        print(f"Computing metrics on dataset B: {dataset_b_basename} ({args.data_type_b})")
        print(f"{'='*80}\n")
        
        results_b, solutions_b = tester.compute_metrics_on_dataset(
            dataset_path=dataset_b,
            truncate_ratio=truncate_ratio,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            data_type=args.data_type_b,
            pregenerated_solutions=cached_solutions_b,
            use_original_index=True  # dataset B must use the original_index field
        )
        
        # Cache solutions after the first generation
        if cached_solutions_b is None:
            cached_solutions_b = solutions_b
            print(f"Cached {len(cached_solutions_b)} solutions from dataset B for reuse in subsequent ratios")
        
        # Filter data by index range (if --index-range is specified)
        if index_start is not None or index_end is not None:
            print(f"\n{'='*80}")
            print(f"Filtering data by index range...")
            print(f"{'='*80}\n")
            
            print("Filtering dataset A:")
            results_a = filter_by_index_range(results_a, index_start, index_end)
            
            print("\nFiltering dataset B:")
            results_b = filter_by_index_range(results_b, index_start, index_end)
            
            if len(results_a) == 0 or len(results_b) == 0:
                print(f"\nWarning: filtered dataset is empty; skipping test for this ratio")
                continue
        
        # Match datasets by original_index to ensure correct paired statistical tests
        print(f"\n{'='*80}")
        print(f"Matching datasets by original_index...")
        print(f"{'='*80}\n")
        
        results_a, results_b = match_datasets_by_original_index(results_a, results_b)
        
        # Perform statistical tests
        print(f"\n{'='*80}")
        print(f"Running statistical tests (truncate_ratio={truncate_ratio})...")
        print(f"{'='*80}\n")
        
        # McNemar's Test for accuracy
        print("\n1. McNemar's Test for Accuracy:")
        accuracy_test = perform_mcnemar_test(results_a, results_b, "accuracy")
        print(f"   Statistic: {accuracy_test['statistic']}")
        print(f"   p-value: {accuracy_test['p_value']}")
        print(f"   Significant (α=0.05): {accuracy_test['significant_at_0.05']}")
        print(f"   Effect Size: {accuracy_test['effect_size']:.4f}")
        print(f"   CCS: {accuracy_test.get('ccs', 'N/A') if accuracy_test.get('ccs') is not None else 'N/A'}")
        print(f"   A performance: {accuracy_test['a_performance']:.4f}")
        print(f"   B performance: {accuracy_test['b_performance']:.4f}")
        print(f"   Performance difference: {accuracy_test['performance_difference']:.4f}")
        
        # McNemar's Test for consistency
        print("\n2. McNemar's Test for Consistency:")
        consistency_test = perform_mcnemar_test(results_a, results_b, "consistency")
        print(f"   Statistic: {consistency_test['statistic']}")
        print(f"   p-value: {consistency_test['p_value']}")
        print(f"   Significant (α=0.05): {consistency_test['significant_at_0.05']}")
        print(f"   Effect Size: {consistency_test['effect_size']:.4f}")
        print(f"   CCS: {consistency_test.get('ccs', 'N/A') if consistency_test.get('ccs') is not None else 'N/A'}")
        print(f"   A performance: {consistency_test['a_performance']:.4f}")
        print(f"   B performance: {consistency_test['b_performance']:.4f}")
        print(f"   Performance difference: {consistency_test['performance_difference']:.4f}")
        
        # Bootstrap Test for first token confidence
        print("\n3. Bootstrap Resampling Test for First Token Confidence:")
        ftc_bootstrap = perform_bootstrap_test(results_a, results_b, "first_token_confidence",
                                               n_bootstrap=10000, seed=args.seed)
        print(f"   Bootstrap resamples: {ftc_bootstrap.get('n_bootstrap', 'N/A')}")
        print(f"   Observed difference (A-B): {ftc_bootstrap.get('observed_difference', 0):.4f}")
        print(f"   p-value: {ftc_bootstrap.get('p_value', 'N/A')}")
        print(f"   Significant (α=0.05): {ftc_bootstrap.get('significant_at_0.05', 'N/A')}")
        ci = ftc_bootstrap.get('confidence_interval', {})
        print(f"   95% CI: [{ci.get('lower', 'N/A')}, {ci.get('upper', 'N/A')}]")
        print(f"   Bootstrap SE: {ftc_bootstrap.get('bootstrap_se', 0):.4f}")
        print(f"   Cohen's d: {ftc_bootstrap.get('cohens_d', 0):.4f}")
        print(f"   CCS: {ftc_bootstrap.get('ccs', 'N/A') if ftc_bootstrap.get('ccs') is not None else 'N/A'}")

        # Bootstrap Test for all token confidence
        print("\n4. Bootstrap Resampling Test for All Token Confidence:")
        atc_bootstrap = perform_bootstrap_test(results_a, results_b, "all_token_confidence",
                                               n_bootstrap=10000, seed=args.seed)
        print(f"   Bootstrap resamples: {atc_bootstrap.get('n_bootstrap', 'N/A')}")
        print(f"   Observed difference (A-B): {atc_bootstrap.get('observed_difference', 0):.4f}")
        print(f"   p-value: {atc_bootstrap.get('p_value', 'N/A')}")
        print(f"   Significant (α=0.05): {atc_bootstrap.get('significant_at_0.05', 'N/A')}")
        ci = atc_bootstrap.get('confidence_interval', {})
        print(f"   95% CI: [{ci.get('lower', 'N/A')}, {ci.get('upper', 'N/A')}]")
        print(f"   Bootstrap SE: {atc_bootstrap.get('bootstrap_se', 0):.4f}")
        print(f"   Cohen's d: {atc_bootstrap.get('cohens_d', 0):.4f}")
        print(f"   CCS: {atc_bootstrap.get('ccs', 'N/A') if atc_bootstrap.get('ccs') is not None else 'N/A'}")
        
        # Compute overall evaluation metrics for datasets A and B
        def compute_dataset_metrics(results):
            """Compute overall metrics for the dataset."""
            full_accuracy_values = [r.get('full_accuracy', 0) for r in results]
            accuracy_values = [r['accuracy'] for r in results]
            consistency_values = [r['consistency'] for r in results]
            ftc_values = [r.get('first_token_confidence', 0.0) for r in results 
                         if r.get('first_token_confidence') is not None and r.get('first_token_confidence', 0.0) > 0]
            atc_values = [r.get('all_token_confidence', 0.0) for r in results 
                         if r.get('all_token_confidence') is not None and r.get('all_token_confidence', 0.0) > 0]
            
            return {
                "full_accuracy": {
                    "mean": float(np.mean(full_accuracy_values)),
                    "std": float(np.std(full_accuracy_values)),
                    "count": len(full_accuracy_values)
                },
                "accuracy": {
                    "mean": float(np.mean(accuracy_values)),
                    "std": float(np.std(accuracy_values)),
                    "count": len(accuracy_values)
                },
                "consistency": {
                    "mean": float(np.mean(consistency_values)),
                    "std": float(np.std(consistency_values)),
                    "count": len(consistency_values)
                },
                "first_token_confidence": {
                    "mean": float(np.mean(ftc_values)) if ftc_values else 0.0,
                    "std": float(np.std(ftc_values)) if ftc_values else 0.0,
                    "count": len(ftc_values),
                    "total_samples": len(results)
                },
                "all_token_confidence": {
                    "mean": float(np.mean(atc_values)) if atc_values else 0.0,
                    "std": float(np.std(atc_values)) if atc_values else 0.0,
                    "count": len(atc_values),
                    "total_samples": len(results)
                }
            }
        
        metrics_a = compute_dataset_metrics(results_a)
        metrics_b = compute_dataset_metrics(results_b)
        
        print(f"\n{'='*80}")
        print(f"Overall metrics summary:")
        print(f"{'='*80}")
        print(f"\nDataset A ({label_a}):")
        print(f"  Full Accuracy: {metrics_a['full_accuracy']['mean']:.4f} ± {metrics_a['full_accuracy']['std']:.4f}")
        print(f"  Accuracy (Truncated): {metrics_a['accuracy']['mean']:.4f} ± {metrics_a['accuracy']['std']:.4f}")
        print(f"  Consistency: {metrics_a['consistency']['mean']:.4f} ± {metrics_a['consistency']['std']:.4f}")
        print(f"  First Token Confidence: {metrics_a['first_token_confidence']['mean']:.4f} ± {metrics_a['first_token_confidence']['std']:.4f} (n={metrics_a['first_token_confidence']['count']})")
        print(f"  All Token Confidence: {metrics_a['all_token_confidence']['mean']:.4f} ± {metrics_a['all_token_confidence']['std']:.4f} (n={metrics_a['all_token_confidence']['count']})")
        
        print(f"\nDataset B ({label_b}):")
        print(f"  Full Accuracy: {metrics_b['full_accuracy']['mean']:.4f} ± {metrics_b['full_accuracy']['std']:.4f}")
        print(f"  Accuracy (Truncated): {metrics_b['accuracy']['mean']:.4f} ± {metrics_b['accuracy']['std']:.4f}")
        print(f"  Consistency: {metrics_b['consistency']['mean']:.4f} ± {metrics_b['consistency']['std']:.4f}")
        print(f"  First Token Confidence: {metrics_b['first_token_confidence']['mean']:.4f} ± {metrics_b['first_token_confidence']['std']:.4f} (n={metrics_b['first_token_confidence']['count']})")
        print(f"  All Token Confidence: {metrics_b['all_token_confidence']['mean']:.4f} ± {metrics_b['all_token_confidence']['std']:.4f} (n={metrics_b['all_token_confidence']['count']})")
        
        # Save results
        output_data = {
            "config": {
                "label_a": label_a,
                "label_b": label_b,
                "model": model_name,
                "lora_path": lora_path,
                "dataset_a": dataset_a,
                "dataset_b": dataset_b,
                "data_type_a": args.data_type_a,
                "data_type_b": args.data_type_b,
                "truncate_ratio": truncate_ratio,
                "index_range": args.index_range if args.index_range else "all",
                "index_start": index_start,
                "index_end": index_end,
                "num_samples_a": len(results_a),
                "num_samples_b": len(results_b)
            },
            "dataset_metrics": {
                "dataset_a": metrics_a,
                "dataset_b": metrics_b
            },
            "statistical_tests": {
                "accuracy_mcnemar": accuracy_test,
                "consistency_mcnemar": consistency_test,
                "first_token_confidence_bootstrap": ftc_bootstrap,
                "all_token_confidence_bootstrap": atc_bootstrap
            },
            "detailed_results_a": results_a,
            "detailed_results_b": results_b
        }
        
        # Create a separate output file for each truncate_ratio
        output_file = os.path.join(args.output_dir, f"statistical_test_results_ratio_{truncate_ratio:.2f}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        # Draw comparison chart
        plot_comparison(results_a, results_b, args.output_dir, 
                       f"{label_a} (ratio={truncate_ratio})", 
                       f"{label_b} (ratio={truncate_ratio})")
        
        # Rename plot file to include truncate_ratio
        old_plot_file = os.path.join(args.output_dir, "comparison_plots.png")
        new_plot_file = os.path.join(args.output_dir, f"comparison_plots_ratio_{truncate_ratio:.2f}.png")
        if os.path.exists(old_plot_file):
            os.rename(old_plot_file, new_plot_file)
            print(f"Comparison chart saved to: {new_plot_file}")
        
        # Before processing the next ratio, fully release all models (except after the last ratio)
        if ratio_idx < len(truncate_ratios) - 1:
            print(f"\nPreparing for the next ratio; fully clearing GPU memory...")
            tester.release_hf_model()
            tester.release_vllm_model()
            
            # Additional cleanup steps
            for _ in range(5):
                gc.collect()
                torch.cuda.empty_cache()
            
            import time
            time.sleep(5)
            
            # Print GPU memory usage
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    print(f"GPU {i}: allocated {allocated:.2f} GB, reserved {reserved:.2f} GB")
            
            print("GPU memory cleared; ready for next ratio\n")
    
    # Release models
    del tester
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"\n{'='*80}")
    print(f"Statistical tests complete for all {len(truncate_ratios)} truncation ratio(s)!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
