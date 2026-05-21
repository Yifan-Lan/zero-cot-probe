"""
Run statistical significance tests comparing ZCP metrics on two datasets using
closed-source API models (OpenAI, Google Gemini, Anthropic Claude).

Statistical tests:
1. McNemar's Test        — for binary metrics (accuracy, consistency)
2. Bootstrap Resampling  — non-parametric test (10,000 resamples) for continuous
                           metrics (first/all token confidence); makes no
                           distributional assumptions.

API keys can be passed via --api-key or through environment variables:
  export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
  export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
  export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY"

Usage examples:

# Test an OpenAI model (e.g., GPT-4o):
python statistical_significance_test_api_model.py \
    --model gpt-4o \
    --model-type openai \
    --dataset-path-a <path/to/paraphrased_dataset.jsonl> \
    --dataset-path-b <path/to/modified_dataset.jsonl> \
    --data-type-a original \
    --data-type-b modified \
    --truncate-ratio 0.0 \
    --output-dir <path/to/output_dir>

# Test a Gemini model:
python statistical_significance_test_api_model.py \
    --model gemini-2.5-flash \
    --model-type google \
    --dataset-path-a <path/to/paraphrased_dataset.jsonl> \
    --dataset-path-b <path/to/modified_dataset.jsonl> \
    --data-type-a original \
    --data-type-b modified \
    --truncate-ratio 0.0 \
    --output-dir <path/to/output_dir>

# Test an Anthropic Claude model:
# The --api-key flag is optional; the ANTHROPIC_API_KEY env var is preferred.
python statistical_significance_test_api_model.py \
    --model claude-sonnet-4-5 \
    --model-type anthropic \
    --dataset-path-a <path/to/paraphrased_dataset.jsonl> \
    --dataset-path-b <path/to/modified_dataset.jsonl> \
    --data-type-a original \
    --data-type-b modified \
    --truncate-ratio 0.0 \
    --output-dir <path/to/output_dir> \
    --force-answer-strict-in-system \
    --skip-full-solution
"""

import os
# Fix MKL threading library conflict
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"

import json
import argparse
import re
import random
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
from google import genai
import seaborn as sns
from openai import OpenAI

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
            orig_idx = item['original_index']
            a_has_original_index = True
        else:
            orig_idx = item['index']
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
        orig_idx = item['original_index']
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
        model_type: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None
    ):
        """
        Initialize the API model tester
        
        Args:
            model_name: API model name (e.g. "gpt-4", "gemini-pro")
            seed: random seed
            model_type: API provider ('openai'/'google'/'auto')
            api_key: API Key (optional; also settable via environment variable)
            api_base: API Base URL (optional; for OpenAI-compatible endpoints)
        """
        self.model_name = model_name
        self.seed = seed
        self.api_key = api_key
        self.api_base = api_base
        
        # Detect model type
        self.model_type = self._detect_model_type(model_name, model_type)
        print(f"Detected API provider: {self.model_type}")
        print(f"Using model: {self.model_name}")
        
        set_random_seed(seed)
        
        # Initialize client
        if self.model_type == 'openai':
            self.client = OpenAI(
                api_key=self.api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=self.api_base or os.environ.get("OPENAI_BASE_URL")
            )
        elif self.model_type == 'google':
             api_key = self.api_key or os.environ.get("GOOGLE_API_KEY")
             if api_key:
                # Use the new google.genai Client (v1.0+)
                try:
                    self.client = genai.Client(api_key=api_key)
                    self.use_new_google_sdk = True
                except AttributeError:
                    # Fall back to the old google.generativeai (v0.x)
                    genai.configure(api_key=api_key)
                    self.use_new_google_sdk = False
             else:
                print("Warning: GOOGLE_API_KEY is not set")
        elif self.model_type == 'anthropic':
            api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                print("Warning: ANTHROPIC_API_KEY (or --api-key) is not set")
                self.client = None
            else:
                try:
                    from anthropic import Anthropic
                except Exception as e:
                    raise ImportError(
                        "Missing anthropic SDK. Please install: pip install anthropic\n"
                        f"Original error: {e}"
                    )

                anthropic_kwargs = {"api_key": api_key}
                # Some corporate/proxy environments may have a custom base_url
                if self.api_base:
                    anthropic_kwargs["base_url"] = self.api_base
                self.client = Anthropic(**anthropic_kwargs)
        
    def _detect_model_type(self, model_name: str, model_type: Optional[str] = None) -> str:
        """
        Detect the API provider type
        """
        if model_type and model_type.lower() != 'auto':
            mt = model_type.lower()
            # Alias compatibility
            if mt in {"claude", "anthropic"}:
                return "anthropic"
            return mt
        
        # Auto-detect
        model_name_lower = model_name.lower()
        if 'gpt' in model_name_lower or 'o1' in model_name_lower:
            return 'openai'
        elif 'gemini' in model_name_lower:
            return 'google'
        elif 'claude' in model_name_lower:
            return 'anthropic'
        else:
            # Default to OpenAI format
            print(f"Warning: cannot auto-detect API type; defaulting to OpenAI format")
            return 'openai'
            
    def generate_api_response(self, prompt: str, system_prompt: str = None, max_tokens: int = None, temperature: float = 0.0) -> str:
        """
        Call the API to generate a response
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"Calling API (attempt {attempt+1}/{max_retries})...")
                if self.model_type == 'openai':
                    model_lower = self.model_name.lower()
                    # Reasoning models use the responses API with low inference overhead
                    is_reasoning_model = any(key in model_lower for key in ["o3", "o4", "reasoning"])
                    if is_reasoning_model:
                        input_msgs = []
                        if system_prompt:
                            input_msgs.append({"role": "developer", "content": [{"type": "input_text", "text": system_prompt}]})
                        input_msgs.append({"role": "user", "content": [{"type": "input_text", "text": prompt}]})

                        reasoning_cfg = {"effort": "none", "verbosity": "low"}
                        resp_kwargs = {
                            "model": self.model_name,
                            "input": input_msgs,
                            "reasoning": reasoning_cfg
                        }
                        if max_tokens:
                            resp_kwargs["max_output_tokens"] = max_tokens

                        try:
                            resp = self.client.responses.create(**resp_kwargs)
                        except Exception as e:
                            err_msg = str(e)
                            # If effort=none is not supported, fall back to low
                            if "effort" in err_msg and "none" in err_msg:
                                print("Warning: effort=none is not supported; falling back to low")
                                reasoning_cfg["effort"] = "low"
                                resp_kwargs["reasoning"] = reasoning_cfg
                                resp = self.client.responses.create(**resp_kwargs)
                            else:
                                raise
                        # responses API returns output_text / output[]
                        if hasattr(resp, "output_text") and resp.output_text:
                            return resp.output_text
                        if hasattr(resp, "output") and resp.output:
                            parts = []
                            for item in resp.output:
                                if hasattr(item, "content") and item.content:
                                    for c in item.content:
                                        if getattr(c, "type", None) == "output_text" and getattr(c, "text", ""):
                                            parts.append(c.text)
                                if getattr(item, "type", None) == "output_text" and getattr(item, "text", ""):
                                    parts.append(item.text)
                            if parts:
                                return "".join(parts)
                        print("Warning: cannot extract text from responses output")
                        return ""
                    
                    # Standard chat-completions call
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})

                    kwargs = {
                        "model": self.model_name,
                        "messages": messages,
                        "temperature": temperature,
                        "seed": self.seed if temperature == 0.0 else None
                    }
                    if max_tokens:
                        kwargs["max_tokens"] = max_tokens

                    # Remove None-valued parameters to avoid passing null values
                    call_kwargs = {k: v for k, v in kwargs.items() if v is not None}
                    response = None
                    for _ in range(2):  # at most two rounds of parameter adjustment retries
                        try:
                            response = self.client.chat.completions.create(**call_kwargs)
                            break
                        except Exception as e:
                            err_msg = str(e)
                            adjusted = False
                            # 1) If model does not support max_tokens, use max_completion_tokens
                            if "max_tokens" in err_msg and "not supported" in err_msg:
                                print("Warning: model does not support max_tokens; retrying with max_completion_tokens...")
                                max_val = call_kwargs.pop("max_tokens", None)
                                if max_val is not None:
                                    call_kwargs["max_completion_tokens"] = max_val
                                adjusted = True
                            # 2) Some models do not support temperature/seed parameters
                            if "temperature" in err_msg and ("unsupported" in err_msg or "does not support" in err_msg):
                                print("Warning: model does not support temperature/seed; retrying without them...")
                                call_kwargs.pop("temperature", None)
                                call_kwargs.pop("seed", None)
                                adjusted = True
                            if not adjusted:
                                raise
                            # If adjustments were made, continue to the next retry round
                    if response is None:
                        # Should have raised or succeeded; this is a fallback
                        raise RuntimeError(f"OpenAI API call failed and parameter adjustment unsuccessful; last params: {call_kwargs}")
                    return response.choices[0].message.content
                    
                elif self.model_type == 'google':
                    full_prompt = prompt
                    
                    if hasattr(self, 'use_new_google_sdk') and self.use_new_google_sdk:
                        # New Google GenAI SDK (google-genai)
                        from google.genai import types
                        
                        config = {
                            "temperature": temperature
                        }
                        if max_tokens:
                            config["max_output_tokens"] = max_tokens
                        if system_prompt:
                            config["system_instruction"] = system_prompt

                        # Configure thinking:
                        # - Series 3: prefer thinking_level="low" (or minimal, if available)
                        # - Series 2.5: use thinking_budget=0 to disable thinking, with include_thoughts=True
                        try:
                            tc_kwargs = {}
                            name_lower = self.model_name.lower()
                            if "3" in name_lower:
                                # Gemini series 3
                                level_val = "low"
                                if hasattr(types.ThinkingConfig, "ThinkingLevel"):
                                    enum_cls = types.ThinkingConfig.ThinkingLevel
                                    level_val = getattr(enum_cls, "LOW", "low")
                                tc_kwargs["thinking_level"] = level_val
                                tc_kwargs["include_thoughts"] = True
                            else:
                                # Gemini series 2.5
                                tc_kwargs["thinking_budget"] = 0  # 0 = disable / minimal thinking
                                tc_kwargs["include_thoughts"] = False
                            if hasattr(types, "ThinkingConfig"):
                                config["thinking_config"] = types.ThinkingConfig(**tc_kwargs)
                        except Exception as tc_err:
                            print(f"Warning: error setting thinking_config: {tc_err}")
                            # Ignore and continue
                        
                        response = self.client.models.generate_content(
                            model=self.model_name,
                            contents=full_prompt,
                            config=types.GenerateContentConfig(**config)
                        )
                        
                        # Extract full response text (including thinking summary)
                        try:
                            if hasattr(response, 'text') and response.text:
                                return response.text
                            elif hasattr(response, 'candidates') and response.candidates:
                                candidate = response.candidates[0]
                                parts_out = []
                                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                    for part in candidate.content.parts:
                                        if not hasattr(part, 'text'):
                                            continue
                                        if getattr(part, 'thought', False):
                                            parts_out.append(f"[THINKING]{part.text}")
                                        else:
                                            parts_out.append(part.text)
                                if parts_out:
                                    return "".join(parts_out)
                            print(f"Warning: cannot extract text from response, response type: {type(response)}")
                            if hasattr(response, '__dict__'):
                                print(f"Response attributes: {response.__dict__.keys()}")
                            return ""
                        except Exception as extract_error:
                            print(f"Error extracting response text: {extract_error}")
                            return ""
                    else:
                        # Old Google Generative AI SDK (google-generativeai)
                        model = genai.GenerativeModel(self.model_name)
                        if system_prompt:
                            # Old SDK can initialize Model with system_instruction, but model is already initialized here
                            # Simple approach: concatenate to prompt
                            full_prompt = f"{system_prompt}\n\n{prompt}"
                            
                        gen_config = {
                            "temperature": temperature
                        }
                        if max_tokens:
                            gen_config["max_output_tokens"] = max_tokens
                            
                        response = model.generate_content(
                            full_prompt,
                            generation_config=genai.types.GenerationConfig(**gen_config)
                        )
                        
                        # Extract full response text (including possible thinking section)
                        try:
                            if hasattr(response, 'text') and response.text:
                                return response.text
                            elif hasattr(response, 'candidates') and response.candidates:
                                candidate = response.candidates[0]
                                thoughts = []
                                texts = []
                                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                    for part in candidate.content.parts:
                                        if hasattr(part, 'text') and part.text:
                                            texts.append(part.text)
                                        if hasattr(part, 'thought') and part.thought:
                                            thoughts.append(part.thought)
                                if thoughts or texts:
                                    joined = "".join([f"[THINKING]{t}" for t in thoughts] + texts)
                                    return joined
                            print(f"Warning: cannot extract text from response")
                            return ""
                        except Exception as extract_error:
                            print(f"Error extracting response text: {extract_error}")
                            return ""
                elif self.model_type == 'anthropic':
                    if self.client is None:
                        raise RuntimeError("Anthropic client not initialized (missing API key or SDK)")

                    def _list_models_for_hint(max_items: int = 50) -> List[str]:
                        """Best-effort list available Anthropic model IDs for hinting."""
                        try:
                            ids: List[str] = []
                            # models.list is paginated and supports iteration
                            for m in self.client.models.list(limit=min(max_items, 100)):
                                mid = getattr(m, "id", None)
                                if mid:
                                    ids.append(str(mid))
                                if len(ids) >= max_items:
                                    break
                            return ids
                        except Exception:
                            return []

                    # Anthropic Messages API
                    # - To minimize thinking: prefer disabling thinking; auto-fallback if model/SDK does not support the field.
                    # - To minimize effort: try setting output_config.effort=low; auto-fallback if unsupported.
                    # - system_prompt goes in the system field; otherwise only a user message.
                    msg_kwargs = {
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": int(max_tokens) if max_tokens else 512,
                    }
                    if system_prompt:
                        msg_kwargs["system"] = system_prompt

                    # Claude output token cost control: effort (low)
                    # https://platform.claude.com/docs/en/build-with-claude/effort
                    msg_kwargs_with_effort = dict(msg_kwargs)
                    msg_kwargs_with_effort["output_config"] = {"effort": "low"}

                    # Claude thinking control: disable/minimize as much as possible
                    # https://platform.claude.com/docs/en/build-with-claude/extended-thinking
                    msg_kwargs_with_effort_and_thinking = dict(msg_kwargs_with_effort)
                    msg_kwargs_with_effort_and_thinking["thinking"] = {"type": "disabled"}

                    # Try in order: effort+thinking -> effort -> thinking -> base
                    variants = [
                        msg_kwargs_with_effort_and_thinking,
                        msg_kwargs_with_effort,
                        {**msg_kwargs, "thinking": {"type": "disabled"}},
                        msg_kwargs,
                    ]

                    resp = None
                    last_err = None
                    for variant in variants:
                        try:
                            resp = self.client.messages.create(**variant)
                            break
                        except Exception as e:
                            last_err = e
                            err_msg = str(e)
                            lower_msg = err_msg.lower()

                            # 404: model not found or no access. Provide hint and abort this call (no retry).
                            try:
                                import anthropic as _anthropic
                                not_found_exc = getattr(_anthropic, "NotFoundError", None)
                            except Exception:
                                not_found_exc = None

                            if (not_found_exc and isinstance(e, not_found_exc)) or (
                                "not_found" in err_msg.lower() and "model" in err_msg.lower()
                            ):
                                available = _list_models_for_hint(max_items=50)
                                print("\nError: Anthropic returned 404 (model not found / no access)")
                                print(f"  Provided model: {self.model_name}")
                                if available:
                                    print("  Available model IDs for your account (first 50):")
                                    for mid in available:
                                        print(f"    - {mid}")
                                else:
                                    print("  Could not automatically list available models (possibly a permissions/network restriction).")
                                    print("  Check the Anthropic console for the model list, or use --list-models with this script.")

                                print("  Common examples: claude-opus-4-6 / claude-sonnet-4-6 / claude-haiku-4-5")
                                return ""

                            # Only downgrade to the next variant if it looks like a field incompatibility.
                            # Otherwise raise to the outer retry/backoff logic to avoid multiple requests per round.
                            if (
                                "thinking" in lower_msg
                                or "output_config" in lower_msg
                                or "output config" in lower_msg
                                or "effort" in lower_msg
                                or "unrecognized" in lower_msg
                                or "unknown" in lower_msg
                                or "unexpected" in lower_msg
                            ):
                                continue
                            raise

                    if resp is None:
                        raise last_err

                    # Extract text
                    try:
                        parts = []
                        content = getattr(resp, "content", None)
                        if isinstance(content, list):
                            for block in content:
                                # anthropic SDK block: has type/text
                                btype = getattr(block, "type", None)
                                btext = getattr(block, "text", None)
                                if btype == "text" and btext:
                                    parts.append(btext)
                                # Some implementations may return a dict directly
                                if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                                    parts.append(block["text"])
                        if parts:
                            return "".join(parts)
                        # Fallback: some versions may have output_text/text
                        if hasattr(resp, "output_text") and getattr(resp, "output_text"):
                            return resp.output_text
                        if hasattr(resp, "text") and getattr(resp, "text"):
                            return resp.text
                        return ""
                    except Exception as extract_error:
                        print(f"Error extracting Claude response text: {extract_error}")
                        return ""
            except Exception as e:
                print(f"API call failed (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(2 * (attempt + 1))
                if attempt == max_retries - 1:
                    return ""
        return ""
    
    # format_chat_prompt and format_continued_prompt removed; prompt logic handled directly in compute_metrics
    # extract_boxed_answer kept here (it is a general utility)
    
    def extract_boxed_answer(self, text: str) -> Optional[str]:
        """
        Extract the answer from text
        Prefer extracting from \\boxed{} (extract the last \\boxed{})
        If that fails, try extracting after "The answer is:"
        """
        # Type check: ensure text is a string
        if text is None or not isinstance(text, str):
            return None
            
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
                if i >= len(text): break
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                i += 1
            
            if brace_count == 0:
                return text[start_pos:i-1].strip()
        
        # Method 2: extract after "The answer is:"
        answer_patterns = [
            r'[Tt]he answer is:\s*\$([^$]+)\$',  # The answer is: $17$
            r'[Tt]he answer is:\s*\\\[([^\]]+)\\\]',  # The answer is: \[17\]
            r'[Tt]he answer is:\s*([^\n\.]+)',  # The answer is: 17
            r'[Tt]he final answer is:\s*\$([^$]+)\$',
            r'[Tt]he final answer is:\s*\\\[([^\]]+)\\\]',
            r'[Tt]he final answer is:\s*([^\n\.]+)',
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

    # GPU memory release function removed
    def release_vllm_model(self): pass
    def release_hf_model(self): pass
    
    # compute_token_confidences changed to a dummy implementation
    def compute_token_confidences(self, prompt: str, answer: str, compute_all_tokens: bool = True) -> Dict:
        """API models do not support token-level confidence; returns default values"""
        return {
            "first_token_confidence": 0.0,
            "all_token_confidence": 0.0,
            "num_answer_tokens": 0,
            "tokens": [],
            "confidences": []
        }

    def compute_metrics_on_dataset(
        self,
        dataset_path: str,
        truncate_ratio: float,
        max_samples: Optional[int] = None,
        batch_size: int = 1, 
        data_type: str = "clean",
        pregenerated_solutions: Optional[List[str]] = None,
        use_original_index: bool = False,
        force_answer_strict_in_system: bool = False,
        skip_full_solution: bool = False,
        max_answer_retries: int = 3,
    ) -> Tuple[List[Dict], List[str]]:
        """
        Compute metrics on a dataset (API version)
        """
        # Parameter validation
        try:
            max_answer_retries = int(max_answer_retries)
        except Exception:
            max_answer_retries = 3
        if max_answer_retries < 1:
            print(f"Warning: max_answer_retries={max_answer_retries} is invalid; auto-corrected to 1")
            max_answer_retries = 1

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
                original_index = item.get('original_index', idx) if use_original_index else idx
                samples.append({
                    'index': idx,
                    'original_index': original_index,
                    'problem': problem,
                    'answer': answer
                })
        
        print(f"Valid sample count: {len(samples)}")
        
        problems = [s['problem'] for s in samples]
        system_content = "Please reason step by step, and put your final answer within \\boxed{}."

        strict_boxed_only = (
            "IMPORTANT: You MUST put your final answer within \\boxed{} format. "
            "Output ONLY the boxed answer without any reasoning, explanation, or other text before or after it."
        )
        
        # Step 1: Generate full solutions
        # - Default: generate full solution for full_accuracy and consistency
        # - When truncate_ratio==0 and skip_full_solution=True: skip full solution generation to save cost
        #   In this mode full_accuracy/consistency are not computed (set to None); only forced-answer accuracy is kept.
        full_solution_skipped = False
        if pregenerated_solutions is None:
            if skip_full_solution and truncate_ratio == 0:
                full_solution_skipped = True
                print("\nStep 1: skipped generating full solutions (skip_full_solution=True and truncate_ratio==0)...")
                all_solutions = [""] * len(problems)
            else:
                print(f"\nStep 1: generating full solutions (even at truncate_ratio=0, needed for consistency)...")
                all_solutions = []
                for problem in tqdm(problems, desc="Generating solutions via API"):
                    solution = self.generate_api_response(problem, system_prompt=system_content)
                    all_solutions.append(solution)
                
                print(f"Successfully generated {len(all_solutions)} solutions")
        else:
            print(f"\nStep 1: using pre-generated solutions (skipping generation step)")
            all_solutions = pregenerated_solutions
            if len(all_solutions) != len(problems):
                print(f"Warning: pre-generated solution count ({len(all_solutions)}) does not match problem count ({len(problems)})")
                all_solutions = all_solutions[:len(problems)]
                samples = samples[:len(all_solutions)]
                problems = problems[:len(all_solutions)]
        
        # Step 2: Truncate and generate forced answers
        print(f"\nStep 2: generating forced answers after truncation...")
        
        all_truncated_solutions = []
        all_forced_answers_full = []
        all_raw_continuations = []  # save raw model output
        
        for idx, (problem, solution) in enumerate(tqdm(zip(problems, all_solutions), total=len(problems), desc="Generating forced answers")):
            truncate_position = int(len(solution) * truncate_ratio)
            truncated_solution = solution[:truncate_position]
            all_truncated_solutions.append(truncated_solution)
            
            # Build prompt for the model to continue completion
            if truncate_ratio == 0:
                prompt_continuation = (
                    f"{problem}\n\nPlease ONLY put your final answer within \\boxed{{}} directly without any other content before or after it (e.g., reasoning or explanation)."
                )
            else:
                prompt_continuation = (
                    f"Question: {problem}\n\n"
                    f"Partial Answer (I have reasoned this far):\n{truncated_solution}\n\n"
                    f"Please continue reasoning from where I left off and provide the final answer within \\boxed{{}}."
                )

            # system prompt for the force-answer stage
            # - Default: follow original logic (no system_prompt when truncate_ratio==0)
            # - When force_answer_strict_in_system is enabled, also inject strict_boxed_only into system_prompt
            force_system_prompt: Optional[str]
            if force_answer_strict_in_system:
                if truncate_ratio == 0:
                    force_system_prompt = strict_boxed_only
                else:
                    force_system_prompt = f"{system_content}\n\n{strict_boxed_only}"
            else:
                force_system_prompt = system_content if truncate_ratio != 0 else None
            
            # Attempt to generate an answer; retry if boxed answer cannot be extracted (configurable)
            continuation = None
            full_forced_text = None
            
            for retry_idx in range(max_answer_retries):
                # First attempt uses temperature=0.0; subsequent retries use higher temperature
                temp = 0.0 if retry_idx == 0 else 0.7
                
                # Use a more emphatic prompt when retrying
                if retry_idx > 0:
                    if truncate_ratio == 0:
                        current_prompt = (
                            f"{problem}\n\n"
                            f"IMPORTANT: You MUST put your final answer within \\boxed{{}} format. "
                            f"Output ONLY the boxed answer without any reasoning, explanation, or other text before or after it. "
                        )
                    else:
                        current_prompt = (
                            f"Question: {problem}\n\n"
                            f"Partial Answer (I have reasoned this far):\n{truncated_solution}\n\n"
                            f"IMPORTANT: Please continue and you MUST put the final answer within \\boxed{{}} format. "
                            f"Do not forget to use \\boxed{{}} around your final answer."
                        )
                else:
                    current_prompt = prompt_continuation
                
                if truncate_ratio == 0:
                    continuation = self.generate_api_response(
                        current_prompt,
                        system_prompt=force_system_prompt,
                        max_tokens=32,
                        temperature=temp,
                    )
                    full_forced_text = continuation if continuation else ""
                else:
                    continuation = self.generate_api_response(
                        current_prompt,
                        system_prompt=force_system_prompt,
                        temperature=temp,
                    )
                    full_forced_text = truncated_solution + "\n[...CONTINUED...]\n" + (continuation if continuation else "")
                
                # Check whether an answer can be extracted
                extracted_answer = self.extract_boxed_answer(full_forced_text) if full_forced_text else None
                if extracted_answer is not None:
                    break
                    
                if retry_idx < max_answer_retries - 1:
                    print(f"\nWarning: sample {idx} failed to extract boxed answer; retrying with temperature={0.7} and emphatic prompt (retry {retry_idx+1})...")
            
            if extracted_answer is None:
                print(f"\nWarning: sample {idx} could not extract boxed answer after {max_answer_retries} attempt(s)")
            
            all_forced_answers_full.append(full_forced_text)
            all_raw_continuations.append(continuation if continuation else "")
        
        print(f"Successfully generated {len(all_forced_answers_full)} forced answers")
        
        # Step 3: Compute metrics
        print(f"\nStep 3: computing metrics...")
        
        results = []
        for idx, sample in enumerate(tqdm(samples, desc="Computing metrics")):
            problem = problems[idx]
            gt_answer = sample['answer']
            solution = all_solutions[idx]
            truncated_sol = all_truncated_solutions[idx]
            forced_full = all_forced_answers_full[idx]
            raw_continuation = all_raw_continuations[idx]
            
            # Extract answers
            full_answer = self.extract_boxed_answer(solution) if not full_solution_skipped else None
            forced_answer = self.extract_boxed_answer(forced_full)
            
            # Compute accuracy and consistency
            full_correct = grade_answer(full_answer, gt_answer) if (full_answer and not full_solution_skipped) else None
            forced_correct = grade_answer(forced_answer, gt_answer) if forced_answer else False
            consistent = grade_answer(full_answer, forced_answer) if (full_answer and forced_answer and not full_solution_skipped) else None
            
            # Dummy confidence
            confidence_result = {
                'first_token_confidence': 0.0,
                'all_token_confidence': 0.0,
                'num_answer_tokens': 0
            }
            
            result = {
                'index': sample['index'],
                'original_index': sample['original_index'],
                'problem': problem,
                'ground_truth_answer': gt_answer,
                'full_solution': None if full_solution_skipped else solution,
                'truncated_solution': truncated_sol,
                'forced_full_response': raw_continuation,
                'full_answer': full_answer,
                'forced_answer': forced_answer,
                'full_accuracy': (1 if full_correct else 0) if full_correct is not None else None,
                'accuracy': 1 if forced_correct else 0,
                'consistency': (1 if consistent else 0) if consistent is not None else None,
                'first_token_confidence': confidence_result['first_token_confidence'],
                'all_token_confidence': confidence_result['all_token_confidence'],
                'num_answer_tokens': confidence_result['num_answer_tokens'],
                'full_solution_skipped': full_solution_skipped,
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
    
    # Extract metric values
    values_a = [r[metric] for r in results_a]
    values_b = [r[metric] for r in results_b]
    
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
                "lower": float(ci_lower),
                "upper": float(ci_upper)
            },
            "bootstrap_se": float(bootstrap_se),
            "bootstrap_mean": float(distribution_mean),
            "bootstrap_median": float(np.median(bootstrap_means)),
            "cohens_d": float(cohens_d),
            "cohens_d_type": "paired (Cohen's d_z based on difference std)",
            "ccs": float(ccs),
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
    
    # Extract metric values
    values_a = [r[metric] for r in results_a]
    values_b = [r[metric] for r in results_b]
    
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
        "ccs": float(ccs),
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
    
    # 1. Accuracy comparison
    ax = axes[0, 0]
    acc_a = [r['accuracy'] for r in results_a]
    acc_b = [r['accuracy'] for r in results_b]
    
    x_pos = [0, 1]
    means = [np.mean(acc_a), np.mean(acc_b)]
    stds = [np.std(acc_a), np.std(acc_b)]
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7,
                   color=['steelblue', 'coral'], edgecolor='black')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([label_a, label_b])
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.3f}±{std:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    # 2. Consistency comparison
    ax = axes[0, 1]
    cons_a = [r.get('consistency') for r in results_a if r.get('consistency') is not None]
    cons_b = [r.get('consistency') for r in results_b if r.get('consistency') is not None]

    ax.set_ylabel('Consistency', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([label_a, label_b])
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    if not cons_a and not cons_b:
        ax.set_title('Consistency Comparison (N/A)', fontsize=13, fontweight='bold')
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14, transform=ax.transAxes)
    else:
        means = [np.mean(cons_a) if cons_a else 0.0, np.mean(cons_b) if cons_b else 0.0]
        stds = [np.std(cons_a) if cons_a else 0.0, np.std(cons_b) if cons_b else 0.0]
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7,
                       color=['steelblue', 'coral'], edgecolor='black')
        ax.set_title('Consistency Comparison', fontsize=13, fontweight='bold')

        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                    f'{mean:.3f}±{std:.3f}',
                    ha='center', va='bottom', fontsize=10)
    
    # 3. First Token Confidence comparison
    ax = axes[1, 0]
    ftc_a = [r['first_token_confidence'] for r in results_a]
    ftc_b = [r['first_token_confidence'] for r in results_b]
    
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
    atc_a = [r['all_token_confidence'] for r in results_a]
    atc_b = [r['all_token_confidence'] for r in results_b]
    
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
    parser = argparse.ArgumentParser(description="Statistical significance test (API version)")
    
    # Model parameters
    parser.add_argument("--model", type=str, required=True,
                        help="API model name (e.g. gpt-4, gemini-pro)")
    parser.add_argument("--model-type", type=str, default="auto", 
                        choices=["openai", "google", "anthropic", "auto"],
                        help="API provider type (openai/google/anthropic/auto)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API Key (or set via environment variable)")
    parser.add_argument("--api-base", type=str, default=None,
                        help="API Base URL (OpenAI-compatible endpoint or Anthropic proxy; leave blank for official default)")

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List model IDs available for the current account and exit (only for model-type=anthropic)"
    )
    
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
    
    # Other parameters
    parser.add_argument("--truncate-ratio", type=float, nargs='+', default=[0.0],
                        help="Truncation ratio (multiple values allowed); 0.0 = Zero-CoT Probe (default)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (for API this is typically concurrency; simplified to 1 here)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    parser.add_argument(
        "--force-answer-strict-in-system",
        action="store_true",
        help=(
            "When enabled, also writes the strict constraint prompt into the system prompt during the force-answer stage: "
            "Requires the final output to contain only the \\boxed{} answer, with no reasoning/explanation text."
        ),
    )

    parser.add_argument(
        "--skip-full-solution",
        action="store_true",
        help=(
            "When enabled, skips full solution generation when truncate_ratio==0 to save API calls."
            "In this mode full_accuracy / consistency are not computed, and McNemar test for consistency is also skipped."
        ),
    )

    parser.add_argument(
        "--max-answer-retries",
        type=int,
        default=3,
        help=(
            "Force-answer stage: maximum retries when a \\boxed{} answer cannot be extracted (default=3)."
        ),
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert paths to absolute paths
    dataset_a = make_absolute_path(args.dataset_path_a)
    dataset_b = make_absolute_path(args.dataset_path_b)
    
    # Process truncate_ratio parameter
    truncate_ratios = args.truncate_ratio if isinstance(args.truncate_ratio, list) else [args.truncate_ratio]
    
    # Generate labels
    model_label = args.model
    dataset_a_basename = os.path.basename(dataset_a)
    dataset_b_basename = os.path.basename(dataset_b)
    
    label_a = f"{model_label} on {dataset_a_basename} ({args.data_type_a})"
    label_b = f"{model_label} on {dataset_b_basename} ({args.data_type_b})"
    
    print(f"\n{'='*80}")
    print(f"Statistical Significance Test (API Mode)")
    print(f"{'='*80}")
    print(f"Model: {model_label}")
    print(f"Dataset A: {dataset_a_basename} (type: {args.data_type_a})")
    print(f"Dataset B: {dataset_b_basename} (type: {args.data_type_b})")
    print(f"Truncation ratios: {truncate_ratios}")
    print(f"Skip full solution (ratio==0): {args.skip_full_solution}")
    print(f"Force-answer max retries: {args.max_answer_retries}")
    print(f"{'='*80}\n")
    
    # Initialize tester
    print(f"\n{'='*80}")
    print(f"Initializing API client...")
    print(f"{'='*80}\n")
    
    tester = StatisticalSignificanceTester(
        model_name=args.model,
        seed=args.seed,
        model_type=args.model_type,
        api_key=args.api_key,
        api_base=args.api_base
    )

    # Quick diagnostic: list available Anthropic models
    if args.list_models:
        if tester.model_type != "anthropic":
            print("--list-models is currently only supported for model-type=anthropic")
            return
        if tester.client is None:
            print("Error: Anthropic client not initialized (please set ANTHROPIC_API_KEY or --api-key)")
            return
        print("Available Anthropic model IDs (auto-paginated, up to 200):")
        shown = 0
        try:
            for m in tester.client.models.list(limit=100):
                mid = getattr(m, "id", None)
                if mid:
                    print(f"- {mid}")
                    shown += 1
                if shown >= 200:
                    break
        except Exception as e:
            print(f"Failed to list models: {e}")
        return
    
    # Variables for caching full solutions
    cached_solutions_a = None
    cached_solutions_b = None
    
    # Test each truncate_ratio
    for ratio_idx, truncate_ratio in enumerate(truncate_ratios):
        print(f"\n{'='*80}")
        print(f"Processing truncation ratio {ratio_idx+1}/{len(truncate_ratios)}: {truncate_ratio}")
        print(f"{'='*80}\n")
        
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
            use_original_index=True,
            force_answer_strict_in_system=args.force_answer_strict_in_system,
            skip_full_solution=args.skip_full_solution,
            max_answer_retries=args.max_answer_retries,
        )
        
        # Cache solutions after the first generation
        if cached_solutions_a is None and not (args.skip_full_solution and truncate_ratio == 0):
            cached_solutions_a = solutions_a
            print(f"Cached {len(cached_solutions_a)} solutions from dataset A for reuse in subsequent ratios")
        
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
            use_original_index=True,
            force_answer_strict_in_system=args.force_answer_strict_in_system,
            skip_full_solution=args.skip_full_solution,
            max_answer_retries=args.max_answer_retries,
        )
        
        # Cache solutions after the first generation
        if cached_solutions_b is None and not (args.skip_full_solution and truncate_ratio == 0):
            cached_solutions_b = solutions_b
            print(f"Cached {len(cached_solutions_b)} solutions from dataset B for reuse in subsequent ratios")
        
        # Match datasets by original_index
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
        can_test_consistency = all(
            (r.get("consistency") in (0, 1)) for r in (results_a + results_b)
        )
        if can_test_consistency:
            consistency_test = perform_mcnemar_test(results_a, results_b, "consistency")
            print(f"   Statistic: {consistency_test['statistic']}")
            print(f"   p-value: {consistency_test['p_value']}")
            print(f"   Significant (α=0.05): {consistency_test['significant_at_0.05']}")
            print(f"   Effect Size: {consistency_test['effect_size']:.4f}")
            print(f"   CCS: {consistency_test.get('ccs', 'N/A') if consistency_test.get('ccs') is not None else 'N/A'}")
            print(f"   A performance: {consistency_test['a_performance']:.4f}")
            print(f"   B performance: {consistency_test['b_performance']:.4f}")
            print(f"   Performance difference: {consistency_test['performance_difference']:.4f}")
        else:
            consistency_test = {
                "skipped": True,
                "reason": "consistency not available (full solution was skipped or missing)",
                "metric": "consistency",
            }
            print("   Skipped: consistency not available (full solution was skipped or missing)")
        
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
            accuracy_values = [r['accuracy'] for r in results]
            consistency_values = [r['consistency'] for r in results if r.get('consistency') is not None]
            ftc_values = [r['first_token_confidence'] for r in results if r['first_token_confidence'] > 0]
            atc_values = [r['all_token_confidence'] for r in results if r['all_token_confidence'] > 0]
            
            return {
                "accuracy": {
                    "mean": float(np.mean(accuracy_values)),
                    "std": float(np.std(accuracy_values)),
                    "count": len(accuracy_values)
                },
                "consistency": {
                    "mean": float(np.mean(consistency_values)) if consistency_values else None,
                    "std": float(np.std(consistency_values)) if consistency_values else None,
                    "count": len(consistency_values),
                    "total_samples": len(results),
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

        def compute_boxed_extraction_stats(results: List[Dict]) -> Dict:
            total = len(results)
            forced_missing = sum(1 for r in results if not r.get('forced_answer'))

            full_skipped = any(bool(r.get('full_solution_skipped')) for r in results)
            if full_skipped:
                full_missing = None
                full_missing_ratio = None
            else:
                full_missing = sum(1 for r in results if not r.get('full_answer'))
                full_missing_ratio = (full_missing / total) if total else 0.0

            forced_missing_ratio = (forced_missing / total) if total else 0.0

            return {
                "total_samples": int(total),
                "forced_answer_boxed_missing_count": int(forced_missing),
                "forced_answer_boxed_missing_ratio": float(forced_missing_ratio),
                "full_answer_skipped": bool(full_skipped),
                "full_answer_boxed_missing_count": (int(full_missing) if full_missing is not None else None),
                "full_answer_boxed_missing_ratio": (float(full_missing_ratio) if full_missing_ratio is not None else None),
            }

        boxed_stats_a = compute_boxed_extraction_stats(results_a)
        boxed_stats_b = compute_boxed_extraction_stats(results_b)
        
        print(f"\n{'='*80}")
        print(f"Overall metrics summary:")
        print(f"{'='*80}")
        print(f"\nDataset A ({label_a}):")
        print(f"  Accuracy: {metrics_a['accuracy']['mean']:.4f} ± {metrics_a['accuracy']['std']:.4f}")
        if metrics_a['consistency']['mean'] is None:
            print(f"  Consistency: N/A (skipped)")
        else:
            print(f"  Consistency: {metrics_a['consistency']['mean']:.4f} ± {metrics_a['consistency']['std']:.4f}")
        print(f"  First Token Confidence: {metrics_a['first_token_confidence']['mean']:.4f} ± {metrics_a['first_token_confidence']['std']:.4f} (n={metrics_a['first_token_confidence']['count']})")
        print(f"  All Token Confidence: {metrics_a['all_token_confidence']['mean']:.4f} ± {metrics_a['all_token_confidence']['std']:.4f} (n={metrics_a['all_token_confidence']['count']})")
        
        print(f"\nDataset B ({label_b}):")
        print(f"  Accuracy: {metrics_b['accuracy']['mean']:.4f} ± {metrics_b['accuracy']['std']:.4f}")
        if metrics_b['consistency']['mean'] is None:
            print(f"  Consistency: N/A (skipped)")
        else:
            print(f"  Consistency: {metrics_b['consistency']['mean']:.4f} ± {metrics_b['consistency']['std']:.4f}")
        print(f"  First Token Confidence: {metrics_b['first_token_confidence']['mean']:.4f} ± {metrics_b['first_token_confidence']['std']:.4f} (n={metrics_b['first_token_confidence']['count']})")
        print(f"  All Token Confidence: {metrics_b['all_token_confidence']['mean']:.4f} ± {metrics_b['all_token_confidence']['std']:.4f} (n={metrics_b['all_token_confidence']['count']})")

        print(f"\n{'-'*80}")
        print("\\boxed{} extraction failure statistics (forced_answer / full_answer):")
        print(f"  Dataset A forced_answer: {boxed_stats_a['forced_answer_boxed_missing_count']}/{boxed_stats_a['total_samples']} = {boxed_stats_a['forced_answer_boxed_missing_ratio']:.2%}")
        if boxed_stats_a["full_answer_skipped"]:
            print("  Dataset A full_answer: skipped")
        else:
            print(f"  Dataset A full_answer: {boxed_stats_a['full_answer_boxed_missing_count']}/{boxed_stats_a['total_samples']} = {boxed_stats_a['full_answer_boxed_missing_ratio']:.2%}")
        print(f"  Dataset B forced_answer: {boxed_stats_b['forced_answer_boxed_missing_count']}/{boxed_stats_b['total_samples']} = {boxed_stats_b['forced_answer_boxed_missing_ratio']:.2%}")
        if boxed_stats_b["full_answer_skipped"]:
            print("  Dataset B full_answer: skipped")
        else:
            print(f"  Dataset B full_answer: {boxed_stats_b['full_answer_boxed_missing_count']}/{boxed_stats_b['total_samples']} = {boxed_stats_b['full_answer_boxed_missing_ratio']:.2%}")
        
        # Save results
        output_data = {
            "config": {
                "label_a": label_a,
                "label_b": label_b,
                "model": args.model,
                "lora_path": None,
                "dataset_a": dataset_a,
                "dataset_b": dataset_b,
                "data_type_a": args.data_type_a,
                "data_type_b": args.data_type_b,
                "truncate_ratio": truncate_ratio,
                "skip_full_solution": bool(args.skip_full_solution),
                "num_samples_a": len(results_a),
                "num_samples_b": len(results_b)
            },
            "dataset_metrics": {
                "dataset_a": metrics_a,
                "dataset_b": metrics_b
            },
            "boxed_answer_extraction": {
                "dataset_a": boxed_stats_a,
                "dataset_b": boxed_stats_b
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
        
    print(f"\n{'='*80}")
    print(f"Statistical tests complete for all {len(truncate_ratios)} truncation ratio(s)!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
