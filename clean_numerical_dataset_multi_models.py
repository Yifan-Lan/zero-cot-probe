"""
Use the GPT API to perturb numerical values in a multi-domain dataset and recalculate
solutions and answers. Supports MMLU-Pro categories (math, physics, chemistry, business)
as well as generic local JSONL files.

This script produces a "cleaned" reference dataset used by the ZCP method to compare
against potentially contaminated data.

Usage examples:

# Clean an MMLU-Pro math subset
python clean_numerical_dataset_multi_models.py \
    --dataset-path <path/to/math/dataset_u.jsonl> \
    --mode mmlu \
    --category math \
    --output-dir <path/to/output_dir> \
    --model o3-mini \
    --verify-model-gemini gemini-2.5-flash \
    --verify-model-gpt o4-mini \
    --display \
    --save-interval 10

# Resume from a previous run using a failed-indices file
python clean_numerical_dataset_multi_models.py \
    --dataset-path <path/to/dataset_u.jsonl> \
    --mode mmlu \
    --category physics \
    --output-dir <path/to/output_dir> \
    --model o3 \
    --verify-model-gemini gemini-2.5-flash \
    --verify-model-gpt o4-mini \
    --save-interval 10 \
    --index-range <path/to/failed_indices.json>

# Supported categories: math, physics, chemistry, business

# Set API keys via environment variables before running:
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
"""

import os
import json
import time
import csv
import random
import re
from typing import Dict, List, Optional
from openai import OpenAI
from tqdm import tqdm
import argparse
from google import genai
from datasets import load_dataset

class DatasetParaphraser:
    def __init__(self, api_key: str, model: str = "gpt-4o", output_dir: str = "./paraphrased_data", display: bool = False,
                 gemini_api_key: Optional[str] = None, verify_model_gpt: str = "gpt-4o-mini", 
                 verify_model_gemini: str = "gemini-2.0-flash-exp", max_verify_retries: int = 3):
        """
        Initialize the dataset modifier
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use
            output_dir: output directory
            display: whether to display the model's full output
            gemini_api_key: Gemini API key
            verify_model_gpt: GPT model for verification
            verify_model_gemini: Gemini model for verification
            max_verify_retries: max retries after verification failure
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.output_dir = output_dir
        self.display = display
        self.verify_model_gpt = verify_model_gpt
        self.verify_model_gemini = verify_model_gemini
        self.max_verify_retries = max_verify_retries
        
        # Initialize Gemini client
        if gemini_api_key:
            self.gemini_client = genai.Client(api_key=gemini_api_key)
        else:
            self.gemini_client = None
            
        self.system_prompt = f"""You are a mathematical data generator specialized in creating diverse training samples. Your task is to create a new sample by paraphrasing and modifying the original problem while maintaining the same difficulty level and solution logic.

**Strict Instructions:**

1.  **Paraphrase and Modify Problem:**
    * **MUST paraphrase**: Rephrase sentences, change wording, and adjust sentence structure to create a distinctly different version.
    * **MUST change context**: Change variable names, object names, or scenarios (e.g., "apples" to "books", "Alice" to "Bob", "students in a class" to "workers in a factory").
    * **Change numerical values** with these constraints:
        - Keep the same ORDER OF MAGNITUDE (e.g., if original is 50, use 30-80, NOT 5 or 500).
        - Keep integers as integers, decimals as decimals with similar precision.
        - For multiple numbers in the problem, scale them proportionally when possible.
        - **CRITICAL:** Aim to keep the final answer's ORDER OF MAGNITUDE similar to the original (e.g., if original answer is 42, target new answer around 20-80, NOT 4 or 400).
    * **CRITICAL:** Do NOT change the mathematical logic, problem type, or solution method. The core mathematical concept must remain identical.
    * **CRITICAL:** Maintain the same difficulty level - if the original requires specific techniques, the modified version must require the same techniques.
    * **CRITICAL:** Preserve ALL formatting, including LaTeX notation ($ signs, \\cdot, \\frac, \\begin, \\end, etc.), Asymptote code ([asy]...[/asy]), and markdown.

2.  **Recalculate Solution:**
    * Rewrite the "Solution" step-by-step using the paraphrased problem and modified numbers.
    * Follow the EXACT same logical reasoning and solution method as the original.
    * Apply the same mathematical techniques and problem-solving steps.
    * Preserve ALL LaTeX formatting and code blocks from the original.
    * Perform all necessary arithmetic correctly to reflect the changes.

3.  **Update Answer:**
    * Calculate the final result based on your new solution.
    * **Verify the answer is in the same ORDER OF MAGNITUDE as the original answer**.
    * Output the new result in the "Answer" field using the SAME format as the original.
    * Ensure the "Answer" matches the recalculated solution.

**Output Format:**
Reasoning: [Describe the changes made, e.g., "Changed 5 to 7, 10 to 14 (kept same order of magnitude), rephrased 'students in class' to 'workers in factory', changed character names from Alice/Bob to Emma/John. Original answer was 15, new answer is 21 (same order of magnitude)."]
New Problem: [Paraphrased problem with new numbers, context, and wording, preserving ALL formatting]
New Solution: [Recalculated solution following the same logic, preserving ALL formatting]
New Answer: [The new final result in same order of magnitude, preserving ALL formatting]
        """

        self.system_prompt_finance = f"""You are a financial problem generator. Your task is to take an existing financial calculation problem, modify the numerical values, and optionally paraphrase the question.

**Strict Instructions:**

1.  **Modify Problem:**
    *   **Change numerical values** with these constraints:
        - Keep the same ORDER OF MAGNITUDE (e.g., if original is 50, use 30-80, NOT 5 or 500).
        - Keep integers as integers, decimals as decimals with similar precision.
        - **CRITICAL:** Aim to keep the final answer's ORDER OF MAGNITUDE similar to the original (e.g., if original answer is 42, target new answer around 20-80, NOT 4 or 400).
    *   **CRITICAL:** Do NOT change the mathematical logic, problem type, or financial principles. 
    *   **CRITICAL:** Maintain the same difficulty level - if the original requires specific techniques, the modified version must require the same techniques.
    *   **CRITICAL:** Ensure the number changes *WILL* affect the final answer.
    *   **Paraphrasing (Optional):** You may rephrase the question text for clarity or variety (wording, context and etc.), as long as the core financial concepts and logic remain EXACTLY the same.

2.  **Generate Solution:**
    *   Since the original problem might not have a solution provided, you must derive the correct step-by-step solution logic based on the financial principles involved.
    *   Apply this logic to the **NEW** numbers.
    *   Show all steps clearly.

3.  **Calculate New Answer:**
    *   Calculate the final result based on your new numbers and solution.
    *   Output the new result in the "Answer" field.

4.  **Formatting:**
    *   Preserve ALL formatting, including LaTeX notation ($ signs, \\cdot, \\frac, \\begin, \\end, etc.), Asymptote code ([asy]...[/asy]), and markdown.

**Output Format:**
Reasoning: [List the number changes made and very briefly the logic used]
New Problem: [The question text with NEW numbers]
New Solution: [Step-by-step solution using the NEW numbers]
Answer: [The final numerical answer]
"""

        self.system_prompt_mmlu = f"""You are a multi-domain data generator specialized in numerical problems across various subjects (mathematics, physics, chemistry, business, etc.). Your task is to take an existing open-ended question with a numerical answer, modify the numbers in the question, and recalculate the correct answer.

**IMPORTANT - When to SKIP:**
If ANY of these conditions apply, output ONLY the word "SKIP" and stop:
- The problem contains NO numerical values that can be reasonably modified
- The numerical values are constants (e.g., fundamental constants like π, e, speed of light)
- Modifying the numbers would violate domain-specific laws (e.g., makes chemistry impossible, physics nonsensical)
- The numbers are years, dates, or historical references that cannot be changed
- The problem structure makes it impossible to maintain the same difficulty when changing numbers
        
**Strict Instructions (if NOT skipping):**

1.  **Modify Question:**
    *   **Change numerical values** with these constraints:
        - Keep the same ORDER OF MAGNITUDE (e.g., if original is 50, use 30-80, NOT 5 or 500).
        - Keep integers as integers, decimals as decimals with similar precision.
        - Preserve units and their appropriateness (e.g., meters, seconds, moles, dollars).
        - **CRITICAL:** Aim to keep the final answer's ORDER OF MAGNITUDE similar to the original.
    *   **CRITICAL:** Do NOT change the problem type, domain-specific laws/principles, or core concepts.
    *   **CRITICAL:** Maintain the same difficulty level and required knowledge.
    *   **CRITICAL:** Ensure the number changes affect the final answer.
    *   **CRITICAL:** For physics problems, ensure modified values still obey physical laws and constraints.
    *   **CRITICAL:** For chemistry problems, keep stoichiometric ratios and chemical principles valid.
    *   **CRITICAL:** For business problems, keep financial metrics and scenarios realistic.
    *   **Paraphrasing (Optional):** You may rephrase the question text for clarity or variety (wording, context and etc.), as long as the domain-specific concepts remain EXACTLY the same.

2.  **Generate Solution:**
    *   Derive the correct step-by-step solution using the NEW numbers.
    *   Show all steps clearly.
    *   Perform all arithmetic correctly.

3.  **Calculate New Answer:**
    *   Calculate the final result based on your new numbers and solution.
    *   Output the new result in similar format to the original answer, preserving any specific formatting (like units, LaTeX, etc.).

4.  **Formatting:**
    *   Preserve ALL formatting, including LaTeX notation ($ signs, \\cdot, \\frac, \\begin, \\end, etc.), scientific notation, chemical formulas, and markdown.
    *   Maintain proper units and notation conventions for the specific domain.

**Output Format:**
If skipping: SKIP

If proceeding:
Reasoning: [List the number changes made and briefly the logic used]
New Problem: [The open-ended question text with NEW numbers]
New Solution: [Step-by-step solution using the NEW numbers]
Answer: [The final answer]
"""
        
        # Fallback: if 'both' mode has no separate prompt defined, reuse the single-sample prompt
        self.system_prompt_both = getattr(self, "system_prompt_both", self.system_prompt)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def _display_block(self, title: str, content: str):
        if not self.display:
            return
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        print(content)
        print("=" * 80 + "\n")

    def _should_fallback_from_chat(self, e: Exception) -> bool:
        msg = str(e).lower()
        # Common patterns:
        # - "only supported in v1/responses and not in v1/chat/completions"
        # - "not a chat model ... not supported in the v1/chat/completions endpoint"
        # - "did you mean to use v1/completions?"
        return (
            "v1/chat/completions" in msg
            and (
                "v1/responses" in msg
                or "only supported" in msg
                or "not a chat model" in msg
                or "v1/completions" in msg
            )
        )

    def _extract_text_from_responses_api(self, response) -> str:
        """Best-effort extraction across OpenAI SDK versions."""
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text

        chunks: List[str] = []
        output = getattr(response, "output", None)
        if isinstance(output, list):
            for item in output:
                content = getattr(item, "content", None)
                if not isinstance(content, list):
                    continue
                for c in content:
                    ctype = getattr(c, "type", None)
                    if ctype == "output_text":
                        t = getattr(c, "text", None)
                        if isinstance(t, str):
                            chunks.append(t)
                    elif ctype == "refusal":
                        # Some models may return refusal blocks
                        t = getattr(c, "refusal", None)
                        if not isinstance(t, str):
                            t = getattr(c, "text", None)
                        if isinstance(t, str) and t.strip():
                            chunks.append(f"[REFUSAL] {t}")
        return "".join(chunks).strip()

    def _truncate_text(self, text: str, max_chars: int = 4000) -> str:
        if not isinstance(text, str):
            return ""
        text = text.strip()
        if len(text) <= max_chars:
            return text
        head = max_chars // 2
        tail = max_chars - head
        return text[:head].rstrip() + "\n\n...[TRUNCATED]...\n\n" + text[-tail:].lstrip()

    def _responses_http_fallback(self, *, model: str, messages: List[Dict], temperature: Optional[float], max_tokens: Optional[int]) -> str:
        """Fallback to raw HTTP /v1/responses for environments where openai SDK lacks client.responses."""
        try:
            import requests  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "The 'requests' package is required for the /v1/responses HTTP fallback. Please install requests or upgrade the openai SDK."
            ) from e

        payload = {
            "model": model,
            "input": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens

        resp = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=180,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"/v1/responses HTTP call failed: {resp.status_code} {resp.text}")

        data = resp.json()
        text = data.get("output_text")
        if isinstance(text, str) and text.strip():
            return text.strip()

        chunks: List[str] = []
        for item in data.get("output", []) or []:
            for c in item.get("content", []) or []:
                if c.get("type") == "output_text" and isinstance(c.get("text"), str):
                    chunks.append(c["text"])
                elif c.get("type") == "refusal":
                    t = c.get("refusal") or c.get("text")
                    if isinstance(t, str) and t.strip():
                        chunks.append(f"[REFUSAL] {t}")
        out = "".join(chunks).strip()
        if not out:
            raise RuntimeError("/v1/responses returned empty content; cannot extract text")
        return out

    def _openai_text(self, *, model: str, messages: List[Dict], temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """Call OpenAI using chat.completions when possible; auto-fallback to responses API for responses-only models."""

        # 1) Try chat.completions first (backward compatible for older models)
        try:
            chat_kwargs = {
                "model": model,
                "messages": messages,
            }
            if temperature is not None:
                chat_kwargs["temperature"] = temperature
            if max_tokens is not None:
                # Some SDKs prefer max_completion_tokens
                chat_kwargs["max_completion_tokens"] = max_tokens

            response = self.client.chat.completions.create(**chat_kwargs)
            text = response.choices[0].message.content
            if isinstance(text, str) and text.strip():
                return text
            # Avoid silently returning empty strings (hard to debug)
            if self.display:
                self._display_block(
                    f"OpenAI chat returned empty output | model={model}",
                    str(response),
                )
            raise RuntimeError("OpenAI chat returned empty output")
        except Exception as e:
            if not self._should_fallback_from_chat(e):
                raise

        # 2) Fallback to responses API (preferred for gpt-5* / o* models)
        if hasattr(self.client, "responses"):
            resp_kwargs = {
                "model": model,
                "input": messages,
            }
            if temperature is not None:
                resp_kwargs["temperature"] = temperature
            if max_tokens is not None:
                resp_kwargs["max_output_tokens"] = max_tokens

            response = self.client.responses.create(**resp_kwargs)
            text = self._extract_text_from_responses_api(response)
            if not text:
                if self.display:
                    raw = None
                    if hasattr(response, "model_dump_json"):
                        try:
                            raw = response.model_dump_json()
                        except Exception:
                            raw = None
                    if raw is None and hasattr(response, "model_dump"):
                        try:
                            import json as _json
                            raw = _json.dumps(response.model_dump(), ensure_ascii=False)
                        except Exception:
                            raw = None
                    if raw is None:
                        raw = str(response)
                    self._display_block(
                        f"OpenAI responses raw response (truncated) | model={model}",
                        self._truncate_text(raw, 8000),
                    )
                raise RuntimeError("responses API returned empty content; cannot extract text output")
            return text

        # 2b) If SDK lacks responses, try raw HTTP /v1/responses; if that fails, try legacy completions
        try:
            return self._responses_http_fallback(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception:
            pass

        if hasattr(self.client, "completions"):
            prompt_parts: List[str] = []
            for m in messages:
                role = (m.get("role") or "").strip()
                content = (m.get("content") or "").strip()
                if not content:
                    continue
                if role:
                    prompt_parts.append(f"{role.upper()}: {content}")
                else:
                    prompt_parts.append(content)
            prompt = "\n\n".join(prompt_parts)

            comp_kwargs = {
                "model": model,
                "prompt": prompt,
            }
            if max_tokens is not None:
                comp_kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                comp_kwargs["temperature"] = temperature

            response = self.client.completions.create(**comp_kwargs)
            text = getattr(response.choices[0], "text", "")
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("completions API returned empty content; cannot extract text output")
            return text

        raise RuntimeError(
            "The current openai SDK does not support responses/completions fallback, but the selected model cannot use v1/chat/completions."
        )
        
    
    def paraphrase_both_versions(
        self, 
        original_problem: str, 
        original_solution: str, 
        original_answer: str,
        paraphrased_problem: str,
        paraphrased_solution: str,
        paraphrased_answer: str,
        max_retries: int = 3
    ) -> Optional[Dict]:
        """
        Modify both original and paraphrased versions simultaneously, ensuring consistent number changes, and verifying answer correctness
        
        Args:
            original_problem: original problem
            original_solution: original solution
            original_answer: original answer
            paraphrased_problem: paraphrased problem
            paraphrased_solution: paraphrased solution
            paraphrased_answer: paraphrased answer
            max_retries: maximum number of retries
            
        Returns:
            dict with modification results for both versions, or None on failure
        """
        user_message = f"""Original Problem: {original_problem}

Original Solution: {original_solution}

Original Answer: {original_answer}

---

Paraphrased Problem: {paraphrased_problem}

Paraphrased Solution: {paraphrased_solution}

Paraphrased Answer: {paraphrased_answer}"""
        
        base_user_message = user_message
        last_generation: Optional[str] = None
        last_feedback: Optional[str] = None

        for attempt in range(max_retries):
            try:
                user_message = base_user_message
                if attempt > 0 and (last_generation or last_feedback):
                    user_message += (
                        "\n\n---\n"
                        "Previous attempt failed verification. Use the verifier feedback below to fix the mistakes and regenerate. "
                        "You MUST follow the required output format exactly.\n\n"
                        f"Previous model output:\n<<<\n{self._truncate_text(last_generation or '', 10000)}\n>>>\n\n"
                        f"Verifier feedback:\n<<<\n{self._truncate_text(last_feedback or '', 10000)}\n>>>\n"
                    )

                gen_messages = [
                    {"role": "system", "content": self.system_prompt_both},
                    {"role": "user", "content": user_message},
                ]
                if self.display:
                    self._display_block(
                        f"Model input (Both Mode) | model={self.model} | attempt={attempt + 1}/{max_retries}",
                        f"[SYSTEM]\n{self.system_prompt_both}\n\n[USER]\n{user_message}",
                    )

                content = self._openai_text(
                    model=self.model,
                    messages=gen_messages,
                    temperature=1,
                )

                self._display_block(
                    f"Model output (Both Mode) | model={self.model} | attempt={attempt + 1}/{max_retries}",
                    content,
                )
                
                # Parse response
                result = self._parse_both_response(content)
                if result:
                    # Verify answers for both versions
                    print(f"\nVerifying generated data (attempt {attempt + 1}/{max_retries})...")
                    
                    # Verify Original version
                    original_valid, original_feedback = self._verify_answer_with_feedback(
                        result["original"]["new_problem"],
                        result["original"]["new_solution"],
                        result["original"]["answer"],
                    )
                    
                    # Verify Paraphrased version
                    paraphrased_valid, paraphrased_feedback = self._verify_answer_with_feedback(
                        result["paraphrased"]["new_problem"],
                        result["paraphrased"]["new_solution"],
                        result["paraphrased"]["answer"],
                    )
                    
                    # Only return result if both versions pass verification
                    if original_valid and paraphrased_valid:
                        print("✓ Verification passed: both versions have correct answers")
                        return result
                    else:
                        print(f"✗ Verification failed: Original={original_valid}, Paraphrased={paraphrased_valid}, retrying...")
                        last_generation = content
                        last_feedback = (
                            f"[Original verifier]\n{original_feedback}\n\n"
                            f"[Paraphrased verifier]\n{paraphrased_feedback}"
                        )
                else:
                    print(f"Parsing failed, attempt {attempt + 1}/{max_retries}")
                    last_generation = content
                    last_feedback = (
                        "Parse failed: missing required markers for both-mode output. "
                        "Please output all required sections exactly (Reasoning / Original New Problem/Solution/Answer / Paraphrased New Problem/Solution/Answer)."
                    )
                    
            except Exception as e:
                print(f"API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        print(f"Reached maximum retries {max_retries}, skipping sample")
        return None
    
    def paraphrase_single_item(self, problem: str, solution: str, answer: str, max_retries: int = 3) -> Optional[Dict]:
        """
        Rewrite a single question and answer using GPT, and verify answer correctness
        
        Args:
            problem: original problem
            solution: original solution process
            answer: original answer
            max_retries: maximum number of retries
            
        Returns:
            dict with rewriting results, or None on failure
        """
        base_user_message = f"Problem: {problem}\n\nSolution: {solution}\n\nAnswer: {answer}"
        last_generation: Optional[str] = None
        last_feedback: Optional[str] = None
        
        for attempt in range(max_retries):
            try:
                user_message = base_user_message
                if attempt > 0 and (last_generation or last_feedback):
                    user_message += (
                        "\n\n---\n"
                        "Previous attempt failed verification. Use the verifier feedback below to fix the mistakes and regenerate. "
                        "You MUST follow the required output format exactly.\n\n"
                        f"Previous model output:\n<<<\n{self._truncate_text(last_generation or '', 10000)}\n>>>\n\n"
                        f"Verifier feedback:\n<<<\n{self._truncate_text(last_feedback or '', 10000)}\n>>>\n"
                    )

                gen_messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message},
                ]
                if self.display:
                    self._display_block(
                        f"Model input (Single Mode) | model={self.model} | attempt={attempt + 1}/{max_retries}",
                        f"[SYSTEM]\n{self.system_prompt}\n\n[USER]\n{user_message}",
                    )

                content = self._openai_text(
                    model=self.model,
                    messages=gen_messages,
                    temperature=1,
                )

                self._display_block(
                    f"Model output (Single Mode) | model={self.model} | attempt={attempt + 1}/{max_retries}",
                    content,
                )
                
                # Parse response
                result = self._parse_response(content)
                if result:
                    # Check if the model returned SKIP (problem not suitable for number modification)
                    if result.get("skip"):
                        print(f"Model determined the problem is not suitable for number modification; returning SKIP")
                        return result
                    
                    # Verify answer
                    print(f"\nVerifying generated data (attempt {attempt + 1}/{max_retries})...")
                    is_valid, feedback = self._verify_answer_with_feedback(
                        result["new_problem"],
                        result["new_solution"],
                        result["answer"],
                    )
                    
                    if is_valid:
                        print("✓ Verification passed: answer is correct")
                        return result
                    else:
                        print(f"✗ Verification failed, retrying...")
                        last_generation = content
                        last_feedback = feedback
                else:
                    print(f"Parsing failed, attempt {attempt + 1}/{max_retries}")
                    last_generation = content
                    last_feedback = (
                        "Parse failed: missing required markers. "
                        "Please output exactly with sections: Reasoning:, New Problem:, New Solution:, Answer:."
                    )
                    
            except Exception as e:
                print(f"API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # exponential backoff
        
        print(f"Reached maximum retries {max_retries}, skipping sample")
        return None
    
    def _parse_response(self, content: str) -> Optional[Dict]:
        """
        Parse GPT response, extracting reasoning, new_problem, new_solution, and answer
        
        Args:
            content: GPT response content
            
        Returns:
            parsed dict, or None on failure; returns a special SKIP marker if applicable
        """
        try:
            # Check for SKIP
            if content.strip().upper() == "SKIP":
                return {"skip": True}
            
            # Find each section
            reasoning_start = content.find("Reasoning:")
            problem_start = content.find("New Problem:")
            solution_start = content.find("New Solution:")
            answer_start = content.find("Answer:")
            
            if reasoning_start == -1 or problem_start == -1 or solution_start == -1 or answer_start == -1:
                return None
            
            reasoning = content[reasoning_start + len("Reasoning:"):problem_start].strip()
            new_problem = content[problem_start + len("New Problem:"):solution_start].strip()
            new_solution = content[solution_start + len("New Solution:"):answer_start].strip()
            answer = content[answer_start + len("Answer:"):].strip()
            
            # Clean up separators in answer
            if answer.endswith("---"):
                answer = answer[:-3].strip()
            
            return {
                "reasoning": reasoning,
                "new_problem": new_problem,
                "new_solution": new_solution,
                "answer": answer
            }
        except Exception as e:
            print(f"Parse error: {str(e)}")
            return None
    
    def _parse_index_range(self, index_range: str) -> Optional[List[int]]:
        """
        Parse the index_range argument
        
        Args:
            index_range: format can be:
                - "start-end" (range)
                - "1,2,3" (comma-separated list)
                - "/path/to/failed_indices.json" (JSON file path)
            
        Returns:
            parsed index list, or None on invalid format
        """
        try:
            # Check if this is a JSON file path
            if index_range.endswith('.json'):
                if not os.path.exists(index_range):
                    print(f"Error: JSON file does not exist: {index_range}")
                    return None
                
                with open(index_range, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Supports direct list or a dict with a 'failed_indices' field
                if isinstance(data, list):
                    indices = [int(x) for x in data]
                elif isinstance(data, dict) and 'failed_indices' in data:
                    indices = [int(x) for x in data['failed_indices']]
                else:
                    print(f"Error: unsupported JSON format, expected a list or a dict with 'failed_indices'")
                    return None
                
                print(f"Loaded {len(indices)} indices from {index_range}")
                return indices
            
            # Original parsing logic
            if '-' in index_range:
                # Range format: "10-20"
                parts = index_range.split('-')
                if len(parts) != 2:
                    return None
                start, end = int(parts[0]), int(parts[1])
                if start > end:
                    return None
                return list(range(start, end + 1))
            elif ',' in index_range:
                # Comma-separated format: "10,15,20"
                return [int(x.strip()) for x in index_range.split(',')]
            else:
                # Single number
                return [int(index_range)]
        except ValueError as e:
            print(f"Failed to parse index range: {str(e)}")
            return None
        except Exception as e:
            print(f"Failed to read JSON file: {str(e)}")
            return None
    
    def _parse_both_response(self, content: str) -> Optional[Dict]:
        """
        Parse GPT response in both-mode, extracting results for two versions
        
        Args:
            content: GPT response content
            
        Returns:
            parsed dict containing original and paraphrased versions, or None on failure
        """
        try:
            # Find each section
            reasoning_start = content.find("Reasoning:")
            
            orig_problem_start = content.find("Original New Problem:")
            orig_solution_start = content.find("Original New Solution:")
            orig_answer_start = content.find("Original New Answer:")
            
            para_problem_start = content.find("Paraphrased New Problem:")
            para_solution_start = content.find("Paraphrased New Solution:")
            para_answer_start = content.find("Paraphrased New Answer:")
            
            # Check if all markers are present
            if (reasoning_start == -1 or orig_problem_start == -1 or orig_solution_start == -1 or 
                orig_answer_start == -1 or para_problem_start == -1 or para_solution_start == -1 or 
                para_answer_start == -1):
                print("Parse failed: missing required marker")
                return None
            
            # Extract reasoning
            reasoning = content[reasoning_start + len("Reasoning:"):orig_problem_start].strip()
            
            # Extract original version
            orig_problem = content[orig_problem_start + len("Original New Problem:"):orig_solution_start].strip()
            orig_solution = content[orig_solution_start + len("Original New Solution:"):orig_answer_start].strip()
            orig_answer = content[orig_answer_start + len("Original New Answer:"):para_problem_start].strip()
            
            # Extract paraphrased version
            para_problem = content[para_problem_start + len("Paraphrased New Problem:"):para_solution_start].strip()
            para_solution = content[para_solution_start + len("Paraphrased New Solution:"):para_answer_start].strip()
            para_answer = content[para_answer_start + len("Paraphrased New Answer:"):].strip()
            
            # Clean up separators in answer
            # Remove trailing "---" or other separators
            if orig_answer.endswith("---"):
                orig_answer = orig_answer[:-3].strip()
            if para_answer.endswith("---"):
                para_answer = para_answer[:-3].strip()
            
            return {
                "reasoning": reasoning,
                "original": {
                    "new_problem": orig_problem,
                    "new_solution": orig_solution,
                    "answer": orig_answer
                },
                "paraphrased": {
                    "new_problem": para_problem,
                    "new_solution": para_solution,
                    "answer": para_answer
                }
            }
        except Exception as e:
            print(f"Parse error: {str(e)}")
            return None
    
    def _parse_verifier_result(self, content: str) -> (bool, str):
        result_line = ""
        reasoning_lines: List[str] = []
        for line in (content or "").splitlines():
            s = line.strip()
            if not result_line and s.startswith("Result:"):
                result_line = s
                continue
            if s.startswith("Reasoning:"):
                reasoning_lines.append(s[len("Reasoning:"):].strip())
                continue
            if reasoning_lines:
                reasoning_lines.append(s)

        if result_line:
            result_value = result_line.replace("Result:", "").strip().upper()
            is_correct = "CORRECT" in result_value and "INCORRECT" not in result_value
        else:
            is_correct = "CORRECT" in (content or "").upper() and "INCORRECT" not in (content or "").upper()

        reasoning = "\n".join([r for r in reasoning_lines if r]).strip()
        if not reasoning:
            reasoning = (content or "").strip()
        return bool(is_correct), reasoning

    def _verify_with_gpt_detail(self, problem: str, solution: str, answer: str) -> (bool, str, str):
        """Return (is_correct, reasoning, raw_output)."""
#         verify_prompt = f"""You are a mathematical answer verifier. Given a math problem, its solution, and the final answer, verify if the solution and answer are correct.

# Problem: {problem}

# Solution: {solution}

# Answer: {answer}

# Please verify if the final answer is mathematically correct based on the problem and solution. Consider:
# 1. Does the solution logic correctly follow from the problem?
# 2. Is the final answer correct based on the solution?

# You MUST respond in the following format:
# Result: [CORRECT or INCORRECT]
# Reasoning: [Brief explanation of your verification]"""
        verify_prompt = f"""You are a mathematical answer verifier. Given a math problem, its solution, and the final answer, verify if the solution and answer are correct.

Problem: {problem}

Solution: {solution}

Answer: {answer}

Please verify if the final answer is mathematically correct. Consider:
1. Is the final answer correct based on the solution?
2. If the final answer is correct, you can ignore the minor mistakes in the solution steps.
3. If the final answer is incorrect, identify the key mistakes in the solution that led to the wrong answer.

You MUST respond in the following format:
Result: [CORRECT or INCORRECT]
Reasoning: [Brief explanation of your verification]"""

        if self.display:
            self._display_block(
                f"Verifier input (OpenAI) | model={self.verify_model_gpt}",
                verify_prompt,
            )

        try:
            reasoning_models = ['o1', 'o3', 'o4']
            is_reasoning_model = any(self.verify_model_gpt.startswith(model) for model in reasoning_models)
            content = self._openai_text(
                model=self.verify_model_gpt,
                messages=[{"role": "user", "content": verify_prompt}],
                temperature=None if is_reasoning_model else 0,
                # No max_tokens limit: allow verifier to give full explanation
            ).strip()
            is_correct, reasoning = self._parse_verifier_result(content)

            if self.display:
                self._display_block(
                    f"Verifier output (OpenAI) | model={self.verify_model_gpt}",
                    content,
                )
            return bool(is_correct), reasoning, content
        except Exception as e:
            msg = f"GPT verification failed: {str(e)}"
            if self.display:
                self._display_block(
                    f"Verifier output (OpenAI) | model={self.verify_model_gpt} | ERROR",
                    msg,
                )
            return False, msg, msg

    def _verify_with_gemini_detail(self, problem: str, solution: str, answer: str) -> (bool, str, str):
        """Return (is_correct, reasoning, raw_output)."""
        if self.gemini_client is None:
            msg = "Gemini client not initialized"
            return False, msg, msg

#         verify_prompt = f"""You are a mathematical answer verifier. Given a math problem, its solution, and the final answer, verify if the answer is correct.

# Problem: {problem}

# Solution: {solution}

# Answer: {answer}

# Please verify if the final answer is mathematically correct based on the problem and solution. Consider:
# 1. Does the solution logic correctly follow from the problem?
# 2. Is the final answer correct based on the solution?

# You MUST respond in the following format:
# Result: [CORRECT or INCORRECT]
# Reasoning: [Brief explanation of your verification]"""
        verify_prompt = f"""You are a mathematical answer verifier. Given a math problem, its solution, and the final answer, verify if the solution and answer are correct.

Problem: {problem}

Solution: {solution}

Answer: {answer}

Please verify if the final answer is mathematically correct. Consider:
1. Is the final answer correct based on the solution?
2. If the final answer is correct, you can ignore the minor mistakes in the solution steps.
3. If the final answer is incorrect, identify the key mistakes in the solution that led to the wrong answer.

You MUST respond in the following format:
Result: [CORRECT or INCORRECT]
Reasoning: [Brief explanation of your verification]"""

        if self.display:
            self._display_block(
                f"Verifier input (Gemini) | model={self.verify_model_gemini}",
                verify_prompt,
            )

        try:
            response = self.gemini_client.models.generate_content(
                model=self.verify_model_gemini,
                contents=verify_prompt
            )
            content = (response.text or "").strip()
            is_correct, reasoning = self._parse_verifier_result(content)

            if self.display:
                self._display_block(
                    f"Verifier output (Gemini) | model={self.verify_model_gemini}",
                    content,
                )
            return bool(is_correct), reasoning, content
        except Exception as e:
            msg = f"Gemini verification failed: {str(e)}"
            if self.display:
                self._display_block(
                    f"Verifier output (Gemini) | model={self.verify_model_gemini} | ERROR",
                    msg,
                )
            return False, msg, msg

    def _verify_with_gpt(self, problem: str, solution: str, answer: str) -> bool:
        """
        Verify whether an answer is correct using GPT
        
        Args:
            problem: math problem
            solution: solution process
            answer: final answer
            
        Returns:
            True if answer is correct, False otherwise
        """
        verify_prompt = f"""You are a mathematical answer verifier. Given a math problem, its solution, and the final answer, verify if the solution and answer are correct.

Problem: {problem}

Solution: {solution}

Answer: {answer}

Please verify if the final answer is mathematically correct based on the problem and solution. Consider:
1. Does the solution logic correctly follow from the problem?
2. Is the final answer correct based on the solution?

You MUST respond in the following format:
Result: [CORRECT or INCORRECT]
Reasoning: [Brief explanation of your verification]"""
        
        try:
            is_correct, _, content = self._verify_with_gpt_detail(problem, solution, answer)
            
            if self.display:
                print(f"\n[GPT verify] model: {self.verify_model_gpt}")
                print(f"Full output:\n{content}")
            
            return bool(is_correct)
            
        except Exception as e:
            print(f"GPT verification failed: {str(e)}")
            return False

    def _verify_answer_with_feedback(self, problem: str, solution: str, answer: str) -> (bool, str):
        """Return (is_valid, feedback_for_retry)."""
        gpt_ok, gpt_reason, gpt_raw = self._verify_with_gpt_detail(problem, solution, answer)
        if not gpt_ok:
            feedback = (
                f"Verifier={self.verify_model_gpt} Result=INCORRECT\n"
                f"Reasoning:\n{gpt_reason}\n\n"
                f"Raw verifier output:\n{self._truncate_text(gpt_raw, 1200)}"
            )
            return False, feedback

        gem_ok, gem_reason, gem_raw = self._verify_with_gemini_detail(problem, solution, answer)
        if not gem_ok:
            feedback = (
                f"Verifier={self.verify_model_gpt} Result=CORRECT\nReasoning:\n{gpt_reason}\n\n"
                f"Verifier={self.verify_model_gemini} Result=INCORRECT\n"
                f"Reasoning:\n{gem_reason}\n\n"
                f"Raw verifier output:\n{self._truncate_text(gem_raw, 1200)}"
            )
            return False, feedback

        return True, "OK"
    
    def _verify_with_gemini(self, problem: str, solution: str, answer: str) -> bool:
        """
        Verify whether an answer is correct using Gemini
        
        Args:
            problem: math problem
            solution: solution process
            answer: final answer
            
        Returns:
            True if answer is correct, False otherwise
        """
        if self.gemini_client is None:
            print("Error: Gemini client not initialized")
            return False
        
        verify_prompt = f"""You are a mathematical answer verifier. Given a math problem, its solution, and the final answer, verify if the answer is correct.

Problem: {problem}

Solution: {solution}

Answer: {answer}

Please verify if the final answer is mathematically correct based on the problem and solution. Consider:
1. Does the solution logic correctly follow from the problem?
2. Is the final answer correct based on the solution?

You MUST respond in the following format:
Result: [CORRECT or INCORRECT]
Reasoning: [Brief explanation of your verification]"""
        
        try:
            response = self.gemini_client.models.generate_content(
                model=self.verify_model_gemini,
                contents=verify_prompt
            )
            
            content = response.text.strip()
            
            if self.display:
                print(f"\n[Gemini verify] model: {self.verify_model_gemini}")
                print(f"Full output:\n{content}")
            
            # Extract Result field
            result_line = ""
            for line in content.split('\n'):
                if line.strip().startswith('Result:'):
                    result_line = line.strip()
                    break
            
            if result_line:
                result_value = result_line.replace('Result:', '').strip().upper()
                is_correct = 'CORRECT' in result_value and 'INCORRECT' not in result_value
            else:
                # If Result field not found, fall back to searching the whole content
                is_correct = 'CORRECT' in content.upper() and 'INCORRECT' not in content.upper()
            
            return is_correct
            
        except Exception as e:
            print(f"Gemini verification failed: {str(e)}")
            return False
    
    def _verify_answer(self, problem: str, solution: str, answer: str) -> bool:
        """
        Verify answers using both GPT and Gemini; return True only if both agree it is correct
        For efficiency: if the first verifier (GPT) returns incorrect, return early without calling Gemini.
        
        Args:
            problem: math problem
            solution: solution process
            answer: final answer
            
        Returns:
            True if both models agree the answer is correct, False if at least one disagrees
        """
        gpt_result = self._verify_with_gpt(problem, solution, answer)
        if not gpt_result:
            if self.display:
                print(f"\n[Verification result] GPT: {gpt_result} (failed, skipping Gemini), final: False")
            return False

        gemini_result = self._verify_with_gemini(problem, solution, answer)
        if self.display:
            print(f"\n[Verification result] GPT: {gpt_result}, Gemini: {gemini_result}, final: {gpt_result and gemini_result}")

        return bool(gemini_result)

    def _is_numerical_text(self, text: str, threshold: float = 0.5) -> bool:
        """
        Check whether the proportion of numeric characters in text exceeds a threshold
        
        Args:
            text: text to check
            threshold: numeric character proportion threshold (default 0.5, i.e. 50%)
            
        Returns:
            True if the proportion of numeric characters >= threshold
        """
        if not text or not isinstance(text, str):
            return False
        
        # Remove spaces to count valid characters
        text_no_space = text.replace(' ', '').replace('\n', '').replace('\t', '')
        if len(text_no_space) == 0:
            return False
        
        # Count digits, decimal points, commas, minus signs, and other numeric characters
        numerical_chars = re.findall(r'[\d.,\-+/()%$]', text_no_space)
        numerical_ratio = len(numerical_chars) / len(text_no_space)
        
        return numerical_ratio >= threshold

    def _contains_modifiable_numbers(self, text: str) -> bool:
        """
        Check whether the text contains at least one modifiable numeric value
        
        Args:
            text: text to check
            
        Returns:
            True if the text contains at least one digit
        """
        if not text or not isinstance(text, str):
            return False
        
        # Check for at least one digit
        return bool(re.search(r'\d', text))

    def _prepare_mmlu_dataset(self, dataset_name: str, category: str, output_dir: str, sample_size: int = 250) -> List[Dict]:
        """
        Prepare data from the MMLU-Pro dataset: filter for numeric answers and create two non-overlapping datasets
        
        Args:
            dataset_name: dataset name, e.g. "TIGER-Lab/MMLU-Pro"
            category: data category, e.g. "math", "physics", etc.
            output_dir: output directory
            sample_size: number of samples per dataset (default 250, max 250)
            
        Returns:
            dataset_c for cleaning (dataset_u is not returned)
        """
        print(f"Loading MMLU-Pro dataset: {dataset_name}, category: {category}")
        
        # Check if preprocessed dataset already exists
        preprocess_dir = os.path.dirname(output_dir) if output_dir.endswith(('cleaned', 'processed')) else output_dir
        path_c = os.path.join(preprocess_dir, "dataset_c.jsonl")
        path_u = os.path.join(preprocess_dir, "dataset_u.jsonl")
        
        if os.path.exists(path_c) and os.path.exists(path_u):
            print(f"Found existing preprocessed datasets:")
            print(f"  - {path_c}")
            print(f"  - {path_u}")
            print(f"Skipping data extraction, loading dataset C directly...")
            
            dataset_c = []
            with open(path_c, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        dataset_c.append(json.loads(line))
            
            print(f"Loaded {len(dataset_c)} items from dataset C")
            return dataset_c
        
        try:
            # Loading dataset
            dataset = load_dataset(dataset_name, split='test')
            
            # Filter by specified category
            filtered_data = []
            for i, item in enumerate(dataset):
                if item.get('category') != category:
                    continue
                
                # Get correct answer index and options
                answer_index = item.get('answer_index', item.get('answer'))
                options = item.get('options', [])
                
                if answer_index is None or not options:
                    continue
                
                # Handle answer index (may be letter A/B/C/D or number 0/1/2/3)
                if isinstance(answer_index, str):
                    # Convert letter index to number
                    answer_idx = ord(answer_index.upper()) - ord('A')
                else:
                    answer_idx = int(answer_index)
                
                # Validate index
                if answer_idx < 0 or answer_idx >= len(options):
                    continue
                
                # Get content of the correct option
                correct_option = options[answer_idx]
                
                # Check if the correct answer is at least 50% numeric
                if not self._is_numerical_text(correct_option, threshold=0.5):
                    continue
                
                # Get question text
                question_text = item.get('question', '')
                
                # Check if the question contains modifiable numeric values
                if not self._contains_modifiable_numbers(question_text):
                    continue
                
                # Build data item
                filtered_item = {
                    "original_index": i,
                    "question_id": item.get('question_id', str(i)),
                    "problem": question_text,
                    "solution": "",  # MMLU datasets typically do not include solution steps
                    "answer": correct_option,  # Use the content of the correct option directly
                    "category": category
                }
                filtered_data.append(filtered_item)
        
        except Exception as e:
            print(f"Error loading MMLU-Pro dataset: {e}")
            return []
        
        print(f"Total matching items (category='{category}', numerical answer >= 50%, problem contains numbers): {len(filtered_data)}")
        
        if len(filtered_data) == 0:
            print("No matching items found!")
            return []
        
        # Randomly sample two non-overlapping datasets
        if len(filtered_data) >= sample_size * 2:
            sampled_items = random.sample(filtered_data, sample_size * 2)
            dataset_c = sampled_items[:sample_size]
            dataset_u = sampled_items[sample_size:]
            print(f"Sampled two disjoint datasets of size {sample_size} each")
        else:
            print(f"Not enough items for two disjoint datasets of size {sample_size}. Using available items split in half.")
            random.shuffle(filtered_data)
            split_idx = len(filtered_data) // 2
            dataset_c = filtered_data[:split_idx]
            dataset_u = filtered_data[split_idx:]
            print(f"Dataset C: {len(dataset_c)} items, Dataset U: {len(dataset_u)} items")
        
        # Sort by original index for consistency
        dataset_c.sort(key=lambda x: x['original_index'])
        dataset_u.sort(key=lambda x: x['original_index'])
        
        # Save preprocessed dataset to parent directory (not the cleaned subdirectory)
        # e.g., if output_dir is /path/to/mmlu/business/cleaned
        # then preprocessed data is saved to /path/to/mmlu/business
        preprocess_dir = os.path.dirname(output_dir) if output_dir.endswith(('cleaned', 'processed')) else output_dir
        os.makedirs(preprocess_dir, exist_ok=True)
        
        path_c = os.path.join(preprocess_dir, "dataset_c.jsonl")
        print(f"Saving dataset C to {path_c}")
        with open(path_c, 'w', encoding='utf-8') as f:
            for item in dataset_c:
                item['dataset_subset'] = 'c'
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        path_u = os.path.join(preprocess_dir, "dataset_u.jsonl")
        print(f"Saving dataset U to {path_u}")
        with open(path_u, 'w', encoding='utf-8') as f:
            for item in dataset_u:
                item['dataset_subset'] = 'u'
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Returning combined dataset for processing (Dataset C only will be cleaned)")
        # Return only dataset_c for subsequent cleaning
        return dataset_c

    def _prepare_finance_dataset_from_csv(self, csv_path: str, output_dir: str, sample_size: int = 250) -> List[Dict]:
        """
        Preprocessing for finance CSV: filter task='calcu' & empty figure, sample TWO disjoint datasets.
        Each dataset size is capped at 250 (default).
        """
        # Check if preprocessed dataset already exists
        preprocess_dir = os.path.dirname(output_dir) if output_dir.endswith(('cleaned', 'processed')) else output_dir
        path_c = os.path.join(preprocess_dir, "dataset_c.jsonl")
        path_u = os.path.join(preprocess_dir, "dataset_u.jsonl")
        
        if os.path.exists(path_c) and os.path.exists(path_u):
            print(f"Found existing preprocessed datasets:")
            print(f"  - {path_c}")
            print(f"  - {path_u}")
            print(f"Skipping data extraction, loading datasets directly...")
            
            dataset_c = []
            dataset_u = []
            
            with open(path_c, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        dataset_c.append(json.loads(line))
            
            with open(path_u, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        dataset_u.append(json.loads(line))
            
            print(f"Loaded {len(dataset_c)} items from dataset C, {len(dataset_u)} items from dataset U")
            return dataset_c + dataset_u
        
        filtered_data = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    task = row.get('task', '').strip().lower()
                    figure = row.get('figure', '').strip()
                    
                    if task == 'calcu' and not figure:
                        item = {
                            "original_index": i,
                            "id": row.get('id', str(i)),
                            "problem": row.get('question', ''),
                            "solution": "",
                            "answer": row.get('ground_truth', '')
                        }
                        filtered_data.append(item)
        except Exception as e:
            print(f"Error reading CSV {csv_path}: {e}")
            return []
            
        print(f"Total matching items (task='calcu', figure=''): {len(filtered_data)}")
        
        # Sample two non-overlapping datasets
        if len(filtered_data) >= sample_size * 2:
            sampled_items = random.sample(filtered_data, sample_size * 2)
            dataset_c = sampled_items[:sample_size]
            dataset_u = sampled_items[sample_size:]
            print(f"Sampled two disjoint datasets of size {sample_size}")
        else:
            print(f"Not enough items for two disjoint datasets of size {sample_size}. Using available items split in half.")
            random.shuffle(filtered_data)
            split_idx = len(filtered_data) // 2
            dataset_c = filtered_data[:split_idx]
            dataset_u = filtered_data[split_idx:]
            
        # Sort by original index for consistency
        dataset_c.sort(key=lambda x: x['original_index'])
        dataset_u.sort(key=lambda x: x['original_index'])
        
        # Save intermediate files to parent directory
        preprocess_dir = os.path.dirname(output_dir) if output_dir.endswith(('cleaned', 'processed')) else output_dir
        os.makedirs(preprocess_dir, exist_ok=True)
        
        path_c = os.path.join(preprocess_dir, "dataset_c.jsonl")
        print(f"Saving dataset C to {path_c}")
        with open(path_c, 'w', encoding='utf-8') as f:
            for item in dataset_c:
                item['dataset_subset'] = 'c'
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        path_u = os.path.join(preprocess_dir, "dataset_u.jsonl")
        print(f"Saving dataset U to {path_u}")
        with open(path_u, 'w', encoding='utf-8') as f:
            for item in dataset_u:
                item['dataset_subset'] = 'u'
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        return dataset_c + dataset_u

    def paraphrase_dataset(
        self, 
        dataset_path: str,
        mode: str = "paraphrased",
        max_samples: Optional[int] = None,
        save_interval: int = 10,
        index_range: Optional[str] = None,
        category: Optional[str] = None
    ):
        """
        Rewrite the entire dataset by modifying numbers and recomputing solutions and answers
        
        Args:
            dataset_path: path to dataset JSON file or HuggingFace dataset name (for MMLU mode)
            mode: modification mode — "paraphrased", "original", "both", "clean", "finance", or "mmlu"
            max_samples: maximum samples to process, None means all
            save_interval: how often (in samples) to save progress
            index_range: specify original_index range, format: "start-end" or "start,end" (e.g. "10-20" or "10,15,20")
            category: category for MMLU mode (e.g. "math", "physics")
        """
        print(f"Loading dataset: {dataset_path} (mode: {mode})")
        
        # Handle MMLU mode
        if mode == "mmlu":
            # Check if it is a local JSONL file
            if os.path.exists(dataset_path) and dataset_path.endswith('.jsonl'):
                print(f"Detected local JSONL file, loading directly: {dataset_path}")
                dataset = []
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            dataset.append(json.loads(line))
                print(f"Loaded {len(dataset)} records")
            else:
                # Prepare data from HuggingFace dataset
                if not category:
                    print("Error: --category is required when loading MMLU dataset from HuggingFace")
                    return
                dataset = self._prepare_mmlu_dataset(dataset_path, category, self.output_dir)
                if not dataset:
                    print("Error: Failed to load MMLU dataset or no matching data found")
                    return
        # Handle finance mode specially for CSV input
        elif mode == "finance" and dataset_path.endswith('.csv'):
            dataset = self._prepare_finance_dataset_from_csv(dataset_path, self.output_dir)

        # Determine which fields to use based on mode
        if mode == "paraphrased":
            problem_field = "paraphrased_problem"
            solution_field = "paraphrased_solution"
            answer_field = "answer"
        elif mode == "original":
            problem_field = "original_problem"
            solution_field = "original_solution"
            answer_field = "original_answer"
        elif mode == "clean":
            problem_field = "question"
            solution_field = "solution"
            answer_field = "answer"
        elif mode == "finance":
            problem_field = "problem"
            solution_field = "solution"
            answer_field = "answer"
            self.system_prompt = getattr(self, "system_prompt_finance", self.system_prompt)
        elif mode == "mmlu":
            problem_field = "problem"
            solution_field = "solution"
            answer_field = "answer"
            self.system_prompt = getattr(self, "system_prompt_mmlu", self.system_prompt)
        elif mode == "both":
            # both mode processes both versions simultaneously
            pass
        else:
            print(f"Error: Unsupported mode '{mode}'. Use 'paraphrased', 'original', 'both', 'clean', 'finance', or 'mmlu'")
            return
        
        # Avoid reloading if already loaded (e.g. from CSV, MMLU)
        if 'dataset' not in locals():
            try:
                # Auto-detect file format or decide based on mode
                if mode == "clean" or mode == "finance" or dataset_path.endswith('.jsonl'):
                    print(f"Loading dataset in JSONL format: {dataset_path}")
                    dataset = []
                    with open(dataset_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                dataset.append(json.loads(line))
                else:
                    with open(dataset_path, 'r', encoding='utf-8') as f:
                        dataset = json.load(f)
            except Exception as e:
                print(f"Failed to load dataset: {str(e)}")
                return
        
        # Ensure all items have original_index
        for i, item in enumerate(dataset):
            if 'original_index' not in item:
                item['original_index'] = i
        
        # Parse the index_range argument
        target_indices = None
        if index_range:
            target_indices = self._parse_index_range(index_range)
            if target_indices is None:
                print(f"Error: invalid index_range format '{index_range}'")
                return
            print(f"Target indices: {target_indices}")
            
            # Filter samples to the specified original_index values
            original_dataset = dataset
            dataset = [item for item in original_dataset if isinstance(item, dict) and item.get('original_index') in target_indices]
            print(f"Filtered to {len(dataset)} samples from {len(original_dataset)}")

        # max_samples is always applied after index_range filtering
        if max_samples is not None:
            if index_range:
                print(f"Note: both index_range and max_samples specified; only the first {max_samples} samples after filtering will be used")
            dataset = dataset[:max_samples]
        
        print(f"Dataset size: {len(dataset)}")
        
        modified_data = []
        failed_indices = []
        skipped_indices = []  # Track skipped samples
        
        # Load existing progress if available
        progress_file = os.path.join(self.output_dir, "progress.json")
        if os.path.exists(progress_file):
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                modified_data = progress.get("modified_data", [])
                failed_indices = progress.get("failed_indices", [])
                skipped_indices = progress.get("skipped_indices", [])  # Load previously skipped indices
                start_idx = len(modified_data) + len(failed_indices) + len(skipped_indices)
                print(f"Resuming from {start_idx}/{len(dataset)}")
        else:
            start_idx = 0

        # If start_idx exceeds dataset length due to max_samples/index_range change, warn and exit
        if start_idx >= len(dataset):
            print(
                f"Current progress already reached/exceeds dataset size ({start_idx}/{len(dataset)}). "
                f"To rerun with new --max-samples/--index-range, delete {progress_file} or use a different --output-dir."
            )
            return
        
        # Process the dataset
        for idx in tqdm(range(start_idx, len(dataset)), desc=f"Modifying dataset (mode: {mode})"):
            item = dataset[idx]
            if not isinstance(item, dict):
                raise ValueError(f"Data format error: sample {idx} is not a dict, actual type={type(item)}")

            # Failed cases must record the original_index from the source data (not the loop idx)
            original_idx = item['original_index']
            
            if mode == "both":
                # Process both original and paraphrased simultaneously; one API call ensures consistent number changes
                result = self.paraphrase_both_versions(
                    item["original_problem"],
                    item["original_solution"],
                    item["original_answer"],
                    item["paraphrased_problem"],
                    item["paraphrased_solution"],
                    item["answer"]
                )
                
                if result:
                    # Create new item containing modification results for both versions
                    modified_item = {
                        "index": idx,
                        "original_index": original_idx,
                        
                        # Original version
                        "original_problem": item["original_problem"],
                        "original_solution": item["original_solution"],
                        "original_answer": item["original_answer"],
                        "modified_original_problem": result["original"]["new_problem"],
                        "modified_original_solution": result["original"]["new_solution"],
                        "modified_original_answer": result["original"]["answer"],
                        
                        # Paraphrased version
                        "paraphrased_problem": item["paraphrased_problem"],
                        "paraphrased_solution": item["paraphrased_solution"],
                        "paraphrased_answer": item["answer"],
                        "modified_paraphrased_problem": result["paraphrased"]["new_problem"],
                        "modified_paraphrased_solution": result["paraphrased"]["new_solution"],
                        "modified_paraphrased_answer": result["paraphrased"]["answer"],
                        
                        "reasoning": result["reasoning"]
                    }
                    modified_data.append(modified_item)
                else:
                    print(f"Index {original_idx} (original_index) modification failed")
                    failed_indices.append(original_idx)
            else:
                # Single mode: process only one version
                problem = item[problem_field]
                solution = item[solution_field]
                answer = item[answer_field]
                
                result = self.paraphrase_single_item(problem, solution, answer)
                
                if result:
                    # Check for SKIP
                    if result.get("skip"):
                        skipped_indices.append(original_idx)
                        print(f"⊘ Skipping sample {idx} (original_index: {original_idx}) - numeric values cannot be reasonably modified")
                    else:
                        # Create new item retaining only necessary fields
                        modified_item = {
                            "index": idx,
                            "original_index": original_idx,
                            "original_problem": problem,
                            "original_solution": solution,
                            "original_answer": answer,
                            "modified_problem": result["new_problem"],
                            "modified_solution": result["new_solution"],
                            "modified_answer": result["answer"],
                            "reasoning": result["reasoning"]
                        }
                        
                        modified_data.append(modified_item)
                        print(f"✓ Successfully modified sample {idx} (original_index: {original_idx})")
                else:
                    print(f"✗ Failed to modify sample {idx} (original_index: {original_idx})")
                    failed_indices.append(original_idx)
            
            # Periodically save progress
            if (idx + 1) % save_interval == 0:
                self._save_progress(modified_data, failed_indices, skipped_indices)
        
        # Final save
        self._save_final_results(modified_data, failed_indices, f"{dataset_path} (mode: {mode})", mode=mode, skipped_indices=skipped_indices)
        
        if mode != "both":
            print(f"\nModification complete!")
            print(f"Successful: {len(modified_data)}/{len(dataset)}")
            print(f"Skipped: {len(skipped_indices)}/{len(dataset)}")
            print(f"Failed: {len(failed_indices)}/{len(dataset)}")
            print(f"Results saved to: {self.output_dir}")
    
    def _save_progress(self, modified_data: List[Dict], failed_indices: List[int], skipped_indices: List[int] = None):
        """Save progress"""
        if skipped_indices is None:
            skipped_indices = []
        progress_file = os.path.join(self.output_dir, "progress.json")
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump({
                "modified_data": modified_data,
                "failed_indices": failed_indices,
                "skipped_indices": skipped_indices
            }, f, ensure_ascii=False, indent=2)
    
    def _save_final_results(
        self, 
        modified_data: List[Dict], 
        failed_indices: List[int],
        dataset_name: str,
        mode: str = "single",
        skipped_indices: List[int] = None
    ):
        """Save final results"""
        if skipped_indices is None:
            skipped_indices = []
        if mode == "both":
            # Both mode: save separately to original_modified and paraphrased_modified directories
            self._save_both_mode_results(modified_data, failed_indices, dataset_name)
        else:
            # Single mode: save to a unified directory
            self._save_single_mode_results(modified_data, failed_indices, dataset_name, skipped_indices)
        
        # Delete progress file
        progress_file = os.path.join(self.output_dir, "progress.json")
        if os.path.exists(progress_file):
            os.remove(progress_file)
    
    def _save_single_mode_results(
        self,
        modified_data: List[Dict],
        failed_indices: List[int],
        dataset_name: str,
        skipped_indices: List[int] = None
    ):
        """Save results for single mode"""
        if skipped_indices is None:
            skipped_indices = []
        
        # Save as JSON
        output_file = os.path.join(self.output_dir, "modified_dataset.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(modified_data, f, ensure_ascii=False, indent=2)
        
        # Save as JSONL
        output_file_jsonl = os.path.join(self.output_dir, "modified_dataset.jsonl")
        with open(output_file_jsonl, 'w', encoding='utf-8') as f:
            for item in modified_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Save failed indices
        if failed_indices:
            failed_file = os.path.join(self.output_dir, "failed_indices.json")
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(failed_indices, f, indent=2)
        
        # Save skipped indices
        if skipped_indices:
            skipped_file = os.path.join(self.output_dir, "skipped_indices.json")
            with open(skipped_file, 'w', encoding='utf-8') as f:
                json.dump(skipped_indices, f, ensure_ascii=False, indent=2)
        
        # Save statistics
        total_samples = len(modified_data) + len(failed_indices) + len(skipped_indices)
        stats = {
            "dataset_name": dataset_name,
            "total_samples": total_samples,
            "successful": len(modified_data),
            "skipped": len(skipped_indices),
            "failed": len(failed_indices),
            "success_rate": len(modified_data) / total_samples * 100 if total_samples > 0 else 0
        }
        stats_file = os.path.join(self.output_dir, "statistics.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # Print statistics summary
        print("\n" + "="*60)
        print("Dataset modification complete")
        print("="*60)
        print(f"Successfully modified: {len(modified_data)}/{total_samples}")
        print(f"Skipped: {len(skipped_indices)}/{total_samples}")
        print(f"Failed: {len(failed_indices)}/{total_samples}")
        print(f"Success rate: {stats['success_rate']:.2f}%")
        print("="*60)
    
    def _save_both_mode_results(
        self,
        modified_data: List[Dict],
        failed_indices: List[int],
        dataset_name: str
    ):
        """Save both-mode results to two separate directories"""
        # Create two subdirectories
        original_dir = os.path.join(self.output_dir, "original_modified")
        paraphrased_dir = os.path.join(self.output_dir, "paraphrased_modified")
        os.makedirs(original_dir, exist_ok=True)
        os.makedirs(paraphrased_dir, exist_ok=True)
        
        # Separate original and paraphrased data
        original_data = []
        paraphrased_data = []
        
        for item in modified_data:
            # Original version
            original_item = {
                "index": item["index"],
                "original_problem": item["original_problem"],
                "original_solution": item["original_solution"],
                "original_answer": item["original_answer"],
                "modified_problem": item["modified_original_problem"],
                "modified_solution": item["modified_original_solution"],
                "modified_answer": item["modified_original_answer"],
                "reasoning": item["reasoning"]
            }
            original_data.append(original_item)
            
            # Paraphrased version
            paraphrased_item = {
                "index": item["index"],
                "original_problem": item["paraphrased_problem"],
                "original_solution": item["paraphrased_solution"],
                "original_answer": item["paraphrased_answer"],
                "modified_problem": item["modified_paraphrased_problem"],
                "modified_solution": item["modified_paraphrased_solution"],
                "modified_answer": item["modified_paraphrased_answer"],
                "reasoning": item["reasoning"]
            }
            paraphrased_data.append(paraphrased_item)
        
        # Save Original version
        with open(os.path.join(original_dir, "modified_dataset.json"), 'w', encoding='utf-8') as f:
            json.dump(original_data, f, ensure_ascii=False, indent=2)
        
        with open(os.path.join(original_dir, "modified_dataset.jsonl"), 'w', encoding='utf-8') as f:
            for item in original_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Save Paraphrased version
        with open(os.path.join(paraphrased_dir, "modified_dataset.json"), 'w', encoding='utf-8') as f:
            json.dump(paraphrased_data, f, ensure_ascii=False, indent=2)
        
        with open(os.path.join(paraphrased_dir, "modified_dataset.jsonl"), 'w', encoding='utf-8') as f:
            for item in paraphrased_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Save failed indices (in main directory)
        if failed_indices:
            failed_file = os.path.join(self.output_dir, "failed_indices.json")
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(failed_indices, f, indent=2)
        
        # Save statistics
        stats = {
            "dataset_name": dataset_name,
            "total_samples": len(modified_data) + len(failed_indices),
            "successful": len(modified_data),
            "failed": len(failed_indices),
            "success_rate": len(modified_data) / (len(modified_data) + len(failed_indices)) * 100 if (len(modified_data) + len(failed_indices)) > 0 else 0,
            "note": "Data split into original_modified/ and paraphrased_modified/ directories"
        }
        
        # Save statistics in main directory
        stats_file = os.path.join(self.output_dir, "statistics.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # Also save statistics in each subdirectory
        for subdir, data in [(original_dir, original_data), (paraphrased_dir, paraphrased_data)]:
            sub_stats = {
                "dataset_name": dataset_name,
                "total_samples": len(data),
                "note": "This is one version from both mode processing"
            }
            with open(os.path.join(subdir, "statistics.json"), 'w', encoding='utf-8') as f:
                json.dump(sub_stats, f, ensure_ascii=False, indent=2)
        
        # Print statistics summary
        print("\n" + "="*60)
        print("Dataset modification complete (Both mode)")
        print("="*60)
        print(f"Successfully modified: {len(modified_data)}/{len(modified_data) + len(failed_indices)}")
        print(f"Failed: {len(failed_indices)}/{len(modified_data) + len(failed_indices)}")
        print(f"Success rate: {stats['success_rate']:.2f}%")
        print(f"\nResults saved to:")
        print(f"  - Original version: {original_dir}")
        print(f"  - Paraphrased version: {paraphrased_dir}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Use GPT to modify numbers in a math dataset")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset JSON file, local JSONL file, or HuggingFace dataset name (for MMLU mode, e.g. 'TIGER-Lab/MMLU-Pro'). In MMLU mode, if a local JSONL file is given it is loaded directly; otherwise the dataset is downloaded and preprocessed from HuggingFace"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["paraphrased", "original", "both", "clean", "finance", "mmlu"],
        default="paraphrased",
        help="Modification mode: 'paraphrased' modifies the paraphrased version, 'original' modifies the original version, 'both' modifies both simultaneously using the same number mapping, 'clean' modifies question-answer-solution format data, 'finance' modifies finance data, 'mmlu' processes MMLU-Pro datasets"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Category to use when downloading data from HuggingFace in MMLU mode (e.g. 'math', 'physics', 'chemistry', 'business'). Not required when loading a local JSONL file directly"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (if not provided, read from OPENAI_API_KEY env variable)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="o4-mini",
        help="GPT model to use (e.g. gpt-4, gpt-3.5-turbo)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: ./modified_{mode}_math_500)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to process (None = all)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save progress every N samples"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the model's full output"
    )
    parser.add_argument(
        "--index-range",
        type=str,
        default=None,
        help="Specify original_index range: '10-20' (range), '10,15,20' (comma-separated), '10' (single), or '/path/to/failed_indices.json' (JSON file)"
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Gemini API key (if not provided, read from GEMINI_API_KEY env variable)"
    )
    parser.add_argument(
        "--verify-model-gpt",
        type=str,
        default="gpt-4o-mini",
        help="GPT model for verification (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--verify-model-gemini",
        type=str,
        default="gemini-2.0-flash-exp",
        help="Gemini model for verification (default: gemini-2.0-flash-exp)"
    )
    parser.add_argument(
        "--max-verify-retries",
        type=int,
        default=3,
        help="Maximum retries after verification failure (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = f"./modified_{args.mode}_math_500"
    
    # Retrieve API keys
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: please provide an OpenAI API key (via --api-key or the OPENAI_API_KEY environment variable)")
        return
    
    gemini_api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        print("Error: please provide a Gemini API key (via --gemini-api-key or the GEMINI_API_KEY environment variable)")
        return
    
    # Create modifier and run
    paraphraser = DatasetParaphraser(
        api_key=api_key,
        model=args.model,
        output_dir=args.output_dir,
        display=args.display,
        gemini_api_key=gemini_api_key,
        verify_model_gpt=args.verify_model_gpt,
        verify_model_gemini=args.verify_model_gemini,
        max_verify_retries=args.max_verify_retries
    )
    
    paraphraser.paraphrase_dataset(
        dataset_path=args.dataset_path,
        mode=args.mode,
        max_samples=args.max_samples,
        save_interval=args.save_interval,
        index_range=args.index_range,
        category=args.category
    )


if __name__ == "__main__":
    main()
