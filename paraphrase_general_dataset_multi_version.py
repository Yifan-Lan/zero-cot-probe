"""
Use the GPT API to paraphrase a multi-domain dataset. Supports loading from Hugging Face
or from a local JSONL file. Generates N paraphrased versions per sample and automatically
retries failed samples. Supports index-range arguments for parallel processing.
Supported domains: math, finance, physics, chemistry, business.

Usage examples:

# Example 1 — load from Hugging Face (3 versions per sample):
python paraphrase_general_dataset_multi_version.py \
    --dataset HuggingFaceH4/MATH-500 \
    --domain math \
    --max-samples 10 \
    --num-versions 3 \
    --max-retries-per-sample 5 \
    --output-dir ./paraphrased_output

# Example 2 — load a local finance dataset (6 versions per sample):
python paraphrase_general_dataset_multi_version.py \
    --local-file <path/to/modified_dataset.jsonl> \
    --problem-field original_problem \
    --solution-field original_solution \
    --answer-field original_answer \
    --domain finance \
    --model gpt-4o \
    --num-versions 6 \
    --max-retries-per-sample 3 \
    --save-interval 20 \
    --output-dir <path/to/output_dir>

# Example 3 — parallel processing (split by index range):
python paraphrase_general_dataset_multi_version.py \
    --local-file <path/to/dataset.jsonl> \
    --domain math \
    --model o4-mini \
    --num-versions 2 \
    --start-index 0 \
    --end-index 100 \
    --output-dir <path/to/output_dir>

# Example 4 — retry specific failed indices:
python paraphrase_general_dataset_multi_version.py \
    --local-file data.jsonl \
    --domain business \
    --indices 5,12,23,45,67,89 \
    --output-dir ./output

# Set your API key before running:
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
"""

import os
import json
import time
from typing import Dict, List, Optional
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import argparse
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Import answer equivalence check function
from math_grade import grade_answer


class DatasetParaphraser:
    def __init__(self, api_key: str, model: str = "gpt-4o", output_dir: str = "./paraphrased_data", num_versions: int = 1, domain: str = "math"):
        """
        Initialize the dataset paraphraser
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use
            output_dir: output directory
            num_versions: Number of paraphrase versions to generate per sample
            domain: Dataset domain (math, finance, physics, chemistry, business)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.output_dir = output_dir
        self.num_versions = num_versions
        self.domain = domain.lower()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1

        # Define domain-specific system prompts
        domain_descriptions = {
            "math": "mathematics",
            "finance": "finance and economics",
            "physics": "physics",
            "chemistry": "chemistry",
            "business": "business and management"
        }
        
        domain_name = domain_descriptions.get(self.domain, "general academic")
        
        self.system_prompt = rf"""You are an expert {domain_name} editor and data augmentation assistant. Your goal is to create a training dataset that teaches a model to solve {domain_name} problems robustly, regardless of how the question is phrased.

**Task:**
1. **Paraphrase the "Problem"** to be linguistically distinct and diverse.
2. **Rewrite the "Solution"** to be the most standard, canonical, and rigorous derivation possible.

**Detailed Instructions:**

1. The Problem: Aggressive Variation & Entity Swapping
   - **Textual Rewriting:** Rephrase the narrative. Vary sentence length, syntactic structure, and vocabulary. Use synonyms and different phrasing styles (e.g., change from imperative "Find X" to interrogative "What is X?").
   - **Entity Substitution (Crucial):** Where applicable, **change the non-numerical entities** (context) while keeping the logic identical.
     - Example (Math): Change "Alice buys 5 apples" to "A machine processes 5 units" or "A particle moves 5 meters".
     - Example (Finance): Change "Company A" to "Corporation B" or change "stocks" to "securities".
     - Example (Physics): Change "particle A" to "object B" or change experimental setup details.
     - **Constraint:** Do NOT change any numerical values, constants, fundamental relationships, or formulas. The answer and logic must remain exactly the same.
   - **Formula/Equation Fidelity:** In the paraphrased problem, every LaTeX math segment from the original problem must be copied verbatim (character-for-character), including delimiters, spacing, and internal formatting. Do NOT introduce new math segments, and do NOT move content into or out of math mode. (i.e., keep exactly the same parts inside $...$, \(...\), \[...\] as in the original problem.)

2. The Solution: Standardization & Rigor
   - **Goal:** Unlike the problem, **do NOT** try to make the solution "linguistically distinct" or "unique." Instead, rewrite it to sound like a **standard, high-quality textbook**.
   - **Style:** Use standard academic English appropriate for {domain_name}. Avoid colloquialisms.
   - **Structure:** Ensure the solution is **step-by-step** as the original solution.
   - **Consistency:** Even though you changed entities in the Problem (e.g., Company A -> Corporation B), you must update the Solution to reflect these new entities so the logic holds.

3. Constraints & Safety
   - **Equivalence:** The final result must be strictly identical to the original.
   - **Formatting:** Keep the exact LaTeX formatting for equations and formulas.

**Output Format:**
Reasoning: [Brief plan: 1. How to rephrase/swap entities in the problem. 2. How to standardize the solution style.]
New Problem: [The aggressively paraphrased problem with entity swaps]
New Solution: [The canonical, rigorous, step-by-step solution matching the new context]
Answer: [Must be equivalent to the original answer]
"""

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def paraphrase_single_item(self, problem: str, solution: str, answer: str, version_num: int = 1, max_retries: int = 3) -> Optional[Dict]:
        """
        Use GPT to paraphrase a single question and answer and verify answer equivalence
        
        Args:
            problem: original problem
            solution: original solution process
            answer: original answer (kept unchanged)
            version_num: current version number (used in progress messages)
            max_retries: maximum number of retries
            
        Returns:
            dict with rewriting results, or None on failure
        """
        user_message = f"Problem: {problem}\n\nSolution: {solution}\n\nAnswer: {answer}"
        
        # If generating multiple versions, add version info to the prompt to increase diversity
        if self.num_versions > 1:
            user_message += f"\n\nNote: This is version {version_num} of {self.num_versions}. Please make this version significantly different from other versions in style, structure, and wording while maintaining correctness."
        
        for attempt in range(max_retries):
            try:
                # o1/o3/o4/gpt-5 series models use max_completion_tokens instead of max_tokens
                if (self.model.startswith('o1') or self.model.startswith('o3') or 
                    self.model.startswith('o4') or self.model.startswith('gpt-5')):
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "user", "content": self.system_prompt + "\n\n" + user_message}
                        ],
                        max_completion_tokens=4096
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        temperature=0.7,
                        max_tokens=4096
                    )
                
                content = response.choices[0].message.content
                
                # Parse response
                result = self._parse_response(content)
                if result:
                    # Verify answer equivalence
                    paraphrased_answer = result['answer']
                    is_equivalent = grade_answer(paraphrased_answer, answer)
                    
                    if is_equivalent:
                        print(f"✓ Answer equivalence verified (version {version_num}, attempt {attempt + 1}/{max_retries})")
                        return result
                    else:
                        print(f"✗ Answer mismatch, retrying (version {version_num}, attempt {attempt + 1}/{max_retries})")
                        print(f"   Original answer: {answer}")
                        print(f"   Paraphrased answer: {paraphrased_answer}")
                        if attempt < max_retries - 1:
                            # Update prompt to explicitly require the answer to remain unchanged
                            user_message = (
                                f"Problem: {problem}\n\nSolution: {solution}\n\nAnswer: {answer}\n\n"
                                f"IMPORTANT: The 'Answer' field MUST remain EXACTLY as '{answer}'. "
                                f"Do not modify, paraphrase, or change the answer in any way."
                            )
                            if self.num_versions > 1:
                                user_message += f"\n\nNote: This is version {version_num} of {self.num_versions}. Please make this version significantly different from other versions."
                else:
                    print(f"Parsing failed, attempt {attempt + 1}/{max_retries}")
                    
            except Exception as e:
                print(f"API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # exponential backoff
                    
        return None
    
    def _parse_response(self, content: str) -> Optional[Dict]:
        """
        Parse GPT response, extracting reasoning, new_problem, new_solution, and answer
        
        Args:
            content: GPT response content
            
        Returns:
            parsed dict, or None on failure
        """
        try:
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
            
            return {
                "reasoning": reasoning,
                "new_problem": new_problem,
                "new_solution": new_solution,
                "answer": answer
            }
        except Exception as e:
            print(f"Parse error: {str(e)}")
            return None
    
    def calculate_bleu(self, reference: str, hypothesis: str) -> float:
        """
        Compute BLEU score
        
        Args:
            reference: reference text (original text)
            hypothesis: hypothesis text (paraphrased text)
            
        Returns:
            BLEU score (0-1)
        """
        reference_tokens = reference.split()
        hypothesis_tokens = hypothesis.split()
        
        if len(hypothesis_tokens) == 0:
            return 0.0
        
        # Use smoothing to avoid zero scores
        score = sentence_bleu([reference_tokens], hypothesis_tokens, 
                             smoothing_function=self.smoothing)
        return score
    
    def calculate_rouge_l(self, reference: str, hypothesis: str) -> float:
        """
        Compute ROUGE-L score
        
        Args:
            reference: reference text (original text)
            hypothesis: hypothesis text (paraphrased text)
            
        Returns:
            ROUGE-L F1 score (0-1)
        """
        scores = self.rouge_scorer.score(reference, hypothesis)
        return scores['rougeL'].fmeasure
    
    def verify_answer_match(self, original_answer: str, paraphrased_answer: str) -> bool:
        """
        Verify whether two answers are identical (ignoring whitespace and case)
        
        Args:
            original_answer: original answer
            paraphrased_answer: paraphrased answer
            
        Returns:
            whether they match
        """
        # Normalize answers: strip whitespace, convert to lowercase
        orig = original_answer.strip().lower()
        para = paraphrased_answer.strip().lower()
        return orig == para
    
    def evaluate_paraphrase_quality(self, original_problem: str, original_solution: str,
                                    paraphrased_problem: str, paraphrased_solution: str,
                                    original_answer: str, paraphrased_answer: str) -> Dict:
        """
        Evaluate paraphrase quality
        
        Args:
            original_problem: original problem
            original_solution: original solution
            paraphrased_problem: paraphrased question
            paraphrased_solution: paraphrased solution
            original_answer: original answer
            paraphrased_answer: paraphrased answer
            
        Returns:
            dictionary containing evaluation metrics
        """
        # Use grade_answer to verify mathematical equivalence
        answer_equivalent = grade_answer(paraphrased_answer, original_answer)
        
        metrics = {
            'problem_bleu': self.calculate_bleu(original_problem, paraphrased_problem),
            'problem_rouge_l': self.calculate_rouge_l(original_problem, paraphrased_problem),
            'solution_bleu': self.calculate_bleu(original_solution, paraphrased_solution),
            'solution_rouge_l': self.calculate_rouge_l(original_solution, paraphrased_solution),
            'answer_match': self.verify_answer_match(original_answer, paraphrased_answer),  # Simple string matching
            'answer_equivalent': answer_equivalent,  # Mathematical equivalence verification
            'original_answer': original_answer,
            'paraphrased_answer': paraphrased_answer
        }
        return metrics
    
    def load_local_jsonl(self, file_path: str) -> List[Dict]:
        """
        Load dataset from a local JSONL file
        
        Args:
            file_path: path to the JSONL file
            
        Returns:
            list of data records
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def paraphrase_dataset(
        self, 
        dataset_name: str = "HuggingFaceH4/MATH-500",
        split: str = "test",
        problem_field: str = "problem",
        solution_field: str = "solution",
        answer_field: str = "answer",
        max_samples: Optional[int] = None,
        save_interval: int = 10,
        local_file: Optional[str] = None,
        max_retries_per_sample: int = 3,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        indices: Optional[List[int]] = None
    ):
        """
        Paraphrase the entire dataset
        
        Args:
            dataset_name: HuggingFace dataset name (used when local_file is None)
            split: dataset split (train/test/validation)
            problem_field: field name for questions
            solution_field: field name for solutions
            answer_field: field name for answers
            max_samples: maximum samples to process, None means all
            save_interval: how often (in samples) to save progress
            local_file: path to local JSONL file; if provided, load locally instead of from HuggingFace
            max_retries_per_sample: maximum retry attempts per sample
            start_index: start index (inclusive); None means start from the beginning (mutually exclusive with indices)
            end_index: end index (exclusive); None means process to the end (mutually exclusive with indices)
            indices: specific list of indices to process (mutually exclusive with start_index/end_index)
        """
        if local_file:
            print(f"Loading dataset from local file: {local_file}")
            try:
                dataset = self.load_local_jsonl(local_file)
            except Exception as e:
                print(f"Failed to load local file: {str(e)}")
                return
        else:
            print(f"Loading dataset: {dataset_name} (split: {split})")
            try:
                dataset = load_dataset(dataset_name, split=split)
            except Exception as e:
                print(f"Failed to load dataset: {str(e)}")
                return
        
        if max_samples:
            if isinstance(dataset, list):
                dataset = dataset[:max_samples]
            else:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        # Determine the list of indices to process
        if indices is not None:
            # Use the specified index list
            if start_index is not None or end_index is not None:
                print("Warning: Both --indices and --start-index/--end-index specified; --start-index/--end-index will be ignored")
            
            # Validate and filter indices
            indices_to_process = []
            for idx in indices:
                if 0 <= idx < len(dataset):
                    indices_to_process.append(idx)
                else:
                    print(f"Warning: Index {idx} is out of dataset range [0, {len(dataset)-1}] and will be skipped")
            
            if not indices_to_process:
                print("Error: No valid indices to process")
                return
            
            indices_to_process.sort()  # Sort for sequential processing
            print(f"Dataset size: {len(dataset)}")
            print(f"Processing {len(indices_to_process)} specified indices: {indices_to_process[:10]}{'...' if len(indices_to_process) > 10 else ''}")
            
        else:
            # Use index range
            actual_start_index = start_index if start_index is not None else 0
            actual_end_index = end_index if end_index is not None else len(dataset)
            
            # Validate index range
            if actual_start_index < 0:
                actual_start_index = 0
            if actual_end_index > len(dataset):
                actual_end_index = len(dataset)
            if actual_start_index >= actual_end_index:
                print(f"Error: Start index ({actual_start_index}) must be less than end index ({actual_end_index})")
                return
            
            indices_to_process = list(range(actual_start_index, actual_end_index))
            print(f"Dataset size: {len(dataset)}")
            print(f"Processing range: index {actual_start_index} to {actual_end_index-1} (total {actual_end_index - actual_start_index} samples)")
        
        paraphrased_data = []
        failed_indices = []
        
        # Load existing progress if available
        progress_file = os.path.join(self.output_dir, "progress.json")
        if os.path.exists(progress_file):
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                paraphrased_data = progress.get("paraphrased_data", [])
                failed_indices = progress.get("failed_indices", [])
                # Compute already-processed indices
                processed_indices = set()
                for item in paraphrased_data:
                    processed_indices.add(item['original_index'])
                processed_indices.update(failed_indices)
                
                # Remove already-processed indices from the to-do list
                remaining_indices = [idx for idx in indices_to_process if idx not in processed_indices]
                
                if remaining_indices:
                    print(f"Resuming from checkpoint: {len(processed_indices)} indices done, {len(remaining_indices)} remaining")
                    indices_to_process = remaining_indices
                else:
                    print(f"All indices have already been processed")
                    return
        
        # Process the dataset
        for idx in tqdm(indices_to_process, desc="Paraphrasing dataset"):
            item = dataset[idx]
            
            # Support two data formats:
            # 1. Fields directly in item: item[problem_field]
            # 2. Nested under full_data: item['full_data'][problem_field]
            if 'full_data' in item and isinstance(item['full_data'], dict):
                # Extract from full_data
                problem = item['full_data'].get(problem_field, item.get(problem_field, ""))
                solution = item['full_data'].get(solution_field, item.get(solution_field, ""))
                answer = item['full_data'].get(answer_field, item.get(answer_field, ""))
            else:
                # Extract directly from item
                problem = item.get(problem_field, "")
                solution = item.get(solution_field, "")
                answer = item.get(answer_field, "")
            
            # Generate multiple versions per sample, with retry support
            sample_success = False
            for retry_attempt in range(max_retries_per_sample):
                if retry_attempt > 0:
                    print(f"\n⚠️  Retrying index {idx} (attempt {retry_attempt}/{max_retries_per_sample-1})")
                
                versions_for_item = []
                all_versions_success = True
                
                for version_num in range(1, self.num_versions + 1):
                    print(f"\nProcessing index {idx}, version {version_num}/{self.num_versions}")
                    result = self.paraphrase_single_item(problem, solution, answer, version_num=version_num)
                    
                    if result:
                        # Evaluate paraphrase quality
                        evaluation_metrics = self.evaluate_paraphrase_quality(
                            problem, solution,
                            result["new_problem"], result["new_solution"],
                            answer, result["answer"]
                        )
                        
                        paraphrased_item = {
                            "original_index": idx,
                            "version_number": version_num,
                            "original_problem": problem,
                            "original_solution": solution,
                            "original_answer": answer,
                            "reasoning": result["reasoning"],
                            "paraphrased_problem": result["new_problem"],
                            "paraphrased_solution": result["new_solution"],
                            "answer": result["answer"],  # Answer kept unchanged
                            "evaluation": evaluation_metrics
                        }
                        
                        # Preserve other fields from the original dataset
                        for key, value in item.items():
                            if key not in [problem_field, solution_field, answer_field]:
                                paraphrased_item[f"original_{key}"] = value
                        
                        versions_for_item.append(paraphrased_item)
                    else:
                        print(f"Index {idx}, version {version_num} paraphrase failed")
                        all_versions_success = False
                        break  # Stop generating further versions if one fails
                
                # Only add to results when all versions succeed
                if all_versions_success and len(versions_for_item) == self.num_versions:
                    paraphrased_data.extend(versions_for_item)
                    print(f"✓ All {self.num_versions} versions for index {idx} generated successfully")
                    sample_success = True
                    break  # Break out of retry loop on success
                else:
                    print(f"✗ Index {idx} failed: only {len(versions_for_item)}/{self.num_versions} versions generated")
                    if retry_attempt < max_retries_per_sample - 1:
                        print(f"   Retrying...")
                        time.sleep(1)  # Brief pause before retry
            
            # If all retries failed, record the failed index
            if not sample_success:
                failed_indices.append(idx)
                print(f"✗✗✗ Index {idx} still failed after {max_retries_per_sample} attempts")
            
            # Periodically save progress
            if (idx + 1) % save_interval == 0:
                self._save_progress(paraphrased_data, failed_indices)
        
        # Final save
        dataset_label = local_file if local_file else dataset_name
        
        # Compute actual start and end indices (for statistics)
        if indices is not None:
            actual_start_index = min(indices) if indices else 0
            actual_end_index = max(indices) + 1 if indices else 0
        
        self._save_final_results(paraphrased_data, failed_indices, dataset_label, actual_start_index, actual_end_index)
        
        print(f"\nParaphrasing complete!")
        
        if indices is not None:
            print(f"Processing mode: specified index list")
            print(f"Requested index count: {len(indices)}")
        else:
            print(f"Processing range: index {actual_start_index} to {actual_end_index-1}")
            print(f"Samples in range: {actual_end_index - actual_start_index}")
        
        print(f"Required versions per sample: {self.num_versions}")
        print(f"Total generated versions: {len(paraphrased_data)}")
        successful_samples = len(paraphrased_data) // self.num_versions if self.num_versions > 0 else 0
        
        if indices is not None:
            total_requested = len(indices)
        else:
            total_requested = actual_end_index - actual_start_index
        
        print(f"Successfully processed samples: {successful_samples}/{total_requested}")
        print(f"Failed samples: {len(failed_indices)}/{total_requested}")
        if successful_samples > 0:
            print(f"Versions per successful sample: {self.num_versions} (guaranteed consistent)")
        
        # If there are failed samples, suggest reprocessing
        if failed_indices:
            print(f"\n⚠️  Warning: {len(failed_indices)} samples failed to generate all versions")
            print(f"Failed indices saved to: {os.path.join(self.output_dir, 'failed_indices.json')}")
            print(f"\nSuggestion: Increase --max-retries-per-sample to improve success rate")
        else:
            print(f"\n✓ Success! All samples have {self.num_versions} versions generated")
        
        print(f"Results saved to: {self.output_dir}")
    
    def _save_progress(self, paraphrased_data: List[Dict], failed_indices: List[int]):
        """Save progress"""
        progress_file = os.path.join(self.output_dir, "progress.json")
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump({
                "paraphrased_data": paraphrased_data,
                "failed_indices": failed_indices
            }, f, ensure_ascii=False, indent=2)
    
    def _save_final_results(
        self, 
        paraphrased_data: List[Dict], 
        failed_indices: List[int],
        dataset_name: str,
        start_index: int = 0,
        end_index: Optional[int] = None
    ):
        """Save final results"""
        # Save as JSON
        output_file = os.path.join(self.output_dir, "paraphrased_dataset.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(paraphrased_data, f, ensure_ascii=False, indent=2)
        
        # Save as JSONL
        output_file_jsonl = os.path.join(self.output_dir, "paraphrased_dataset.jsonl")
        with open(output_file_jsonl, 'w', encoding='utf-8') as f:
            for item in paraphrased_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Save failed indices
        if failed_indices:
            failed_file = os.path.join(self.output_dir, "failed_indices.json")
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(failed_indices, f, indent=2)
        
        # Compute evaluation statistics
        if paraphrased_data:
            problem_bleu_scores = [item['evaluation']['problem_bleu'] for item in paraphrased_data]
            problem_rouge_scores = [item['evaluation']['problem_rouge_l'] for item in paraphrased_data]
            solution_bleu_scores = [item['evaluation']['solution_bleu'] for item in paraphrased_data]
            solution_rouge_scores = [item['evaluation']['solution_rouge_l'] for item in paraphrased_data]
            answer_matches = [item['evaluation']['answer_match'] for item in paraphrased_data]
            # For non-math domains, answer_equivalent may not be present
            answer_equivalents = [item['evaluation'].get('answer_equivalent', item['evaluation']['answer_match']) for item in paraphrased_data]
            
            # Count versions generated per original sample
            version_counts = {}
            for item in paraphrased_data:
                idx = item['original_index']
                version_counts[idx] = version_counts.get(idx, 0) + 1
            
            # Count successfully processed original samples
            total_original_samples = len(version_counts)
            
            # Ensure all samples have exactly the same number of versions
            expected_versions = self.num_versions
            actual_version_counts = list(version_counts.values())
            all_consistent = all(count == expected_versions for count in actual_version_counts)
            
            evaluation_stats = {
                "version_consistency": {
                    "all_samples_have_same_versions": all_consistent,
                    "expected_versions_per_sample": expected_versions,
                    "actual_min_versions": min(actual_version_counts) if actual_version_counts else 0,
                    "actual_max_versions": max(actual_version_counts) if actual_version_counts else 0,
                },
                "problem_metrics": {
                    "bleu_mean": float(np.mean(problem_bleu_scores)),
                    "bleu_std": float(np.std(problem_bleu_scores)),
                    "bleu_min": float(np.min(problem_bleu_scores)),
                    "bleu_max": float(np.max(problem_bleu_scores)),
                    "rouge_l_mean": float(np.mean(problem_rouge_scores)),
                    "rouge_l_std": float(np.std(problem_rouge_scores)),
                    "rouge_l_min": float(np.min(problem_rouge_scores)),
                    "rouge_l_max": float(np.max(problem_rouge_scores)),
                },
                "solution_metrics": {
                    "bleu_mean": float(np.mean(solution_bleu_scores)),
                    "bleu_std": float(np.std(solution_bleu_scores)),
                    "bleu_min": float(np.min(solution_bleu_scores)),
                    "bleu_max": float(np.max(solution_bleu_scores)),
                    "rouge_l_mean": float(np.mean(solution_rouge_scores)),
                    "rouge_l_std": float(np.std(solution_rouge_scores)),
                    "rouge_l_min": float(np.min(solution_rouge_scores)),
                    "rouge_l_max": float(np.max(solution_rouge_scores)),
                },
                "answer_verification": {
                    "total_samples": len(answer_matches),
                    "string_matched": sum(answer_matches),
                    "string_mismatched": len(answer_matches) - sum(answer_matches),
                    "string_match_rate": sum(answer_matches) / len(answer_matches) * 100,
                    "mathematically_equivalent": sum(answer_equivalents),
                    "mathematically_not_equivalent": len(answer_equivalents) - sum(answer_equivalents),
                    "mathematical_equivalence_rate": sum(answer_equivalents) / len(answer_equivalents) * 100
                }
            }
        else:
            evaluation_stats = {}
            total_original_samples = 0
        
        # Save statistics
        stats = {
            "dataset_name": dataset_name,
            "index_range": {
                "start": start_index,
                "end": end_index,
                "total_in_range": (end_index - start_index) if end_index else "unknown"
            },
            "total_original_samples": total_original_samples + len(failed_indices),
            "versions_per_sample": self.num_versions,
            "total_versions_generated": len(paraphrased_data),
            "successful_original_samples": total_original_samples,
            "failed_original_samples": len(failed_indices),
            "success_rate": (total_original_samples / (total_original_samples + len(failed_indices)) * 100) if (total_original_samples + len(failed_indices)) > 0 else 0,
            "evaluation_statistics": evaluation_stats
        }
        stats_file = os.path.join(self.output_dir, "statistics.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # Print evaluation summary
        if evaluation_stats:
            print("\n" + "="*60)
            print("Evaluation Statistics Summary")
            print("="*60)
            print(f"\nVersion consistency check:")
            print(f"  Expected versions per sample: {evaluation_stats['version_consistency']['expected_versions_per_sample']}")
            print(f"  All samples have same version count: {'✓ Yes' if evaluation_stats['version_consistency']['all_samples_have_same_versions'] else '✗ No'}")
            print(f"  Actual minimum versions: {evaluation_stats['version_consistency']['actual_min_versions']}")
            print(f"  Actual maximum versions: {evaluation_stats['version_consistency']['actual_max_versions']}")
            print(f"\nProblem metrics:")
            print(f"  BLEU: {evaluation_stats['problem_metrics']['bleu_mean']:.4f} ± {evaluation_stats['problem_metrics']['bleu_std']:.4f}")
            print(f"  Rouge-L: {evaluation_stats['problem_metrics']['rouge_l_mean']:.4f} ± {evaluation_stats['problem_metrics']['rouge_l_std']:.4f}")
            print(f"\nSolution metrics:")
            print(f"  BLEU: {evaluation_stats['solution_metrics']['bleu_mean']:.4f} ± {evaluation_stats['solution_metrics']['bleu_std']:.4f}")
            print(f"  Rouge-L: {evaluation_stats['solution_metrics']['rouge_l_mean']:.4f} ± {evaluation_stats['solution_metrics']['rouge_l_std']:.4f}")
            print(f"\nAnswer verification:")
            print(f"  String match: {evaluation_stats['answer_verification']['string_matched']}/{evaluation_stats['answer_verification']['total_samples']} ({evaluation_stats['answer_verification']['string_match_rate']:.2f}%)")
            print(f"  Mathematical equivalence: {evaluation_stats['answer_verification']['mathematically_equivalent']}/{evaluation_stats['answer_verification']['total_samples']} ({evaluation_stats['answer_verification']['mathematical_equivalence_rate']:.2f}%)")
            if evaluation_stats['answer_verification']['mathematical_equivalence_rate'] < 100:
                print(f"  ⚠️  Warning: {evaluation_stats['answer_verification']['mathematically_not_equivalent']} answers are not mathematically equivalent!")
            else:
                print(f"  ✓ All answers are mathematically equivalent!")
            print("="*60)
        
        # Delete progress file
        progress_file = os.path.join(self.output_dir, "progress.json")
        if os.path.exists(progress_file):
            os.remove(progress_file)


def main():
    parser = argparse.ArgumentParser(description="Use GPT to paraphrase multi-domain datasets (math, finance, physics, chemistry, business, etc.)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceH4/MATH-500",
        help="HuggingFace dataset name (used when --local-file is not specified)"
    )
    parser.add_argument(
        "--local-file",
        type=str,
        default=None,
        help="Path to local JSONL file (e.g., /path/to/dataset.jsonl)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (train/test/validation), only for HuggingFace datasets"
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
        default="gpt-4o",
        help="GPT model to use (e.g. gpt-4, gpt-3.5-turbo)"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="math",
        choices=["math", "finance", "physics", "chemistry", "business"],
        help="Dataset domain: math, finance, physics, chemistry, or business"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./paraphrased_data",
        help="Output directory"
    )
    parser.add_argument(
        "--problem-field",
        type=str,
        default="question",
        help="Field name for questions"
    )
    parser.add_argument(
        "--solution-field",
        type=str,
        default="solution",
        help="Field name for solution steps"
    )
    parser.add_argument(
        "--answer-field",
        type=str,
        default="answer",
        help="Field name for answers"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (None means process all)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save progress every N samples"
    )
    parser.add_argument(
        "--num-versions",
        type=int,
        default=1,
        help="Number of paraphrase versions to generate per sample"
    )
    parser.add_argument(
        "--max-retries-per-sample",
        type=int,
        default=3,
        help="Maximum retries per sample (to ensure N versions are generated for every sample)"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Start index (inclusive) for parallel processing. None means start from the beginning. Mutually exclusive with --indices"
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="End index (exclusive) for parallel processing. None means process to the end. Mutually exclusive with --indices"
    )
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated list of specific indices to process (e.g., '5,12,23,45,67'). Mutually exclusive with --start-index/--end-index"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please provide an OpenAI API key (via --api-key or the OPENAI_API_KEY environment variable)")
        return
    
    # Parse index list
    indices_list = None
    if args.indices:
        try:
            indices_list = [int(idx.strip()) for idx in args.indices.split(',') if idx.strip()]
            print(f"Parsed {len(indices_list)} indices")
        except ValueError as e:
            print(f"Error: Failed to parse index list '{args.indices}': {e}")
            print("Please ensure the index format is correct, e.g.: '5,12,23,45,67'")
            return
    
    # Check mutually exclusive arguments
    if indices_list is not None and (args.start_index is not None or args.end_index is not None):
        print("Warning: Both --indices and --start-index/--end-index specified")
        print("Using --indices; --start-index and --end-index will be ignored")
    
    # If an index range or index list is specified, add range info to the output directory name
    output_dir = args.output_dir
    if indices_list is not None:
        # Naming when using an index list
        if len(indices_list) <= 5:
            idx_str = '_'.join(map(str, indices_list))
        else:
            idx_str = f"{indices_list[0]}_to_{indices_list[-1]}_and_{len(indices_list)}_indices"
        output_dir = f"{output_dir}_indices_{idx_str}"
        print(f"Output directory adjusted to: {output_dir}")
    elif args.start_index is not None or args.end_index is not None:
        start_idx = args.start_index if args.start_index is not None else 0
        end_idx = args.end_index if args.end_index is not None else "end"
        output_dir = f"{output_dir}_idx{start_idx}_to_{end_idx}"
        print(f"Output directory adjusted to: {output_dir}")
    
    # Create paraphraser and run
    paraphraser = DatasetParaphraser(
        api_key=api_key,
        model=args.model,
        output_dir=output_dir,
        num_versions=args.num_versions,
        domain=args.domain
    )
    
    paraphraser.paraphrase_dataset(
        dataset_name=args.dataset,
        split=args.split,
        problem_field=args.problem_field,
        solution_field=args.solution_field,
        answer_field=args.answer_field,
        max_samples=args.max_samples,
        save_interval=args.save_interval,
        local_file=args.local_file,
        max_retries_per_sample=args.max_retries_per_sample,
        start_index=args.start_index,
        end_index=args.end_index,
        indices=indices_list
    )


if __name__ == "__main__":
    main()
