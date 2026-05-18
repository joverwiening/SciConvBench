import json
from typing import Dict, List, Any
from config import Config
from utils import LLMService
import os
import argparse
import re


class LLMJudgeInconsistency:
    def __init__(self):
        self.config = Config()
        self.llm = LLMService(self.config)

    def read_file(self, path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            return ""

    def evaluate(self, generated_text: str, inconsistencies: List[str], incomplete_prompt: str, correct_prompt: str) -> Dict[str, Any]:
        """
        Checks if the Generated Result resolved ALL known inconsistencies by comparing it against the Ground Truth.
        """
        stats_before = self.llm.get_statistics()
        if not generated_text:
            return {
                "score": "0/1",
                "status": "ERROR",
                "reasoning": "No generated text found.",
                "breakdown": [],
                "llm_usage": {
                    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                    "total_calls": 0, "failed_calls": 0, "total_retries": 0
                }
            }

        inconsistency_block = ""
        for i, inc in enumerate(inconsistencies, 1):
            inconsistency_block += f"Inconsistency #{i}: {inc}\n"

        system_prompt = """
        You are an expert AI Evaluator.
        You will be provided with:
        1. [INCOMPLETE_PROMPT]: The original flawed request given to the Agent.
        2. [GROUND_TRUTH_PROMPT]: The perfect, correct version of the request.
        3. [KNOWN_INCONSISTENCIES]: A list of specific errors that were present in the incomplete prompt.
        4. [GENERATED_SPECIFICATION]: The final output produced by the Agent.

        **YOUR GOAL:**
        - Compare the [GENERATED_SPECIFICATION] against the [GROUND_TRUTH_PROMPT].
        - Determine if the Agent resolved **ALL** of the listed inconsistencies in the [KNOWN_INCONSISTENCIES] list based on how the ground truth handled it.
        - Did the Agent **avoid** repeating the user's incorrect instructions for EVERY inconsistency.
        - The generated text does not need to be a word-for-word copy of the ground truth, but it MUST capture the correct meaning.

        
        **SCORING:**
        - **Score = "1/1"**: The Agent successfully identified and fixed **ALL** inconsistencies listed, aligning with the ground truth.
        - **Score = "0/1"**: The Agent failed to fix **AT LEAST ONE** of the inconsistencies (or repeated the original error).

        **OUTPUT FORMAT:**
        CRITICAL: The "breakdown" list MUST contain exactly one entry for EACH item in the [KNOWN_INCONSISTENCIES] list. 
        If there are 2 inconsistencies, the breakdown array must have 2 objects.
        
        You must output your final evaluation as a JSON object enclosed EXACTLY within <FINAL_JSON> tags. 
        <FINAL_JSON>
        {
            "score": "1/1",  // "1/1" ONLY if ALL items in breakdown are resolved: true. Otherwise "0/1".
            "status": "RESOLVED", // "RESOLVED" if score is "1/1", otherwise "FAILED"
            "breakdown": [
                {
                    "inconsistency_index": 1,
                    "resolved": true, // true if fixed to match ground truth, false if error persisted
                    "reasoning": "Specific explanation for why Inconsistency #1 is resolved or failed."
                },
                {
                    "inconsistency_index": 2,
                    "resolved": false,
                    "reasoning": "Specific explanation for why Inconsistency #2 is resolved or failed"
                }
                // ... continue for all inconsistencies
            ]
        }
        </FINAL_JSON>
        """

        user_prompt = f"""
        <INCOMPLETE_PROMPT>
        {incomplete_prompt}
        </INCOMPLETE_PROMPT>

        <GROUND_TRUTH_PROMPT>
        {correct_prompt}
        </GROUND_TRUTH_PROMPT>

        <KNOWN_INCONSISTENCIES>
        {inconsistency_block}
        </KNOWN_INCONSISTENCIES>

        <GENERATED_SPECIFICATION>
        {generated_text}
        </GENERATED_SPECIFICATION>
        
        Evaluate: Did the generated text resolve ALL listed inconsistencies to match the ground truth? Provide a breakdown for each.
        """

        # Invoke LLM
        response_text = self.llm.invoke(user_prompt, system_prompt)
        stats_after = self.llm.get_statistics()
        case_usage = {
            "prompt_tokens": stats_after["prompt_tokens"] - stats_before["prompt_tokens"],
            "completion_tokens": stats_after["completion_tokens"] - stats_before["completion_tokens"],
            "total_tokens": stats_after["total_tokens"] - stats_before["total_tokens"],
            "total_calls": stats_after["total_calls"] - stats_before["total_calls"],
            "failed_calls": stats_after["failed_calls"] - stats_before["failed_calls"],
            "total_retries": stats_after["total_retries"] - stats_before["total_retries"]
        }
        try:
            # Use Regex to extract ONLY the text between <FINAL_JSON> tags
            clean_json_str = ""

            matches = re.findall(
                r'<FINAL_JSON>(.*?)</FINAL_JSON>', response_text, re.DOTALL)

            if matches:
                clean_json_str = matches[-1].strip()
            else:  # Fallback:
                cleaned_text = re.sub(r'```json\s*', '', response_text)
                cleaned_text = re.sub(r'```\s*', '', cleaned_text)
                start_idx = cleaned_text.find('{')
                end_idx = cleaned_text.rfind('}')

                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    clean_json_str = cleaned_text[start_idx:end_idx+1]
                else:
                    raise ValueError("Could not find JSON block in response.")
            # Clean up any lingering markdown backticks inside the block
            clean_json_str = re.sub(r'```json\s*', '', clean_json_str)
            clean_json_str = re.sub(r'```\s*', '', clean_json_str)
            clean_json_str = re.sub(r'\\(?!["\\/])', r'\\\\', clean_json_str)
            parsed_json = json.loads(clean_json_str, strict=False)

            final_result = {
                "score": "0/1",
                "status": "ERROR",
                "breakdown": []
            }

            final_result.update(parsed_json)
            final_result["llm_usage"] = case_usage

            return final_result

        except (json.JSONDecodeError, ValueError) as e:
            print("Failed to parse LLM response as JSON. Raw response:")
            print(response_text)
            return {
                "score": "0/1",
                "status": "ERROR",
                "breakdown": [],
                "llm_usage": case_usage
            }


def main():
    parser = argparse.ArgumentParser(description="Run Evaluation")
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--json_path", required=True, type=str)

    args = parser.parse_args()
    test_runs_dir = args.input_dir
    json_path = args.json_path
    output_summary_file = "llm_judge_summary.json"

    if not os.path.exists(json_path):
        print(
            f"Error: {json_path} not found. Cannot determine inconsistencies.")
        return

    with open(json_path, 'r') as f:
        raw_list = json.load(f)

    cases_data = {}
    for item in raw_list:
        case_id = item.get("id")
        if case_id:
            cases_data[case_id] = item
        else:
            print("Warning: Found item in JSON without an 'id' field. Skipping.")

    judge = LLMJudgeInconsistency()

    print(f"\nSTARTING BATCH LLM INCONSISTENCY EVALUATION")
    print("="*60)

    report_data = {
        "cases": []
    }

    total_cases_run = len(cases_data)
    strict_score_accum = 0
    global_resolved = 0
    global_total_inconsistencies = 0

    if total_cases_run == 0:
        print("\nNo cases were found. Nothing can be evaluated\n")
        return
    missing_generated = []
    for case_id in sorted(cases_data.keys()):

        folder_path = os.path.join(test_runs_dir, case_id)
        gen_path = os.path.join(folder_path, "generated_prompt.txt")
        # Handle cases where the agent failed to generate a prompt
        if not os.path.exists(gen_path):
            print(
                f"   [!] THE AGENT DID NOT RUN {case_id}. Please double check what went wrong.")
            missing_generated.append(case_id)
            continue

        output_path = os.path.join(
            folder_path, "llm_judge_inconsistency_score.json")
        case_data = cases_data[case_id]
        inconsistencies_list = case_data.get("inconsistencies", [])
        incomplete_prompt = case_data.get("inconsistent_prompt", "")
        correct_prompt = case_data.get("correct_prompt", "")

        print(f"\nLLM Judging {case_id}...")

        result = {}
        clean_breakdown = []

        # Standard Evaluation if the file exists
        gen_text = judge.read_file(gen_path)
        result = judge.evaluate(
            gen_text, inconsistencies_list, incomplete_prompt, correct_prompt)

        raw_breakdown = result.get("breakdown", [])

        for i, inconsistency_text in enumerate(inconsistencies_list):
            found_match = None
            for item in raw_breakdown:
                idx = item.get("inconsistency_index", -1)
                if idx == (i + 1):
                    found_match = item
                    break

            if found_match:
                clean_entry = {
                    "description": inconsistency_text,
                    "resolved": found_match.get("resolved", False),
                    "reason": found_match.get("reasoning", "No reasoning provided.")
                }
            else:
                clean_entry = {
                    "description": inconsistency_text,
                    "resolved": False,
                    "reason": "LLM failed to return a verdict for this specific item."
                }
            clean_breakdown.append(clean_entry)

        result["breakdown"] = clean_breakdown
        result["case_id"] = case_id

        # Calculate Strict and Partial
        strict_score = result.get("score", "0/1")
        case_total_inc = len(clean_breakdown)
        case_resolved = sum(
            1 for item in clean_breakdown if item.get("resolved") is True)
        partial_score = f"{case_resolved}/{case_total_inc}"

        result["strict_score"] = strict_score
        result["partial_score"] = partial_score

        # Save Individual Result
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4)

        status = "SUCCESS" if strict_score == "1/1" else "FAIL"
        print(
            f"   {status} | Strict [{strict_score}] | Partial [{partial_score}]")

        # Add to Summary Data
        case_entry = {
            "case_id": case_id,
            "strict_score": strict_score,
            "partial_score": partial_score,
            "status": result.get("status", "UNKNOWN"),
            "breakdown": clean_breakdown,
            "llm_usage": result.get('llm_usage', {})
        }
        report_data["cases"].append(case_entry)

        if strict_score == "1/1":
            strict_score_accum += 1

        global_resolved += case_resolved
        global_total_inconsistencies += case_total_inc

    if total_cases_run > 0:
        strict_pct = (strict_score_accum / total_cases_run) * 100
        partial_pct = (global_resolved / global_total_inconsistencies *
                       100) if global_total_inconsistencies > 0 else 0
        final_llm_stats = judge.llm.get_statistics()

        report_data["final_summary"] = {
            "strict_accuracy_percentage": f"{strict_pct:.2f}%",
            "strict_aggregate_score": f"{strict_score_accum}/{total_cases_run}",
            "partial_accuracy_percentage": f"{partial_pct:.2f}%",
            "partial_aggregate_score": f"{global_resolved}/{global_total_inconsistencies}",
            "total_cases_evaluated": total_cases_run,
            "total_llm_usage": final_llm_stats
        }

        final_output_path = os.path.join(test_runs_dir, output_summary_file)

        with open(final_output_path, "w") as f:
            json.dump(report_data, f, indent=4)

        print("\n" + "="*60)
        print(
            f"STRICT ACCURACY: {strict_score_accum}/{total_cases_run} ({strict_pct:.1f}%)")
        print(
            f"PARTIAL RESOLVED: {global_resolved}/{global_total_inconsistencies} ({partial_pct:.1f}%)")
        print(f"REPORT SAVED TO: {final_output_path}")
        print("="*60)

    if len(missing_generated) > 0:
        print("\nWARNING: The agent did not run the following cases so they were not evaluated:")
        print(missing_generated)


if __name__ == "__main__":
    main()
