import json
from typing import Dict, Any
from config import Config
from utils import LLMService
import os
import argparse
import re

class LLMJudgeIntent:
    def __init__(self):
        self.config = Config()
        self.llm = LLMService(self.config)

    def read_file(self, path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            return ""

    def evaluate(self, generated_text: str, ground_truth: str) -> Dict[str, Any]:
        """
        Evaluates if the Generated Specification is faithful to the Ground Truth intent.
        """
        stats_before = self.llm.get_statistics()
        if not generated_text:
            return {
                "score": "0/1",
                "status": "UNFAITHFUL",
                "reasoning": "No generated text found.",
                "llm_usage": {
                    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                    "total_calls": 0, "failed_calls": 0, "total_retries": 0
                }
            }

        system_prompt = """
        You are an expert AI Evaluator.
        You will be provided with:
        1. [GROUND_TRUTH_PROMPT]: The perfect, correct version of the user's intent.
        2. [GENERATED_SPECIFICATION]: The final output produced by an Agent.

        **YOUR GOAL:**
        - Evaluate if the [GENERATED_SPECIFICATION] is faithful to the overall intent of the [GROUND_TRUTH_PROMPT].
        - Does the generated text capture all the essential instructions?

        **SCORING:**
        - **Score = "1/1"**: The Agent successfully captured all the core intent and requirements of the ground truth.
        - **Score = "0/1"**: The Agent missed critical parameters, or fundamentally altered the ground truth.

        **OUTPUT FORMAT:**
        You must output your final evaluation as a JSON object enclosed EXACTLY within <FINAL_JSON> tags. 
        <FINAL_JSON>
        {
            "score": "1/1",  // "1/1" if faithful, "0/1" if unfaithful
            "status": "FAITHFUL", // "FAITHFUL" if score is "1/1", otherwise "UNFAITHFUL"
            "reasoning": "Specific explanation comparing the generated text to the ground truth, highlighting what was captured successfully or what was missed."
        }
        </FINAL_JSON>
        """

        user_prompt = f"""
        <GROUND_TRUTH_PROMPT>
        {ground_truth}
        </GROUND_TRUTH_PROMPT>

        <GENERATED_SPECIFICATION>
        {generated_text}
        </GENERATED_SPECIFICATION>
        
        Evaluate: Is the generated specification faithful to the ground truth intent?
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

            matches = re.findall(r'<FINAL_JSON>(.*?)</FINAL_JSON>', response_text, re.DOTALL)
            
            if matches:
                clean_json_str = matches[-1].strip()
            else: # Fallback
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
                "reasoning": "Not provided by LLM."
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
                "reasoning": "Failed to parse evaluation response from LLM.",
                "llm_usage": case_usage
            }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--json_path", required=True)

    args = parser.parse_args()
    test_runs_dir = args.input_dir
    json_path = args.json_path
    output_summary_file = "llm_judge_intent_summary.json"
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    # Load Dataset
    with open(json_path, 'r') as f:
        raw_list = json.load(f)

    cases_data = {}
    for item in raw_list:
        case_id = item.get("id")
        if case_id:
            cases_data[case_id] = item
        else:
            print("Warning: Found item in JSON without an 'id' field. Skipping.")

    judge = LLMJudgeIntent()
    
    print(f"\nSTARTING BATCH LLM INTENT EVALUATION")
    print("="*60)

    report_data = {
        "cases": []
    }
    
    total_cases_run = len(cases_data)
    score_accum = 0

    if total_cases_run == 0:
        print("\nNo cases were found. Nothing can be evaluated\n")
        return

    missing_generated = []

    for case_id in sorted(cases_data.keys()):
        folder_path = os.path.join(test_runs_dir, case_id)
        gen_path = os.path.join(folder_path, "generated_prompt.txt")

        if not os.path.exists(gen_path):
            print(f"   [!] THE AGENT DID NOT RUN {case_id}. Please double check what went wrong.")
            missing_generated.append(case_id)                
            continue

        ground_truth_path = os.path.join(folder_path, "complete_prompt.txt")
        output_path = os.path.join(folder_path, "llm_judge_intent_score.json")

        # Read Texts
        gen_text = judge.read_file(gen_path)
        ground_truth = judge.read_file(ground_truth_path)
        # Run Evaluation
        print(f"\nLLM Judging Intent for {case_id}...")        
        result = judge.evaluate(gen_text, ground_truth)
        result["case_id"] = case_id
        
        # Save Individual Result
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4)
            
        score_str = result.get("score", "0/1")
        status = result.get("status", "UNKNOWN")
        
        print(f"   Status: {status} | Score: {score_str}")
        print(f"   Saved to: {os.path.basename(output_path)}")

        # Add to Summary Data
        case_entry = {
            "case_id": case_id,
            "score": score_str,
            "status": status,
            "reasoning": result.get("reasoning", ""),
            "llm_usage": result.get('llm_usage', {})
        }
        report_data["cases"].append(case_entry)

        if score_str == "1/1":
            score_accum += 1

    # Final Summary & Save
    if total_cases_run > 0:
        avg_percentage = (score_accum / total_cases_run) * 100
        final_llm_stats = judge.llm.get_statistics()
        report_data["final_summary"] = {
            "average_percentage": f"{avg_percentage:.2f}%",
            "aggregate_score": f"{score_accum}/{total_cases_run}",
            "total_cases_evaluated": total_cases_run,
            "total_llm_usage": final_llm_stats
        }
        
        final_output_path = os.path.join(test_runs_dir, output_summary_file)
        
        with open(final_output_path, "w") as f:
            json.dump(report_data, f, indent=4)
            
        print("\n" + "="*60)
        print(f"INTENT/FAITHFULNESS SCORE: {score_accum}/{total_cases_run} ({avg_percentage:.1f}%)")
        print(f"REPORT SAVED TO: {final_output_path}")
        print("="*60)
        
    if len(missing_generated) > 0:
        print("\nWARNING: The agent did not run the following cases so they were evaluated as 0/UNFAITHFUL:")
        print(missing_generated)

if __name__ == "__main__":
    main()