import json
import os
from typing import Dict, Any, List
from config import Config
from utils import LLMService
import argparse
import re

class LLMJudgeDisambiguation:
    def __init__(self):
        self.config = Config()
        self.llm = LLMService(self.config)

    def read_file(self, path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            return ""

    def evaluate(self, ground_truth: str, final_result: str, expected_missing: List[str]) -> Dict[str, Any]:
        """
        Compares the Ground Truth vs the Generated Result.
        Focuses specifically on whether the 'expected_missing' entities were recovered.
        """
        stats_before = self.llm.get_statistics()
        if not ground_truth or not final_result:
            return {
                "score": "0/0",
                "recovered_entities": [],
                "still_missing_entities": expected_missing,
                "llm_usage": {
                    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                    "total_calls": 0, "failed_calls": 0, "total_retries": 0
                }
            }

        total_expected = len(expected_missing)
        missing_list_str = ", ".join(expected_missing)

        system_prompt = f"""
        You are an expert AI Evaluator. 
        Your specific task is to check if a [GENERATED_SPECIFICATION] has successfully recovered information that was known to be missing.

        You are provided with a list of [EXPECTED_MISSING_ENTITIES]. 
        For each entity in that list, determine if the [GENERATED_SPECIFICATION] contains the specific values or details found in the [GROUND_TRUTH].

        **SCORING RULE:**
        - The total possible score is {total_expected} (the number of expected missing entities).
        - Count how many of these entities are effectively present and correct in the Generated version.
        - Output the score strictly as a string "X/{total_expected}".

        **OUTPUT FORMAT:**
        You must output your final evaluation as a JSON object enclosed EXACTLY within <FINAL_JSON> tags. 
        
        <FINAL_JSON>
        {{
            "score": "X/{total_expected}",
            "recovered_entities": ["list", "of", "entities", "from", "the", "expected", "list", "that", "were", "found"],
            "still_missing_entities": ["list", "of", "entities", "from", "the", "expected", "list", "that", "are", "still", "absent"]
        }}
        </FINAL_JSON>        
        """

        user_prompt = f"""
        <EXPECTED_MISSING_ENTITIES>
        {missing_list_str}
        </EXPECTED_MISSING_ENTITIES>

        <GROUND_TRUTH>
        {ground_truth}
        </GROUND_TRUTH>

        <GENERATED_SPECIFICATION>
        {final_result}
        </GENERATED_SPECIFICATION>
        
        Evaluate now.
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
                "score": f"0/{total_expected}", 
                "recovered_entities": [],
                "still_missing_entities": expected_missing
            }
            
            final_result.update(parsed_json)
            final_result["llm_usage"] = case_usage

            return final_result

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse LLM response as JSON. Error: {e}")
            print("Raw response:")
            print(response_text)
            return {
                "score": f"0/{total_expected}", 
                "recovered_entities": [],
                "still_missing_entities": expected_missing,
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
        print(f"Error: {json_path} not found.")
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

    judge = LLMJudgeDisambiguation()
    
    print(f"\nSTARTING BATCH LLM DISAMBIGUATION EVALUATION")
    print("="*60)

    report_data = {
        "cases": []
    }
    
    total_cases_run = len(cases_data)
    strict_score_accum = 0
    global_recovered = 0
    global_total_entities = 0

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

        gt_path = os.path.join(folder_path, "complete_prompt.txt")
        output_path = os.path.join(folder_path, "llm_judge_disambiguation_score.json")
        case_data = cases_data[case_id]
        missing_entities = case_data.get("missing_entities", [])


        print(f"\nLLM Judging {case_id}...")     
        # Read Texts   
        gt_text = judge.read_file(gt_path)
        gen_text = judge.read_file(gen_path)
        # Run Evaluation
        result = judge.evaluate(gt_text, gen_text, missing_entities)

        score_str = result.get('score', '0/0')
        try:
            recovered_val = int(score_str.split('/')[0])
            total_val = int(score_str.split('/')[1])
        except (ValueError, IndexError):
            recovered_val = 0
            total_val = len(missing_entities)

        # Calculate Strict and Partial
        is_perfect = (recovered_val == total_val and total_val > 0)
        strict_score = "1/1" if is_perfect else "0/1"
        partial_score = f"{recovered_val}/{total_val}"

        result["strict_score"] = strict_score
        result["partial_score"] = partial_score

        # Save Individual Result
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4)
        
        status = "SUCCESS" if is_perfect else "FAIL"
        print(f"   {status} | Strict [{strict_score}] | Partial [{partial_score}]")

        # Add to Summary Data
        case_entry = {
            "case_id": case_id,
            "strict_score": strict_score,
            "partial_score": partial_score,
            "recovered": result.get('recovered_entities', []),
            "missing": result.get('still_missing_entities', []),
            "llm_usage": result.get('llm_usage', {})
        }
        report_data["cases"].append(case_entry)

        if is_perfect:
            strict_score_accum += 1
            
        global_recovered += recovered_val
        global_total_entities += total_val

    # Final Summary Math
    if total_cases_run > 0:
        strict_pct = (strict_score_accum / total_cases_run) * 100
        partial_pct = (global_recovered / global_total_entities * 100) if global_total_entities > 0 else 0
        final_llm_stats = judge.llm.get_statistics()
        report_data["final_summary"] = {
            "strict_accuracy_percentage": f"{strict_pct:.2f}%",
            "strict_aggregate_score": f"{strict_score_accum}/{total_cases_run}",
            "partial_accuracy_percentage": f"{partial_pct:.2f}%",
            "partial_aggregate_score": f"{global_recovered}/{global_total_entities}",
            "total_cases_evaluated": total_cases_run,
            "total_llm_usage": final_llm_stats

        }
        
        final_output_path = os.path.join(test_runs_dir, output_summary_file)
        
        with open(final_output_path, "w") as f:
            json.dump(report_data, f, indent=4)
            
        print("\n" + "="*60)
        print(f"STRICT ACCURACY: {strict_score_accum}/{total_cases_run} ({strict_pct:.1f}%)")
        print(f"PARTIAL RECOVERY: {global_recovered}/{global_total_entities} ({partial_pct:.1f}%)")
        print(f"REPORT SAVED TO: {final_output_path}")
        print("="*60)
    
    if len(missing_generated) > 0:
        print("\nWARNING: The agent did not run the following cases so they were not evaluated")
        print(missing_generated)

if __name__ == "__main__":
    main()