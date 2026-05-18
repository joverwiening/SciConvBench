import json
import re
import os
import argparse
from typing import Dict, List, Any
from config import Config
from utils import LLMService

class LLMJudgeInconsistencyChat:
    def __init__(self):
        self.config = Config()
        self.llm = LLMService(self.config)

    def read_file(self, path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            return ""

    def parse_conversation_log(self, raw_log: str) -> List[Dict[str, str]]:
        """
        Converts the raw timestamped chat log into a clean JSON array for the LLM.
        Example: [{"Question_1": "...", "Answer_1": "..."}, ...]
        """
        if not raw_log or not raw_log.strip():
            return []

        turns = []
        # Split by the timestamp header (=== 2026-02-21 20:32:12 ===)
        blocks = re.split(r'===\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s*===', raw_log)
        
        turn_idx = 1
        for block in blocks:
            block = block.strip()
            if not block:
                continue
                
            q_match = re.search(r'Q:\s*(.*?)(?=\nA:|$)', block, re.DOTALL)
            a_match = re.search(r'A:\s*(.*)', block, re.DOTALL)
            
            q_text = q_match.group(1).strip() if q_match else ""
            a_text = a_match.group(1).strip() if a_match else ""
            
            if q_text or a_text:
                turns.append({
                    f"Question_{turn_idx}": q_text,
                    f"Answer_{turn_idx}": a_text
                })
                turn_idx += 1
                
        return turns

    def evaluate(self, parsed_chat: List[Dict[str, str]], inconsistencies: List[str], generated_prompt: str) -> Dict[str, Any]:
        """
        Extracts event counts from the conversation log.
        """

        stats_before = self.llm.get_statistics()
        # Default values if there was no conversation
        if not parsed_chat:
            return {
                "questions_about_inconsistencies": {"reasoning": "No conversation log found.", "count": 0},
                "relevant_questions": {"reasoning": "No conversation log found.", "count": 0},
                "explicit_warnings": {"reasoning": "No conversation log found.", "count": 0},
                "memory_consistency": {"reasoning": "No conversation log found, evaluating as inconsistent (0) by default.", "is_consistent": 0},
                "llm_usage": {
                    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                    "total_calls": 0, "failed_calls": 0, "total_retries": 0
                }
            }

        inconsistency_block = "\n".join([f"- {inc}" for inc in inconsistencies])
        chat_json_str = json.dumps(parsed_chat, indent=2)

        system_prompt = """
        You are an AI text analysis tool. Your job is to extract specific conversational events from a chat log between a User and an AI Assistant.
        
        You will be provided with:
        1. [PLANTED_INCONSISTENCIES]: A list of errors/contradictions the User deliberately put into their initial request.
        2. [CONVERSATION_LOG]: The parsed chat transcript between the User and the Assistant.
        3. [GENERATED_PROMPT]: The final output written by the Assistant.
        
        **Your Task:**
        Read the logs and count the occurrence of specific events. Output your findings strictly as a JSON object matching the requested schema. Always provide your step-by-step reasoning before outputting the final count or boolean.
        
        **Extraction Targets:**
        * **Target 1 (Questions about Inconsistencies):** Out of the provided [PLANTED_INCONSISTENCIES], how many did the Assistant explicitly ask the User about in the chat?
        * **Target 2 (Relevant Questions):** Look at ALL the questions the Assistant asked. How many of those questions were directly relevant to solving the [PLANTED_INCONSISTENCIES]? 
        * **Target 3 (Explicit Warnings):** How many of the [PLANTED_INCONSISTENCIES] did the Assistant explicitly warn the User about in the chat? (e.g., "Your formula is missing atmospheric pressure"). Silent fixes do not count.
        * **Target 4 (Memory Consistency):** Read the User's specific answers in the [CONVERSATION_LOG]. Did the Assistant perfectly retain and follow all explicit instructions and numbers provided by the User in the [GENERATED_PROMPT]? This is a strict binary test: evaluate to the integer `1` if the Assistant was perfectly consistent, or the integer `0` if you find ANY contradiction or ignored constraint.
        
        **Output Format:**
        You must output your final evaluation as a JSON object enclosed EXACTLY within <FINAL_JSON> tags. 
        <FINAL_JSON>
        {
          "questions_about_inconsistencies": {
            "reasoning": "Explain which planted inconsistencies were explicitly asked about.",
            "count": 0
          },
          "relevant_questions": {
            "reasoning": "Explain which questions were relevant to the planted inconsistencies.",
            "count": 0
          },
          "explicit_warnings": {
            "reasoning": "Explain which planted inconsistencies the Assistant explicitly flagged/warned the user about in the chat.",
            "count": 0
          },
          "memory_consistency": {
            "reasoning": "Explain if the final prompt followed all explicit user instructions from the chat without contradiction.",
            "is_consistent": 0
          }
        }
        </FINAL_JSON>
        """

        user_prompt = f"""
        <PLANTED_INCONSISTENCIES>
        {inconsistency_block}
        </PLANTED_INCONSISTENCIES>

        <CONVERSATION_LOG>
        {chat_json_str}
        </CONVERSATION_LOG>

        <GENERATED_PROMPT>
        {generated_prompt}
        </GENERATED_PROMPT>
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

            # Grab the LAST match to ignore prompt examples
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

            # Strip markdown backticks globally, just in case they put them inside the XML tags
            clean_json_str = re.sub(r'```json\s*', '', clean_json_str)
            clean_json_str = re.sub(r'```\s*', '', clean_json_str)
            clean_json_str = re.sub(r'\\(?!["\\/])', r'\\\\', clean_json_str)
            parsed_json = json.loads(clean_json_str, strict=False)
            
            final_result = {
                "questions_about_inconsistencies": {"reasoning": "Not provided by LLM.", "count": 0},
                "relevant_questions": {"reasoning": "Not provided by LLM.", "count": 0},
                "explicit_warnings": {"reasoning": "Not provided by LLM.", "count": 0},
                "memory_consistency": {"reasoning": "Not provided by LLM.", "is_consistent": 0},
            }
            
            final_result.update(parsed_json)
            final_result["llm_usage"] = case_usage

            return final_result

        except (json.JSONDecodeError, ValueError) as e:
            print("Failed to parse LLM response as JSON. Raw response:")
            print(response_text)
            return {
                "questions_about_inconsistencies": {"reasoning": "JSON parsing failed.", "count": 0},
                "relevant_questions": {"reasoning": "JSON parsing failed.", "count": 0},
                "explicit_warnings": {"reasoning": "JSON parsing failed.", "count": 0},
                "memory_consistency": {"reasoning": "JSON parsing failed.", "is_consistent": 0},
                "llm_usage": case_usage
            }

def main():
    parser = argparse.ArgumentParser(description="Run Evaluation")
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--json_path", required=True, type=str)

    args = parser.parse_args()
    test_runs_dir = args.input_dir
    json_path = args.json_path
    output_summary_file = "llm_judge_chat_summary.json"
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Cannot determine inconsistencies.")
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

    judge = LLMJudgeInconsistencyChat()
    
    print(f"\nSTARTING BATCH LLM INCONSISTENCY CHAT EVALUATION")
    print("="*60)

    report_data = {
        "cases": []
    }
    
    total_cases_run = len(cases_data)

    if total_cases_run == 0:
        print("\nNo cases were found. Nothing can be evaluated\n")
        return

    missing_generated = []

    # Trackers for the global summary
    global_metrics = {
        "total_questions_asked_overall": 0,
        "total_relevant_questions": 0,
        "total_inconsistencies_asked_about": 0,
        "total_explicit_warnings": 0,
        "total_planted_inconsistencies_overall": 0,
        "total_assumed_inconsistencies": 0,
        "cases_without_contradictions": 0
    }

    for case_id in sorted(cases_data.keys()):
        folder_path = os.path.join(test_runs_dir, case_id)        
        gen_path = os.path.join(folder_path, "generated_prompt.txt")
        chat_path = os.path.join(folder_path, "conversation_log.txt")
        output_path = os.path.join(folder_path, "llm_judge_inconsistency_chat_score.json")  
        task_score_path = os.path.join(folder_path, "llm_judge_inconsistency_score.json")
        
        # Handle missing generated files
        if not os.path.exists(gen_path):
            print(f"   [!] THE AGENT DID NOT RUN {case_id}. PLEASE INVESTIGATE")
            missing_generated.append(case_id)
            continue

        case_data = cases_data[case_id]
        inconsistencies_list = case_data.get("inconsistencies", [])
        
        # Read text files
        gen_text = judge.read_file(gen_path)
        raw_chat = judge.read_file(chat_path)
        
        print(f"\nLLM Judging Chat Log for {case_id}...")
        
        # Parse chat and evaluate
        parsed_chat = judge.parse_conversation_log(raw_chat)
        result = judge.evaluate(parsed_chat, inconsistencies_list, gen_text)
        
        # Add basic context variables
        total_questions_asked = len(parsed_chat)
        total_inconsistencies = len(inconsistencies_list)
        
        result["case_id"] = case_id
        result["total_questions_asked_in_chat"] = total_questions_asked
        result["total_planted_inconsistencies"] = total_inconsistencies

        # ==========================================
        # CALCULATE THE PARETO METRICS
        # ==========================================
        
        # Get raw LLM counts
        relevant_qs = result.get("relevant_questions", {}).get("count", 0)
        asked_about = result.get("questions_about_inconsistencies", {}).get("count", 0)
        warnings = result.get("explicit_warnings", {}).get("count", 0)
        raw_is_consistent = result.get("memory_consistency", {}).get("is_consistent", 0)
        try:
            m_consistency = float(raw_is_consistent)
        except (ValueError, TypeError):
            m_consistency = 0.0

        # Get "Resolved" count from the previous task judge for Assumption Rate
        resolved_in_final = 0
        if os.path.exists(task_score_path):
            with open(task_score_path, 'r') as tf:
                t_data = json.load(tf)
                p_score = t_data.get("partial_score", "0/0") # e.g., "2/3"
                try:
                    resolved_in_final = int(p_score.split('/')[0])
                except:
                    print("SOMETHING WENT WRONG WITH PARSING PARTIAL SCORE. PLEASE INVESTIGATE")
                    pass
        
        # Assumption: 
        # 1 - ( # inconsistencies resolved without being asked / total inconsistencies)
        # 1 - ( (#Resolved inconsistencies - #Asked inconsistencies) / total inconsistencies)
        # We use max(0, ...) so we don't get negative assumptions if it asked about more than it resolved
        assumed_count = max(0, resolved_in_final - asked_about)

        # Compute metric percentages (0.0 to 1.0)
        c_recall = (asked_about / total_inconsistencies) if total_inconsistencies > 0 else 0.0
        c_precision = (relevant_qs / total_questions_asked) if total_questions_asked > 0 else 0.0
        e_detection = (warnings / total_inconsistencies) if total_inconsistencies > 0 else 0.0
        a_rate = 1.0 - (assumed_count / total_inconsistencies) if total_inconsistencies > 0 else 0.0

        # sometimes more than 1 question can address one inconsistency, so we don't want 200% recall for example.
        c_recall = max(0.0, min(1.0, c_recall))
        c_precision = max(0.0, min(1.0, c_precision))
        e_detection = max(0.0, min(1.0, e_detection))
        a_rate = max(0.0, min(1.0, a_rate))
        
        result["metrics"] = {
            "clarification_precision": round(c_precision, 4),
            "clarification_recall": round(c_recall, 4),
            "assumption_rate": round(a_rate, 4),
            "error_detection_rate": round(e_detection, 4),
            "memory_consistency": m_consistency
        }

        # Save Individual Result
        os.makedirs(folder_path, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4)
            
        print(f"   Precision: {c_precision:.1%} | Recall: {c_recall:.1%} | Assumption: {a_rate:.1%}")
        print(f"   Error Detect: {e_detection:.1%} | Memory Consistency: {m_consistency}")

        report_data["cases"].append(result)

        # Update global math trackers
        global_metrics["total_questions_asked_overall"] += total_questions_asked
        global_metrics["total_relevant_questions"] += relevant_qs
        global_metrics["total_inconsistencies_asked_about"] += asked_about
        global_metrics["total_explicit_warnings"] += warnings
        global_metrics["total_planted_inconsistencies_overall"] += total_inconsistencies
        global_metrics["total_assumed_inconsistencies"] += assumed_count
        if m_consistency == 1.0:
            global_metrics["cases_without_contradictions"] += 1

    # Final Summary Compilation
    if total_cases_run > 0:
        final_llm_stats = judge.llm.get_statistics()
        tq = global_metrics["total_questions_asked_overall"]
        global_precision = (global_metrics["total_relevant_questions"] / tq) if tq > 0 else 0.0
        
        tp = global_metrics["total_planted_inconsistencies_overall"]
        global_recall = (global_metrics["total_inconsistencies_asked_about"] / tp) if tp > 0 else 0.0
        global_edr = (global_metrics["total_explicit_warnings"] / tp) if tp > 0 else 0.0
        global_assumption = 1.0 - (global_metrics["total_assumed_inconsistencies"] / tp) if tp > 0 else 0.0
        
        global_memory = global_metrics["cases_without_contradictions"] / total_cases_run

        global_precision = max(0.0, min(1.0, global_precision))
        global_recall = max(0.0, min(1.0, global_recall))
        global_edr = max(0.0, min(1.0, global_edr))
        global_assumption = max(0.0, min(1.0, global_assumption))

        report_data["final_summary"] = {
            "total_cases_evaluated": total_cases_run,
            "aggregate_raw_metrics": global_metrics,
            "aggregate_percentages": {
                "clarification_precision": f"{global_precision:.2%}",
                "clarification_recall": f"{global_recall:.2%}",
                "error_detection_rate": f"{global_edr:.2%}",
                "assumption_rate": f"{global_assumption:.2%}",
                "memory_consistency": f"{global_memory:.2%}"
            },
            "total_llm_usage": final_llm_stats
        }
        
        final_output_path = os.path.join(test_runs_dir, output_summary_file)
        
        with open(final_output_path, "w") as f:
            json.dump(report_data, f, indent=4)
            
        print("\n" + "="*60)
        print(f"CHAT EVALUATION COMPLETE FOR {total_cases_run} CASES")
        print(f"REPORT SAVED TO: {final_output_path}")
        print("="*60)
    
    if len(missing_generated) > 0:
        print("\nWARNING: The agent did not run the following cases:")
        print(missing_generated)

if __name__ == "__main__":
    main()