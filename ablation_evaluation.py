import json
import argparse

from config import Config
from llm_judge_disambiguation import LLMJudgeDisambiguation
from llm_judge_inconsistency import LLMJudgeInconsistency
from llm_judge_disambiguation_chat import LLMJudgeDisambiguationChat
from llm_judge_inconsistency_chat import LLMJudgeInconsistencyChat
from llm_judge_intent import LLMJudgeIntent


def safe_get(d, key, default=None):
    """Safely access a dictionary key and print a warning if it doesn't exist."""
    if not isinstance(d, dict):
        print(
            f"  [Warning] Expected dictionary for key '{key}', got {type(d).__name__}")
        return default
    if key not in d:
        print(
            f"  [Warning] Key '{key}' not found in dictionary! Falling back to default: {default}")
        return default
    return d[key]


def get_judges():
    return {
        "intent": LLMJudgeIntent(),
        "inc_task": LLMJudgeInconsistency(),
        "dis_task": LLMJudgeDisambiguation(),
        "inc_chat": LLMJudgeInconsistencyChat(),
        "dis_chat": LLMJudgeDisambiguationChat()
    }


def adapt_chat_format(conversation_turns):
    """
    Translates the JSON chat format into the format your Chat Judges expect:
    [{"Question_1": "...", "Answer_1": "..."}, ...]
    """
    adapted_chat = []
    for turn in conversation_turns:
        idx = safe_get(turn, "turn", 1)
        adapted_chat.append({
            f"Question_{idx}": safe_get(turn, "Q", ""),
            f"Answer_{idx}": safe_get(turn, "A", "")
        })
    return adapted_chat


def run_jury(input_json: str, judge_model: str, output_filename: str):
    with open(input_json, 'r') as f:
        data = json.load(f)

    llm_data = {
        "cases": []
    }

    print(f"\n{'='*50}")
    print(f"STARTING JURY RUN: {judge_model}")
    print(f"{'='*50}")

    judges = get_judges()

    for case in safe_get(data, "cases", []):
        case_id = safe_get(case, "case_id")
        task = safe_get(case, "task")
        print(f"Judging {case_id} ({task})...")

        has_error = False  # Flag to track if any judge fails

        # Extract inputs from the JSON
        planted_gt = safe_get(case, "planted_ground_truth", {})
        if task == "Disambiguation":
            incomplete_prompt = safe_get(
                planted_gt, "original_underspecified_prompt", "")
            ground_truth = safe_get(
                planted_gt, "ground_truth_complete_prompt", "")
        else:
            incomplete_prompt = safe_get(
                planted_gt, "original_inconsistent_prompt", "")
            ground_truth = safe_get(
                planted_gt, "ground_truth_correct_prompt", "")

        agent_output = safe_get(case, "agent_final_output", "")
        raw_chat = safe_get(case, "conversation_turns", [])
        parsed_chat = adapt_chat_format(raw_chat)

        # Setup Variables
        fr_val, cgr_val, ic_val, mc_val, cr_val, cp_val, dr_val = 0, 0, 0, 0, 0.0, 0.0, 0.0
        total_issues = 0
        addressed = 0

        # INTENT JUDGE
        try:
            intent_res = safe_get(judges, "intent").evaluate(
                generated_text=agent_output,
                ground_truth=ground_truth
            )
            intent_status = safe_get(intent_res, "status")
            if intent_status == "FAITHFUL":
                ic_val = 1
            elif intent_status == "ERROR":
                print(f"  [!] INTENT JUDGE ERROR on {case_id}")
                has_error = True
        except Exception as e:
            print(f"  [!] CRITICAL INTENT FAILURE on {case_id}: {e}")
            has_error = True
            intent_res = {}

        # TASK & CHAT JUDGES (By Mode)
        try:
            if task == "Inconsistency":
                inconsistencies = safe_get(
                    planted_gt, "planted_inconsistencies", [])
                total_issues = len(inconsistencies)

                # TASK JUDGE
                task_res = safe_get(judges, "inc_task").evaluate(
                    generated_text=agent_output,
                    inconsistencies=inconsistencies,
                    incomplete_prompt=incomplete_prompt,
                    correct_prompt=ground_truth
                )

                task_status = safe_get(task_res, "status")
                if task_status == "ERROR":
                    print(f"  [!] INC TASK ERROR on {case_id}")
                    has_error = True

                task_score = safe_get(task_res, "score")
                if task_score == "1/1":
                    fr_val = 1

                # CHAT JUDGE
                chat_res = safe_get(judges, "inc_chat").evaluate(
                    parsed_chat=parsed_chat,
                    inconsistencies=inconsistencies,
                    generated_prompt=agent_output
                )
                q_inc = safe_get(
                    chat_res, "questions_about_inconsistencies", {})
                addressed = safe_get(q_inc, "count", 0)

            else:  # Disambiguation
                missing_ents = safe_get(
                    planted_gt, "planted_missing_entities", [])
                total_issues = len(missing_ents)

                # TASK JUDGE
                task_res = safe_get(judges, "dis_task").evaluate(
                    ground_truth=ground_truth,
                    final_result=agent_output,
                    expected_missing=missing_ents
                )

                # A strict 1/1 means it recovered everything
                task_score = safe_get(task_res, "score")
                if task_score == f"{total_issues}/{total_issues}":
                    fr_val = 1

                # CHAT JUDGE
                chat_res = safe_get(judges, "dis_chat").evaluate(
                    parsed_chat=parsed_chat,
                    missing_entities=missing_ents,
                    generated_prompt=agent_output
                )
                q_miss = safe_get(
                    chat_res, "questions_about_missing_entities", {})
                addressed = safe_get(q_miss, "count", 0)

        except Exception as e:
            print(f"  [!] CRITICAL TASK/CHAT FAILURE on {case_id}: {e}")
            has_error = True
            task_res, chat_res = {}, {}

        # METRICS
        if not has_error:
            # MC (Memory Consistency)
            try:
                mem_const = safe_get(chat_res, "memory_consistency", {})
                is_consistent = safe_get(mem_const, "is_consistent", 0)
                if float(is_consistent) == 1.0:
                    mc_val = 1
            except:
                pass

            # CP (Clarification Precision)
            total_qs = len(parsed_chat)
            rel_qs_dict = safe_get(chat_res, "relevant_questions", {})
            relevant_qs = safe_get(rel_qs_dict, "count", 0)
            if total_qs > 0:
                cp_val = max(0.0, min(1.0, relevant_qs / total_qs))

            # CR (Clarification Recall) - Fraction
            if total_issues > 0:
                cr_val = max(0.0, min(1.0, addressed / total_issues))

            # CGR (Conversation Grounded Resolution) - Still Binary (Strict)
            if fr_val == 1 and addressed >= total_issues:
                cgr_val = 1

        llm_scores = {
            "FR": {"value": fr_val, "reason": "Task judge evaluated all issues as resolved." if fr_val else "Failed to resolve all issues."},
            "CGR": {"value": cgr_val, "reason": f"FR is {fr_val}, Addressed {addressed}/{total_issues}"},
            "IC": {"value": ic_val, "reason": f"Intent Judge status was {safe_get(intent_res, 'status', 'ERROR')}"},
            "MC": {"value": mc_val, "reason": f"Chat Consistency evaluation returned {mc_val}"},
            "CR": {"value": round(cr_val, 4), "reason": f"Addressed {addressed} out of {total_issues} issues."},
            "CP": {"value": round(cp_val, 4), "reason": f"Asked {relevant_qs} relevant questions out of {total_qs} total questions."}
        }
        if task == "Inconsistency":
            exp_warn = safe_get(chat_res, "explicit_warnings", {})
            warnings = safe_get(exp_warn, "count", 0)
            dr_val = 0.0
            if total_issues > 0:
                dr_val = max(0.0, min(1.0, warnings / total_issues))
            llm_scores["DR"] = {
                "value": round(dr_val, 4),
                "reason": f"Explicit warnings {warnings} were given out of {total_issues}."
            }
        new_case = case.copy()
        new_case["llm_scores"] = llm_scores

        # Inject an Error Flag if anything broke
        if has_error:
            new_case["ERROR_FLAG"] = True
            new_case["rater_notes"] = "One or more judges failed or returned ERROR during parsing."

        new_case.pop("human_scores", None)
        new_case.pop("annotation_time_minutes", None)
        if not has_error:
            new_case.pop("rater_notes", None)

        safe_get(llm_data, "cases").append(new_case)

    with open(output_filename, 'w') as f:
        json.dump(llm_data, f, indent=2)

    print(f"\nDone: Saved results to {output_filename}")


if __name__ == "__main__":
    # JUDGE ABLATION:
    # python ablation_evaluation.py --model bedrock_4_6_sonnet --input judge_ablation/human_annotation_form_annotated.json --output judge_ablation/bedrock_4_6_sonnet_results.json
    # python ablation_evaluation.py --model openai_gpt5 --input judge_ablation/human_annotation_form_annotated.json --output judge_ablation/openai_gpt5_results.json

    # USER SIMULATED LLM ABLATION:
    # python ablation_evaluation.py --model codex_gpt5 --input user_llm_ablation/GEMINI_PRO/gemini_pro_results.json --output user_llm_ablation/GEMINI_PRO/gemini_pro_results.json
    # python ablation_evaluation.py --model codex_gpt5 --input user_llm_ablation/GPT_5_2/openai_gpt5_results.json --output user_llm_ablation/GPT_5_2/openai_gpt5_results.json
    # python ablation_evaluation.py --model codex_gpt5 --input user_llm_ablation/SONNET_4_6/bedrock_4_6_sonnet_results.json --output user_llm_ablation/SONNET_4_6/bedrock_4_6_sonnet_results.json

    # SYSTEM PROMPT ABLATION:
    # python ablation_evaluation.py --model codex_gpt5 --input prompt_ablation/OPTION_1/paraphrased_option_1_results.json --output prompt_ablation/OPTION_1/paraphrased_option_1_results.json
    # python ablation_evaluation.py --model codex_gpt5 --input prompt_ablation/OPTION_2/paraphrased_option_2_results.json --output prompt_ablation/OPTION_2/paraphrased_option_2_results.json

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model = args.model
    if model in Config.LLM_PROFILES:
        print(f"Overriding default Config. Switching to: {model}")
        Config.ACTIVE_PROFILE = model
        Config.active_conf = Config.LLM_PROFILES[model]
        Config.model_provider = safe_get(Config.active_conf, "provider")
        Config.model_version = safe_get(Config.active_conf, "model_version")
        Config.temperature = safe_get(Config.active_conf, "temperature")
    else:
        raise ValueError(f"Model '{model}' not found in Config.LLM_PROFILES")

    input_json = args.input
    output_filename = args.output
    run_jury(input_json, model, output_filename)
