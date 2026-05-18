import os
import json

MODELS = [
    # "CLAUDE_SONNET_4.0",
    # "CLAUDE_HAIKU_3.5",
    "CLAUDE_SONNET_4.6",
    "GEMINI_2_5_PRO",
    "GEMINI_2_5_FLASH",
    "GPT_5_2",
    "OLLAMA"
]

MODES = ["Inconsistency", "Disambiguation"]


def is_valid_and_non_empty(filepath):
    """
    Checks if a file exists and is non-empty. 
    Supports .json (checks for valid data) and .txt (checks for text).
    """

    if not os.path.exists(filepath):
        return False, "File missing"

    filename = os.path.basename(filepath)
    ext = os.path.splitext(filepath)[1].lower()

    # Handle JSON (Always needs to be non-empty and valid)
    if ext == '.json':
        if os.path.getsize(filepath) == 0:
            return False, "JSON file is 0 bytes"
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return (True, "OK") if data else (False, "JSON is empty")
        except json.JSONDecodeError:
            return False, "Invalid JSON"

    # Handle TXT (Allow 0 bytes IF it is the conversation log)
    elif ext == '.txt':
        if os.path.getsize(filepath) == 0 and filename != "conversation_log.txt":
            return False, "Text file is 0 bytes"

        # If it's not 0 bytes, check for whitespace
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content and filename != "conversation_log.txt":
                return False, "Text file contains only whitespace"

        return True, "OK"

    return False, "Unsupported file type"


def check_for_llm_failures(filepath, mode):
    """
    Scans valid JSON files for specific LLM failure/fallback values 
    in their reasoning or status fields.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return False, ""  # Handled by is_valid_and_non_empty

    # Safe retrieval for nested dicts
    def get_reasoning(key):
        if key in data and isinstance(data[key], dict):
            return data[key].get("reasoning", "")
        return ""

    filename = os.path.basename(filepath)

    # Chat Score Checks
    if filename == f"llm_judge_{mode.lower()}_chat_score.json":
        # Mode-specific chat keys
        if mode == "Disambiguation":
            r_miss = get_reasoning("questions_about_missing_entities")
            if r_miss in ["JSON parsing failed.", "Not provided by LLM."]:
                return True, f"questions_about_missing_entities: {r_miss}"
        elif mode == "Inconsistency":
            r_inc = get_reasoning("questions_about_inconsistencies")
            if r_inc in ["JSON parsing failed.", "Not provided by LLM."]:
                return True, f"questions_about_inconsistencies: {r_inc}"

            r_warn = get_reasoning("explicit_warnings")
            if r_warn == "Not provided by LLM.":
                return True, f"explicit_warnings: {r_warn}"

        # Common chat keys
        r_rel = get_reasoning("relevant_questions")
        if r_rel == "Not provided by LLM.":
            return True, f"relevant_questions: {r_rel}"

        r_mem = get_reasoning("memory_consistency")
        if r_mem == "Not provided by LLM.":
            return True, f"memory_consistency: {r_mem}"

    # Intent Score Checks
    elif filename == "llm_judge_intent_score.json":
        r_intent = data.get("reasoning", "")
        if r_intent in ["No generated text found.", "Not provided by LLM.", "Failed to parse evaluation response from LLM."]:
            return True, f"Intent reasoning: {r_intent}"

    # Evaluator Score Checks (Inconsistency or Disambiguation)
    elif filename == f"llm_judge_{mode.lower()}_score.json":
        r_eval = data.get("reasoning", "")
        if r_eval == "No generated text found.":
            return True, f"Evaluator reasoning: {r_eval}"
        if data.get("status") == "ERROR":
            return True, "Evaluator status: ERROR"

    return False, ""


def get_expected_case_ids(domain, mode):
    json_path = os.path.join(domain, f"{mode.lower()}_{domain}.json")
    if not os.path.exists(json_path):
        print(
            f"[!] Warning: Could not find source JSON dataset at {json_path}")
        return []

    with open(json_path, 'r') as f:
        cases_data = json.load(f)
        return [item["id"] for item in cases_data if "id" in item]


def run_sanity_check(domain, baseline=False):
    print(f"\n{'='*60}")
    print(f" STARTING DEEP SANITY CHECK FOR {domain.upper()}")
    print(f"{'='*60}")

    all_passed = True
    errors = []

    for mode in MODES:
        expected_case_ids = get_expected_case_ids(domain, mode)
        expected_count = len(expected_case_ids)
        expected_cases_set = set(expected_case_ids)

        if expected_count == 0:
            print(
                f"\nSkipping {mode} because no case IDs were found in the source JSON.\n")
            continue

        for model in MODELS:
            folder_name = f"{domain}/{model}_{mode}"
            if baseline:
                folder_name = f"{domain}_baseline/{model}_{mode}"
            print(
                f"Checking: {model}_{mode} (Expecting exactly {expected_count} targeted cases)")

            if not os.path.exists(folder_name):
                all_passed = False
                msg = f"[!] Missing entirely: {folder_name}"
                print(f"  {msg}")
                errors.append(msg)
                continue

            # Check Root Summary Files
            root_files = [
                "llm_judge_summary.json",
                "llm_judge_intent_summary.json",
                "llm_judge_chat_summary.json"
            ]
            if domain in ["fluids", "foam", "pde"]:
                if mode == "Disambiguation":
                    root_files.append("disambiguation_coverage_summary.json")

            for rf in root_files:
                rf_path = os.path.join(folder_name, rf)
                valid, msg = is_valid_and_non_empty(rf_path)

                if not valid:
                    all_passed = False
                    err_msg = f"[!] {folder_name}: {rf} - {msg}"
                    print(f"  {err_msg}")
                    errors.append(err_msg)
                else:
                    # Verify all expected case IDs are present in the summary JSON
                    try:
                        with open(rf_path, 'r', encoding='utf-8') as f:
                            summary_data = json.load(f)

                        found_cases = set()
                        if "cases" in summary_data:
                            for case_entry in summary_data["cases"]:
                                if "case_id" in case_entry:
                                    found_cases.add(case_entry["case_id"])

                        missing_in_summary = expected_cases_set - found_cases
                        if missing_in_summary:
                            all_passed = False
                            missing_list = sorted(list(missing_in_summary))
                            err_msg = f"[!] {folder_name}: {rf} is missing expected case IDs -> {missing_list}"
                            print(f"  {err_msg}")
                            errors.append(err_msg)

                    except json.JSONDecodeError:
                        all_passed = False
                        err_msg = f"[!] {folder_name}: {rf} - Corrupted JSON, cannot verify cases."
                        print(f"  {err_msg}")
                        errors.append(err_msg)
                    except Exception as e:
                        all_passed = False
                        err_msg = f"[!] {folder_name}: {rf} - Error reading cases for validation: {e}"
                        print(f"  {err_msg}")
                        errors.append(err_msg)

            # Check Individual Case Files
            mode_lower = mode.lower()
            expected_case_files = [
                f"llm_judge_{mode_lower}_score.json",
                f"llm_judge_{mode_lower}_chat_score.json",
                f"llm_judge_intent_score.json",
                "statistics.json",
                "generated_prompt.txt",
                "complete_prompt.txt",
                "conversation_log.txt"
            ]
            if domain in ["fluids", "foam", "pde"]:
                if mode == "Disambiguation":
                    root_files.append("disambiguation_coverage_score.json")

            missing_folders_count = 0
            missing_case_files = 0
            failed_llm_cases = 0
            for case_id in expected_case_ids:
                case_dir = os.path.join(folder_name, case_id)

                if not os.path.exists(case_dir):
                    all_passed = False
                    missing_folders_count += 1
                    err_msg = f"[!] {folder_name}: Missing expected folder for {case_id}"
                    errors.append(err_msg)
                    continue

                for cf in expected_case_files:
                    cf_path = os.path.join(case_dir, cf)
                    valid, msg = is_valid_and_non_empty(cf_path)
                    if not valid:
                        all_passed = False
                        missing_case_files += 1
                        err_msg = f"[!] {case_dir}: {cf} - {msg}"
                        errors.append(err_msg)
                    elif cf.endswith('.json'):
                        has_failure, fail_msg = check_for_llm_failures(
                            cf_path, mode)
                        if has_failure:
                            all_passed = False
                            failed_llm_cases += 1
                            err_msg = f"[!] LLM EVAL FAILURE -> {case_dir}/{cf} | {fail_msg}"
                            errors.append(err_msg)

            if missing_folders_count > 0:
                print(
                    f"  [!] Found {missing_folders_count} missing case folders that were expected by the JSON.")
            if missing_case_files > 0:
                print(
                    f"  [!] Found {missing_case_files} missing/empty/corrupt score files inside valid cases.")
            if failed_llm_cases > 0:
                print(
                    f"  [!] Found {failed_llm_cases} files with failed LLM reasonings/status.")
            if missing_folders_count == 0 and missing_case_files == 0 and failed_llm_cases == 0:
                print(
                    "  [OK] All targeted case files present, non-empty, and LLM evaluations succeeded.")
    # Final Report
    print(f"\n{'='*60}")
    dir = domain
    if baseline:
        dir = f"{domain}_baseline"
    print(f"SANITY CHECK SUMMARY for {dir}")
    print(f"{'='*60}")

    if all_passed:
        print("\nPERFECT! All models, targeted cases, and JSON files are present, valid, and non-empty.\n")
    else:
        print(f"\nFAILED. Found {len(errors)} issues:\n")
        # Print up to 20 errors
        for e in errors[:20]:
            print(e)
        if len(errors) > 20:
            print(f"...and {len(errors) - 20} more issues.")
        print("\nPlease fix the above errors before running your plotting scripts.\n")


if __name__ == "__main__":
    run_sanity_check("foam", baseline=False)
    run_sanity_check("foam", baseline=True)

    run_sanity_check("fluids", baseline=False)
    run_sanity_check("fluids", baseline=True)

    run_sanity_check("pde", baseline=False)
    run_sanity_check("pde", baseline=True)

    run_sanity_check("solMech", baseline=False)
    # run_sanity_check("solMech", baseline=True)

    run_sanity_check("solToolUse", baseline=False)
    # run_sanity_check("solToolUse", baseline=True)

    run_sanity_check("matSci", baseline=False)
    # run_sanity_check("matSci", baseline=True)

    run_sanity_check("matToolUse", baseline=False)
    # run_sanity_check("matToolUse", baseline=True)
