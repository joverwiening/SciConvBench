#########################################################################################################
#################################### CALCULATING current COST of bedrock NORMAL RUNS ####################
#########################################################################################################

# import json
# from pathlib import Path


# base_dirs = [
#     # "matSci",
#     # "matToolUse",
#     # "solMech",
#     # "solToolUse",
#     # "pde",
#     # "pde_baseline",
#     # "foam",
#     # "foam_baseline",
#     "fluids",
#     # "fluids_baseline"
# ]
# models = [
#     # "CLAUDE_SONNET_4.0_Inconsistency",
#     # "CLAUDE_SONNET_4.0_Disambiguation",
#     # "CLAUDE_HAIKU_3.5_Inconsistency",
#     # "CLAUDE_HAIKU_3.5_Disambiguation",
#     # "CLAUDE_SONNET_4.6_Inconsistency",
#     # "CLAUDE_SONNET_4.6_Disambiguation",
#     # "CLAUDE_SONNET_.0_Inconsistency",
#     # "CLAUDE_SONNET_4.0_Disambiguation",
#     "QWEN_3_5_Disambiguation",
#     "QWEN_3_5_Inconsistency"
# ]

# input = 0
# output = 0
# for dir_path in base_dirs:
#     base_dir = Path(dir_path)

#     if base_dir.exists() and base_dir.is_dir():
#         for model in models:

#             # # Skip Claude 4.6 models if the current directory is "solMech"
#             # if dir_path == "solMech" and model in [
#             #     "CLAUDE_SONNET_4.6_Inconsistency",
#             #     "CLAUDE_SONNET_4.6_Disambiguation"
#             # ]:
#             #     continue

#             sub_dir = base_dir / model

#             if sub_dir.is_dir():
#                 for case_dir in sub_dir.glob("case_*"):
#                     if case_dir.is_dir():
#                         stats_file = case_dir / "statistics.json"

#                         if stats_file.exists():
#                             try:
#                                 stats_data = json.loads(
#                                     stats_file.read_text(encoding="utf-8"))

#                                 usage = stats_data.get("agent_llm_usage", {})

#                                 input += usage.get("prompt_tokens", 0)
#                                 output += usage.get("completion_tokens", 0)

#                             except json.JSONDecodeError:
#                                 print(f"ERROR: Could not parse {stats_file}")
#                         else:
#                             print(
#                                 f"SOMETHING IS WRONG (No stats.json found in {case_dir})")

#     else:
#         print(f"Skipping (not found): {dir_path}")

# print(f"Total Adjusted Input Tokens:  {0.53*input/1000000}")
# print(f"Total Adjusted Output Tokens: {2.66*output/1000000}")

#########################################################################################################
#################################### CALCULATING COST OF JUDGE RUNS ####################################
#########################################################################################################


# import json
# from pathlib import Path

# base_dirs = ["matSci", "matToolUse", "solMech", "solToolUse"]
# existing_models = [
#     "CLAUDE_SONNET_4.6_Inconsistency",
#     "CLAUDE_SONNET_4.6_Disambiguation",
#     "GEMINI_2_5_PRO_Disambiguation",
#     "GEMINI_2_5_PRO_Inconsistency",
#     "GPT_5_2_Disambiguation",
#     "GPT_5_2_Inconsistency"
# ]
# target_files = [
#     "llm_judge_summary.json",
#     "llm_judge_chat_summary.json",
#     "llm_judge_intent_summary.json"
# ]

# INPUT_PRICE = 3.00
# OUTPUT_PRICE = 15.00

# known_input_tokens = 0
# known_output_tokens = 0
# completed_runs = 0

# for dir_path in base_dirs:
#     base_dir = Path(dir_path)
#     if base_dir.exists() and base_dir.is_dir():
#         for model in existing_models:
#             sub_dir = base_dir / model
#             print(sub_dir)
#             if sub_dir.exists() and sub_dir.is_dir():
#                 run_has_files = False
#                 for file_name in target_files:
#                     stats_file = sub_dir / file_name
#                     if stats_file.exists():
#                         run_has_files = True
#                         try:
#                             stats_data = json.loads(stats_file.read_text(encoding="utf-8"))
#                             usage = stats_data.get("final_summary", {}).get("total_llm_usage", {})
#                             known_input_tokens += usage.get("prompt_tokens", 0)
#                             known_output_tokens += usage.get("completion_tokens", 0)
#                         except json.JSONDecodeError:
#                             pass

#             # If we found at least one summary file, count this as a completed run
#             if run_has_files:
#                 completed_runs += 1
# print(completed_runs)
# avg_input_per_run = known_input_tokens / completed_runs
# avg_output_per_run = known_output_tokens / completed_runs
# # # 4 normal and 4 baselines  = 8 dir, 10 model variants (dis and inconsistency for 2 gemini + 1 bedrock + 1 gpt + 1 ollama), 10 * 8 = 80 total runs.
# # # i completed 6 * 4 = 24 , i didnt judge any baseline but i judged 8 extra from sol mech, so
# # # The Math: 80 Total Runs - 24  -  8 Completed Runs = 48 Missing Runs
# missing_runs = 48

# raw_missing_input = avg_input_per_run * missing_runs
# raw_missing_output = avg_output_per_run * missing_runs


# est_input_cost = (raw_missing_input / 1_000_000) * INPUT_PRICE
# est_output_cost = (raw_missing_output / 1_000_000) * OUTPUT_PRICE
# total_est_cost = est_input_cost + est_output_cost

# Output Results
# print("="*50)
# print(" MISSING DATA ESTIMATION")
# print("="*50)
# print(f"Completed Runs Found: {completed_runs}")
# # print(f"Missing Runs Needed:  {missing_runs}")
# # print(f"Applied Pricing Tier: {tier_name}")
# print("-"*50)
# print(f"Projected Input Tokens:  {known_input_tokens:,.0f}")
# print(f"Projected Output Tokens: {known_output_tokens:,.0f}")
# print("-"*50)
# print(f"Estimated Input Cost:  ${est_input_cost:.2f} (@ ${INPUT_PRICE}/1M)")
# print(f"Estimated Output Cost: ${est_output_cost:.2f} (@ ${OUTPUT_PRICE}/1M)")
# print(f"TOTAL PROJECTED COST:  ${total_est_cost:.2f}")
# print("="*50)

##########################################################################################
#################################### GET COMMAND #########################################
##########################################################################################


# from pathlib import Path
# import json
# filepath = "out.txt"
# with open(filepath, 'r') as file:
#     lines = file.readlines()
# commands = [line.strip() for line in lines if line.strip()]
# joined_commands = "; ".join(commands)
# final_output = f"(export PYTHONUNBUFFERED=1; {joined_commands}) > out.txt 2>&1"
# with open(filepath, 'w') as file:
#     file.write(final_output)


##########################################################################################################################################
#################################### CHECK MISSING ENTITIES OR INCONSISTENCIES AVERAGE PER DOMAIN #########################################
##########################################################################################################################################


# import json
# from pathlib import Path

# # Setup our individual domains and macro-domain groupings
# DOMAINS = ['fluids', 'foam', 'pde', 'solMech', 'solToolUse', 'matSci', 'matToolUse']

# MACRO_DOMAINS = {
#     'Fluid mechanics':   ['fluids', 'foam'],
#     'Solid mechanics':   ['solMech', 'solToolUse'],
#     'Materials science': ['matSci', 'matToolUse'],
#     'PDEs':              ['pde'],
# }

# def calculate_averages():
#     # {domain: {'disambiguation': (total_len, num_cases), 'inconsistency': (total_len, num_cases)}}
#     domain_stats = {}

#     for domain in DOMAINS:
#         domain_stats[domain] = {
#             'disambiguation': {'total_len': 0, 'cases': 0},
#             'inconsistency': {'total_len': 0, 'cases': 0}
#         }

#         # Calculate Disambiguation (missing_entities)
#         disambiguation_file = Path(domain) / f"disambiguation_{domain}.json"
#         if disambiguation_file.exists():
#             with open(disambiguation_file, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#                 for case in data:
#                     domain_stats[domain]['disambiguation']['cases'] += 1
#                     domain_stats[domain]['disambiguation']['total_len'] += len(case.get('missing_entities', []))

#         # Calculate Inconsistency (inconsistencies)
#         inconsistency_file = Path(domain) / f"inconsistency_{domain}.json"
#         if inconsistency_file.exists():
#             with open(inconsistency_file, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#                 for case in data:
#                     domain_stats[domain]['inconsistency']['cases'] += 1
#                     domain_stats[domain]['inconsistency']['total_len'] += len(case.get('inconsistencies', []))

#     print(f"{'='*50}\nINDIVIDUAL DOMAIN AVERAGES\n{'='*50}")
#     for domain in DOMAINS:
#         d_stats = domain_stats[domain]['disambiguation']
#         i_stats = domain_stats[domain]['inconsistency']

#         # Calculate averages
#         d_avg = d_stats['total_len'] / d_stats['cases'] if d_stats['cases'] > 0 else 0
#         i_avg = i_stats['total_len'] / i_stats['cases'] if i_stats['cases'] > 0 else 0

#         print(f"Domain: {domain}")
#         print(f"  Disambiguation (missing_entities avg): {d_avg:.2f} (over {d_stats['cases']} cases)")
#         print(f"  Inconsistency (inconsistencies avg):   {i_avg:.2f} (over {i_stats['cases']} cases)")
#         print("-" * 30)

#     print(f"\n{'='*50}\nCOMBINED MACRO-DOMAIN AVERAGES\n{'='*50}")
#     for macro_name, sub_domains in MACRO_DOMAINS.items():
#         # Aggregate stats across all sub-domains that belong to this macro-domain
#         d_total_len = sum(domain_stats[sub]['disambiguation']['total_len'] for sub in sub_domains)
#         d_total_cases = sum(domain_stats[sub]['disambiguation']['cases'] for sub in sub_domains)

#         i_total_len = sum(domain_stats[sub]['inconsistency']['total_len'] for sub in sub_domains)
#         i_total_cases = sum(domain_stats[sub]['inconsistency']['cases'] for sub in sub_domains)

#         # Calculate macro averages
#         d_avg = d_total_len / d_total_cases if d_total_cases > 0 else 0
#         i_avg = i_total_len / i_total_cases if i_total_cases > 0 else 0

#         print(f"Macro Domain: {macro_name} (Includes: {', '.join(sub_domains)})")
#         print(f"  Disambiguation avg: {d_avg:.2f} (over {d_total_cases} combined cases)")
#         print(f"  Inconsistency avg:  {i_avg:.2f} (over {i_total_cases} combined cases)")
#         print("-" * 30)

# if __name__ == "__main__":
#     calculate_averages()

####################################################################################################################
#################################### GET PARTIAL RESOLUTION AVG PER DOMAIN #########################################
####################################################################################################################

# import os
# import json

# MODELS = ['CLAUDE_SONNET_4.6', 'GPT_5_2', 'GEMINI_2_5_PRO']
# MODEL_LABELS = ['Claude 4.6', 'GPT-5.2', 'Gemini 2.5 Pro']
# MODES = ['Disambiguation', 'Inconsistency']

# MACRO_DOMAINS = {
#     'Fluid mechanics':   ['fluids', 'foam'],
#     'Solid mechanics':   ['solMech', 'solToolUse'],
#     'Materials science': ['matSci', 'matToolUse'],
#     'PDEs':              ['pde'],
# }

# def extract_counts(domain, model, mode):
#     """
#     Returns (total_recovered_hits, total_expected_issues) across the entire domain.
#     """
#     json_path = os.path.join(domain, f"{mode.lower()}_{domain}.json")
#     if not os.path.exists(json_path):
#         print("SOMETHIGN IS WRONG")
#         return 0, 0

#     with open(json_path, 'r') as f:
#         cases_data = json.load(f)
#         expected_cases = [item["id"] for item in cases_data if "id" in item]

#     folder_path = os.path.join(domain, f"{model}_{mode}")

#     total_resolved = 0
#     total_expected = 0

#     for case_id in expected_cases:
#         case_folder = os.path.join(folder_path, case_id)
#         if not os.path.exists(case_folder):
#             print("SOMETHIGN IS WRONG")
#             continue

#         # Get Recovered from Task Judge
#         task_file = os.path.join(case_folder, f"llm_judge_{mode.lower()}_score.json")
#         recovered = 0
#         expected = 0
#         if os.path.exists(task_file):
#             with open(task_file, 'r') as f:
#                 t_data = json.load(f)
#                 p_score = t_data.get("partial_score", "0/0")
#                 try:
#                     num, den = map(int, p_score.split('/'))
#                     recovered = num
#                     expected = den
#                 except:
#                     pass


#         total_resolved += recovered
#         total_expected += expected

#     return total_resolved, total_expected

# def main():
#     for mode in MODES:
#         print(f"\n{'='*105}")
#         print(f"MODE: {mode.upper()} - PARTIAL RESOLUTION (Resolved / Total Planted)")
#         print(f"{'='*105}")

#         # Build Table Headers
#         headers = f"{'Macro Domain':<18} | {'Domain':<12} | " + " | ".join([f"{m:<18}" for m in MODEL_LABELS]) + " | Avg %"
#         print(headers)
#         print("-" * len(headers))

#         grand_totals = {m: {'num': 0, 'den': 0} for m in MODELS}

#         for macro_domain, domains in MACRO_DOMAINS.items():
#             macro_totals = {m: {'num': 0, 'den': 0} for m in MODELS}

#             for i, domain in enumerate(domains):
#                 # Only print the Macro Domain name on the first row of its group
#                 md_label = macro_domain if i == 0 else ""
#                 row_str = f"{md_label:<18} | {domain:<12} | "

#                 domain_pcts = []
#                 for model in MODELS:
#                     recovered, exp = extract_counts(domain, model, mode)

#                     # Accumulate for Macro Domain and Grand Total
#                     macro_totals[model]['num'] += recovered
#                     macro_totals[model]['den'] += exp
#                     grand_totals[model]['num'] += recovered
#                     grand_totals[model]['den'] += exp

#                     if exp > 0:
#                         pct = (recovered / exp) * 100
#                         domain_pcts.append(pct)
#                         val_str = f"{recovered}/{exp} ({pct:.1f}%)"
#                     else:
#                         val_str = "0/0 (N/A)"

#                     row_str += f"{val_str:<18} | "

#                 # 3-Model Average per domain
#                 if domain_pcts:
#                     avg_pct = sum(domain_pcts) / len(domain_pcts)
#                     row_str += f"{avg_pct:.1f}%"
#                 else:
#                     print("SOMETHIGN IS WRONG")
#                     row_str += "N/A"

#                 print(row_str)

#             # MACRO DOMAIN ROLL-UP ROW
#             macro_row = f"{' '*18} | {'[MACRO AVG]':<12} | "
#             macro_pcts = []
#             for model in MODELS:
#                 num = macro_totals[model]['num']
#                 den = macro_totals[model]['den']
#                 if den > 0:
#                     pct = (num / den) * 100
#                     macro_pcts.append(pct)
#                     val_str = f"{num}/{den} ({pct:.1f}%)"
#                 else:
#                     print("SOMETHIGN IS WRONG")
#                     val_str = "0/0 (N/A)"
#                 macro_row += f"{val_str:<18} | "

#             if macro_pcts:
#                 avg_pct = sum(macro_pcts) / len(macro_pcts)
#                 macro_row += f"{avg_pct:.1f}%"
#             else:
#                 print("SOMETHIGN IS WRONG")
#                 macro_row += "N/A"

#             print(macro_row)
#             print("-" * len(headers))

#         # GRAND TOTAL ROLL-UP ROW
#         grand_row = f"{'**ALL DOMAINS**':<18} | {'[GRAND AVG]':<12} | "
#         grand_pcts = []
#         for model in MODELS:
#             num = grand_totals[model]['num']
#             den = grand_totals[model]['den']
#             if den > 0:
#                 pct = (num / den) * 100
#                 grand_pcts.append(pct)
#                 val_str = f"{num}/{den} ({pct:.1f}%)"
#             else:
#                 print("SOMETHIGN IS WRONG")
#                 val_str = "0/0 (N/A)"
#             grand_row += f"{val_str:<18} | "

#         if grand_pcts:
#             avg_pct = sum(grand_pcts) / len(grand_pcts)
#             grand_row += f"{avg_pct:.1f}%"
#         else:
#             print("SOMETHIGN IS WRONG")
#             grand_row += "N/A"

#         print(grand_row)
#         print("=" * len(headers) + "\n")

# if __name__ == "__main__":
#     main()


####################################################################################################################
#################################### FIND WRONG CONVERSATION LOG FILES (THINKING...) ###############################
####################################################################################################################
# ALSO MAKE SURE ALL EMPTY LOGS == ONLY 1 TURN NEEDED

# import json
# from pathlib import Path

# base_dirs = [
#     "pde",
#     "pde_baseline",
#     "foam",
#     "foam_baseline",
#     "fluids",
#     "fluids_baseline",
#     "matSci",
#     "matSci_baseline",
#     "matToolUse",
#     "matToolUse_baseline",
#     "solMech",
#     "solMech_baseline",
#     "solToolUse",
#     "solToolUse_baseline"
# ]

# total_cases = 0
# warnings_found = 0

# for dir_path in base_dirs:
#     base_dir = Path(dir_path)

#     if base_dir.exists() and base_dir.is_dir():
#         for sub_dir in base_dir.iterdir():
#             if sub_dir.is_dir():

#                 for case_dir in sub_dir.glob("case_*"):
#                     if case_dir.is_dir():
#                         total_cases += 1
#                         log_file = case_dir / "conversation_log.txt"
#                         stats_file = case_dir / "statistics.json"

#                         if not log_file.exists():
#                             print(f"NOT FOUND: {log_file}")
#                         else:
#                             content = log_file.read_text(encoding="utf-8").strip()

#                             # If the conversation log is empty
#                             if not content:
#                                 if stats_file.exists():
#                                     try:
#                                         # Parse the JSON data
#                                         stats_data = json.loads(stats_file.read_text(encoding="utf-8"))

#                                         # Safely get turns_needed (defaults to None if missing)
#                                         turns_needed = stats_data.get("turns_needed")

#                                         if turns_needed is not None and turns_needed != 1:
#                                             print(f"WARNING: Empty log but turns_needed={turns_needed} -> {case_dir}")
#                                             warnings_found += 1
#                                         # else:
#                                         #     print(f"EMPTY FILE (Turns=1): {log_file}")

#                                     except json.JSONDecodeError:
#                                         print(f"ERROR: Could not parse {stats_file}")
#                                 else:
#                                     print(f"EMPTY FILE (No stats.json found): {log_file}")

#                             else:
#                                 target_pattern = "a: thinking..."
#                                 pattern_count = content.lower().count(target_pattern)

#                                 if pattern_count > 0:
#                                     print(f"FOUND {pattern_count} 'Thinking...' pair(s) in: {log_file}")
#                                 else:
#                                     pass # Optional: print(f"NO MATCH: File exists but no 'Thinking...' pattern in: {log_file}")
#     else:
#         print(f"Skipping (not found): {dir_path}")

# print(f"\nAll directories processed! Total cases checked: {total_cases}")
# if warnings_found > 0:
#     print(f"Total Warnings: {warnings_found}")


######################################################################################
#################################### FIND EASY CASES #################################
######################################################################################


# import os
# import json
# import glob

# MAJOR_MODELS = [
#     "CLAUDE_SONNET_4.6",
#     "GEMINI_2_5_PRO",
#     "GEMINI_2_5_FLASH"
#     "GPT_5_2"
# ]

# MODES = ["Inconsistency", "Disambiguation"]

# def trim_dataset(domain, mode, easy_cases):
#     """
#     Reads the original JSON file, removes the easy cases,
#     and saves a new '_hard.json' file.
#     """
#     # Assuming your original files are named "disambiguation.json" and "inconsistency.json"
#     source_file = os.path.join(domain, f"{mode.lower()}_{domain}.json")
#     output_file = os.path.join(domain, f"{mode.lower()}_trimmed_{domain}.json")

#     if not os.path.exists(source_file):
#         print(f" [!] Could not find original dataset at {source_file} to trim.")
#         return

#     with open(source_file, 'r') as f:
#         original_data = json.load(f)

#     # Filter out the easy cases
#     hard_data = [case for case in original_data if case.get("id") not in easy_cases]

#     # Save the new trimmed dataset
#     with open(output_file, 'w') as f:
#         json.dump(hard_data, f, indent=4)

#     print(f" [+] Trimmed dataset saved to: {output_file}")
#     print(f"     (Original cases: {len(original_data)} | New 'Hard' cases: {len(hard_data)})")

# def find_easy_cases(domain):
#     print("="*60)
#     print(f"ANALYZING {domain.upper()} DATASET")
#     print("="*60)

#     for mode in MODES:
#         all_case_ids = set()
#         for model in MAJOR_MODELS:
#             folder_path = os.path.join(domain, f"{model}_{mode}")
#             if os.path.exists(folder_path):
#                 cases = [os.path.basename(c) for c in glob.glob(os.path.join(folder_path, "case_*"))]
#                 all_case_ids.update(cases)

#         all_case_ids = sorted(list(all_case_ids))
#         total_cases = len(all_case_ids)

#         easy_cases = []

#         # Check each case to see if ALL major models passed it perfectly
#         for case_id in all_case_ids:
#             solved_by_all = True

#             for model in MAJOR_MODELS:
#                 case_folder = os.path.join(domain, f"{model}_{mode}", case_id)

#                 # If a model didn't even run this case, we can't say it's solved by all
#                 if not os.path.exists(case_folder):
#                     solved_by_all = False
#                     break

#                 # Check Task Specific Judge (Strict Score)
#                 task_file = os.path.join(case_folder, f"llm_judge_{mode.lower()}_score.json")
#                 task_strict = "0/1"
#                 if os.path.exists(task_file):
#                     with open(task_file, 'r') as f:
#                         task_strict = json.load(f).get("strict_score", "0/1")

#                 # Check Intent Judge (Faithfulness)
#                 intent_file = os.path.join(case_folder, "llm_judge_intent_score.json")
#                 intent_status = "UNFAITHFUL"
#                 if os.path.exists(intent_file):
#                     with open(intent_file, 'r') as f:
#                         intent_status = json.load(f).get("status", "UNFAITHFUL")

#                 # If ANY major model failed either test, this is NOT an easy case
#                 if task_strict != "1/1" or intent_status != "FAITHFUL":
#                     solved_by_all = False
#                     break

#             # If we made it through all models and they all passed, it's an easy case
#             if solved_by_all:
#                 easy_cases.append(case_id)

#         # Print the Statistics
#         print(f"\n   MODE: {mode.upper()}")
#         print(f"Total Cases Found: {total_cases}")
#         print(f"Cases solved perfectly by ALL major models: {len(easy_cases)}")

#         if easy_cases:
#             percentage = (len(easy_cases) / total_cases) * 100
#             print(f"Dataset Easy Percentage: {percentage:.1f}%")
#             print("EASY CASE IDs (the ones to skip):")
#             for i in range(0, len(easy_cases), 10):
#                 print(*easy_cases[i:i+10])

#             # Call the new function to create the trimmed JSON files
#             trim_dataset(domain, mode, easy_cases)

#         else:
#             print("No cases were solved perfectly by all major models. The dataset is already tough!")

# if __name__ == "__main__":
#     # You can change these domains as needed
#     find_easy_cases("matSci")
#     find_easy_cases("matToolUse")
#     find_easy_cases("solMech")
#     find_easy_cases("solToolUse")

######################################################################################
#################################### REBUILD SUMMARIES ###############################
######################################################################################
# this script makes sure the summary files are up-to-date with the correct cases per domain
# also if you re-judge some cases again, you can run this script to update the summaries.

# import json
# from pathlib import Path

# # Modify this list to include only the directories you want to process
# TARGET_DIRS = [
# "fluids/QWEN_3_Inconsistency",
# "fluids/QWEN_3_Disambiguation",
# "foam/QWEN_3_Inconsistency",
# "foam/QWEN_3_Disambiguation",
# "pde/QWEN_3_Inconsistency",
# "pde/QWEN_3_Disambiguation",
# "matSci/QWEN_3_Inconsistency",
# "matSci/QWEN_3_Disambiguation",
# 'solMech/QWEN_3_Inconsistency',
# "solMech/QWEN_3_Disambiguation",
# "solToolUse/QWEN_3_Disambiguation",
# "solToolUse/QWEN_3_Inconsistency",
# "fluids/KIMI_2_5_Inconsistency",
# "fluids/KIMI_2_5_Disambiguation",
# "foam/KIMI_2_5_Inconsistency",
# "foam/KIMI_2_5_Disambiguation",
# "pde/KIMI_2_5_Inconsistency",
# "pde/KIMI_2_5_Disambiguation",
# "matSci/KIMI_2_5_Inconsistency",
# "matSci/KIMI_2_5_Disambiguation",
# 'solMech/KIMI_2_5_Inconsistency',
# "solMech/KIMI_2_5_Disambiguation",
# "solToolUse/KIMI_2_5_Disambiguation",
# "solToolUse/KIMI_2_5_Inconsistency",

# "fluids/DEEPSEEK_V3_Inconsistency",
# "fluids/DEEPSEEK_V3_Disambiguation",
# "foam/DEEPSEEK_V3_Inconsistency",
# "foam/DEEPSEEK_V3_Disambiguation",
# "pde/DEEPSEEK_V3_Inconsistency",
# "pde/DEEPSEEK_V3_Disambiguation",
# "matSci/DEEPSEEK_V3_Inconsistency",
# "matSci/DEEPSEEK_V3_Disambiguation",
# 'solMech/DEEPSEEK_V3_Inconsistency',
# "solMech/DEEPSEEK_V3_Disambiguation",
# "solToolUse/DEEPSEEK_V3_Disambiguation",
# "solToolUse/DEEPSEEK_V3_Inconsistency"
# ]
#     "clamber/OLLAMA_Disambiguation",
#     "clamber/GEMINI_2_5_PRO_Disambiguation",
#     "solToolUse/OLLAMA_Inconsistency",
#     "solToolUse/CLAUDE_SONNET_4.6_Inconsistency",
#     "solToolUse/GEMINI_2_5_PRO_Inconsistency",
#     "solToolUse/GEMINI_2_5_FLASH_Inconsistency",
#     "solToolUse/GPT_5_2_Inconsistency",
#     "solToolUse/OLLAMA_Disambiguation",
#     "solToolUse/CLAUDE_SONNET_4.6_Disambiguation",
#     "solToolUse/GEMINI_2_5_PRO_Disambiguation",
#     "solToolUse/GEMINI_2_5_FLASH_Disambiguation",
#     "solToolUse/GPT_5_2_Disambiguation",

#     "matSci/OLLAMA_Inconsistency",
#     "matSci/CLAUDE_SONNET_4.6_Inconsistency",
#     "matSci/GEMINI_2_5_PRO_Inconsistency",
#     "matSci/GEMINI_2_5_FLASH_Inconsistency",
#     "matSci/GPT_5_2_Inconsistency",
#     "matSci/OLLAMA_Disambiguation",
#     "matSci/CLAUDE_SONNET_4.6_Disambiguation",
#     "matSci/GEMINI_2_5_PRO_Disambiguation",
#     "matSci/GEMINI_2_5_FLASH_Disambiguation",
#     "matSci/GPT_5_2_Disambiguation",

#     "matToolUse/OLLAMA_Inconsistency",
#     "matToolUse/CLAUDE_SONNET_4.6_Inconsistency",
#     "matToolUse/GEMINI_2_5_PRO_Inconsistency",
#     "matToolUse/GEMINI_2_5_FLASH_Inconsistency",
#     "matToolUse/GPT_5_2_Inconsistency",
#     "matToolUse/OLLAMA_Disambiguation",
#     "matToolUse/CLAUDE_SONNET_4.6_Disambiguation",
#     "matToolUse/GEMINI_2_5_PRO_Disambiguation",
#     "matToolUse/GEMINI_2_5_FLASH_Disambiguation",
#     "matToolUse/GPT_5_2_Disambiguation",

#     "solMech/OLLAMA_Inconsistency",
#     "solMech/CLAUDE_SONNET_4.6_Inconsistency",
#     "solMech/GEMINI_2_5_PRO_Inconsistency",
#     "solMech/GEMINI_2_5_FLASH_Inconsistency",
#     "solMech/GPT_5_2_Inconsistency",
#     "solMech/OLLAMA_Disambiguation",
#     "solMech/CLAUDE_SONNET_4.6_Disambiguation",
#     "solMech/GEMINI_2_5_PRO_Disambiguation",
#     "solMech/GEMINI_2_5_FLASH_Disambiguation",
#     "solMech/GPT_5_2_Disambiguation",

#     "pde/OLLAMA_Inconsistency",
#     "pde/CLAUDE_SONNET_4.6_Inconsistency",
#     "pde/GEMINI_2_5_PRO_Inconsistency",
#     "pde/GEMINI_2_5_FLASH_Inconsistency",
#     "pde/GPT_5_2_Inconsistency",
#     "pde/OLLAMA_Disambiguation",
#     "pde/CLAUDE_SONNET_4.6_Disambiguation",
#     "pde/GEMINI_2_5_PRO_Disambiguation",
#     "pde/GEMINI_2_5_FLASH_Disambiguation",
#     "pde/GPT_5_2_Disambiguation",

#     "fluids/OLLAMA_Inconsistency",
#     "fluids/CLAUDE_SONNET_4.6_Inconsistency",
#     "fluids/GEMINI_2_5_PRO_Inconsistency",
#     "fluids/GEMINI_2_5_FLASH_Inconsistency",
#     "fluids/GPT_5_2_Inconsistency",
#     "fluids/OLLAMA_Disambiguation",
#     "fluids/CLAUDE_SONNET_4.6_Disambiguation",
#     "fluids/GEMINI_2_5_PRO_Disambiguation",
#     "fluids/GEMINI_2_5_FLASH_Disambiguation",
#     "fluids/GPT_5_2_Disambiguation",

#     "foam/OLLAMA_Inconsistency",
#     "foam/CLAUDE_SONNET_4.6_Inconsistency",
#     "foam/GEMINI_2_5_PRO_Inconsistency",
#     "foam/GEMINI_2_5_FLASH_Inconsistency",
#     "foam/GPT_5_2_Inconsistency",
#     "foam/OLLAMA_Disambiguation",
#     "foam/CLAUDE_SONNET_4.6_Disambiguation",
#     "foam/GEMINI_2_5_PRO_Disambiguation",
#     "foam/GEMINI_2_5_FLASH_Disambiguation",
#     "foam/GPT_5_2_Disambiguation"

#     # BASELINES
#     "solToolUse_baseline/OLLAMA_Inconsistency",
#     "solToolUse_baseline/CLAUDE_SONNET_4.6_Inconsistency",
#     "solToolUse_baseline/GEMINI_2_5_PRO_Inconsistency",
#     "solToolUse_baseline/GEMINI_2_5_FLASH_Inconsistency",
#     "solToolUse_baseline/GPT_5_2_Inconsistency",
#     "solToolUse_baseline/OLLAMA_Disambiguation",
#     "solToolUse_baseline/CLAUDE_SONNET_4.6_Disambiguation",
#     "solToolUse_baseline/GEMINI_2_5_PRO_Disambiguation",
#     "solToolUse_baseline/GEMINI_2_5_FLASH_Disambiguation",
#     "solToolUse_baseline/GPT_5_2_Disambiguation",

#     "solMech_baseline/OLLAMA_Inconsistency",
#     "solMech_baseline/CLAUDE_SONNET_4.6_Inconsistency",
#     "solMech_baseline/GEMINI_2_5_PRO_Inconsistency",
#     "solMech_baseline/GEMINI_2_5_FLASH_Inconsistency",
#     "solMech_baseline/GPT_5_2_Inconsistency",
#     "solMech_baseline/OLLAMA_Disambiguation",
#     "solMech_baseline/CLAUDE_SONNET_4.6_Disambiguation",
#     "solMech_baseline/GEMINI_2_5_PRO_Disambiguation",
#     "solMech_baseline/GEMINI_2_5_FLASH_Disambiguation",
#     "solMech_baseline/GPT_5_2_Disambiguation",

#     "matSci_baseline/OLLAMA_Inconsistency",
#     "matSci_baseline/CLAUDE_SONNET_4.6_Inconsistency",
#     "matSci_baseline/GEMINI_2_5_PRO_Inconsistency",
#     "matSci_baseline/GEMINI_2_5_FLASH_Inconsistency",
#     "matSci_baseline/GPT_5_2_Inconsistency",
#     "matSci_baseline/OLLAMA_Disambiguation",
#     "matSci_baseline/CLAUDE_SONNET_4.6_Disambiguation",
#     "matSci_baseline/GEMINI_2_5_PRO_Disambiguation",
#     "matSci_baseline/GEMINI_2_5_FLASH_Disambiguation",
#     "matSci_baseline/GPT_5_2_Disambiguation",

#     "matToolUse_baseline/OLLAMA_Inconsistency",
#     "matToolUse_baseline/CLAUDE_SONNET_4.6_Inconsistency",
#     "matToolUse_baseline/GEMINI_2_5_PRO_Inconsistency",
#     "matToolUse_baseline/GEMINI_2_5_FLASH_Inconsistency",
#     "matToolUse_baseline/GPT_5_2_Inconsistency",
#     "matToolUse_baseline/OLLAMA_Disambiguation",
#     "matToolUse_baseline/CLAUDE_SONNET_4.6_Disambiguation",
#     "matToolUse_baseline/GEMINI_2_5_PRO_Disambiguation",
#     "matToolUse_baseline/GEMINI_2_5_FLASH_Disambiguation",
#     "matToolUse_baseline/GPT_5_2_Disambiguation",

#     "pde_baseline/OLLAMA_Inconsistency",
#     "pde_baseline/CLAUDE_SONNET_4.6_Inconsistency",
#     "pde_baseline/GEMINI_2_5_PRO_Inconsistency",
#     "pde_baseline/GEMINI_2_5_FLASH_Inconsistency",
#     "pde_baseline/GPT_5_2_Inconsistency",
#     "pde_baseline/OLLAMA_Disambiguation",
#     "pde_baseline/CLAUDE_SONNET_4.6_Disambiguation",
#     "pde_baseline/GEMINI_2_5_PRO_Disambiguation",
#     "pde_baseline/GEMINI_2_5_FLASH_Disambiguation",
#     "pde_baseline/GPT_5_2_Disambiguation",

#     "fluids_baseline/OLLAMA_Inconsistency",
#     "fluids_baseline/CLAUDE_SONNET_4.6_Inconsistency",
#     "fluids_baseline/GEMINI_2_5_PRO_Inconsistency",
#     "fluids_baseline/GEMINI_2_5_FLASH_Inconsistency",
#     "fluids_baseline/GPT_5_2_Inconsistency",
#     "fluids_baseline/OLLAMA_Disambiguation",
#     "fluids_baseline/CLAUDE_SONNET_4.6_Disambiguation",
#     "fluids_baseline/GEMINI_2_5_PRO_Disambiguation",
#     "fluids_baseline/GEMINI_2_5_FLASH_Disambiguation",
#     "fluids_baseline/GPT_5_2_Disambiguation",

#     "foam_baseline/OLLAMA_Inconsistency",
#     "foam_baseline/CLAUDE_SONNET_4.6_Inconsistency",
#     "foam_baseline/GEMINI_2_5_PRO_Inconsistency",
#     "foam_baseline/GEMINI_2_5_FLASH_Inconsistency",
#     "foam_baseline/GPT_5_2_Inconsistency",
#     "foam_baseline/OLLAMA_Disambiguation",
#     "foam_baseline/CLAUDE_SONNET_4.6_Disambiguation",
#     "foam_baseline/GEMINI_2_5_PRO_Disambiguation",
#     "foam_baseline/GEMINI_2_5_FLASH_Disambiguation",
#     "foam_baseline/GPT_5_2_Disambiguation"

# ]


# def save_json_if_changed(filepath, new_data):
#     if isinstance(filepath, str):
#         filepath = Path(filepath)

#     if filepath.exists():
#         try:
#             with open(filepath, 'r', encoding='utf-8') as f:
#                 existing_data = json.load(f)

#             if existing_data == new_data:
#                 return False  # Content is identical, skip writing
#         except json.JSONDecodeError:
#             pass  # If the existing file is corrupted, we want to overwrite it anyway

#     # If we reach here, it either didn't exist, was corrupted, or content actually changed
#     with open(filepath, 'w', encoding='utf-8') as f:
#         json.dump(new_data, f, indent=4)

#     return True


# def load_valid_cases(domain: str, task_type: str) -> list:
#     """
#     Loads the ground truth JSON file and extracts the list of valid case IDs.
#     Expects files like: foam/inconsistency_foam.json or foam/disambiguation_foam.json
#     """
#     # Strip '_baseline' suffix if it exists to get the true domain name
#     base_domain = domain.replace("_baseline", "")

#     # Construct the path using the base domain (e.g., 'fluids' instead of 'fluids_baseline')
#     ref_filename = f"{task_type.lower()}_{base_domain}.json"
#     ref_file_path = Path(base_domain) / ref_filename

#     if not ref_file_path.exists():
#         print(f"  [!] Reference file missing: {ref_file_path}")
#         return []

#     try:
#         with open(ref_file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#             # Assuming the JSON is a list of dicts with an 'id' field
#             return [item['id'] for item in data if 'id' in item]
#     except Exception as e:
#         print(f"  [!] Error loading reference file {ref_file_path}: {e}")
#         return []


# def update_intent_summary(base_dir: Path, valid_cases: list):
#     """Generates llm_judge_intent_summary.json"""
#     cases_list = []
#     score_accum = 0
#     total_cases_run = 0

#     llm_usage_totals = {
#         "total_calls": 0, "failed_calls": 0, "total_retries": 0,
#         "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0
#     }

#     for case_id in valid_cases:
#         score_file = base_dir / case_id / "llm_judge_intent_score.json"

#         if not score_file.exists():
#             continue

#         try:
#             with open(score_file, 'r', encoding='utf-8') as f:
#                 case_data = json.load(f)

#             cases_list.append(case_data)
#             total_cases_run += 1

#             # Parse score (e.g., "1/1")
#             score_str = case_data.get("score", "0/1")
#             try:
#                 num, den = map(int, score_str.split('/'))
#                 if num == den and den > 0:
#                     score_accum += 1
#             except ValueError:
#                 pass

#             # Accumulate LLM usage
#             usage = case_data.get("llm_usage", {})
#             for k in llm_usage_totals.keys():
#                 llm_usage_totals[k] += usage.get(k, 0)

#         except Exception as e:
#             print(f"    [!] Error processing {score_file}: {e}")

#     if total_cases_run == 0:
#         return

#     avg_percentage = (score_accum / total_cases_run) * 100

#     final_summary = {
#         "average_percentage": f"{avg_percentage:.2f}%",
#         "aggregate_score": f"{score_accum}/{total_cases_run}",
#         "total_cases_evaluated": total_cases_run,
#         "total_llm_usage": llm_usage_totals
#     }

#     output_file = base_dir / "llm_judge_intent_summary.json"
#     report_data = {
#         "cases": cases_list,
#         "final_summary": final_summary
#     }
#     save_json_if_changed(output_file, report_data)

#     print(f"    Updated Intent Summary ({total_cases_run} cases)")


# def update_chat_summary(base_dir: Path, valid_cases: list, is_inconsistency: bool):
#     """Generates llm_judge_chat_summary.json"""
#     target_file = "llm_judge_inconsistency_chat_score.json" if is_inconsistency else "llm_judge_disambiguation_chat_score.json"
#     cases_list = []

#     # Tracking
#     llm_usage_totals = {k: 0 for k in ["total_calls", "failed_calls",
#                                        "total_retries", "total_tokens", "prompt_tokens", "completion_tokens"]}
#     tq_overall, relevant_q_overall, cases_without_contradictions = 0, 0, 0
#     asked_about_overall, warnings_overall, planted_overall, assumed_overall = 0, 0, 0, 0

#     for case_id in valid_cases:
#         score_file = base_dir / case_id / target_file
#         if not score_file.exists():
#             continue

#         try:
#             with open(score_file, 'r', encoding='utf-8') as f:
#                 case_data = json.load(f)

#             cases_list.append(case_data)

#             # Base LLM usage and general chat metrics
#             usage = case_data.get("llm_usage", {})
#             for k in llm_usage_totals.keys():
#                 llm_usage_totals[k] += usage.get(k, 0)

#             tq_overall += case_data.get("total_questions_asked_in_chat", 0)
#             relevant_q_overall += case_data.get(
#                 "relevant_questions", {}).get("count", 0)

#             if float(case_data.get("metrics", {}).get("memory_consistency", 0.0)) == 1.0:
#                 cases_without_contradictions += 1

#             # Mode-specific extraction
#             if is_inconsistency:
#                 target_planted = case_data.get(
#                     "total_planted_inconsistencies", 0)
#                 asked = case_data.get(
#                     "questions_about_inconsistencies", {}).get("count", 0)
#                 warnings_overall += case_data.get(
#                     "explicit_warnings", {}).get("count", 0)
#             else:
#                 target_planted = case_data.get("total_missing_entities", 0)
#                 asked = case_data.get(
#                     "questions_about_missing_entities", {}).get("count", 0)

#             planted_overall += target_planted
#             asked_about_overall += asked

#             # Assumption rate applies to both
#             a_rate = case_data.get("metrics", {}).get("assumption_rate", 1.0)
#             assumed_overall += round((1.0 - a_rate) * target_planted)

#         except Exception as e:
#             print(f"    [!] Error processing {score_file}: {e}")

#     total_cases_evaluated = len(cases_list)
#     if total_cases_evaluated == 0:
#         return

#     # Calculate Percentages
#     global_memory = cases_without_contradictions / total_cases_evaluated
#     global_precision = min(
#         1.0, max(0.0, (relevant_q_overall / tq_overall) if tq_overall > 0 else 0.0))
#     global_recall = min(1.0, max(
#         0.0, (asked_about_overall / planted_overall) if planted_overall > 0 else 0.0))
#     global_assumption = min(1.0, max(
#         0.0, 1.0 - (assumed_overall / planted_overall) if planted_overall > 0 else 0.0))

#     # Base Summary Structure
#     final_summary = {
#         "total_cases_evaluated": total_cases_evaluated,
#         "aggregate_raw_metrics": {
#             "total_questions_asked_overall": tq_overall,
#             "total_relevant_questions": relevant_q_overall,
#             "cases_without_contradictions": cases_without_contradictions
#         },
#         "aggregate_percentages": {
#             "clarification_precision": f"{global_precision:.2%}",
#             "clarification_recall": f"{global_recall:.2%}",
#             "assumption_rate": f"{global_assumption:.2%}",
#             "memory_consistency": f"{global_memory:.2%}"
#         },
#         "total_llm_usage": llm_usage_totals
#     }

#     # Add Mode-Specific Keys
#     if is_inconsistency:
#         global_edr = min(1.0, max(
#             0.0, (warnings_overall / planted_overall) if planted_overall > 0 else 0.0))
#         final_summary["aggregate_raw_metrics"].update({
#             "total_inconsistencies_asked_about": asked_about_overall,
#             "total_explicit_warnings": warnings_overall,
#             "total_planted_inconsistencies_overall": planted_overall,
#             "total_assumed_inconsistencies": assumed_overall
#         })
#         final_summary["aggregate_percentages"].update({
#             "error_detection_rate": f"{global_edr:.2%}"
#         })
#     else:
#         final_summary["aggregate_raw_metrics"].update({
#             "total_entities_asked_about": asked_about_overall,
#             "total_missing_entities_overall": planted_overall,
#             "total_assumed_entities": assumed_overall
#         })

#     # Save to disk using the anti-git-pollution helper
#     output_file = base_dir / "llm_judge_chat_summary.json"
#     report_data = {
#         "cases": cases_list,
#         "final_summary": final_summary
#     }

#     save_json_if_changed(output_file, report_data)

#     print(
#         f"    Checked/Updated Chat Summary ({total_cases_evaluated} cases)")


# def update_evaluator_summary(base_dir: Path, valid_cases: list, is_inconsistency: bool):
#     """Generates llm_judge_summary.json (Standardized name for disambiguation/inconsistency base eval)"""
#     target_file = "llm_judge_inconsistency_score.json" if is_inconsistency else "llm_judge_disambiguation_score.json"

#     strict_passes, total_cases, partial_hits, partial_total = 0, 0, 0, 0
#     report_data = {"cases": []}

#     # Initialize LLM usage tracker
#     llm_usage_totals = {
#         "total_calls": 0, "failed_calls": 0, "total_retries": 0,
#         "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0
#     }

#     for case_id in valid_cases:
#         score_file = base_dir / case_id / target_file
#         if not score_file.exists():
#             continue

#         try:
#             with open(score_file, 'r', encoding='utf-8') as f:
#                 data = json.load(f)

#             # Accumulate LLM usage
#             usage = data.get("llm_usage", {})
#             for k in llm_usage_totals.keys():
#                 llm_usage_totals[k] += usage.get(k, 0)

#             if is_inconsistency:
#                 strict_score = data.get("score", "0/1")
#                 breakdown = data.get("breakdown", [])
#                 case_total = len(breakdown)
#                 case_hits = sum(
#                     1 for item in breakdown if item.get("resolved") is True)
#                 partial_score = f"{case_hits}/{case_total}"

#                 data["strict_score"] = strict_score
#                 data["partial_score"] = partial_score

#                 report_data["cases"].append({
#                     "case_id": case_id,
#                     "strict_score": strict_score,
#                     "partial_score": partial_score,
#                     "status": data.get("status", "UNKNOWN"),
#                     "breakdown": breakdown,
#                     "llm_usage": usage
#                 })
#             else:
#                 score_str = data.get("score", "0/0")
#                 try:
#                     case_hits, case_total = map(int, score_str.split('/'))
#                 except:
#                     case_hits, case_total = 0, 0

#                 is_perfect = (case_hits == case_total and case_total > 0)
#                 strict_score = "1/1" if is_perfect else "0/1"
#                 partial_score = score_str

#                 data["strict_score"] = strict_score
#                 data["partial_score"] = partial_score

#                 report_data["cases"].append({
#                     "case_id": case_id,
#                     "strict_score": strict_score,
#                     "partial_score": partial_score,
#                     "recovered": data.get("recovered_entities", []),
#                     "missing": data.get("still_missing_entities", []),
#                     "llm_usage": usage
#                 })

#             # Save individual case score updates
#             save_json_if_changed(score_file, data)

#             total_cases += 1
#             partial_hits += case_hits
#             partial_total += case_total
#             if strict_score == "1/1":
#                 strict_passes += 1

#         except Exception as e:
#             print(f"    [!] Error processing {score_file}: {e}")

#     if total_cases == 0:
#         return

#     strict_pct = (strict_passes / total_cases) * 100
#     partial_pct = (partial_hits / partial_total *
#                    100) if partial_total > 0 else 0

#     report_data["final_summary"] = {
#         "strict_accuracy_percentage": f"{strict_pct:.2f}%",
#         "strict_aggregate_score": f"{strict_passes}/{total_cases}",
#         "partial_accuracy_percentage": f"{partial_pct:.2f}%",
#         "partial_aggregate_score": f"{partial_hits}/{partial_total}",
#         "total_cases_evaluated": total_cases,
#         "total_llm_usage": llm_usage_totals
#     }

#     output_file = base_dir / "llm_judge_summary.json"
#     save_json_if_changed(output_file, report_data)

#     print(f"    Updated Evaluator Summary ({total_cases} cases)")


# def process_directories(directories):
#     """Main Controller Function"""
#     for dir_path in directories:
#         base_dir = Path(dir_path)

#         if not base_dir.exists() or not base_dir.is_dir():
#             print(f"\n[SKIP] Directory not found: {base_dir}")
#             continue

#         print(f"\n[RUN] Processing: {base_dir}")

#         domain = base_dir.parts[0]
#         task_type = "Inconsistency" if "Inconsistency" in base_dir.name else "Disambiguation"
#         is_inconsistency = (task_type == "Inconsistency")

#         valid_cases = load_valid_cases(domain, task_type)
#         if not valid_cases:
#             print(
#                 f"  [!] No valid cases found for {domain} {task_type}. Skipping.")
#             continue

#         print(f"  Loaded {len(valid_cases)} valid cases from reference file.")

#         update_evaluator_summary(base_dir, valid_cases, is_inconsistency)
#         update_chat_summary(base_dir, valid_cases, is_inconsistency)
#         update_intent_summary(base_dir, valid_cases)


# if __name__ == "__main__":
#     process_directories(TARGET_DIRS)


##################################################################################################
#################################### GROUNDED RESOLUTION RATE ####################################
##################################################################################################

# """
# Compute CGR (Conversation-Grounded Resolution) from raw per-case JSON files.

# CGR Rate = (Successful grounded Cases) / (Total Number of Cases in Dataset)
# Successful = strict_score is "1/1" AND asked >= total.
# """

# from pathlib import Path
# import json
# import pandas as pd

# ROOT = Path(__file__).resolve().parent

# # MODELS = ['CLAUDE_SONNET_4.6', 'GEMINI_2_5_PRO',
# #           "GEMINI_2_5_FLASH", 'GPT_5_2', "OLLAMA"]

# # MODEL_DISPLAY = {
# #     'CLAUDE_SONNET_4.6': 'Claude Sonnet 4.6',
# #     'GPT_5_2':           'GPT-5.2',
# #     'GEMINI_2_5_PRO':    'Gemini 2.5 Pro',
# #     'GEMINI_2_5_FLASH':  'Gemini 2.5 Flash',
# #     'OLLAMA':            'GPT-OSS 120B'
# # }

# # DOMAINS = ["fluids", "foam", "solMech",
# #            "solToolUse", "matSci", "matToolUse", "pde"]
# MODELS = ['GEMINI_2_5_PRO', "OLLAMA"]

# MODEL_DISPLAY = {
#     'GEMINI_2_5_PRO':    'Gemini 2.5 Pro',
#     'OLLAMA':            'GPT-OSS 120B'
# }

# DOMAINS = ["clamber"]


# def load_json(path: Path) -> dict:
#     with open(path) as f:
#         return json.load(f)


# per-case CGR helpers

# def cgr_for_disambiguation(case_dir: Path, domain: str) -> int:
#     """
#     Returns 1 (CGR), 0 (not CGR).
#     Grounding check strictly depends on the domain.
#     """
#     score_p = case_dir / 'llm_judge_disambiguation_score.json'

#     if not score_p.exists():
#         return 0

#     strict = load_json(score_p).get('strict_score', '0/1') == '1/1'
#     if not strict:
#         return 0   # not fully resolved -> cannot be CGR

#     asked = None
#     total = None

#     if domain in ["foam", "fluids", "pde"]:
#         target_p = case_dir / 'disambiguation_coverage_score.json'
#         if not target_p.exists():
#             print(f"  [WARN] Missing file: {target_p}")
#             return 0

#         data = load_json(target_p)
#         if 'ambiguities_total' not in data or 'ambiguities_addressed_in_conversation' not in data:
#             print(f"  [WARN] Missing expected keys in: {target_p}")
#             return 0

#         total = data['ambiguities_total']
#         asked = data['ambiguities_addressed_in_conversation']

#     else:
#         target_p = case_dir / 'llm_judge_disambiguation_chat_score.json'
#         if not target_p.exists():
#             print(f"  [WARN] Missing file: {target_p}")
#             return 0

#         data = load_json(target_p)
#         if 'total_missing_entities' not in data or 'questions_about_missing_entities' not in data:
#             print(f"  [WARN] Missing expected keys in: {target_p}")
#             return 0

#         total = data['total_missing_entities']
#         asked = data['questions_about_missing_entities'].get('count', 0)

#     if not total:
#         return 0   # Cannot confirm grounding if there are no totals recorded

#     return int(asked >= total)


# def cgr_for_inconsistency(case_dir: Path) -> int:
#     """
#     Returns 1 (CGR), 0 (not CGR).
#     Grounding check: questions_about_inconsistencies.count >= total_planted_inconsistencies
#     """
#     score_p = case_dir / 'llm_judge_inconsistency_score.json'
#     chat_p = case_dir / 'llm_judge_inconsistency_chat_score.json'

#     if not score_p.exists():
#         return 0

#     strict = load_json(score_p).get('strict_score', '0/1') == '1/1'
#     if not strict:
#         return 0

#     if not chat_p.exists():
#         return 0   # resolved but no conversation evidence -> SRR not CGR

#     cdat = load_json(chat_p)
#     total = cdat.get('total_planted_inconsistencies', 0)
#     asked = cdat.get('questions_about_inconsistencies', {}).get('count', 0)

#     if not total:
#         return 0

#     return int(asked >= total)


# main aggregation

# def compute() -> dict:
#     """
#     Returns {(domain, model, task): float (rate)}
#     """
#     results = {}

#     for task in ['Disambiguation', 'Inconsistency']:
#         for domain in DOMAINS:
#             for model in MODELS:
#                 cgr_sum = 0
#                 total = 0

#                 domain_dir = ROOT / domain
#                 run_dir = domain_dir / f'{model}_{task}'
#                 dataset_p = domain_dir / f'{task.lower()}_{domain}.json'

#                 if not dataset_p.exists():
#                     print(f'  [WARN] missing dataset: {dataset_p}')
#                     results[(domain, model, task)] = None
#                     continue

#                 dataset = load_json(dataset_p)
#                 valid_ids = [item['id'] for item in dataset if 'id' in item]

#                 for cid in valid_ids:
#                     case_dir = run_dir / cid
#                     total += 1          # Increment total for every case in dataset

#                     if not case_dir.exists():
#                         continue        # Case exists in dataset but no output generated

#                     if task == 'Disambiguation':
#                         cgr_sum += cgr_for_disambiguation(case_dir, domain)
#                     else:
#                         cgr_sum += cgr_for_inconsistency(case_dir)

#                 rate = (cgr_sum / total) if total > 0 else None
#                 results[(domain, model, task)] = rate

#     return results


# def print_results(results: dict) -> None:
#     col_width = 18

#     for task in ['Disambiguation', 'Inconsistency']:
#         print(f'\n\n=== {task} ===')

#         # Build Table Header
#         header = f'{"Domain":<14} | ' + \
#             ' | '.join(f'{MODEL_DISPLAY[m]:<{col_width}}' for m in MODELS)
#         print(header)
#         print('-' * len(header))

#         # Build Rows
#         for domain in DOMAINS:
#             row_str = f'{domain:<14} | '
#             cols = []
#             for model in MODELS:
#                 rate = results.get((domain, model, task))
#                 if rate is not None:
#                     # Format as percentage with 1 decimal place
#                     cols.append(f'{rate * 100:>17.1f}%')
#                 else:
#                     cols.append(f'{"N/A":>18}')

#             row_str += ' | '.join(cols)
#             print(row_str)


# def create_dataframes(results: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Converts the results dictionary into two pandas DataFrames.
#     """
#     data_disambiguation = {model: [] for model in MODELS}
#     data_inconsistency = {model: [] for model in MODELS}

#     for domain in DOMAINS:
#         for model in MODELS:
#             # Append Disambiguation rates
#             rate_d = results.get((domain, model, 'Disambiguation'))
#             data_disambiguation[model].append(rate_d)

#             # Append Inconsistency rates
#             rate_i = results.get((domain, model, 'Inconsistency'))
#             data_inconsistency[model].append(rate_i)

#     # Build the DataFrames using standard row/column operations
#     df_disambiguation = pd.DataFrame(data_disambiguation, index=DOMAINS)
#     df_inconsistency = pd.DataFrame(data_inconsistency, index=DOMAINS)

#     # Rename the columns to their display names
#     df_disambiguation.rename(columns=MODEL_DISPLAY, inplace=True)
#     df_inconsistency.rename(columns=MODEL_DISPLAY, inplace=True)

#     return df_disambiguation, df_inconsistency


# if __name__ == '__main__':
#     results = compute()
#     print_results(results)
#     # df_disambiguation, df_inconsistency = create_dataframes(results)
#     # print("=== Disambiguation ===")
#     # print(df_disambiguation)
#     # print("\n=== Inconsistency ===")
#     # print(df_inconsistency)

#     # df_disambiguation.to_csv("disambiguation_results.csv")
#     # df_inconsistency.to_csv("inconsistency_results.csv")
