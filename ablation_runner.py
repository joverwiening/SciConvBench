import os
import json
import time
import argparse
from config import Config
from main import ConvAgent
from utils import LLMService
from llm_user_interface import LLMUserInterface

class CustomConfig:
    """Creates an isolated config instance to allow different models to run concurrently."""
    def __init__(self, profile_name):
        self.ACTIVE_PROFILE = profile_name
        self.active_conf = Config.LLM_PROFILES[profile_name]
        self.model_provider = self.active_conf["provider"]
        self.model_version = self.active_conf["model_version"]
        self.temperature = self.active_conf["temperature"]
        
        self.COMPLETE_PROMPT_PATH = Config.COMPLETE_PROMPT_PATH
        self.INCOMPLETE_PROMPT = Config.INCOMPLETE_PROMPT
        self.LOG_FILE = Config.LOG_FILE

def setup_case_folder(output_dir, case_id):
    """Creates a clean folder for the specific case run."""
    folder_path = os.path.join(output_dir, case_id)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def save_statistics(file_path, agent_stats, turns_needed, max_turns_allowed, success, task_type):
    """Saves run metrics to statistics.json."""
    stats_data = {
        "test_type": task_type,
        "result": "SUCCESS" if success else "FAILED/INCOMPLETE",
        "turns_needed": turns_needed,
        "max_turns_allowed": max_turns_allowed,
        "agent_llm_usage": agent_stats
    }
    
    with open(file_path, "w") as f:
        json.dump(stats_data, f, indent=4)

def run_single_case(case_data, config_keys, task_mode, output_root, agent_model, user_model, prompt_mode=None):
    """Runs the simulation for one specific case and updates the JSON dictionary."""
    
    case_id = case_data.get('case_id', 'unknown_case')
    planted_gt = case_data.get('planted_ground_truth', {})
    
    incomplete_req = planted_gt.get(config_keys['input_prompt'], "")
    complete_req = planted_gt.get(config_keys['ground_truth'], "")
    max_turns = 10

    if not incomplete_req:
        print(f"Skipping {case_id}: Missing input prompt in JSON under key '{config_keys['input_prompt']}'.")
        return case_data

    print(f"\nSTARTING CASE: {case_id} ({task_mode})")
    
    # Setup Files using the passed output_root
    case_folder = setup_case_folder(output_root, case_id)
    log_file = os.path.join(case_folder, "conversation_log.txt")
    result_file = os.path.join(case_folder, "generated_prompt.txt")
    ground_truth_file = os.path.join(case_folder, "complete_prompt.txt")
    stats_file = os.path.join(case_folder, "statistics.json")

    # Save Ground Truth locally
    with open(ground_truth_file, "w") as f:
        f.write(complete_req)

    # INITIALIZE SEPARATE CONFIGS
    agent_cfg = CustomConfig(agent_model)
    agent_cfg.LOG_FILE = log_file
    agent_cfg.INCOMPLETE_PROMPT = incomplete_req

    user_cfg = CustomConfig(user_model)
    user_cfg.LOG_FILE = log_file

    print(50*"=")
    print(f"AGENT Model Provider: {agent_cfg.model_provider} | Version: {agent_cfg.model_version}")
    print(f"USER Model Provider: {user_cfg.model_provider} | Version: {user_cfg.model_version}")
    print(f"PROMPT MODE: {prompt_mode if prompt_mode else 'Standard (None)'}")
    print(50*"=")

    
    agent = ConvAgent(agent_cfg, prompt_mode=prompt_mode)

    # Initialize User
    user_llm_service = LLMService(user_cfg)
    simulated_user = LLMUserInterface(
        llm_service=user_llm_service,
        incomplete_requirement=incomplete_req,
        complete_requirement=complete_req,
        log_file=log_file
    )

    # Run Loop
    current_input = incomplete_req
    turn = 0
    success = False
    final_summary = ""
    
    try:
        while turn < max_turns:
            turn += 1
            print(f"  [Turn {turn}] Agent Thinking...")

            agent_response = agent.chat(current_input)

            # Check Completion
            if "[COMPLETE]" in agent_response:
                print(f"  Case {case_id} COMPLETED in {turn} turns.")
                final_summary = agent_response.replace("[COMPLETE]", "").strip()
                success = True
                break

            # User Turn
            user_response = simulated_user.answer_clarification_question(agent_response)
            current_input = user_response
            time.sleep(0.5) 

        if not success:
            print(f"  Case {case_id} reached max turns ({max_turns}).")
            final_summary = agent.get_final_attempt().strip()
            success = True

    except Exception as e:
        print(f"  Error in case {case_id}: {e}")    

    # Save local text files
    with open(result_file, "w") as f:
        f.write(final_summary)

    agent_stats = agent.llm.get_statistics()
    save_statistics(stats_file, agent_stats, turn, max_turns, success, task_mode)
    print(f"  Saved local outputs to {case_folder}/")

    # UPDATE CASE DICTIONARY FOR JSON 
    formatted_turns = []
    for idx, clar in enumerate(simulated_user.clarifications, 1):
        formatted_turns.append({
            "turn": idx,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(clar.get("timestamp", time.time()))),
            "Q": clar.get("question", ""),
            "A": clar.get("answer", "")
        })

    # update summary
    case_data["conversation_turns"] = formatted_turns
    case_data["agent_final_output"] = final_summary
    case_data["user_simulated_llm"] = user_cfg.model_version
    case_data["model"] = agent_cfg.model_version 
    
    # Only write the prompt_mode key if it was explicitly used
    if prompt_mode:
        case_data["prompt_mode"] = prompt_mode
    
    case_data.pop("paths", None)
    case_data.pop("human_scores", None)
    case_data.pop("rater_notes", None)
    case_data.pop("annotation_time_minutes", None)
    return case_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Root directory where case folders and final JSON will be created")
    
    parser.add_argument("--prompt_mode", default=None, choices=["paraphrased_option_1", "paraphrased_option_2"],
                        help="Which system prompt option to use for ConvAgent. Leave blank to use the standard prompt.")
    
    parser.add_argument("--user_model", default="gemini_pro", choices=["bedrock_4_6_sonnet", "gemini_pro", "openai_gpt5"],
                        help="Model profile to use for the Simulated User")
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    prompt_mode = args.prompt_mode
    user_model = args.user_model
    input_file = "judge_ablation/human_annotation_form_annotated.json"
    agent_model = "gemini_pro"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Loading cases from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    data.pop("meta", None)
    cases = data.get("cases", [])
    print(f"Found {len(cases)} cases.")
    print(f"Output Directory: {output_dir}")
    print("="*60)
    
    updated_cases = []
    # Run All
    for case in cases:
        task_type = case.get("task", "").lower()
        
        # Determine the keys dynamically based on the task type specified in the case
        if "inconsistency" in task_type:
            config_keys = {
                "input_prompt": "original_inconsistent_prompt",
                "ground_truth": "ground_truth_correct_prompt"
            }
        elif "disambiguation" in task_type:
            config_keys = {
                "input_prompt": "original_underspecified_prompt",
                "ground_truth": "ground_truth_complete_prompt"
            }
        else:
            print(f"Unknown task type '{task_type}' for case {case.get('case_id')}. Skipping.")
            continue

        updated_case = run_single_case(
            case_data=case, 
            config_keys=config_keys, 
            task_mode=task_type, 
            output_root=output_dir, 
            agent_model=agent_model, 
            user_model=user_model,
            prompt_mode=prompt_mode
        )
        if updated_case:
            updated_cases.append(updated_case)

    # Save the updated JSON to the output directory
    data["cases"] = updated_cases    
    
    file_prefix = prompt_mode if prompt_mode else user_model
    output_json_path = os.path.join(output_dir, f"{file_prefix}_results.json")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"\nAblation run complete. Updated JSON saved to: {output_json_path}")

if __name__ == "__main__":
    # python ablation_runner.py --output_dir user_llm_ablation/GEMINI_PRO --user_model gemini_pro
    # python ablation_runner.py --output_dir user_llm_ablation/GPT_5_2 --user_model openai_gpt5
    # python ablation_runner.py --output_dir user_llm_ablation/SONNET_4_6 --user_model bedrock_4_6_sonnet
    
    # python ablation_runner.py --output_dir prompt_ablation/OPTION_1 --prompt_mode paraphrased_option_1
    # python ablation_runner.py --output_dir prompt_ablation/OPTION_2 --prompt_mode paraphrased_option_2

    main()