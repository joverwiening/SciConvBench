import os
import json
import time
import argparse
from config import Config
from main import ConvAgent
from utils import LLMService
from llm_user_interface import LLMUserInterface

# --- CONFIGURATION MAPPING ---
TASK_CONFIG = {
    "disambiguation": {
        "filename": "disambiguation",
        "keys": {
            "input_prompt": "incomplete_user_req",
            "ground_truth": "complete_user_req",
            "max_turns": "max_turns_to_resolve"
        }
    },
    "inconsistency": {
        "filename": "inconsistency",
        "keys": {
            "input_prompt": "inconsistent_prompt",
            "ground_truth": "correct_prompt",
            "max_turns": "max_turns_to_resolve"
        }
    }
}

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

def run_single_case(case_data, config, task_mode, output_root, baseline=False):
    """Runs the simulation for one specific case."""
    
    keys = config['keys']
    case_id = case_data.get('id', 'unknown_case')
    incomplete_req = case_data.get(keys['input_prompt'], "")
    complete_req = case_data.get(keys['ground_truth'], "")
    max_turns = 10 # Hardcoded

    if not incomplete_req:
        print(f"Skipping {case_id}: Missing input prompt in JSON.")
        return

    print(f"\nSTARTING CASE: {case_id} ({task_mode})")
    
    # Setup Files using the passed output_root
    case_folder = setup_case_folder(output_root, case_id)
    log_file = os.path.join(case_folder, "conversation_log.txt")
    result_file = os.path.join(case_folder, "generated_prompt.txt")
    ground_truth_file = os.path.join(case_folder, "complete_prompt.txt")
    stats_file = os.path.join(case_folder, "statistics.json")

    # Save Ground Truth
    with open(ground_truth_file, "w") as f:
        f.write(complete_req)

    # Configure Components
    cfg = Config()
    cfg.LOG_FILE = log_file 
    cfg.INCOMPLETE_PROMPT = incomplete_req
    print(50*"=")
    print("Double checking model provider:", cfg.model_provider)
    print("Double checking model service:", cfg.model_version)
    print(50*"=")
    if baseline:
        print("Running baseline (implicit) agent")
    else:
        print("Running expert (explicit) agent")
    prompt_mode = "baseline" if baseline else None
    agent = ConvAgent(cfg, prompt_mode)
    
    user_llm_service = LLMService(cfg)
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
    
    try:
        while turn < max_turns:
            turn += 1
            print(f"  [Turn {turn}] Agent Thinking...")

            agent_response = agent.chat(current_input)

            # Check Completion
            if "[COMPLETE]" in agent_response:
                print(f"  Case {case_id} COMPLETED in {turn} turns.")
                final_summary = agent_response.replace("[COMPLETE]", "").strip()
                
                with open(result_file, "w") as f:
                    f.write(final_summary)
                
                success = True
                break

            # User Turn
            user_response = simulated_user.answer_clarification_question(agent_response)
            current_input = user_response
            time.sleep(0.5) 

        if not success:
            print(f"  Case {case_id} reached max turns ({max_turns}).")
            final_summary = agent.get_final_attempt()
            with open(result_file, "w") as f:
                f.write(final_summary)
            success = True

    except Exception as e:
        print(f"  Error in case {case_id}: {e}")    

    # Save Stats
    agent_stats = agent.llm.get_statistics()
    save_statistics(stats_file, agent_stats, turn, max_turns, success, task_mode)
    print(f"  Saved outputs to {case_folder}/")

def main():
    parser = argparse.ArgumentParser(description="Run Agent Tests")
    
    parser.add_argument("--mode", choices=["disambiguation", "inconsistency"], 
                        help="Which test set to run")
    
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Root directory where case folders will be created")

    parser.add_argument("--domain", required=True, type=str,
                        help="The domain folder containing the test files (e.g., foam, fluids, pde)")
    
    parser.add_argument("--model", required=True, choices=[
        "bedrock_3_5_sonnet",
        "bedrock_4_0_sonnet", 
        "bedrock_3_5_haiku",
        "bedrock_4_6_sonnet",
        "bedrock_4_6_opus",
        "gemini_pro",
        "gemini_flash",
        "openai_gpt5",
        "ollama_local"
    ], help="Model profile to use (overrides config.py)")

    parser.add_argument("--baseline", action="store_true",
                        help="Run the baseline agent instead of the expert agent.")
    
    args = parser.parse_args()
    
    mode = args.mode
    output_dir = args.output_dir
    domain = args.domain
    selected_profile = args.model
    print(f"Override: Switching model to [{selected_profile}]")
    baseline = args.baseline
    Config.ACTIVE_PROFILE = selected_profile
    Config.active_conf = Config.LLM_PROFILES[selected_profile]
    Config.model_provider = Config.active_conf["provider"]
    Config.model_version = Config.active_conf["model_version"]
    Config.temperature = Config.active_conf["temperature"]

    task_cfg = TASK_CONFIG[mode]
    json_filename = f"{task_cfg['filename']}_{domain}.json"
    json_path = os.path.join(domain, json_filename)

    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    print(f"Loading {mode} cases from {json_path}...")
    with open(json_path, 'r') as f:
        cases = json.load(f)
    
    print(f"Found {len(cases)} cases.")
    print(f"Output Directory: {output_dir}")
    print("="*60)
    
    # Run All
    for case in cases:
        id_int = int(case["id"][-3] + case["id"][-2] + case["id"][-1])
        # if id_int in [94, 96, 100, 102]:
        run_single_case(case, task_cfg, mode, output_dir, baseline)
if __name__ == "__main__":
    main()