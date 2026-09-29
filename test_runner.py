import os
import json
import time
import argparse
from dataclasses import dataclass
from typing import Optional
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


@dataclass
class DialogueResult:
    final_spec: Optional[str]
    turns: int
    completed_by_agent: bool
    cap_hit: bool
    error: Optional[str] = None
    final_attempts: int = 0
    final_valid: Optional[bool] = None


FINAL_MIN_CHARS = 300
FINAL_RETRY_INSTRUCTION = ("Your previous output was not a specification. Output only the final specification, "
                           "one sentence per line, with no questions.")


def final_spec_valid(text: str) -> bool:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return len(text) >= FINAL_MIN_CHARS and not (lines and lines[-1].endswith("?"))


def forced_final_fixed(agent, last_user_reply):
    """(spec, attempts, valid): the cap-hit final spec with the user's last reply in history, one retry if invalid."""
    agent.history.append(f"User: {last_user_reply}")
    spec = agent.get_final_attempt_fixed().replace("[COMPLETE]", "").strip()
    if final_spec_valid(spec):
        return spec, 1, True
    spec = agent.get_final_attempt_fixed(FINAL_RETRY_INSTRUCTION).replace("[COMPLETE]", "").strip()
    return spec, 2, final_spec_valid(spec)


def run_dialogue(agent, simulated_user, initial_input, max_turns, sleep=0.5, final_fix=False):
    """Upstream question loop; on the turn cap the agent is forced to write its final spec.
    final_fix: the forced call sees the user's last reply and is retried once if it is not a spec."""
    current_input = initial_input
    turn = 0
    try:
        while turn < max_turns:
            turn += 1
            print(f"  [Turn {turn}] Agent Thinking...")

            agent_response = agent.chat(current_input)

            if "[COMPLETE]" in agent_response:
                final_summary = agent_response.replace(
                    "[COMPLETE]", "").strip()
                return DialogueResult(final_summary, turn, True, False)

            user_response = simulated_user.answer_clarification_question(
                agent_response)
            current_input = user_response
            time.sleep(sleep)

        if final_fix:
            spec, attempts, valid = forced_final_fixed(agent, current_input)
            return DialogueResult(spec, turn, False, True, final_attempts=attempts, final_valid=valid)
        return DialogueResult(agent.get_final_attempt(), turn, False, True, final_attempts=1)

    except Exception as e:
        return DialogueResult(None, turn, False, False, str(e))


def run_single_case(case_data, config, task_mode, output_root, baseline=False,
                    user_profile=None, max_turns=10):
    """Runs the simulation for one specific case."""

    keys = config['keys']
    case_id = case_data.get('id', 'unknown_case')
    incomplete_req = case_data.get(keys['input_prompt'], "")
    complete_req = case_data.get(keys['ground_truth'], "")

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

    # upstream built the user from the agent's cfg; the paper (sec 4) uses a fixed Sonnet 4.6 user
    if user_profile:
        user_cfg = Config()
        prof = Config.LLM_PROFILES[user_profile]
        user_cfg.model_provider = prof["provider"]
        user_cfg.model_version = prof["model_version"]
        user_cfg.temperature = prof["temperature"]
    else:
        user_cfg = cfg
    print("Simulated user model:", user_cfg.model_provider, user_cfg.model_version)
    user_llm_service = LLMService(user_cfg)
    simulated_user = LLMUserInterface(
        llm_service=user_llm_service,
        incomplete_requirement=incomplete_req,
        complete_requirement=complete_req,
        log_file=log_file
    )

    result = run_dialogue(agent, simulated_user, incomplete_req, max_turns)
    if result.error is not None:
        print(f"  Error in case {case_id}: {result.error}")
    elif result.completed_by_agent:
        print(f"  Case {case_id} COMPLETED in {result.turns} turns.")
    else:
        print(f"  Case {case_id} reached max turns ({max_turns}).")
    if result.final_spec is not None:
        with open(result_file, "w") as f:
            f.write(result.final_spec)
    turn = result.turns
    success = result.final_spec is not None

    # Save Stats
    agent_stats = agent.llm.get_statistics()
    save_statistics(stats_file, agent_stats, turn,
                    max_turns, success, task_mode)
    with open(os.path.join(case_folder, "user_statistics.json"), "w") as f:
        json.dump({"user_model": user_cfg.model_version,
                   "user_llm_usage": user_llm_service.get_statistics()}, f, indent=4)
    print(f"  Saved outputs to {case_folder}/")


def main():
    parser = argparse.ArgumentParser(description="Run Agent Tests")

    parser.add_argument("--mode", choices=["disambiguation", "inconsistency"],
                        help="Which test set to run")

    parser.add_argument("--output_dir", required=True, type=str,
                        help="Root directory where case folders will be created")

    parser.add_argument("--domain", required=True, type=str,
                        help="The domain folder containing the test files (e.g., foam, fluids, pde)")

    parser.add_argument("--model", required=True, choices=list(Config.LLM_PROFILES),
                        help="Model profile to use (overrides config.py)")

    parser.add_argument("--user_model", default=None, choices=list(Config.LLM_PROFILES),
                        help="Simulated-user profile; default = same as --model (upstream behaviour)")

    # 10 question turns + 1 forced final call = the paper's 11-turn budget; released runs record 10
    parser.add_argument("--max_turns", type=int, default=10)

    parser.add_argument("--cases", default=None,
                        help="Comma-separated case ids to run (default: all)")

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
    wanted = set(args.cases.split(",")) if args.cases else None
    for case in cases:
        if wanted and case["id"] not in wanted:
            continue
        run_single_case(case, task_cfg, mode, output_dir, baseline,
                        user_profile=args.user_model, max_turns=args.max_turns)


if __name__ == "__main__":
    main()
