import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    # ==========================================
    # Change ACTIVE_PROFILE to switch providers
    # Options:
    # bedrock_3_5_sonnet
    # bedrock_4_0_sonnet
    # bedrock_3_5_haiku
    # bedrock_4_6_sonnet
    # bedrock_4_6_opus
    # bedrock_4_6_sonnet_sdk
    # gemini_pro
    # gemini_flash
    # openai_gpt5
    # codex_gpt5
    # ollama_local
    # ==========================================
    # judges build Config() with no arguments, so this env var is how the judge model is chosen per run
    ACTIVE_PROFILE = os.getenv("SCICONV_PROFILE", "gemini_2_5_pro_vertex")

    LLM_PROFILES = {
        "or_sonnet_4_6": {
            "provider": "openrouter",
            "model_version": "anthropic/claude-sonnet-4.6",
            "temperature": 0.0
        },
        "or_gemini_2_5_pro": {
            "provider": "openrouter",
            "model_version": "google/gemini-2.5-pro",
            "temperature": 0.0
        },
        "or_gemini_2_5_flash": {
            "provider": "openrouter",
            "model_version": "google/gemini-2.5-flash",
            "temperature": 0.0
        },
        "or_gpt_5_2": {
            "provider": "openrouter",
            "model_version": "openai/gpt-5.2",
            "temperature": 0.0
        },
        "or_gpt_6_luna": {
            "provider": "openrouter",
            "model_version": "openai/gpt-6-luna",
            "temperature": 0.0
        },
        "bedrock_3_5_sonnet": {
            "provider": "bedrock",
            "model_version": "arn:aws:bedrock:us-west-2:<ACCOUNT_ID>:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "temperature": 0.0
        },
        "bedrock_4_0_sonnet": {
            "provider": "bedrock",
            "model_version": "arn:aws:bedrock:us-west-2:<ACCOUNT_ID>:inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0",
            "temperature": 0.0
        },
        "bedrock_3_5_haiku": {
            "provider": "bedrock",
            "model_version": "arn:aws:bedrock:us-west-2:<ACCOUNT_ID>:inference-profile/us.anthropic.claude-3-5-haiku-20241022-v1:0",
            "temperature": 0.0
        },
        "bedrock_4_6_sonnet": {
            "provider": "bedrock",
            "model_version": "arn:aws:bedrock:us-west-2:<ACCOUNT_ID>:inference-profile/us.anthropic.claude-sonnet-4-6",
            "temperature": 0.0
        },
        "bedrock_4_6_opus": {
            "provider": "bedrock",
            "model_version": "arn:aws:bedrock:us-west-2:<ACCOUNT_ID>:inference-profile/us.anthropic.claude-opus-4-6-v1",
            "temperature": 0.0
        },
        "bedrock_4_6_sonnet_sdk": {
            "provider": "claude_sdk",
            "model_version": "claude-sonnet-4-6",
            "temperature": 0.0
        },
        "gemini_pro": {
            "provider": "google",
            "model_version": "gemini-2.5-pro",
            "temperature": 0.0
        },
        "gemini_flash": {
            "provider": "google",
            "model_version": "gemini-2.5-flash",
            "temperature": 0.0
        },
        "openai_gpt5": {
            "provider": "openai",
            "model_version": "gpt-5.2",
            "temperature": 0.0
        },
        "codex_gpt5": {
            "provider": "codex",
            "model_version": "gpt-5.3-codex",
            "temperature": 0.0
        },
        "ollama_local": {
            "provider": "ollama",
            "model_version": "gpt-oss:120b",
            "temperature": 0.0
        },
        "openrouter_deepseek": {
            "provider": "openrouter",
            "model_version": "deepseek/deepseek-v4-pro",
            "temperature": 0.0
        },
        "openrouter_qwen": {
            "provider": "openrouter",
            "model_version": "qwen/qwen3.5-122b-a10b",
            "temperature": 0.0
        },
        "openrouter_llama": {
            "provider": "openrouter",
            "model_version": "meta-llama/llama-4-maverick",
            "temperature": 0.0
        },
        "bedrock_qwen": {
            "provider": "bedrock_boto3",
            "model_version": "qwen.qwen3-235b-a22b-2507-v1:0",
            "temperature": 0.0
        },
        "bedrock_deepseek": {
            "provider": "bedrock_boto3",
            "model_version": "deepseek.v3.2",
            "temperature": 0.0
        },
        "bedrock_llama": {
            "provider": "bedrock_boto3",
            "model_version": "meta.llama3-3-70b-instruct-v1:0",
            "temperature": 0.0
        },
        "bedrock_kimi": {
            "provider": "bedrock_boto3",
            "model_version": "moonshotai.kimi-k2.5",
            "temperature": 0.0
        },
        "gemini_2_5_pro_vertex": {
            "provider": "vertex_gemini",
            "model_version": "gemini-2.5-pro",
            "temperature": 0.0
        },
        "bedrock_glm5": {
            "provider": "bedrock_boto3",
            "model_version": "zai.glm-5",
            "temperature": 0.0,
            "max_tokens": 8192
        }

    }

    active_conf = LLM_PROFILES[ACTIVE_PROFILE]
    model_provider = active_conf["provider"]
    model_version = active_conf["model_version"]
    temperature = active_conf["temperature"]
    max_tokens = active_conf.get("max_tokens")

    # Paths
    COMPLETE_PROMPT_PATH = "ground_truth.txt"
    INCOMPLETE_PROMPT = "Run a two-dimensional Rayleigh Benard convection cell simulation in a rectangular cavity that is 9 units wide and 1 unit tall, extruded a single cell in the spanwise direction with the front and back patches treated as empty to enforce 2D. Use 90 cells in x direction, 10 cells in y and 1 in z. Use the simpleFoam solver for steady-state simulation with gravity acting downward in the negative y direction. The reference state should be air-like with density 1.0 kg/m³, reference temperature 300 K, thermal expansion coefficient 3.3e-3 1/K, kinematic viscosity 1.5e-5 m²/s, and thermal diffusivity 2.1e-5 m²/s (Pr ≈ 0.7). Temperature of bottom wall is fixed at 305 K and the top wall at 295 K, while the vertical side walls are adiabatic. Apply no-slip velocity conditions on all solid walls, initialize the flow at rest with a uniform temperature field of 300 K. Use a fixed timestep of 1s (set deltaT = 1 s and do not enable adjustTimeStep, maxCo, or set a maxDeltaT), simulate up to 1000 s of physical time, and write results every 50 steps. Use k-epsilon turbulence model. Use RAS k-epsilon explicitly: Set constant/momentumTransport to simulationType RAS with model kEpsilon, turbulence on, and printCoeffs on (do not use laminar anywhere). In constant/physicalProperties use heRhoThermo with equationOfState Boussinesq; set molWeight 29, rho0 1.0, T0 300, beta 3.3e-3, Cv 714.3, Hf 0, mu 1.5e-5, and Pr 0.714. In 0/k set internalField uniform 1e-5 with kqRWallFunction on walls and empty front/back; in 0/epsilon set internalField uniform 1e-5 with epsilonWallFunction on walls and empty front/back; in 0/nut set internalField uniform 0 with nutkWallFunction on walls and empty front/back; in 0/alphat set internalField uniform 0 with compressible::alphatJayatillekeWallFunction on walls and Prt 0.85, and empty front/back. In 0/U set internalField uniform (0 0 0) with noSlip on all solid walls and empty front/back. In 0/T set internalField uniform 300, bottom fixedValue 305, top fixedValue 295, sides zeroGradient, and empty front/back. In 0/p set internalField uniform 0 with walls calculated and empty front/back; in 0/p_rgh set internalField uniform 0 with walls fixedFluxPressure and empty front/back. In system/fvSolution set p_rgh and (U|e|k|epsilon) solvers to tolerance 1e-8 and relTol 0.01, and set the corresponding *Final solvers to relTol 0"

    # Logging
    LOG_FILE = "conversation_log.txt"
