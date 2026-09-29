import os
import time
import random
import asyncio
import threading
from typing import Optional
import requests
import json
from pathlib import Path
import boto3
from botocore.config import Config as BotoConfig
from google import genai
from langchain_aws import ChatBedrockConverse
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama

from claude_agent_sdk import query, ClaudeAgentOptions

try:
    import tiktoken
    _enc = tiktoken.get_encoding("o200k_base")

    def count_tokens(text: str) -> int:
        return len(_enc.encode(text or ""))
except ImportError:
    def count_tokens(text: str) -> int:
        return max(1, len((text or "").split()))

# boto3 client creation is not thread-safe
_BOTO3_LOCK = threading.Lock()
BOTO3_CONFIG = dict(read_timeout=300, connect_timeout=30, retries={"max_attempts": 3, "mode": "adaptive"})
CUSTOM_PROVIDERS = ["claude_sdk", "codex", "bedrock_boto3", "vertex_gemini"]


class LLMService:
    def __init__(self, config: object):
        self.config = config
        self.model_provider = config.model_provider
        self.model_version = config.model_version
        self.temperature = config.temperature
        self.max_tokens = getattr(config, "max_tokens", None)
        self.llm = None
        self._boto3_client = None

        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.failed_calls = 0
        self.retry_count = 0
        # provider-reported usage (langchain usage_metadata); includes reasoning tokens the estimates miss
        self.api_usage_calls = 0
        self.api_input_tokens = 0
        self.api_output_tokens = 0
        self.api_reasoning_tokens = 0
        self.api_reasoning_chars = 0

        self._initialize_llm()

    def _initialize_llm(self):
        if self.model_provider.lower() == "bedrock":
            aws_region = os.getenv("AWS_DEFAULT_REGION", "us-west-2").strip()
            self.llm = ChatBedrockConverse(
                provider="amazon",
                model_id=self.model_version,
                temperature=self.temperature,
                region_name=aws_region
            )

        elif self.model_provider == "google":
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_version,
                temperature=self.temperature,
                google_api_key=os.getenv("GEMINI_API_KEY")
            )

        elif self.model_provider == "openai":
            self.llm = ChatOpenAI(
                model=self.model_version,
                temperature=self.temperature,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif self.model_provider == "openrouter":
            self.llm = ChatOpenAI(
                model=self.model_version,
                temperature=self.temperature,
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1"
            )

        elif self.model_provider.lower() == "ollama":
            # Get Ollama base URL from environment (for remote instances, e.g. an HPC compute node)
            # Default to localhost if not set
            ollama_base_url = os.getenv(
                "OLLAMA_BASE_URL", "http://127.0.0.1:11434")

            # Check if Ollama server is reachable
            try:
                response = requests.get(
                    f"{ollama_base_url}/api/version", timeout=5)
                response.raise_for_status()
                print(f"Connected to Ollama server at {ollama_base_url}")
            except requests.exceptions.RequestException as e:
                print(
                    f"Error: Cannot connect to Ollama server at {ollama_base_url}: {e}")
                print(
                    "Make sure Ollama is running. On a remote node, start with: ollama serve")
                raise ConnectionError(
                    f"Cannot connect to Ollama at {ollama_base_url}. Please ensure Ollama is running.")

            # Check if model exists, and pull if needed
            try:
                models_response = requests.get(
                    f"{ollama_base_url}/api/tags", timeout=5)
                models_response.raise_for_status()
                models_data = models_response.json()
                available_models = {m.get("name")
                                    for m in models_data.get("models", [])}

                if self.model_version not in available_models:
                    print(
                        f"Model '{self.model_version}' not found. Pulling model...")
                    # Pull the model (streaming)
                    pull_response = requests.post(
                        f"{ollama_base_url}/api/pull",
                        json={"name": self.model_version},
                        stream=True,
                        timeout=600
                    )
                    pull_response.raise_for_status()
                    for line in pull_response.iter_lines():
                        if line:
                            try:
                                msg = json.loads(line.decode("utf-8"))
                                status = msg.get("status") or msg.get(
                                    "error") or str(msg)
                                if status:
                                    print(f"[Ollama pull] {status}")
                            except (json.JSONDecodeError, KeyError):
                                pass
                    print(f"Model '{self.model_version}' pulled successfully")
                else:
                    print(f"Model '{self.model_version}' is already available")
            except requests.exceptions.RequestException as e:
                print(f"Warning: Could not check/pull model: {e}")
                print("Proceeding with model initialization anyway...")

            # Initialize ChatOllama with the specified model
            self.llm = ChatOllama(
                model=self.model_version,
                base_url=ollama_base_url,
                temperature=self.temperature
            )
        elif self.model_provider in ["claude_sdk", "codex", "bedrock_boto3", "vertex_gemini"]:
            self.llm = self.model_provider
        else:
            raise ValueError(
                f"{self.model_provider} is not a supported model_provider")

    def _get_boto3_client(self):
        with _BOTO3_LOCK:
            if self._boto3_client is None:
                aws_region = os.getenv("AWS_DEFAULT_REGION", "us-west-2").strip()
                self._boto3_client = boto3.client("bedrock-runtime", region_name=aws_region,
                                                  config=BotoConfig(**BOTO3_CONFIG))
            return self._boto3_client

    def _invoke_boto3_converse(self, system_prompt: Optional[str], user_prompt: str) -> str:
        return self._boto3_converse_messages(system_prompt, [{"role": "user", "content": user_prompt}])

    def _boto3_converse_messages(self, system_prompt: Optional[str], messages: list) -> str:
        client = self._get_boto3_client()

        system_msgs = [{"text": system_prompt}] if system_prompt else []
        converse_messages = [{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages]
        inference_config = {"temperature": self.temperature}
        if self.max_tokens:
            inference_config["maxTokens"] = self.max_tokens

        resp = client.converse(
            modelId=self.model_version,
            system=system_msgs,
            messages=converse_messages,
            inferenceConfig=inference_config
        )

        # recorded before the empty-response check: empty responses are billed too
        usage = resp.get("usage") or {}
        self.api_usage_calls += 1
        self.api_input_tokens += usage.get("inputTokens", 0) or 0
        self.api_output_tokens += usage.get("outputTokens", 0) or 0
        content = resp["output"]["message"]["content"]
        for p in content:
            reasoning = p.get("reasoningContent")
            if isinstance(reasoning, dict):
                self.api_reasoning_chars += len((reasoning.get("reasoningText") or {}).get("text", "") or "")

        text = "".join(p.get("text", "") for p in content if "text" in p).strip()
        return text

    def _invoke_vertex_gemini(self, system_prompt: Optional[str], user_prompt: str) -> str:
        project = os.environ.get(
            "GOOGLE_CLOUD_PROJECT", "gen-lang-client-0168697037")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

        client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )

        config = {}
        if system_prompt:
            config["system_instruction"] = system_prompt

        resp = client.models.generate_content(
            model=self.model_version,
            contents=user_prompt,
            config=config if config else None,
        )

        text = (resp.text or "").strip()
        return text

    def _load_codex_token(self) -> tuple[str, Optional[str]]:
        candidates = [
            Path(os.getenv("CODEX_HOME", "")) /
            "auth.json" if os.getenv("CODEX_HOME") else None,
            Path.home() / ".codex" / "auth.json",
        ]
        for p in candidates:
            if p and p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                token, account_id = self._scan_auth_json(data)
                if token:
                    return token, account_id
        raise FileNotFoundError(
            "No Codex auth file found. Run `codex` and sign in with ChatGPT.")

    @staticmethod
    def _scan_auth_json(obj):
        access, fallback, account_id = None, None, None

        def walk(o):
            nonlocal access, fallback, account_id
            if isinstance(o, dict):
                for k, v in o.items():
                    low = k.lower()
                    if isinstance(v, str) and v:
                        if low in {"access_token", "access"} and access is None:
                            access = v
                        elif low in {"token", "id_token"} and fallback is None:
                            fallback = v
                        elif low in {"account_id", "accountid"} and account_id is None:
                            account_id = v
                    else:
                        walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(obj)
        return (access or fallback), account_id

    @staticmethod
    def _extract_codex_text(resp_json: dict) -> str:
        if isinstance(resp_json.get("output_text"), str):
            return resp_json["output_text"]
        out = []
        for item in resp_json.get("output", []) or []:
            for c in item.get("content", []) or []:
                if c.get("type") in {"output_text", "text"} and isinstance(c.get("text"), str):
                    out.append(c["text"])
        return "\n".join(out).strip()

    def _invoke_codex(self, system_prompt: Optional[str], user_prompt: str) -> str:
        token, account_id = self._load_codex_token()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [
                            {"type": "input_text", "text": system_prompt}]})
        messages.append({"role": "user", "content": [
                        {"type": "input_text", "text": user_prompt}]})

        payload = {
            "model": self.model_version,
            "input": messages,
            "instructions": "You are running as a coding agent. Be concise and accurate.",
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "reasoning": {"summary": "auto"},
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
        }

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "standalone-codex/1.0",
        }
        if account_id:
            headers["ChatGPT-Account-Id"] = account_id

        r = requests.post("https://chatgpt.com/backend-api/codex/responses",
                          headers=headers, json=payload, timeout=120, stream=True)
        if not r.ok:
            raise requests.HTTPError(
                f"HTTP {r.status_code}: {r.text[:2000]}", response=r)

        chunks = []
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            line = str(raw).strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                evt = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(evt, dict):
                continue

            t = evt.get("type")
            if t == "response.output_text.delta" and isinstance(evt.get("delta"), str):
                chunks.append(evt["delta"])
            elif t == "response.output_text.done" and isinstance(evt.get("text"), str) and not chunks:
                chunks.append(evt["text"])
            else:
                txt = self._extract_codex_text(evt)
                if txt:
                    chunks.append(txt)

        return "".join(chunks).strip()

    # Internal Async Claude SDK Invoker
    async def _invoke_claude_sdk(
        self,
        system_prompt: Optional[str],
        user_prompt: str
    ) -> str:
        opts = ClaudeAgentOptions(
            model=self.model_version,
            system_prompt=system_prompt or "",
            tools=[],
        )

        chunks = []
        # Pass the user_prompt to query
        async for msg in query(prompt=user_prompt, options=opts):
            content = getattr(msg, "content", None)
            if not isinstance(content, list):
                continue
            for block in content:
                text = getattr(block, "text", None)
                if isinstance(text, str) and text:
                    chunks.append(text)
        return "\n".join(chunks).strip()

    def invoke(self,
               user_prompt: str,
               system_prompt: Optional[str] = None,
               max_retries: int = 5) -> str:

        self.total_calls += 1
        prompt_tokens = 0
        messages = []

        if self.model_provider in CUSTOM_PROVIDERS:
            prompt_tokens = count_tokens(
                system_prompt or "") + count_tokens(user_prompt)
        else:
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=user_prompt))

            for m in messages:
                prompt_tokens += self.llm.get_num_tokens(m.content)

        def call():
            # Custom Integrations (Claude & Codex)
            if self.model_provider == "claude_sdk":
                return asyncio.run(self._invoke_claude_sdk(system_prompt, user_prompt))
            if self.model_provider == "codex":
                return self._invoke_codex(system_prompt, user_prompt)
            if self.model_provider == "bedrock_boto3":
                return self._invoke_boto3_converse(system_prompt, user_prompt)
            if self.model_provider == "vertex_gemini":
                return self._invoke_vertex_gemini(system_prompt, user_prompt)
            # Standard Langchain Integration
            return self._invoke_langchain(messages)

        return self._run_with_retries(call, prompt_tokens, max_retries)

    def invoke_messages(self,
                        messages: list,
                        system_prompt: Optional[str] = None,
                        max_retries: int = 5) -> str:
        """messages: [{"role": "user"|"assistant", "content": str}], starting with "user" and alternating."""
        if not messages:
            raise ValueError("messages must not be empty")
        for i, m in enumerate(messages):
            want = "user" if i % 2 == 0 else "assistant"
            if m.get("role") != want:
                raise ValueError(f"message {i} has role {m.get('role')!r}; expected {want!r} "
                                 "(messages must start with 'user' and alternate)")
        if self.model_provider in CUSTOM_PROVIDERS and self.model_provider != "bedrock_boto3":
            raise NotImplementedError(f"invoke_messages is not supported for {self.model_provider}")

        self.total_calls += 1
        if self.model_provider == "bedrock_boto3":
            prompt_tokens = count_tokens(system_prompt or "") + sum(count_tokens(m["content"]) for m in messages)

            def call():
                return self._boto3_converse_messages(system_prompt, messages)
        else:
            lc_messages = [SystemMessage(content=system_prompt)] if system_prompt else []
            lc_messages += [HumanMessage(content=m["content"]) if m["role"] == "user"
                            else AIMessage(content=m["content"]) for m in messages]
            prompt_tokens = sum(self.llm.get_num_tokens(m.content) for m in lc_messages)

            def call():
                return self._invoke_langchain(lc_messages)

        return self._run_with_retries(call, prompt_tokens, max_retries)

    def _invoke_langchain(self, messages: list) -> str:
        response = self.llm.invoke(messages)
        self._record_api_usage(response)
        raw_content = response.content
        if isinstance(raw_content, list):
            text_parts = []
            for part in raw_content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            return "".join(text_parts)
        return str(raw_content)

    def _run_with_retries(self, call, prompt_tokens: int, max_retries: int) -> str:
        retry_count = 0
        while True:
            try:
                final_content = call()

                # Post-processing and Token Counting
                if not final_content or not final_content.strip():
                    raise ValueError(
                        f"Received empty response from {self.model_provider}.")

                if self.model_provider in CUSTOM_PROVIDERS:
                    completion_tokens = count_tokens(final_content)
                else:
                    completion_tokens = self.llm.get_num_tokens(final_content)

                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_tokens += (prompt_tokens + completion_tokens)
                return final_content

            except Exception as e:
                retry_count += 1
                self.retry_count += 1
                if retry_count > max_retries:
                    self.failed_calls += 1
                    raise Exception(
                        f"Maximum retries ({max_retries}) exceeded: {str(e)}")

                base_delay = 1.0
                delay = min(60.0, base_delay * (2 ** (retry_count - 1)))
                jitter = random.uniform(0, 0.1 * delay)
                sleep_time = delay + jitter

                print(
                    f"Error: {str(e)}. Retrying in {sleep_time:.2f}s (attempt {retry_count}/{max_retries})")
                time.sleep(sleep_time)

    def _record_api_usage(self, response) -> None:
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            return
        self.api_usage_calls += 1
        self.api_input_tokens += usage.get("input_tokens", 0) or 0
        self.api_output_tokens += usage.get("output_tokens", 0) or 0
        details = usage.get("output_token_details") or {}
        self.api_reasoning_tokens += details.get("reasoning", 0) or 0

    def print_statistics(self, name: str = "LLM Service") -> None:
        """Print the current statistics of the LLM service."""
        print(f"\n<Statistics: {name}>")
        print(f"Total calls: {self.total_calls}")
        print(f"Failed calls: {self.failed_calls}")
        print(f"Total retries: {self.retry_count}")
        print(f"Total tokens (approx): {self.total_tokens}")
        print(f"  - Prompt: {self.total_prompt_tokens}")
        print(f"  - Completion: {self.total_completion_tokens}")
        print(f"</Statistics: {name}>\n")

    def get_statistics(self, name: str = "LLM Service") -> None:
        """Return the current statistics of the LLM service."""
        return {
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "total_retries": self.retry_count,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "api_usage_calls": self.api_usage_calls,
            "api_input_tokens": self.api_input_tokens,
            "api_output_tokens": self.api_output_tokens,
            "api_reasoning_tokens": self.api_reasoning_tokens,
            "api_reasoning_chars": self.api_reasoning_chars
        }
