import time
from typing import Dict, List, Any
from dataclasses import dataclass, field
from utils import LLMService
import os

@dataclass
class LLMUserInterface:
    """
    Simulated User that answers clarification questions based on a hidden ground truth.
    """
    llm_service: LLMService
    incomplete_requirement: str
    complete_requirement: str
    clarifications: List[Dict[str, Any]] = field(default_factory=list)
    log_file: str = "conversation_log.txt"

    def __post_init__(self):
        """Ensures the log file exists, even if no conversation happens"""
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)        
        # Create the file if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, "a", encoding="utf-8") as f:
                pass

    def get_full_context(self) -> str:
        """Constructs the XML context for the Simulated User."""
        context_parts = [
            "<incomplete_requirement>",
            self.incomplete_requirement,
            "</incomplete_requirement>",
            "",
            "<complete_requirement>",
            self.complete_requirement,
            "</complete_requirement>",
            "",
            "<clarifications>",
        ]
        
        if self.clarifications:
            for entry in self.clarifications:
                q = entry.get("question", "")
                a = entry.get("answer", "")
                context_parts.append(f"<exchange>")
                context_parts.append(f"<question>{q}</question>")
                context_parts.append(f"<answer>{a}</answer>")
                context_parts.append("</exchange>")
        else:
            context_parts.append("No clarifications yet.")
            
        context_parts.append("</clarifications>")
        return "\n".join(context_parts)
    
    def answer_clarification_question(self, question: str) -> str:
        # Log the question immediately
        self._log_qa(question, "Thinking...")
        
        system_prompt = f"""You are a User who knows the COMPLETE requirement for a task.

        The Agent only has an INCOMPLETE version and is asking clarification questions.
        You have access to the COMPLETE requirement and should answer based strictly on that.

        IMPORTANT GUIDELINES:
        1. Answer questions based on the COMPLETE user requirement.
        2. Be specific and provide concrete values if they exist in the COMPLETE requirement.
        3. If the complete requirement doesn't specify something, tell the agent: "Make a reasonable assumption based on best practices."
        4. Be consistent with previous clarifications.

        CONTEXT:
        {self.get_full_context()}

        CRITICAL INSTRUCTIONS:
        - Be concise.
        - Do not explain your reasoning.
        - Do not provide information not asked for.
        - Do not make up new constraints that aren't in the complete requirement.
        - Answer like a human user would.
        """

        user_prompt = f"""The Agent asks:
        "{question}"

        Please provide a clear answer based on the COMPLETE requirement."""

        try:
            answer = self.llm_service.invoke(user_prompt, system_prompt)
            
            # Store history
            self.clarifications.append({
                "question": question,
                "answer": answer,
                "timestamp": time.time()
            })
            
            # Update log with actual answer
            self._update_last_log_entry(answer)
            return answer
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "I am unable to answer that at this time."
    
    def _log_qa(self, question: str, answer: str):
        """Append Q&A to a log file."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"=== {timestamp} ===\n")
            f.write(f"Q: {question}\n")
            f.write(f"A: {answer}\n\n")
    
    def _update_last_log_entry(self, answer: str):
        """Updates the last log entry (which currently says 'Thinking...') with the real answer."""
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Simple replacement of the last "A: Thinking..." line
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].startswith("A: Thinking..."):
                    lines[i] = f"A: {answer}\n"
                    break
            
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.writelines(lines)
        except Exception:
            pass