# src/state.py
from typing import Annotated, Any, Dict, List, Literal, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator

# Import constants if needed, or keep literals directly
# from .config import ASSUMED_ENTRY_POINT_FUNC

class AgentState(BaseModel):
    """Defines the structure of the state including execution results."""
    problem_description: str = Field(description="The initial problem description.")
    improvement_prompt: str = Field(description="The prompt used for requesting improvements.")
    messages: Annotated[List[BaseMessage], add_messages] = Field(
        default_factory=list,
        description="History of messages for LLM context."
    )
    current_code: str = Field(default="", description="The latest generated code.")
    previous_code: Optional[str] = Field(default=None, description="The code before the last improvement attempt.")
    iteration_count: int = Field(default=-1, description="Current improvement iteration count.")
    max_iterations: int = Field(description="Maximum number of improvement iterations allowed.")

    # Fields for execution results
    execution_status: Optional[Literal["success", "syntax_error", "runtime_error", "timeout", "not_run"]] = Field(
        default="not_run", description="Status of the last code execution attempt."
    )
    execution_output: Optional[Any] = Field(default=None, description="Output captured from the last execution.")
    execution_error: Optional[str] = Field(default=None, description="Error message if execution failed.")
    execution_time_ms: Optional[float] = Field(default=None, description="Execution time in milliseconds.")

    # You might add more fields later for testing, static analysis, etc.
    # test_results: Optional[Dict] = Field(default=None, description="Results from unit tests.")
    # static_analysis_report: Optional[Dict] = Field(default=None, description="Results from static analysis.")

    @field_validator('iteration_count')
    @classmethod
    def check_iteration_count(cls, v: int) -> int:
        if v < -1: raise ValueError('iteration_count must be >= -1')
        return v

    @field_validator('max_iterations')
    @classmethod
    def check_max_iterations(cls, v: int) -> int:
        if v < 0: raise ValueError('max_iterations must be >= 0')
        return v

    # Helper method to easily create the initial state dictionary
    @classmethod
    def initial_state(cls, problem: str, max_iter: int, improvement: str) -> Dict[str, Any]:
         # Basic validation before creating the state dict
        if not problem: raise ValueError("Problem description cannot be empty.")
        if max_iter < 0: raise ValueError("Maximum iterations must be non-negative.")
        return {
            "problem_description": problem,
            "max_iterations": max_iter,
            "improvement_prompt": improvement,
            "messages": [], # Ensure messages list exists initially
            "current_code": "",
            "previous_code": None,
            "iteration_count": -1,
            "execution_status": "not_run",
            "execution_output": None,
            "execution_error": None,
            "execution_time_ms": None,
        }