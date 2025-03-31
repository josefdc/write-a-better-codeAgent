# src/graph_nodes.py
import logging
from typing import Dict, Any, Literal

from langchain_core.messages import HumanMessage # No AIMessage needed here directly
# If llm_client is needed directly here in future: from langchain_anthropic import ChatAnthropic
from langgraph.errors import GraphInterrupt
from langgraph.graph import END
from langgraph.types import interrupt  # Add import for interrupt function

# Import components using relative paths
from .state import AgentState
from .llm_utils import call_llm, extract_python_code
from .execution import execute_code_docker
from .config import ASSUMED_ENTRY_POINT_FUNC, HUMAN_REVIEW_PROMPT, HUMAN_REVIEW_OPTIONS

logger = logging.getLogger(__name__)

# --- Node Definitions ---

async def generate_initial_code_node(state: AgentState, llm_client: Any) -> Dict[str, Any]:
    """Generates the initial Python code based on the problem description."""
    logger.info("Node: Generating Initial Code")
    problem = state.problem_description
    if not problem:
        raise ValueError("Problem description is missing in the state.")

    # Construct the prompt for initial code generation
    initial_prompt = HumanMessage(
        content=f"Write Python code to solve this problem:\n\n{problem}\n\n"
                f"Ensure the code defines a main function called '{ASSUMED_ENTRY_POINT_FUNC}' that takes no arguments and returns the final result. "
                "The code should be functional and follow standard Python practices. "
                "ONLY output the Python code inside a single ```python ... ``` block."
    )

    try:
        response = await call_llm(llm_client, [initial_prompt])
        extracted_code = extract_python_code(response.content)

        if extracted_code is None:
            logger.error("Failed to extract code from the initial LLM response. Cannot proceed.")
            # How to handle this? Raise error? Return special state?
            # For now, raise an error to stop the graph here.
            raise ValueError("Initial code generation failed to produce an extractable code block.")

        # Prepare state update
        update_dict = {
            "messages": [initial_prompt, response],
            "current_code": extracted_code,
            "iteration_count": 0,
            "previous_code": None, # No previous code for initial generation
            "execution_status": "not_run", # Reset execution status for the new code
            "execution_output": None,
            "execution_error": None,
            "execution_time_ms": None,
        }
        return update_dict

    except Exception as e:
        logger.error(f"Error during initial code generation: {e}", exc_info=True)
        # Propagate the error or handle it (e.g., return state indicating failure)
        raise # Re-raise for now


async def improve_code_node(state: AgentState, llm_client: Any) -> Dict[str, Any]:
    """Improves the current code based on history and the improvement prompt."""
    iteration = state.iteration_count + 1
    logger.info(f"Node: Improving Code (Iteration {iteration})")

    code_before_improvement = state.current_code
    current_messages = state.messages
    improvement_prompt_str = state.improvement_prompt
    if not improvement_prompt_str:
        logger.warning("Improvement prompt is empty, using default.")
        improvement_prompt_str = "write better code" # Fallback from config could be used here too

    improvement_prompt_msg = HumanMessage(content=improvement_prompt_str)

    # Construct message history for the LLM call
    history = list(current_messages) if current_messages else []
    # Add specific instruction for output format *if needed*, LLM might learn context
    # history.append(HumanMessage(content=f"ONLY output the improved python code...")) # Optional refinement

    # Include previous execution results in the history for context if failed?
    if state.execution_status != 'success' and state.execution_error:
         error_context = f"The previous code execution failed with status '{state.execution_status}' and error:\n{state.execution_error}"
         history.append(HumanMessage(content=error_context))
         logger.info("Added execution error context to improvement prompt.")

    history.append(improvement_prompt_msg)

    try:
        # Call LLM with updated history
        response = await call_llm(llm_client, history) # Pass combined history
        extracted_code = extract_python_code(response.content)

        update_dict = {
            "messages": [improvement_prompt_msg, response], # Append only new user prompt + AI response
            "iteration_count": iteration,
            "previous_code": code_before_improvement,
            "execution_status": "not_run", # Reset execution status
            "execution_output": None,
            "execution_error": None,
            "execution_time_ms": None,
        }

        if extracted_code is None:
            logger.warning(f"Failed to extract improved code in iteration {iteration}. Keeping previous code version.")
            # No need to set current_code, merge preserves the old one
        else:
            if extracted_code == code_before_improvement:
                logger.info(f"LLM did not change the code in iteration {iteration}.")
            update_dict["current_code"] = extracted_code # Update with new code

        return update_dict

    except Exception as e:
        logger.error(f"Error during code improvement: {e}", exc_info=True)
        raise



async def execute_code_node(state: AgentState) -> Dict[str, Any]:
    """Executes the current code using the Docker sandbox execution utility."""
    logger.info("Node: Execute Code (via Docker)")
    code_to_run = state.current_code
    exec_result = await execute_code_docker(code_to_run)

    # Mapear el resultado a las claves del AgentState
    return {
        "execution_status": exec_result.get("status", "error"),
        "execution_output": exec_result.get("output"),
        "execution_error": exec_result.get("error"),
        "execution_time_ms": exec_result.get("time_ms"),
    }


async def human_review_node(state: AgentState) -> Dict[str, Any]:
    """
    Node that pauses using interrupt() for human review and prepares state update upon resumption.
    """
    logger.info("Node: Human Review - Requesting human decision via interrupt()")

    # Information to show to the user
    interrupt_payload = {
        "code_to_review": state.current_code,
        "previous_code_available": state.previous_code is not None,
        "iteration": state.iteration_count,
        "last_exec_status": state.execution_status,
        "review_prompt": HUMAN_REVIEW_PROMPT
    }

    # --- Pause and get decision ---
    # First time this pauses. When resumed, it returns what was passed in Command(resume=...)
    user_decision = interrupt(value=interrupt_payload)
    logger.info(f"Resumed after human review. Decision: '{user_decision}'")

    # --- Prepare state updates based on the decision ---
    updates_to_state = {}
    if user_decision == HUMAN_REVIEW_OPTIONS['reject']:
        logger.info("User rejected improvement. Preparing state to revert code.")
        if state.previous_code is not None:
            updates_to_state["current_code"] = state.previous_code  # Revert to previous code
            updates_to_state["messages"] = [HumanMessage(content="User rejected the last improvement. Code reverted.")]
            # Reset execution status since code changed
            updates_to_state["execution_status"] = "not_run"
            updates_to_state["execution_output"] = None
            updates_to_state["execution_error"] = None
            updates_to_state["execution_time_ms"] = None
        else:
            logger.warning("User rejected, but no previous code found to revert to. Keeping current code.")
            updates_to_state["messages"] = [HumanMessage(content="User rejected the last improvement, but no previous version available.")]
    elif user_decision == HUMAN_REVIEW_OPTIONS['accept']:
        logger.info("User accepted improvement.")
        updates_to_state["messages"] = [HumanMessage(content="User accepted the last improvement.")]
    else:
        # Unexpected case if validation in the 'run' loop fails or is skipped
        logger.error(f"Received unexpected decision '{user_decision}' after interrupt. Proceeding without changes.")
        updates_to_state["messages"] = [HumanMessage(content=f"Proceeding after receiving unexpected review input: {user_decision}")]

    # Return the updates. The graph will continue to the next node (`execute_code`)
    # with the state updated according to the decision.
    return updates_to_state


def should_continue_node(state: AgentState) -> Literal["continue_improve", END]:
    """Decides whether to continue the improvement loop or end."""
    max_iterations = state.max_iterations
    current_iteration = state.iteration_count
    exec_status = state.execution_status

    logger.info(f"Condition Check: Iteration {current_iteration}/{max_iterations}, Status: {exec_status}")

    # Decision logic:
    if current_iteration < 0: # Should only happen if graph starts differently
        logger.warning("should_continue called before first iteration (iteration_count = -1).")
        # This likely means initial generation failed and graph ended prematurely
        # Or the flow is wrong. Let's assume it should end if exec never ran.
        return END

    if current_iteration >= max_iterations:
        logger.info("Condition: Reached max iterations, ending.")
        return END
    else:
        # Simple logic for now: continue unless max iterations reached.
        # Could add: stop if status is 'success' and user accepted?
        # Could add: stop if status is 'syntax_error' for 2 consecutive times?
        logger.info("Condition: Continuing improvement cycle.")
        return "continue_improve"