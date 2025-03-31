import argparse
import asyncio
import contextlib
import getpass
import logging
import operator
import os
import re
import time  # For timing execution
import traceback  # For formatting exceptions
import uuid
from io import StringIO  # To potentially capture stdout
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence

# Third-party libraries
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                    wait_exponential)

# --- Constants ---
ANTHROPIC_API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_IMPROVEMENT_PROMPT = "write better code"
LOG_FORMAT = '%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s'
EXECUTION_TIMEOUT_SECONDS = 10  # Basic timeout for execution attempt
ASSUMED_ENTRY_POINT_FUNC = "solve_problem"  # Function name the agent assumes LLM creates

# Model Configuration
PREFERRED_ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620"
FALLBACK_ANTHROPIC_MODEL = "claude-3-haiku-20240307"

# Configure logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# --- State Definition using Pydantic ---

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
    # New fields for execution results
    execution_status: Optional[Literal["success", "syntax_error", "runtime_error", "timeout", "not_run"]] = Field(
        default="not_run", description="Status of the last code execution attempt."
    )
    execution_output: Optional[Any] = Field(default=None, description="Output captured from the last execution.")
    execution_error: Optional[str] = Field(default=None, description="Error message if execution failed.")
    execution_time_ms: Optional[float] = Field(default=None, description="Execution time in milliseconds.")

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


# --- Agent Class ---

class CodeImprovingAgent:
    """
    Encapsulates the logic for the code improving agent.
    (Async, Streaming, Checkpointed, HITL, Execution version).
    """

    def __init__(self, llm_client: ChatAnthropic):
        self.llm_client = llm_client
        self.node_generate_initial = "generate_initial_code"
        self.node_improve_code = "improve_code"
        self.node_human_review = "human_review"
        self.node_execute_code = "execute_code"  # New node name
        self.graph: Optional[CompiledGraph] = None

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying async LLM call after error ({type(retry_state.outcome.exception()).__name__}). Attempt #{retry_state.attempt_number}"
        )
    )
    async def _call_llm(self, messages: Sequence[BaseMessage]) -> AIMessage:
        logger.info(f"Calling LLM async with {len(messages)} messages.")
        response = await self.llm_client.ainvoke(messages)
        if not isinstance(response, AIMessage): raise TypeError(f"LLM expected AIMessage, got {type(response)}")
        if not response.content or not isinstance(response.content, str) or not response.content.strip(): logger.warning("LLM response content is empty or invalid.")
        logger.debug(f"LLM raw response content: {response.content[:500]}...")
        return response

    def _extract_python_code(self, text: str) -> Optional[str]:
        match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
        if match:
            extracted_code = match.group(1).strip()
            logger.info(f"Successfully extracted Python code block ({len(extracted_code)} chars).")
            return extracted_code
        else:
            lines = text.strip().split('\n')
            if len(lines) > 1 and (lines[0].strip().startswith(('import ', 'def ', 'class ')) or (len(lines) > 1 and lines[1].strip().startswith('    '))):
                logger.warning("No ```python ... ``` block found, but content looks like code. Using raw content.")
                return text.strip()
            logger.warning("Could not extract Python code block using ```python ... ```.")
            return None

    # --- Node Definitions (as async methods) ---

    async def generate_initial_code(self, state: AgentState) -> dict:
        logger.info("Node: Generating Initial Code")
        problem = state.problem_description
        initial_prompt = HumanMessage(
            content=f"Write Python code to solve this problem:\n\n{problem}\n\n"
                    f"Ensure the code defines a main function called '{ASSUMED_ENTRY_POINT_FUNC}' that takes no arguments and returns the final result. "
                    "The code should be functional, follow PEP 8, include comments/docstrings. "
                    "ONLY output the python code inside a single ```python ... ``` block."
        )
        response = await self._call_llm([initial_prompt])
        extracted_code = self._extract_python_code(response.content)
        if extracted_code is None:
            logger.error("Failed to extract code from the initial LLM response. Cannot proceed.")
            raise ValueError("Initial code generation failed to produce an extractable code block.")
        # Reset execution status for the new code
        return {
            "messages": [initial_prompt, response],
            "current_code": extracted_code,
            "iteration_count": 0,
            "execution_status": "not_run",
            "execution_output": None,
            "execution_error": None,
            "execution_time_ms": None,
            "previous_code": None  # No previous code for initial generation
        }

    async def improve_code(self, state: AgentState) -> dict:
        iteration = state.iteration_count + 1
        logger.info(f"Node: Improving Code (Iteration {iteration})")
        code_before_improvement = state.current_code
        current_messages = state.messages
        improvement_prompt_msg = HumanMessage(content=state.improvement_prompt)
        history = list(current_messages) if current_messages else []
        history.append(HumanMessage(content=f"ONLY output the improved python code, containing the '{ASSUMED_ENTRY_POINT_FUNC}' function, inside a single ```python ... ``` block."))
        response = await self._call_llm(history + [improvement_prompt_msg])
        extracted_code = self._extract_python_code(response.content)
        update_dict = {
            "messages": [improvement_prompt_msg, response],
            "iteration_count": iteration,
            "previous_code": code_before_improvement,
            # Reset execution status as code has changed
            "execution_status": "not_run",
            "execution_output": None,
            "execution_error": None,
            "execution_time_ms": None,
        }
        if extracted_code is None:
            logger.warning(f"Failed to extract improved code in iteration {iteration}. Keeping previous code version.")
            update_dict["current_code"] = code_before_improvement
        else:
            update_dict["current_code"] = extracted_code
        return update_dict

    async def human_review(self, state: AgentState) -> dict:
        """Node that triggers the interrupt for human review."""
        logger.info("Node: Human Review - Interrupting for input.")
        current_code = state.current_code
        # The actual input gathering and state update happens in the run loop
        # after the interrupt is caught and resumed.
        # This node simply raises the interrupt.
        raise GraphInterrupt(value={"code_to_review": current_code})
        # The code below this line won't be executed due to the interrupt.
        # It's kept here conceptually but the logic moved to the run loop.
        # return {} # Return empty dict, state unchanged until resume + next node

    # Note: The _get_human_decision method is removed as its logic is now
    # handled directly within the run loop's GraphInterrupt handler.

    async def execute_code(self, state: AgentState) -> dict:
        """Async Node: Executes the current code (UNSAFE)."""
        logger.warning("Node: Executing Code - WARNING: Using exec() is insecure!")
        code_to_run = state.current_code
        if not code_to_run:
            logger.warning("No code found to execute.")
            return {"execution_status": "not_run"}

        start_time = time.perf_counter()
        status = "runtime_error"  # Default to error
        output = None
        error_msg = None

        # Create a restricted namespace for exec, but it's NOT a true sandbox
        local_namespace = {}
        # Import necessary modules into the global namespace for exec
        import random # Example problem might need random
        global_namespace = {
            '__builtins__': __builtins__,  # Provide standard builtins
            # Add safe modules if needed, e.g. 'math': math
            'random': random,
            'time': time # Allow timing within the code itself if needed
        }

        # Use StringIO to capture stdout if needed, though we assume a return value
        stdout_capture = StringIO()
        try:
            # First, try to compile the code to catch syntax errors
            compiled_code = compile(code_to_run, '<string>', 'exec')

            # Execute the compiled code to define functions/classes
            # Use asyncio.to_thread to run sync exec in a separate thread
            # Timeout is complex with exec; skipping for now.
            await asyncio.to_thread(
                exec, compiled_code, global_namespace, local_namespace
            )

            # Check if the assumed entry point function exists
            if ASSUMED_ENTRY_POINT_FUNC not in local_namespace:
                raise NameError(f"Function '{ASSUMED_ENTRY_POINT_FUNC}' not found in executed code.")

            # Call the entry point function
            # Run the potentially blocking function call in a thread
            with contextlib.redirect_stdout(stdout_capture):  # Capture prints
                output = await asyncio.to_thread(local_namespace[ASSUMED_ENTRY_POINT_FUNC])

            status = "success"
            logger.info(f"Code execution successful. Output type: {type(output)}")

        except SyntaxError as e:
            status = "syntax_error"
            error_msg = f"Syntax Error: {e}\n{traceback.format_exc(limit=1)}"
            logger.error(error_msg)
        except Exception as e:
            status = "runtime_error"
            error_msg = f"Runtime Error: {e}\n{traceback.format_exc(limit=5)}"
            logger.error(error_msg)
        finally:
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            logger.info(f"Execution finished with status: {status}, Time: {execution_time_ms:.2f} ms")

        # Capture stdout in output if return value is None
        captured_stdout = stdout_capture.getvalue()
        if output is None and captured_stdout:
            output = captured_stdout
            logger.info("Captured stdout as execution output.")

        return {
            "execution_status": status,
            "execution_output": repr(output) if output is not None else None,  # Store repr for safety
            "execution_error": error_msg,
            "execution_time_ms": execution_time_ms,
        }

    async def should_continue(self, state: AgentState) -> Literal["continue", "__end__"]:
        """Async Node Logic: Decides whether to continue iterating or end."""
        max_iterations = state.max_iterations
        current_iteration = state.iteration_count
        logger.info(f"Condition Check: Iteration {current_iteration}/{max_iterations}")
        if current_iteration >= max_iterations:
            logger.info("Condition: Reached max iterations, ending.")
            return END
        else:
            logger.info("Condition: Continuing to improve code.")
            return "continue"

    # --- Graph Construction & Compilation ---

    def _build_graph_definition(self) -> StateGraph:
        """Defines the graph structure including execution and human review steps."""
        workflow = StateGraph(AgentState)
        workflow.add_node(self.node_generate_initial, self.generate_initial_code)
        workflow.add_node(self.node_improve_code, self.improve_code)
        workflow.add_node(self.node_human_review, self.human_review)
        workflow.add_node(self.node_execute_code, self.execute_code)  # Add execution node

        workflow.add_edge(START, self.node_generate_initial)
        # After initial generation, execute it
        workflow.add_edge(self.node_generate_initial, self.node_execute_code)
        # After execution, check if we should improve or end
        workflow.add_conditional_edges(
            self.node_execute_code,
            self.should_continue,
            {"continue": self.node_improve_code, END: END}
        )
        # After improving, go to human review (which interrupts)
        workflow.add_edge(self.node_improve_code, self.node_human_review)
        # After human review (interrupt handled in run loop), execute the potentially reverted/accepted code
        # The graph resumes *after* the human_review node, so the next step is execution.
        workflow.add_edge(self.node_human_review, self.node_execute_code)
        # The conditional edge to loop/end now comes *after* execution

        return workflow

    def compile(self, checkpointer: Optional[BaseCheckpointSaver] = None):
        """Compiles the graph with an optional checkpointer."""
        workflow = self._build_graph_definition()
        logger.info(f"Compiling the agent graph {'with' if checkpointer else 'without'} checkpointer.")
        # Add interrupt_before=[self.node_human_review] to pause before the node runs
        # Or interrupt_after=[self.node_human_review] to pause after it runs (and potentially raises GraphInterrupt)
        # Let's interrupt *before* the human review node runs its logic (which is now just raising the interrupt)
        self.graph = workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=[self.node_human_review] # Interrupt before human_review node
        )

    # --- Execution Logic ---

    async def run(self, problem_description: str, max_iterations: int, improvement_prompt: str, thread_id: str):
        """Async runs the agent graph, handling interrupts and streaming results."""
        if self.graph is None:
            logger.error("Graph not compiled. Call agent.compile(checkpointer) first.")
            return

        logger.info(f"Starting Agent Execution Async for thread_id='{thread_id}' (Max Improve Iterations: {max_iterations})")
        initial_state_dict = {
            "problem_description": problem_description,
            "max_iterations": max_iterations,
            "improvement_prompt": improvement_prompt,
            "messages": []  # Ensure messages list exists initially
        }
        config = {"configurable": {"thread_id": thread_id}}
        current_input: Any = initial_state_dict
        last_printed_code: Optional[str] = None
        last_known_state_dict: Optional[Dict] = None # Store last known state dict from output

        while True:
            should_resume_with: Optional[Any] = None
            try:
                logger.debug(f"Streaming events with input/resume type: {type(current_input)}")
                async for event in self.graph.astream_events(current_input, config=config, version="v1"):
                    kind = event["event"]
                    if kind == "on_chain_end":
                        node_name = event["name"]
                        output_data = event["data"].get("output")
                        if output_data and isinstance(output_data, dict):
                            # Store the full output dict as the last known state
                            last_known_state_dict = output_data
                            current_code = output_data.get("current_code")
                            current_iter = output_data.get("iteration_count", -1)

                            # Print code only if it changed after generation/improvement
                            if node_name in [self.node_generate_initial, self.node_improve_code]:
                                if current_code is not None and current_code != last_printed_code:
                                    print(f"\n--- Code after {node_name} (Iteration {current_iter}) ---")
                                    print("-" * 35)
                                    print(current_code)
                                    print("-" * 35)
                                    last_printed_code = current_code

                            # Print execution results after execution node
                            if node_name == self.node_execute_code:
                                status = output_data.get('execution_status', 'unknown')
                                exec_time = output_data.get('execution_time_ms')
                                exec_output = output_data.get('execution_output')
                                exec_error = output_data.get('execution_error')
                                print(f"\n--- Execution Result (Iteration {current_iter}) ---")
                                print(f"Status: {status.upper()}")
                                if exec_time is not None: print(f"Time:   {exec_time:.2f} ms")
                                if status == 'success': print(f"Output: {exec_output}")
                                if exec_error: print(f"Error:  {exec_error}")
                                print("-" * 35)

                # Stream finished without interruption
                logger.info("Agent stream finished.")
                break # Exit the while loop

            except GraphInterrupt as interrupt_request:
                logger.warning(f"Graph interrupted before node: {self.node_human_review}")
                # Retrieve the current state at the point of interruption
                current_state_snapshot = await self.graph.aget_state(config)
                if not current_state_snapshot:
                    logger.error("Could not retrieve state during interrupt. Aborting.")
                    break
                current_state = AgentState(**current_state_snapshot.values)
                code_to_review = current_state.current_code
                previous_code = current_state.previous_code

                print("\n--- ACTION REQUIRED ---")
                print("Code requires review:")
                print("-" * 35)
                print(code_to_review)
                print("-" * 35)
                # Get user decision (synchronous input in a thread)
                user_decision = await asyncio.to_thread(
                    input, "Type 'accept' to continue, 'reject' to revert code: "
                )
                user_decision = user_decision.strip().lower()

                updates_for_resume = {}
                if user_decision == "reject":
                    logger.info("User rejected improvement. Reverting code.")
                    if previous_code is not None:
                        updates_for_resume["current_code"] = previous_code
                        # Reset execution status as code has changed
                        updates_for_resume["execution_status"] = "not_run"
                        updates_for_resume["execution_output"] = None
                        updates_for_resume["execution_error"] = None
                        updates_for_resume["execution_time_ms"] = None
                    else:
                        logger.warning("Cannot revert code, no previous code found in state.")
                    updates_for_resume["messages"] = [HumanMessage(content="User rejected the last improvement.")]
                else: # Accept or any other input treated as accept
                    logger.info("User accepted improvement or provided other input.")
                    updates_for_resume["messages"] = [HumanMessage(content="User accepted the last improvement.")]

                # Prepare to resume the graph *after* the human_review node
                # by providing the updates. LangGraph applies these updates to the state
                # before continuing to the next node (execute_code).
                should_resume_with = updates_for_resume
                logger.info(f"Resuming graph with updates: {list(should_resume_with.keys())}")


            except ValueError as ve:
                logger.error(f"Agent execution failed due to ValueError: {ve}", exc_info=False)
                print(f"\nERROR: Agent execution failed - {ve}")
                break
            except Exception as e:
                logger.error(f"Agent execution failed: {e}", exc_info=True)
                print("\nERROR: Agent execution encountered an unexpected error. Check logs.")
                break

            # If we caught an interrupt and got a decision, set it as input for the next loop iteration
            if should_resume_with is not None:
                current_input = should_resume_with # This will be passed to graph.astream_events to resume
            else:
                # If the loop finished normally (no interrupt), break
                logger.info("Loop finished normally.")
                break


        # Retrieve and print final state after loop finishes
        logger.info(f"Retrieving final state for thread_id='{thread_id}'.")
        try:
            final_state_snapshot = await self.graph.aget_state(config)
            if final_state_snapshot:
                final_state = AgentState(**final_state_snapshot.values)
                print(f"\nFinal Code Summary (Thread: {thread_id}, Iterations: {final_state.iteration_count}):")
                print("-" * 35)
                print(final_state.current_code)
                print("-" * 35)
                print(f"Final Execution Status: {final_state.execution_status}")
            # Fallback using last known state from stream if aget_state fails
            elif last_known_state_dict:
                final_state = AgentState(**last_known_state_dict)
                logger.warning("Could not retrieve definitive final state via aget_state, using last known state from stream.")
                print(f"\nFinal Code Summary (Fallback, Thread: {thread_id}, Iterations: {final_state.iteration_count}):")
                print("-" * 35)
                print(final_state.current_code)
                print("-" * 35)
                print(f"Last Execution Status: {final_state.execution_status}")
            else:
                print("\nNo final state captured.")
        except Exception as e:
            logger.error(f"Failed to retrieve final state: {e}", exc_info=True)
            print("\nError retrieving final state.")

    def display_graph_visualization(self):
        if self.graph is None: logger.warning("Graph not compiled yet."); return
        try:
            from IPython.display import Image, display
            logger.info("Generating graph visualization...")
            # Ensure the graph object used for drawing is the underlying StateGraph
            # The CompiledGraph itself might not have the draw method directly in all versions/setups.
            # Accessing the internal graph definition might be needed if `self.graph.get_graph()` fails.
            # Assuming `self.graph.get_graph()` returns the drawable object.
            display(Image(self.graph.get_graph().draw_mermaid_png()))
        except ImportError:
             logger.warning("IPython.display not found. Cannot display visualization inline.")
             logger.info("Install 'ipython' and 'pygraphviz' or 'mermaid-cli' for visualization.")
             self._print_graph_text_flow()
        except Exception as e:
            logger.warning(f"Could not generate graph visualization: {e}.", exc_info=False)
            self._print_graph_text_flow()

    def _print_graph_text_flow(self):
        """Prints a textual representation of the graph flow."""
        print("\n--- Graph Flow ---")
        print("START -> generate_initial_code -> execute_code")
        print("  `-> should_continue")
        print("      |-- (if END) ----> END")
        print("      `-- (if continue) -> improve_code -> human_review (Interrupt)")
        print("                           `-- (Resume) -> execute_code -> (back to should_continue)")
        print("-" * 18)


# --- Helper Functions ---
def setup_anthropic_client() -> ChatAnthropic:
    if ANTHROPIC_API_KEY_ENV_VAR not in os.environ:
        logger.info(f"{ANTHROPIC_API_KEY_ENV_VAR} not found in environment variables.")
        try:
            api_key = getpass.getpass(f"Enter {ANTHROPIC_API_KEY_ENV_VAR}: ")
            os.environ[ANTHROPIC_API_KEY_ENV_VAR] = api_key
        except Exception as e:
            logger.error(f"Could not get API key via getpass: {e}")
            raise ValueError(f"Anthropic API Key not provided via environment variable or prompt.") from e
    else:
        logger.info(f"Using {ANTHROPIC_API_KEY_ENV_VAR} from environment.")

    try:
        model = ChatAnthropic(model=PREFERRED_ANTHROPIC_MODEL)
        logger.info(f"Using Anthropic model: {PREFERRED_ANTHROPIC_MODEL}")
    except Exception:
        logger.warning(f"Could not initialize {PREFERRED_ANTHROPIC_MODEL}. Falling back to {FALLBACK_ANTHROPIC_MODEL}.", exc_info=False)
        try:
            model = ChatAnthropic(model=FALLBACK_ANTHROPIC_MODEL)
            logger.info(f"Using Anthropic model: {FALLBACK_ANTHROPIC_MODEL}")
        except Exception as fallback_e:
            logger.error(f"Failed to initialize fallback model {FALLBACK_ANTHROPIC_MODEL}: {fallback_e}", exc_info=True)
            raise ValueError("Could not initialize any Anthropic model. Check API key and model availability.") from fallback_e
    return model


# --- Script Entry Point ---
async def main():  # Async main
    parser = argparse.ArgumentParser(description="Iteratively improve code with HITL and Execution.")
    parser.add_argument("-p", "--problem", type=str, required=True, help="Problem description.")
    parser.add_argument("-i", "--iterations", type=int, default=DEFAULT_MAX_ITERATIONS, help=f"Max improvement iterations (default: {DEFAULT_MAX_ITERATIONS}).")
    parser.add_argument("--prompt", type=str, default=DEFAULT_IMPROVEMENT_PROMPT, help=f"Improvement prompt (default: '{DEFAULT_IMPROVEMENT_PROMPT}').")
    parser.add_argument("--thread-id", type=str, default=None, help="Thread ID (allows resuming). Generates new if omitted.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")

    # 1. Setup
    try:
        llm = setup_anthropic_client()
    except ValueError as e:
        print(f"Error setting up LLM Client: {e}")
        return # Exit if setup fails

    checkpointer = MemorySaver()
    agent = CodeImprovingAgent(llm_client=llm)
    agent.compile(checkpointer=checkpointer)
    thread_id = args.thread_id if args.thread_id else str(uuid.uuid4())
    logger.info(f"Using Thread ID: {thread_id}")

    # Optionally display graph structure
    # agent.display_graph_visualization() # Requires IPython/Mermaid

    # 2. Run
    await agent.run(
        problem_description=args.problem,
        max_iterations=args.iterations,
        improvement_prompt=args.prompt,
        thread_id=thread_id
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()
