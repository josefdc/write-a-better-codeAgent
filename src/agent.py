# src/agent.py
import asyncio
import logging
import sys
from typing import Any, Dict, Optional
import functools # For partial

from langchain_core.messages import HumanMessage
# If needed directly: from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.graph.graph import CompiledGraph

# Import components using relative paths
from .state import AgentState
from .graph_nodes import (generate_initial_code_node, improve_code_node,
                          execute_code_node, human_review_node,
                          should_continue_node)
from .config import (HUMAN_REVIEW_OPTIONS, HUMAN_REVIEW_PROMPT, # Import review details
                     DEFAULT_IMPROVEMENT_PROMPT)

logger = logging.getLogger(__name__)

class CodeImprovingAgent:
    """
    Manages the LangGraph workflow for iterative code improvement.
    Handles graph definition, compilation, and execution loop with interrupts.
    """
    def __init__(self, llm_client: Any):
        """
        Initializes the agent with an LLM client.

        Args:
            llm_client: An initialized LangChain compatible chat model client.
        """
        if llm_client is None:
             raise ValueError("LLM client cannot be None.")
        self.llm_client = llm_client
        self.graph: Optional[CompiledGraph] = None

        # Define node names used in the graph
        self.node_generate_initial = "generate_initial_code"
        self.node_improve_code = "improve_code"
        self.node_execute_code = "execute_code"
        self.node_human_review = "human_review" # Node that raises interrupt
        self.conditional_edge_entry = "should_continue" # Entry point for conditional edge

    def _build_graph_definition(self) -> StateGraph:
        """Defines the graph structure using nodes from graph_nodes.py."""
        workflow = StateGraph(AgentState)

        # Bind the LLM client to the nodes that need it using functools.partial
        # Nodes expect `state` and optionally `config`. `partial` fixes the `llm_client` argument.
        generate_with_llm = functools.partial(generate_initial_code_node, llm_client=self.llm_client)
        improve_with_llm = functools.partial(improve_code_node, llm_client=self.llm_client)

        # Add nodes to the workflow
        # Use the partial functions where the LLM client is needed
        workflow.add_node(self.node_generate_initial, generate_with_llm)
        workflow.add_node(self.node_improve_code, improve_with_llm)
        # These nodes don't directly need the llm_client instance passed in
        workflow.add_node(self.node_execute_code, execute_code_node)
        workflow.add_node(self.node_human_review, human_review_node)

        # Define the graph edges and control flow
        workflow.add_edge(START, self.node_generate_initial)

        # After initial generation, always execute the code
        workflow.add_edge(self.node_generate_initial, self.node_execute_code)

        # After execution, decide whether to improve or end
        workflow.add_conditional_edges(
            self.node_execute_code,
            should_continue_node, # Function from graph_nodes
            {
                "continue_improve": self.node_improve_code, # If continue, try to improve
                END: END, # Otherwise, end the graph
            },
        
        )

        # After attempting improvement, go to human review (which interrupts)
        workflow.add_edge(self.node_improve_code, self.node_human_review)

        # After human review (interrupt handled in run loop), execute the accepted/reverted code
        # The graph resumes *after* the human_review node was interrupted *before*.
        # The next node needs to be execution.
        workflow.add_edge(self.node_human_review, self.node_execute_code)

        return workflow

    def compile(self, checkpointer: Optional[BaseCheckpointSaver] = None):
        """Compiles the graph with an optional checkpointer."""
        if self.graph:
            logger.warning("Graph already compiled. Re-compiling.")

        workflow = self._build_graph_definition()
        logger.info(f"Compiling the agent graph {'with' if checkpointer else 'without'} checkpointer.")

        # Compile without explicit interrupts. The human_review_node will handle interruption using interrupt()
        self.graph = workflow.compile(
            checkpointer=checkpointer
            # No interrupt_after needed as we're using interrupt() function
        )
        logger.info("Graph compiled successfully.")


    async def run(
        self,
        problem_description: str,
        max_iterations: int,
        improvement_prompt: str = DEFAULT_IMPROVEMENT_PROMPT,
        thread_id: Optional[str] = None
    ):
        """
        Runs the agent graph asynchronously for a given problem.

        Handles the execution loop, streaming of events, and human-in-the-loop interrupts.

        Args:
            problem_description: The problem for the agent to solve.
            max_iterations: Maximum number of improvement iterations.
            improvement_prompt: The prompt used for asking for improvements.
            thread_id: The specific thread ID for conversation history and state.
                       Must be provided if a checkpointer is used.
        """
        if self.graph is None:
            logger.error("Graph not compiled. Call agent.compile(checkpointer) first.")
            raise RuntimeError("Agent graph must be compiled before running.")
        if self.graph.checkpointer and not thread_id:
             logger.error("A thread_id is required when running a compiled graph with a checkpointer.")
             raise ValueError("A thread_id must be provided when using a checkpointer.")
        elif not self.graph.checkpointer and thread_id:
             logger.warning("A thread_id was provided, but the graph has no checkpointer. It will be ignored.")

        logger.info(f"Starting Agent Execution Async for thread_id='{thread_id}' (Max Improve Iterations: {max_iterations})")

        # Prepare initial state and config
        initial_state = AgentState.initial_state(
            problem=problem_description,
            max_iter=max_iterations,
            improvement=improvement_prompt
        )
        config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
        current_input: Any = initial_state # Start with the full initial state
        last_printed_code: Optional[str] = None
        last_known_state_values: Optional[Dict] = None # Store last known state values from output

        run_count = 0
        # More robust loop limit based on expected steps per iteration + buffer
        max_runs = max_iterations * 4 + 5 # (improve, review, execute, condition) per iter + initial gen/exec

        # --- Main Execution Loop ---
        while run_count < max_runs :
            run_count += 1
            resume_value: Optional[Any] = None # Value to pass to Command(resume=...)
            interrupt_payload: Optional[Dict] = None # Data received from interrupt()
            processed_interrupt = False

            logger.debug(f"Run loop iteration {run_count}. Input type: {type(current_input).__name__}")

            try:
                # Use astream_events to process outputs and interrupts
                async for event in self.graph.astream_events(current_input, config=config, version="v1"):
                    kind = event["event"]
                    logger.debug(f"Event received: {kind}, Node: {event.get('name')}")

                    if kind == "on_chain_end":
                        node_name = event["name"]
                        logger.debug(f"Finished node: {node_name}")
                        output_data = event["data"].get("output")
                        if output_data and isinstance(output_data, dict):
                            # Keep track of the latest state values observed
                            last_known_state_values = output_data.copy()
                            # --- Print Updates ---
                            self._print_node_output(node_name, output_data, last_printed_code)
                            if node_name in [self.node_generate_initial, self.node_improve_code]:
                                last_printed_code = output_data.get("current_code", last_printed_code)

                    elif kind == "on_interrupt":
                        logger.warning(f"Graph interrupted by node: {event['name']}. Checkpoint: {event['checkpoint']}")
                        interrupt_payload = event['data'] # Data passed from interrupt()
                        processed_interrupt = True
                        # Break the inner loop to handle the interrupt in the outer loop
                        break

                # --- Interrupt Handling (after breaking from inner loop) ---
                if processed_interrupt and interrupt_payload is not None:
                    logger.info("Handling graph interrupt triggered by interrupt() function.")

                    # Extract info to show to the user from the payload
                    code_to_review = interrupt_payload.get("code_to_review")
                    iteration_num = interrupt_payload.get("iteration", "?")
                    last_status = interrupt_payload.get("last_exec_status", "unknown")
                    review_prompt_text = interrupt_payload.get('review_prompt', HUMAN_REVIEW_PROMPT)

                    print("\n--- ACTION REQUIRED (Human Review) ---")
                    print(f"Code after iteration {iteration_num} (Last execution: {str(last_status).upper()}):")
                    print("-" * 35)
                    print(code_to_review if code_to_review else "[No code available for review]")
                    print("-" * 35)

                    # Get user decision
                    while True:
                         user_decision = await asyncio.to_thread(input, review_prompt_text)
                         user_decision = user_decision.strip().lower()
                         if user_decision in HUMAN_REVIEW_OPTIONS.values():
                             break
                         else:
                             print(f"Invalid input. Please enter '{HUMAN_REVIEW_OPTIONS['accept']}' or '{HUMAN_REVIEW_OPTIONS['reject']}'.")

                    # Prepare the value to resume with (the node will handle the logic)
                    resume_value = user_decision # Simply pass the decision
                    logger.info(f"Prepared resume value: '{resume_value}'")

                # --- End of Stream Handling ---
                elif not processed_interrupt:
                    logger.info("Agent stream finished normally in this iteration.")
                    break # Exit the main while loop

            except Exception as e:
                logger.error(f"Agent execution failed in run loop iteration {run_count}: {e}", exc_info=True)
                print(f"\nERROR: Agent execution encountered an unexpected error: {e}")
                # Attempt to retrieve state before breaking
                final_state = await self._get_final_state(config, last_known_state_values)
                self._print_final_state(final_state, thread_id, "ERROR")
                raise # Re-raise the exception after attempting to print state

            # --- Prepare for Next Loop Iteration ---
            if resume_value is not None:
                # Use Command(resume=...) to resume execution
                from langgraph.types import Command
                current_input = Command(resume=resume_value)
                logger.info(f"Resuming graph with Command(resume='{resume_value}')")
            elif processed_interrupt:
                 # This shouldn't happen if we prepared resume_value
                 logger.error("Interrupt processed but no resume value prepared. Aborting.")
                 break
            else:
                 # The stream finished normally
                 logger.debug("No resume data needed, assuming stream finished.")
                 break # Exit while loop

            # Safety break
            if run_count >= max_runs:
                 logger.error(f"Agent exceeded maximum run loop iterations ({max_runs}). Preventing potential infinite loop.")
                 print("\nERROR: Agent seems stuck in a loop. Aborting.", file=sys.stderr)
                 break

        # --- End of Execution ---
        logger.info(f"Agent execution loop finished for thread_id='{thread_id}'.")
        final_state = await self._get_final_state(config, last_known_state_values)
        self._print_final_state(final_state, thread_id, "Final")


    def _print_node_output(self, node_name: str, output_data: Dict, last_printed_code: Optional[str]):
         """Helper to print relevant output from nodes."""
         current_code = output_data.get("current_code")
         current_iter = output_data.get("iteration_count", -1)
         exec_status = output_data.get("execution_status")

         # Print code only if it changed after generation/improvement
         if node_name in [self.node_generate_initial, self.node_improve_code]:
             if current_code is not None and current_code != last_printed_code:
                 title = f"--- Code after {node_name} (Iteration {current_iter}) ---"
                 print(f"\n{title}")
                 # print("-" * len(title)) # Dynamic separator
                 print(current_code)
                 print("-" * len(title))


         # Print execution results after execution node
         if node_name == self.node_execute_code and exec_status != 'not_run':
             exec_time = output_data.get('time_ms') # Ensure key matches execution.py
             exec_output = output_data.get('execution_output')
             exec_error = output_data.get('execution_error')
             title = f"--- Execution Result (Iteration {current_iter}) ---"
             print(f"\n{title}")
             # print("-" * len(title)) # Dynamic separator
             print(f"Status: {str(exec_status).upper()}")
             if exec_time is not None: print(f"Time:   {exec_time:.2f} ms")
             if exec_status == 'success': print(f"Output: {exec_output}")
             if exec_error: print(f"Error:  {exec_error}")
             print("-" * len(title))


    async def _get_final_state(self, config: Dict, fallback_state_values: Optional[Dict]) -> Optional[AgentState]:
        """Attempts to retrieve the final state, falling back to last known values."""
        if not self.graph or not self.graph.checkpointer:
             logger.warning("Cannot get final state: Graph not compiled or no checkpointer.")
             if fallback_state_values:
                  logger.warning("Attempting to use last known state values from stream.")
                  try:
                      return AgentState(**fallback_state_values)
                  except Exception as e:
                      logger.error(f"Failed to create AgentState from fallback values: {e}")
             return None

        logger.info(f"Retrieving final state for config: {config}.")
        final_state = None
        try:
            final_state_snapshot = await self.graph.aget_state(config)
            if final_state_snapshot:
                final_state = AgentState(**final_state_snapshot.values)
                logger.info("Successfully retrieved final state via aget_state.")
            else:
                logger.warning("aget_state returned None for final state.")

        except Exception as e:
            logger.error(f"Failed to retrieve final state via aget_state: {e}", exc_info=True)

        # Fallback to last known state from stream if aget_state failed or returned None
        if final_state is None and fallback_state_values:
            try:
                final_state = AgentState(**fallback_state_values)
                logger.warning("Using last known state values from stream as final state.")
            except Exception as fallback_e:
                 logger.error(f"Failed to create AgentState from fallback values: {fallback_e}")
                 final_state = None # Ensure it's None if fallback fails

        return final_state

    def _print_final_state(self, final_state: Optional[AgentState], thread_id: str, status_prefix: str):
         """Prints the final state summary."""
         if final_state:
            title = f"--- {status_prefix} State (Thread: {thread_id}) ---"
            print(f"\n{title}")
            # print("-" * len(title))
            print(f"Iterations Completed: {final_state.iteration_count}")
            print(f"Final Execution Status: {final_state.execution_status}")
            if final_state.execution_time_ms is not None: print(f"Final Execution Time: {final_state.execution_time_ms:.2f} ms")
            if final_state.execution_error: print(f"Final Execution Error: {final_state.execution_error}")
            print("\nFinal Code:")
            # print("-" * 35)
            print(final_state.current_code if final_state.current_code else "[No code]")
            print("-" * len(title))
         else:
            print(f"\n--- {status_prefix} State (Thread: {thread_id}) ---")
            print("[No definitive final state could be retrieved]")
            print("-" * 36)


    def display_graph_visualization(self):
        """Displays a PNG visualization of the graph if possible."""
        if self.graph is None:
            logger.warning("Cannot display visualization: Graph not compiled yet.")
            return
        print("\nAttempting to display graph visualization...")
        try:
            # Check if running in an environment that supports IPython display
            from IPython.display import Image, display
            img_data = self.graph.get_graph().draw_mermaid_png()
            if img_data:
                logger.info("Displaying graph visualization.")
                display(Image(img_data))
            else:
                logger.warning("draw_mermaid_png() returned no data.")
                self._print_graph_text_flow()
        except ImportError:
            logger.warning("IPython.display not available. Cannot display visualization inline.")
            self._print_graph_text_flow()
            logger.info("Install 'ipython' for inline visualization.")
        except Exception as e:
            logger.warning(f"Could not generate or display graph visualization: {e}", exc_info=False)
            self._print_graph_text_flow()

    def _print_graph_text_flow(self):
        """Prints a textual representation of the graph flow."""
        print("\n--- Graph Flow (Text Representation) ---")
        print(f"START -> {self.node_generate_initial} -> {self.node_execute_code}")
        print(f"  `-- {self.conditional_edge_entry} edge -->")
        print(f"      |-- (if END) ----> END")
        print(f"      `-- (if continue_improve) -> {self.node_improve_code} -> {self.node_human_review} (Interrupts)")
        print(f"                                      `-- (Resume) -> {self.node_execute_code} -> (back to conditional edge)")
        print("-" * 36)