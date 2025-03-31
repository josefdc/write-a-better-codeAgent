# Code Improving Agent Project

This project implements a LangGraph agent designed to iteratively generate and improve Python code based on a problem description. It features secure code execution using Docker and includes a human-in-the-loop (HITL) step for reviewing and accepting/rejecting code improvements.

## Features

*   **Iterative Code Improvement:** Starts with an initial code generation and refines it over multiple iterations.
*   **LangGraph Framework:** Built using LangGraph for defining and managing the agent's stateful workflow.
*   **Secure Code Execution:** Executes the generated Python code within an isolated Docker container to prevent security risks.
*   **Human-in-the-Loop (HITL):** Pauses the workflow to allow a human user to review proposed code improvements and decide whether to accept or reject them.
*   **Configurable:** Uses environment variables (`.env`) for API keys and Python constants (`src/config.py`) for agent behavior (e.g., max iterations, prompts, models).
*   **Anthropic LLM Integration:** Uses Anthropic's Claude models (configurable) via `langchain-anthropic`.
*   **LangGraph CLI Ready:** Exposes the compiled graph via `src/main.py` for easy serving with `langgraph dev`.

## Architecture Overview

The agent operates as a state machine defined by a LangGraph graph. Key components include:

1.  **Agent State (`src/state.py`):** A Pydantic model defining the data tracked throughout the workflow (problem description, code versions, execution results, messages, etc.).
2.  **Graph Nodes (`src/graph_nodes.py`):** Functions representing individual steps in the workflow:
    *   `generate_initial_code_node`: Generates the first version of the code.
    *   `improve_code_node`: Asks the LLM to improve the current code based on history and prompts.
    *   `execute_code_node`: Executes the current code using the Docker sandbox (`src/execution.py`).
    *   `human_review_node`: Interrupts the graph, presenting the code for human review and awaiting a decision (accept/reject).
    *   `should_continue_node`: A conditional node deciding whether to loop for another improvement iteration or end based on iteration count.
3.  **LLM Utilities (`src/llm_utils.py`):** Handles setting up the Anthropic client and provides robust functions for calling the LLM and extracting code from its responses.
4.  **Docker Execution (`src/execution.py`, `Dockerfile`, `docker_exec_script.py`):** Manages the secure execution of generated code.
    *   `Dockerfile`: Defines a minimal Python environment with a non-root user.
    *   `docker_exec_script.py`: The script run inside the container. It reads the code, executes the target function (`solve_problem` by default), captures output/errors/timing, and prints results as JSON.
    *   `execution.py`: Contains the host-side logic to interact with the Docker client, run the container with the generated code mounted, and parse the JSON result.
5.  **Agent Logic (`src/agent.py`):** Ties everything together. Defines the graph structure, compiles it, and provides the `run` method (though direct execution is now handled by `langgraph dev`).
6.  **Configuration (`src/config.py`, `.env`):** Manages settings like API keys, model names, prompts, and iteration limits.
7.  **Server Entry Point (`src/main.py`):** Sets up logging, loads the environment, initializes the agent, compiles the graph, and exposes it as `app` for the LangGraph CLI.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd code_agent_project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # Or if using pyproject.toml with editable install:
    # pip install -e .
    ```

4.  **Configure Environment Variables:**
    *   Copy the example environment file (if one exists) or create a `.env` file in the project root.
    *   Add your API keys:
        ```properties
        # .env
        ANTHROPIC_API_KEY="sk-ant-api03-..."
        LANGSMITH_API_KEY="lsv2_pt_..." # Optional, for LangSmith tracing
        ```
    *   Ensure the `.env` file is listed in your `.gitignore`!

5.  **Install Docker:**
    *   Make sure you have Docker installed and the Docker daemon is running. Follow the official Docker installation guide for your operating system.

## Building the Docker Image

Before running the agent, you need to build the Docker image used for code execution:

```bash
docker build -t code-executor-image:latest .
```

This command builds the image defined in `Dockerfile` and tags it as `code-executor-image:latest`, which is the name expected by `src/execution.py`.

## Running the Agent

This project is designed to be run using the LangGraph CLI.

1.  **Start the LangGraph Server:**
    From the project root directory, run:
    ```bash
    langgraph dev
    ```
    This command reads `langgraph.json`, finds the graph instance (`app` in `src/main.py`), and starts a FastAPI server with a UI (usually at `http://127.0.0.1:8000`).

2.  **Interact via the UI:**
    *   Open your web browser to the address provided by `langgraph dev`.
    *   You should see the "code_agent" graph listed.
    *   Click on it to access the interaction UI.
    *   Input the required initial state fields:
        *   `problem_description`: The task for the agent (e.g., "Write a Python function that takes a list of numbers and returns their sum").
        *   `max_iterations`: The maximum number of improvement loops (e.g., `3`).
        *   `improvement_prompt`: The instruction for improvement (e.g., "Make the code more efficient" or use the default "write better code").
    *   Start the run. The UI will show the graph execution step-by-step.
    *   When the `human_review` node is reached, the execution will pause, and the UI will display the code and prompt for your decision ("accept" or "reject"). Enter your choice to resume the graph.

## Configuration

*   **API Keys:** Set in the `.env` file (see Setup).
*   **Agent Behavior:** Modify constants in `src/config.py`:
    *   `DEFAULT_MAX_ITERATIONS`: Default loop count if not provided.
    *   `DEFAULT_IMPROVEMENT_PROMPT`: Default prompt for the improvement step.
    *   `LOG_LEVEL`: Change logging verbosity (e.g., `logging.DEBUG`).
    *   `ASSUMED_ENTRY_POINT_FUNC`: The function name the agent expects the LLM to generate and which `docker_exec_script.py` tries to call.
    *   `PREFERRED_ANTHROPIC_MODEL`, `FALLBACK_ANTHROPIC_MODEL`: Specify Anthropic models to use.
    *   `HUMAN_REVIEW_OPTIONS`, `HUMAN_REVIEW_PROMPT`: Customize the HITL interaction.
*   **Docker Execution:** Modify constants in `src/execution.py`:
    *   `CONTAINER_TIMEOUT_SECONDS`: (Note: Currently informational, not strictly enforced by `docker.run`).
    *   `CONTAINER_MEM_LIMIT`: Memory limit for the execution container.

## How It Works (Graph Flow)

1.  **Start:** The process begins with the initial state provided by the user.
2.  **Generate Initial Code:** The LLM generates the first version of the Python code based on the `problem_description`.
3.  **Execute Code:** The generated code is run inside the Docker container. Results (status, output/error, time) are captured.
4.  **Should Continue? (Conditional Edge):**
    *   Checks if `iteration_count` has reached `max_iterations`.
    *   If yes -> **End**.
    *   If no -> **Improve Code**.
5.  **Improve Code:** The LLM is asked to improve the `current_code` based on the `improvement_prompt` and the conversation history (which includes previous code and potentially execution errors).
6.  **Human Review:** The graph *interrupts*. The user is shown the newly improved code and asked to `accept` or `reject` it via the LangGraph UI (or console if run differently).
7.  **(Resume):** Based on the user's input:
    *   If `reject`, the state is updated to revert `current_code` to `previous_code`.
    *   If `accept`, the state keeps the improved `current_code`.
8.  **Execute Code:** The (potentially reverted or accepted) code is executed again in Docker.
9.  **Loop:** The flow returns to the **Should Continue?** conditional edge.

