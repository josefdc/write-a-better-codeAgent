# src/config.py
import logging

# --- Constants ---
ANTHROPIC_API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_IMPROVEMENT_PROMPT = "write better code"
LOG_LEVEL = logging.INFO # Cambia a logging.DEBUG para más detalle
LOG_FORMAT = '%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s] %(message)s'
EXECUTION_TIMEOUT_SECONDS = 10  # Aún no implementado efectivamente con exec
ASSUMED_ENTRY_POINT_FUNC = "solve_problem"  # Función que el agente espera que el LLM cree

# Model Configuration
PREFERRED_ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620"
FALLBACK_ANTHROPIC_MODEL = "claude-3-haiku-20240307"

# Human Review Options
HUMAN_REVIEW_OPTIONS = {
    "accept": "accept",
    "reject": "reject"
}
HUMAN_REVIEW_PROMPT = f"Review the code. Type '{HUMAN_REVIEW_OPTIONS['accept']}' to continue, '{HUMAN_REVIEW_OPTIONS['reject']}' to revert: "