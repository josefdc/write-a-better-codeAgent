# src/llm_utils.py
# REMOVED getpass import, no longer needed here
import logging
import os
import re
from typing import Optional, Sequence

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                    wait_exponential)

# Importar config
from .config import (ANTHROPIC_API_KEY_ENV_VAR, FALLBACK_ANTHROPIC_MODEL,
                     PREFERRED_ANTHROPIC_MODEL)

logger = logging.getLogger(__name__)

# --- LLM Client Setup ---
def setup_anthropic_client() -> ChatAnthropic:
    """
    Configura e inicializa el cliente ChatAnthropic.
    ASUME que la clave API ya existe en las variables de entorno (verificada por main.py).
    """
    # Obtener la clave (ahora sabemos que existe gracias a la verificación en main.py)
    api_key = os.getenv(ANTHROPIC_API_KEY_ENV_VAR)
    if not api_key:
        # Este error no debería ocurrir si main.py hizo la verificación,
        # pero es una salvaguarda por si se llama a esta función directamente.
        logger.critical(f"CRITICAL INTERNAL ERROR: {ANTHROPIC_API_KEY_ENV_VAR} was not found, despite prior checks.")
        raise ValueError(f"Internal Error: API key environment variable '{ANTHROPIC_API_KEY_ENV_VAR}' disappeared.")

    logger.info(f"Initializing Anthropic client using key from environment.")

    try:
        # Pasar la clave explícitamente al constructor
        model = ChatAnthropic(model=PREFERRED_ANTHROPIC_MODEL, api_key=api_key)
        logger.info(f"Using Anthropic model: {PREFERRED_ANTHROPIC_MODEL}")
        # Podrías añadir una llamada de prueba aquí si quieres asegurar conectividad
        # try:
        #     model.invoke("Connectivity test prompt")
        # except Exception as test_e:
        #     logger.error(f"Connectivity test failed: {test_e}")
        #     raise ValueError("Failed to connect to Anthropic API.") from test_e

    except Exception as preferred_e:
        logger.warning(f"Could not initialize {PREFERRED_ANTHROPIC_MODEL}: {preferred_e}. Falling back to {FALLBACK_ANTHROPIC_MODEL}.", exc_info=False)
        try:
            # Intentar con el modelo de fallback, pasando la clave también
            model = ChatAnthropic(model=FALLBACK_ANTHROPIC_MODEL, api_key=api_key)
            logger.info(f"Using Anthropic model: {FALLBACK_ANTHROPIC_MODEL}")
        except Exception as fallback_e:
            logger.error(f"Failed to initialize fallback model {FALLBACK_ANTHROPIC_MODEL}: {fallback_e}", exc_info=True)
            raise ValueError("Could not initialize any Anthropic model. Check API key and model availability.") from fallback_e
    return model


# --- LLM Interaction (call_llm) ---
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(Exception), # Ser más específico si es posible (ej. anthropic.APIError)
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        f"Retrying LLM call after error ({type(retry_state.outcome.exception()).__name__}). Attempt #{retry_state.attempt_number}"
    )
)
async def call_llm(llm_client: ChatAnthropic, messages: Sequence[BaseMessage]) -> AIMessage:
    """Calls the LLM asynchronously with retry logic."""
    if not messages:
        logger.error("LLM call attempted with empty message list.")
        raise ValueError("Cannot call LLM with empty messages.")
    logger.info(f"Calling LLM async with {len(messages)} messages.")
    # La clave API ya está configurada en el cliente llm_client
    response = await llm_client.ainvoke(messages)
    if not isinstance(response, AIMessage):
        raise TypeError(f"LLM expected AIMessage, got {type(response)}")
    if not response.content or not isinstance(response.content, str) or not response.content.strip():
        logger.warning("LLM response content is empty or invalid.")
    logger.debug(f"LLM raw response content: {response.content[:500]}...")
    return response


# --- Code Extraction (extract_python_code) ---
# ... (sin cambios) ...
def extract_python_code(text: Optional[str]) -> Optional[str]:
    """Extracts Python code from a string, looking for ```python blocks or inferring."""
    if not text:
        logger.warning("Attempted to extract code from empty or None text.")
        return None

    # Pattern to find ```python ... ``` blocks
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if match:
        extracted_code = match.group(1).strip()
        logger.info(f"Successfully extracted Python code block using ```python marker ({len(extracted_code)} chars).")
        return extracted_code
    else:
        # Fallback: Check if the whole text looks like Python code
        lines = text.strip().split('\n')
        if len(lines) > 0 and (
            lines[0].strip().startswith(('import ', 'def ', 'class ', '#', '"""', "'''")) or
            (len(lines) > 1 and lines[1].strip().startswith('    '))
           ):
            logger.warning("No ```python ... ``` block found, but content looks like Python code. Using raw content.")
            return text.strip()

        logger.warning("Could not extract Python code block using ```python marker or heuristic.")
        return None