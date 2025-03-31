# src/main.py
import argparse # Lo quitamos pero dejamos el import por si se reutiliza
import asyncio
import logging
import sys
import uuid
import traceback
from dotenv import load_dotenv
import os

# Tus imports relativos
from .config import (LOG_LEVEL, LOG_FORMAT, ANTHROPIC_API_KEY_ENV_VAR,
                     DEFAULT_MAX_ITERATIONS, DEFAULT_IMPROVEMENT_PROMPT) # Quitar defaults si no se usan aquí
from .llm_utils import setup_anthropic_client
from .agent import CodeImprovingAgent
from langgraph.checkpoint.memory import MemorySaver # O el checkpointer que uses

# --- Cargar .env ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dotenv_path = os.path.join(project_root, '.env')
if os.path.exists(dotenv_path):
    # Usar logging básico antes de configurar completamente
    print(f"INFO: Loading environment variables from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
else:
    print("INFO: .env file not found, relying on system environment variables.")

# --- Configurar Logging ---
# Configuración básica, se puede refinar más adelante si es necesario
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__) # Logger para este módulo

# --- Definición y Compilación del Grafo (para la CLI/Server) ---

app = None # Inicializar app como None

try:
    logger.info("Setting up LLM client...")
    if not os.getenv(ANTHROPIC_API_KEY_ENV_VAR):
         raise ValueError(f"Setup failed: {ANTHROPIC_API_KEY_ENV_VAR} not set.")
    llm_client = setup_anthropic_client()

    logger.info("Using checkpointer: MemorySaver")
    checkpointer = MemorySaver() # O tu checkpointer persistente

    logger.info("Initializing Agent...")
    agent = CodeImprovingAgent(llm_client=llm_client)

    logger.info("Compiling Agent Graph...")
    agent.compile(checkpointer=checkpointer)

    # <<< EXponer la instancia compilada >>>
    # La CLI buscará una variable llamada 'app' o 'graph' por defecto
    app = agent.graph # Asignar el grafo compilado a la variable 'app'
    if app is None:
         raise RuntimeError("Agent compilation failed, graph is None.")
    logger.info("Compiled graph 'app' is ready for LangGraph server.")

except ValueError as e:
    logger.critical(f"Setup failed: {e}", exc_info=False)
    print(f"\nCRITICAL ERROR during setup: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
     logger.critical(f"Unexpected error during setup: {e}", exc_info=True)
     print(f"\nCRITICAL UNEXPECTED ERROR during setup: {e}", file=sys.stderr)
     traceback.print_exc()
     sys.exit(1)

# --- Bloque Opcional para Ejecución Directa (Comentado) ---
# Si quisieras poder ejecutar `python -m src.main` para hacer algo directamente
# (como un test rápido), podrías añadir un bloque if __name__ == "__main__": aquí.
# Pero para usar con `langgraph dev`, no es estrictamente necesario.

# if __name__ == "__main__":
#      print("Graph 'app' defined. Run using 'langgraph dev'.")
#      # Podrías añadir un pequeño test aquí si quisieras, ej:
#      # print(app.get_graph().draw_mermaid())