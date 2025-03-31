# src/execution.py
import asyncio
import logging
import tempfile
import os
import json
import time
import docker # Importar la librería Docker
from docker.errors import ContainerError, ImageNotFound, APIError
from docker.types import Mount
from typing import Any, Dict, Literal

# Importar constantes de configuración (ASSUMED_ENTRY_POINT_FUNC ya no es necesaria aquí)
# from .config import ASSUMED_ENTRY_POINT_FUNC # <<< CORRECCIÓN: Quitar import no usado

logger = logging.getLogger(__name__)

# Nombre de la imagen Docker que construiremos
DOCKER_IMAGE_NAME = "code-executor-image:latest"
# Script interno que se ejecutará dentro del contenedor
# INTERNAL_SCRIPT_NAME = "docker_exec_script.py" # No se necesita aquí
# Path donde montaremos el código dentro del contenedor
CONTAINER_CODE_PATH = "/code/script.py"
# Límite de tiempo para la ejecución del contenedor (en segundos) - Nota: No forzado activamente por run()
CONTAINER_TIMEOUT_SECONDS = 10
# Límites de memoria (ej. '128m', '1g')
CONTAINER_MEM_LIMIT = '256m'
# Límites de CPU (opcional, requiere más config)
# CONTAINER_CPU_QUOTA = 50000
# CONTAINER_CPU_PERIOD = 100000

# Type alias for execution result
ExecResult = Dict[str, Any]

# --- Docker Client Initialization ---
try:
    docker_client = docker.from_env()
    docker_client.ping()
    logger.info("Docker client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Docker client: {e}. Docker execution will fail.", exc_info=False)
    docker_client = None

# --- Code Execution via Docker ---
async def execute_code_docker(code_to_run: str) -> ExecResult:
    """
    Executes Python code inside an isolated Docker container.
    """
    if docker_client is None:
        logger.error("Docker client not available. Cannot execute code.")
        return {"status": "error", "output": None, "error": "Docker client not initialized.", "time_ms": 0.0}

    if not code_to_run or not code_to_run.strip():
        logger.warning("No code provided to execute in Docker.")
        return {"status": "not_run", "output": None, "error": "No code provided.", "time_ms": 0.0}

    # Usar archivo temporal en el host
    tmp_file_descriptor, host_code_path = tempfile.mkstemp(suffix='.py', text=True)
    logger.debug(f"Code written to temporary host file: {host_code_path}")
    with os.fdopen(tmp_file_descriptor, 'w') as tmp_code_file:
        tmp_code_file.write(code_to_run)


    container = None
    result: ExecResult = { # Default error result
        "status": "error", "output": None, "error": "Container execution failed.", "time_ms": 0.0
    }
    start_time = time.perf_counter() # Tiempo total de interacción Docker

    try:
        logger.info(f"Running code in Docker container (Image: {DOCKER_IMAGE_NAME})...")
        mounts = [Mount(target=CONTAINER_CODE_PATH, source=host_code_path, type='bind', read_only=True)]

        # Ejecutar el contenedor
        container_output_bytes = docker_client.containers.run(
            image=DOCKER_IMAGE_NAME,
            command=None, # Usa ENTRYPOINT
            mounts=mounts,
            mem_limit=CONTAINER_MEM_LIMIT,
            network_mode="none",
            remove=True,
            stdout=True,
            stderr=True,
            detach=False,
            user="appuser" # <<< CORRECCIÓN: Ejecutar como usuario no root
        )

        stdout_str = container_output_bytes.decode('utf-8').strip()
        logger.debug(f"Container stdout:\n{stdout_str}")

        # Parsear resultado JSON
        try:
            last_line = stdout_str.splitlines()[-1] if stdout_str else "{}"
            parsed_result = json.loads(last_line)
            if all(k in parsed_result for k in ["status", "output", "error", "time_ms"]):
                 result = parsed_result
                 logger.info(f"Successfully parsed result from container: Status '{result.get('status')}'")
            else:
                 logger.error(f"JSON from container missing expected keys: {last_line}")
                 result["error"] = f"Invalid JSON result structure from container: {last_line}"
        except (json.JSONDecodeError, IndexError) as parse_err:
            logger.error(f"Failed to parse JSON from container stdout: {parse_err}")
            logger.error(f"Container stdout was: {stdout_str}")
            result["error"] = f"ParseError from container stdout: {stdout_str}"

    # ... (Manejo de ContainerError, ImageNotFound, APIError sin cambios) ...
    except ContainerError as e:
        logger.error(f"ContainerError: Exit code {e.exit_status}. Stderr:\n{e.stderr.decode('utf-8') if e.stderr else 'N/A'}")
        result["error"] = f"Container exited with status {e.exit_status}:\n{e.stderr.decode('utf-8') if e.stderr else 'No stderr'}"
    except ImageNotFound:
        logger.error(f"Docker image '{DOCKER_IMAGE_NAME}' not found. Build it first.")
        result["error"] = f"Docker image '{DOCKER_IMAGE_NAME}' not found."
    except APIError as e:
        logger.error(f"Docker APIError: {e}", exc_info=True)
        result["error"] = f"Docker API Error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error during Docker execution: {e}", exc_info=True)
        result["error"] = f"Unexpected Docker execution error: {e}"
    finally:
        # Limpiar archivo temporal
        if 'host_code_path' in locals() and os.path.exists(host_code_path):
            try:
                os.remove(host_code_path)
                logger.debug(f"Temporary host file deleted: {host_code_path}")
            except OSError as e:
                logger.warning(f"Could not delete temporary file {host_code_path}: {e}")

        # Calcular tiempo total si el script interno no lo hizo (ej. en error)
        # Usar el tiempo interno si fue exitoso
        internal_time = result.get("time_ms")
        if internal_time is None or result.get("status") != "success":
             end_time = time.perf_counter()
             result["time_ms"] = (end_time - start_time) * 1000
             logger.info(f"Using total Docker interaction time: {result['time_ms']:.2f} ms")
        else:
             logger.info(f"Using internal script execution time: {internal_time:.2f} ms")


    return result