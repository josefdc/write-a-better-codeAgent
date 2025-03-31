# docker_exec_script.py
import time
import traceback
import json
import sys
import os
import contextlib
from io import StringIO

# --- Constantes (deben coincidir con las del agente si es necesario) ---
# Asume que el código a ejecutar estará en /code/script.py
CODE_FILE_PATH = "/code/script.py"
# Nombre de la función que esperamos encontrar y ejecutar
ASSUMED_ENTRY_POINT_FUNC = "solve_problem"

def main():
    # --- Preparar el resultado por defecto ---
    result = {
        "status": "error", # Empezar asumiendo error
        "output": None,
        "error": "Execution did not complete.", # Mensaje de error por defecto
        "time_ms": 0.0
    }
    start_time = time.perf_counter()
    # Definir stdout_capture aquí para que esté disponible en finally
    stdout_capture = StringIO()

    try:
        # --- Leer el código ---
        if not os.path.exists(CODE_FILE_PATH):
            raise FileNotFoundError(f"Code file not found at {CODE_FILE_PATH}")

        with open(CODE_FILE_PATH, 'r') as f:
            code_to_run = f.read()

        if not code_to_run.strip():
             raise ValueError("Code file is empty.")

        # --- Ejecutar el código ---
        local_namespace = {}
         # Permitir import random si es necesario por el problema base
        import random
        global_namespace = {
            '__builtins__': __builtins__,
            'random': random,
            'time': time # Permitir timing dentro del código si es necesario
            }

        exec_output = None

        # Compilar y ejecutar
        compiled_code = compile(code_to_run, CODE_FILE_PATH, 'exec')

        # Ejecutar para definir funciones/clases
        exec(compiled_code, local_namespace, local_namespace)

        # Verificar y obtener la función de entrada
        if ASSUMED_ENTRY_POINT_FUNC not in local_namespace:
            raise NameError(f"Entry point function '{ASSUMED_ENTRY_POINT_FUNC}' not found in executed code.")

        entry_point_func = local_namespace[ASSUMED_ENTRY_POINT_FUNC]
        if not callable(entry_point_func):
             raise TypeError(f"'{ASSUMED_ENTRY_POINT_FUNC}' is not a callable function.")

        # Llamar a la función de entrada capturando stdout
        with contextlib.redirect_stdout(stdout_capture):
            exec_output = entry_point_func() # Ejecutar la función

        # --- Éxito ---
        result["status"] = "success"
        result["error"] = None # <<< CORRECCIÓN: Limpiar el error si la ejecución fue exitosa
        captured_stdout = stdout_capture.getvalue()

        # Guardar resultado o stdout
        if exec_output is not None:
            result["output"] = repr(exec_output) # Usar repr por seguridad/consistencia
        elif captured_stdout:
             result["output"] = repr(captured_stdout)
        else:
             result["output"] = None

    except FileNotFoundError as e:
         result["status"] = "error"
         result["error"] = f"Setup Error: {e}"
    except ValueError as e:
         result["status"] = "error"
         result["error"] = f"Setup Error: {e}"
    except SyntaxError as e:
        result["status"] = "syntax_error"
        result["error"] = f"Syntax Error: {e}\n{traceback.format_exc(limit=1)}"
    except NameError as e:
         result["status"] = "runtime_error"
         result["error"] = f"Runtime Error (NameError): {e}\n{traceback.format_exc(limit=2)}"
    except TypeError as e:
         result["status"] = "runtime_error"
         result["error"] = f"Runtime Error (TypeError): {e}\n{traceback.format_exc(limit=2)}"
    except Exception as e:
        result["status"] = "runtime_error"
        result["error"] = f"Runtime Error: {e}\n{traceback.format_exc(limit=5)}"
    finally:
        # Calcular tiempo y cerrar el capturador de stdout
        end_time = time.perf_counter()
        result["time_ms"] = (end_time - start_time) * 1000
        stdout_capture.close()

        # --- Imprimir resultado como JSON a stdout ---
        # Esto es lo que capturará el agente LangGraph
        try:
            print(json.dumps(result))
        except TypeError as e:
             # Fallback si algo en el resultado no es serializable a JSON
             print(json.dumps({
                 "status": "error",
                 "output": None,
                 "error": f"Failed to serialize result to JSON: {e}",
                 "time_ms": result["time_ms"]
             }))

if __name__ == "__main__":
    main()