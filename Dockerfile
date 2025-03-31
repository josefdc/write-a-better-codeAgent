# Dockerfile

# Usar una imagen base de Python ligera y oficial
FROM python:3.11-slim as builder

# Establecer directorio de trabajo
WORKDIR /app

# Copiar el script de ejecución interno (que crearemos a continuación)
COPY docker_exec_script.py .

# (Opcional) Instalar bibliotecas comunes y seguras si se espera que el código las necesite.
# Mantenlo al mínimo absoluto. Ejemplo:
# RUN pip install --no-cache-dir numpy scipy # ¡Cuidado con lo que instalas!

# Crear un usuario no root para mayor seguridad
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Punto de entrada: ejecutar el script interno con Python
ENTRYPOINT ["python", "docker_exec_script.py"]