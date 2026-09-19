# Survival agent server (task 1) for free cloud hosting -- built from the repo root so
# hosting UIs need no root-directory / context settings.  See survival-simulator/Dockerfile.
FROM python:3.11-slim
WORKDIR /app
COPY survival-simulator/requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt
COPY survival-simulator/agent_server.py .
COPY survival-simulator/src/utils/DTOs.py src/utils/DTOs.py
COPY survival-simulator/src/utils/controllers/ src/utils/controllers/
ENV PORT=9052
EXPOSE 9052
CMD ["python", "agent_server.py"]
