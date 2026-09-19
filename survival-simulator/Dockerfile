# Survival agent server only (no simulator, no pygame) -- for cheap/free cloud hosting.
#   docker build -t survival-agent . && docker run -p 9052:9052 survival-agent
FROM python:3.11-slim
WORKDIR /app
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt
COPY agent_server.py .
COPY src/__init__.py src/__init__.py
COPY src/utils/__init__.py src/utils/__init__.py
COPY src/utils/DTOs.py src/utils/DTOs.py
COPY src/utils/controllers/ src/utils/controllers/
ENV PORT=9052
EXPOSE 9052
CMD ["python", "agent_server.py"]
