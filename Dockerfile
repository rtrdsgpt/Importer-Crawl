# Runs the FastAPI wrapper (api.py) by default. The same image also has
# main.py/app.py's dependencies, so it works for those too -- just override
# the command, e.g.:
#   docker run --env-file .env myimage python main.py --product "Ceramic Tiles" --country Germany --provider groq
#   docker run --env-file .env -p 8501:8501 myimage streamlit run app.py --server.address 0.0.0.0
FROM python:3.11-slim

WORKDIR /app

# Layer dependency install separately from source so a code-only change
# doesn't invalidate the (slow, network-heavy) pip install layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py main.py app.py mcp_server.py ./
COPY src/ ./src/

# Results/checkpoints written here by main.py/api.py -- mount a volume to
# persist them across container restarts.
RUN mkdir -p data

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
