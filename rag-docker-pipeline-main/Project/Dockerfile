# ---------------------------------------------------------
# 1. Base Image
# ---------------------------------------------------------
    FROM python:3.10-slim

    # ---------------------------------------------------------
    # 2. Install system dependencies required by rag_pipeline.py
    # ---------------------------------------------------------
    RUN apt-get update && apt-get install -y \
        ffmpeg \
        tesseract-ocr \
        libgl1 \
        poppler-utils \
        build-essential \
        curl \
        git \
        && apt-get clean
    
    # ---------------------------------------------------------
    # 3. Create working directory docker push priyanshu265/rag-pipeline:latest,docker pull priyanshu265/rag-pipeline:latest

    # ---------------------------------------------------------
    WORKDIR /app
    
    # ---------------------------------------------------------
    # 4. Copy requirements
    # ---------------------------------------------------------
    COPY requirements.txt .
    
    RUN pip install --no-cache-dir -r requirements.txt
    
    # ---------------------------------------------------------
    # 5. Copy all project files
    # ---------------------------------------------------------
    COPY . .
    
    # Ensure storage folder exists for persistence
    RUN mkdir -p /app/storage
    
    # ---------------------------------------------------------
    # 6. Set environment variables for Ollama fallback
    # ---------------------------------------------------------
    ENV PYTHONUNBUFFERED=1
    ENV OLLAMA_HOST="http://host.docker.internal:11434"
    
    # ---------------------------------------------------------
    # 7. Default command (runs your CLI RAG pipeline)
    # ---------------------------------------------------------
    CMD ["python", "rag_pipeline.py"]
    