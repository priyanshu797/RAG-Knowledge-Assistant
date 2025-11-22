# RAG Knowledge Assistant

A Retrieval-Augmented Generation (RAG) system combining LlamaIndex, Groq, and Ollama, built with Flask for intelligent document-based Q&A.

## Overview

RAG Knowledge Assistant allows users to upload multiple file formats (PDF, DOCX, TXT, CSV, JSON, Images, Audio, etc.) and ask natural language questions about their content. It uses LlamaIndex for indexing and Groq’s Llama 3.3-70B model for intelligent answers, with fallback to Ollama Gemma2:2B if needed.

## Features

- Multi-format document ingestion: PDF, DOCX, TXT, CSV, JSON, Images, Audio, ZIP.
- OCR and Whisper transcription for non-text data.
- Vector indexing and retrieval using LlamaIndex.
- LLM integration with Groq and Ollama.
- Flask backend with REST APIs for uploading, querying, and managing files.
- Persistent vector storage and chat history.

## Project Structure

📂 Project
│
├── main.py                # Flask backend server
├── rag_pipeline.py        # Core RAG logic (indexing, querying, extraction)
├── Dockerfile             # Container definition
├── requirements.txt       # Dependencies
├── templates/             # Frontend interface
│   └── index.html
└── How_to_Reduce_Stress_Levels.pdf  # Example document

## Setup Instructions

1. Clone the repository

   git clone https://github.com/<your-username>/rag-knowledge-assistant.git
   cd rag-knowledge-assistant/Project

2. Install dependencies

   pip install -r requirements.txt

3. Add your Groq API key in rag_pipeline.py or as an environment variable:

   GROQ_API_KEY=your_groq_api_key_here

## Docker Setup

Build the image:

   docker build -t rag-pipeline .

Run the container:

   docker run -it --rm -p 5000:5000 \
     -v "C:\Users\shukla\Desktop\rag-docker-pipeline-main\Project:/app" \
     priyanshu265/rag-pipeline:latest

Then open in browser: http://localhost:5000

## API Endpoints

| Endpoint | Method | Description |
|-----------|--------|-------------|
| /api/initialize | POST | Initialize LLM (Groq/Ollama) |
| /api/upload | POST | Upload and process files |
| /api/query | POST | Ask a question |
| /api/status | GET | Check processing status |
| /api/chat-history | GET | Retrieve chat history |
| /api/reset | POST | Reset system |
| /api/delete-file | POST | Delete uploaded file |

## Example Usage

Start Flask server:

   python main.py

Upload document(s) via UI or API, then ask questions like:

   "Summarize stress reduction methods."

Example session:

   📁 Enter file paths (comma-separated): /app/How_to_Reduce_Stress_Levels.pdf
   Index ready! Using GROQ for responses.
   Question: What are the top stress reduction techniques?
   Answer: Breathing exercises, mindfulness, and balanced sleep.

## Future Enhancements

- Web chat UI with dynamic updates.
- Integration with vector databases (ChromaDB/Milvus).
- Session-based user management.
- Voice input/output features.

## License

Licensed under the MIT License.

## Author

Priyanshu Shukla  
Generative AI Developer | Flask | RAG | Docker | Groq | LlamaIndex
