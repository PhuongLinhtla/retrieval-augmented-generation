### rag_project/ ├── app/ │ ├── __init__.py │ ├── main.py │ # FastAPI entry point & WebSocket chat endpoint
# retrieval-augmented-generation
### A small FastAPI-based RAG chatbot system with vector search integration.
### It provides a WebSocket-based chat interface, processes user queries by generating embeddings, retrieves the most relevant text chunks using vector similarity search, and returns responses with source context and similarity scores.

# Features
### - FastAPI WebSocket endpoint for real-time chatbot interaction
### - Embedding generation using SentenceTransformers
### - Vector similarity search to retrieve top-k relevant text chunks
### - RAG pipeline combining retrieval and response generation
### - Source tracing with similarity scores for each response
### - Lightweight demo frontend for chat interaction

# Quickstart (local)
### 1.Create and activate a virtual environment:
###     python3 -m venv .venv source .venv/bin/activate  

### 2.Install dependencies:
###     pip install -r requirements.txt  

### 3.Provide environment variables (see section below) in a .env file or your shell.

### 4.Start services with Docker Compose (optional for local development):
### docker-compose up -d
### This can be used to run supporting services such as a vector database or API backend if configured in docker-compose.yml.

### Run the FastAPI app:
### uvicorn app.main:app --reload --host 0.0.0.0 --port 8000  
### Connect to the WebSocket endpoint:
### ws://localhost:8000/ws  
### Use a simple frontend or WebSocket client to send queries and receive responses with context and similarity scores.
