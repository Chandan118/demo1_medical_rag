# 🧬 AI Medical Researcher (RAG System)

A state-of-the-art Retrieval-Augmented Generation (RAG) system designed to act as an AI Medical Researcher. By uploading complex Bio-Medical or Robotics research papers (PDFs), users can ask precise questions and receive accurate answers with citations based directly on the provided texts.

**Developed by Chandan Sheikder, PhD Researcher.**

![UI Mockup](https://via.placeholder.com/800x400.png?text=AI+Medical+Researcher+Dashboard)

## 📌 The Problem
Medical professionals, researchers, and robotics engineers spend thousands of hours reading dense academic papers to find specific methodologies or data points. Existing tools rely on keyword search, which often misses semantic context and fails to synthesize information across multiple documents.

## 💡 The Solution
This AI tool uses an advanced RAG pipeline to "read" up to 10 massive research papers simultaneously. It semantically understands the context and allows users to query the documents in natural language. It provides exact, hallucination-free answers by grounding the LLM entirely on the uploaded dataset.

## 🛠 Tech Stack (100% Free & Local)
- **Language:** Python
- **Framework:** LangChain & Streamlit (Frontend UI)
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
- **Vector Database:** ChromaDB
- **LLM:** Meta Llama-3 (running locally via Ollama)

## 🚀 Quickstart Guide

### 1. Prerequisites
You need to install [Ollama](https://ollama.com/) to run the Llama-3 model locally.
After installing, open your terminal and run:
```bash
ollama run llama3
```
Leave this running in the background.

### 2. Setup the Project
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the Web App
```bash
streamlit run app.py
```
Upload your PDFs through the sidebar, click **"Process Documents"**, and start chatting with your research!
