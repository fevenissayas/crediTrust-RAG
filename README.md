# crediTrust-RAG

A Retrieval-Augmented Generation (RAG) system that enhances LLM answers by grounding them in trusted, external documents. Built with LangChain, FAISS, and OpenAI APIs.

## Features

- Upload your own documents as context
- Ask questions and receive context-aware answers
- Uses vector search to retrieve the most relevant chunks
- Easy integration with OpenAI API

## Project Structure

```
credittrust_complaint_chatbot/
├── data/
├── notebooks/
├── src/
├── vector_store/
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```

## Setup

1. **Clone the repo:**
    ```bash
    git clone <your-repo-url>
    cd credittrust_complaint_chatbot
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Use

**Task 1: Data Analysis & Preprocessing**
- Jupyter Notebook:
    ```bash
    jupyter notebook notebooks/eda_preprocessing.ipynb
    ```
- Script:
    ```bash
    python src/eda_preprocessing.py
    ```

**Task 2: Embedding & Indexing**
- Run:
    ```bash
    python src/chunk_embed_indexing.py
    ```

**Task 3: RAG Core & Evaluation**
- Run:
    ```bash
    python src/rag_application.py
    ```

**Task 4: Chatbot Web Interface**
- Launch:
    ```bash
    python app.py
    ```

> **Note:** If you see `ResourceExhausted` errors, your Google Gemini API quota may be used up. Wait and try again, or upgrade your quota.

---