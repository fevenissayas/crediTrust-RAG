import gradio as gr
import pandas as pd
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
import google.generativeai as genai
import textwrap
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE_PATH = 'vector_store'
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

qa_chain = None
llm_model_name = "Not Loaded" 


def initialize_rag_components():
    """Set up Vector Store, Retriever, and LLM."""
    global qa_chain, llm_model_name

    if qa_chain is not None:
        return "RAG already initialized."

    try:
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if not google_api_key:
            return "Missing GOOGLE_API_KEY. Set in .env or environment."
        genai.configure(api_key=google_api_key)
    except Exception as e:
        return f"Google AI config error: {e}"

    LLM_MODEL_TO_USE = None
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    for name in [
        "models/gemini-1.5-flash",
        "models/gemini-1.5-flash-latest",
        "models/gemini-1.5-pro",
        "models/gemini-1.0-pro"
    ]:
        if name in available_models:
            LLM_MODEL_TO_USE = name
            break
    if not LLM_MODEL_TO_USE and available_models:
        for model_name in available_models:
            if "gemini" in model_name:
                LLM_MODEL_TO_USE = model_name
                break

    if not LLM_MODEL_TO_USE:
        return "No Gemini model found. Check API/model access."

    llm_model_name = LLM_MODEL_TO_USE 

    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    try:
        db = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        return f"Vector store load error: {e}"

    retriever = db.as_retriever(search_kwargs={"k": 5})

    try:
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL_TO_USE, temperature=0.3, google_api_key=google_api_key)
    except Exception as e:
        return f"LLM init error: {e}"

    template = (
        "You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. "
        "Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.\n\n"
        "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    )

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return "RAG initialized."


def ask_chatbot(query: str, history: list):
    """Query the RAG app and return answer and sources."""
    global qa_chain

    if qa_chain is None:
        return "RAG not initialized. Click 'Initialize RAG' first.", []

    try:
        result = qa_chain.invoke({"query": query})
        answer = result["result"]
        source_documents = result["source_documents"]

        sources_display = ""
        if source_documents:
            sources_display += "**Supporting Complaints (Sources):**\n\n"
            for i, doc in enumerate(source_documents):
                wrapped_content = textwrap.fill(doc.page_content, width=100)
                sources_display += (
                    f"**Source {i+1}:**\n"
                    f"- **Complaint ID:** {doc.metadata.get('complaint_id', 'N/A')}\n"
                    f"- **Product:** {doc.metadata.get('product', 'N/A')}\n"
                    f"- **Content Snippet:** \"{wrapped_content[:250]}...\"\n\n"
                )
        else:
            sources_display = "No source documents retrieved."

        history.append((query, answer))
        history.append((None, sources_display))

        return history, sources_display
    except Exception as e:
        history.append((query, f"Error: {e}"))
        return history, f"Error: {e}"

def clear_chat():
    """Reset chat history."""
    return [], ""

with gr.Blocks(title="CrediTrust Complaint Chatbot (RAG)") as demo:
    gr.Markdown(
        """
        # CrediTrust Financial Complaint Chatbot
        Ask a question about complaints for Credit Cards, Personal Loans, BNPL, Savings Accounts, or Money Transfers.
        The AI will summarize answers with supporting complaint narratives.
        """
    )

    status_message = gr.Textbox(label="Status", value="Click 'Initialize RAG' to start.", interactive=False)
    llm_info = gr.Textbox(label="LLM Model In Use", value=llm_model_name, interactive=False)

    initialize_button = gr.Button("Initialize RAG Components")
    initialize_button.click(initialize_rag_components, outputs=status_message)

    chatbot = gr.Chatbot(label="Conversation History", height=400)
    msg = gr.Textbox(label="Your Question", placeholder="e.g., What are common issues with credit card fraud?")
    sources_output = gr.Markdown(label="Retrieved Sources", value="")

    submit_button = gr.Button("Ask CrediTrust")
    clear_button = gr.Button("Clear Chat")

    submit_button.click(
        ask_chatbot,
        inputs=[msg, chatbot],
        outputs=[chatbot, sources_output]
    ).then(lambda: "", inputs=None, outputs=msg) 

    clear_button.click(clear_chat, inputs=None, outputs=[chatbot, sources_output])

    gr.Markdown(
        "**Note:** If you see `ResourceExhausted` errors, your API quota may be exceeded. Wait and try again, or upgrade your Google Cloud project."
    )

demo.launch()