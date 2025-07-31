import pandas as pd
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE_PATH = 'vector_store'
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("Task 3: Initializing RAG system...")

try:
    google_api_key = os.getenv('GOOGLE_API_KEY')
    if not google_api_key:
        print("Missing GOOGLE_API_KEY. Set it in your .env or environment.")
        exit()
    genai.configure(api_key=google_api_key)
except Exception as e:
    print(f"Error configuring Google Generative AI: {e}")
    exit()

LLM_MODEL_TO_USE = None
available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
priority_models = [
    "models/gemini-1.5-flash",
    "models/gemini-1.5-flash-latest",
    "models/gemini-1.5-pro",
    "models/gemini-1.0-pro"
]
for m in priority_models:
    if m in available_models:
        LLM_MODEL_TO_USE = m
        break
if not LLM_MODEL_TO_USE and available_models:
    for model_name in available_models:
        if "gemini" in model_name:
            LLM_MODEL_TO_USE = model_name
            break
if not LLM_MODEL_TO_USE:
    print("No Gemini model with 'generateContent' found. Check API/model access.")
    exit()

print(f"Model: {LLM_MODEL_TO_USE}")

embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
try:
    db = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    print(f"Vector store load error: {e}")
    exit()

retriever = db.as_retriever(search_kwargs={"k": 5})

try:
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL_TO_USE, temperature=0.3, google_api_key=google_api_key)
except Exception as e:
    print(f"LLM init error: {e}")
    exit()

template = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context: {context}

Question: {question}

Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", 
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

def ask_credittrust_rag(query: str):
    print(f"\nQ: {query}")
    try:
        result = qa_chain.invoke({"query": query})
        answer = result["result"]
        source_documents = result["source_documents"]

        print(f"AI: {answer}")
        print("Sources:")
        retrieved_sources_info = []
        if source_documents:
            for i, doc in enumerate(source_documents):
                info = {
                    "content_snippet": doc.page_content[:100] + "...",
                    "complaint_id": doc.metadata.get('complaint_id', 'N/A'),
                    "product": doc.metadata.get('product', 'N/A')
                }
                retrieved_sources_info.append(info)
                print(f"  [{i+1}] ID:{info['complaint_id']} Product:{info['product']}")
        else:
            print("  None")
        return answer, retrieved_sources_info
    except Exception as e:
        print(f"Error: {e}")
        return "Error processing request.", []

evaluation_questions = [
    "What are the most common issues reported by consumers?",
    "Which product category receives the most complaints overall?",
    "What trends can be observed in complaint submission methods?",
    "Are there recurring problems related to customer service?",
    "How have complaint volumes changed over time?",
    "What are the top-requested resolutions by consumers?",
    "Which companies receive the most complaints?",
    "Are there patterns in how quickly companies respond?",
    "What are common issues related to loan products?",
    "What is the general sentiment of the complaints?"
]

evaluation_results = []
for i, q in enumerate(evaluation_questions):
    print(f"\nEval {i+1}/{len(evaluation_questions)}")
    generated_answer, retrieved_sources = ask_credittrust_rag(q)
    evaluation_results.append({
        "Question": q,
        "Generated Answer": generated_answer,
        "Retrieved Sources": retrieved_sources,
        "Quality Score": "N/A",
        "Comments/Analysis": "N/A"
    })

print("\nEvaluation Results Table:")
print("| Question | Generated Answer | Retrieved Sources (1-2) | Quality Score | Comments/Analysis |")
print("|---|---|---|---|---|")
for res in evaluation_results:
    sources_str = ""
    for j, src in enumerate(res["Retrieved Sources"][:2]):
        sources_str += f"[ID:{src['complaint_id']}, {src['product']}] "
    escaped_answer = res["Generated Answer"].replace("|", "\\|")
    print(f"| {res['Question']} | {escaped_answer} | {sources_str.strip()} | {res['Quality Score']} | {res['Comments/Analysis']} |")

print("\nTask 3 complete.")