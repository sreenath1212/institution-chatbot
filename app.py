# chatbot_mini_rag.py

import streamlit as st
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import json
import tempfile
import os

# ---- Settings ----
MODEL_NAME = "mistralai/mistral-7b-instruct"
TEMPERATURE = 0.3
CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 100

# ---- Helper Functions ----

@st.cache_resource
def load_text_file(file) -> str:
    return file.read().decode('utf-8')

@st.cache_resource
def create_vectorstore_from_text(text):
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorstore

def openrouter_chat(messages):
    api_key = st.secrets["OPENROUTER_API_KEY"]
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
    }
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

    try:
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception:
        raise Exception(f"Invalid response: {response.text}")

def ask_mistral(question, context):
    prompt = f"""Answer the question based ONLY on the following context:

{context}

Question: {question}
Answer:"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers ONLY using the provided context."},
        {"role": "user", "content": prompt}
    ]
    return openrouter_chat(messages)

# ---- Streamlit App ----

def main():
    st.set_page_config(page_title="🎓 Institution Chatbot (Mini-RAG)", page_icon="📚", layout="wide")

    st.title("🎓 Institution Information Chatbot (Mini-RAG + LangChain)")
    st.markdown("Upload your institution descriptions file and ask questions!")

    # Sidebar
    with st.sidebar:
        st.title("Settings ⚙️")
        uploaded_file = st.file_uploader("📄 Upload Institution Descriptions (.txt)", type="txt")
        if uploaded_file:
            text_data = load_text_file(uploaded_file)
            vectorstore = create_vectorstore_from_text(text_data)
            st.success("✅ File processed into vector database!")
        else:
            text_data = None
            vectorstore = None
        st.divider()

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input
    user_query = st.chat_input("Type your question here...")

    # Show past chat
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    # New user query
    if user_query and vectorstore:
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Search relevant chunks
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        relevant_docs = retriever.get_relevant_documents(user_query)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Ask Mistral
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = ask_mistral(user_query, context)
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")

    elif user_query and not vectorstore:
        st.error("🚫 Please upload a text file first!")

if __name__ == "__main__":
    main()
