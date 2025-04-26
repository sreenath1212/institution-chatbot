import os
import re
import sys
import streamlit as st
import faiss
import openai
import numpy as np
from sentence_transformers import SentenceTransformer

# === OpenRouter API Setup ===
openai.api_key = st.secrets.get("OPENAI_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

# === Load embedding model ===
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# === Load and embed data ===
@st.cache_resource
def load_and_embed_data():
    try:
        with open("institution_descriptions.txt", "r", encoding="utf-8") as file:
            raw_text = file.read()
    except FileNotFoundError:
        st.error("❌ 'institution_descriptions.txt' not found.")
        return None, None, None

    chunks = [block.strip() for block in re.split(r"-{5,}", raw_text) if len(block.strip()) > 50]

    documents = []
    vectors = []
    for chunk in chunks:
        embedding = embedder.encode(chunk)
        vectors.append(embedding)
        documents.append(chunk)

    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors))

    return index, vectors, documents

index, vector_data, documents = load_and_embed_data()

# === Retrieve top-k documents ===
def get_chunks(query, k=4):
    if index is None or documents is None:
        return []

    query_vector = embedder.encode(query).astype("float32").reshape(1, -1)
    _, indices = index.search(query_vector, k)
    return [documents[i] for i in indices[0] if i < len(documents)]

# === Ask Mistral LLM ===
def ask_mistral(question, context):
    try:
        prompt = f"""Use the following context to answer the question accurately.

Context:
{context}

Question:
{question}

Answer:"""

        response = openai.ChatCompletion.create(
            model="mistralai/mistral-7b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"❌ Error communicating with Mistral LLM: {e}")
        return "Sorry, I couldn't get an answer from the AI model."

# === Streamlit UI ===
st.set_page_config(page_title="Institution Chatbot", page_icon="🎓")
st.title("🎓 Institution Info Chatbot")
st.markdown("Ask me anything about colleges, courses, placements, and more!")

# Store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
user_input = st.text_input("🔎 Ask a question:")

# Button to clear chat
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []

if user_input:
    with st.spinner("Thinking..."):
        context_chunks = get_chunks(user_input, k=4)
        combined_context = "\n\n".join(context_chunks)
        answer = ask_mistral(user_input, combined_context)

        # Add to chat history
        st.session_state.chat_history.append((user_input, answer))

# Display chat history
if st.session_state.chat_history:
    st.markdown("### 💬 Chat History")
    for i, (q, a) in enumerate(reversed(st.session_state.chat_history), 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a}")
        st.markdown("---")
