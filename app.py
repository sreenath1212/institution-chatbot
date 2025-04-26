import streamlit as st
import os
import chromadb
from sentence_transformers import SentenceTransformer
import openai

# === OpenRouter API Setup ===
openai.api_key = st.secrets["OPENAI_API_KEY"]
openai.api_base = "https://openrouter.ai/api/v1"

# === Setup ChromaDB client ===
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="institutions")

# === Load embedding model ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === Helper: Get top k chunks ===
def get_chunks(query, k=4):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    return results["documents"][0]

# === Helper: Ask Mistral LLM ===
def ask_mistral(question, context):
    prompt = f"""Use the following context to answer the question accurately.

Context:
{context}

Question:
{question}

Answer:"""

    response = openai.ChatCompletion.create(
        model="mistralai/mistral-7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response['choices'][0]['message']['content']

# === Streamlit UI ===
st.set_page_config(page_title="Institution Chatbot", page_icon="🎓")
st.title("🎓 Institution Info Chatbot")
st.markdown("Ask me anything about colleges, courses, placement, etc!")

user_input = st.text_input("🔎 Ask a question:")

if user_input:
    with st.spinner("Fetching answer..."):
        context_chunks = get_chunks(user_input, k=4)
        combined_context = "\n\n".join(context_chunks)
        answer = ask_mistral(user_input, combined_context)

        st.markdown("### ✅ Answer")
        st.write(answer)

        with st.expander("📄 Source Context"):
            for i, chunk in enumerate(context_chunks):
                st.markdown(f"**Chunk {i+1}:**\n{chunk}")
