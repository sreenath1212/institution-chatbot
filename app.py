import streamlit as st
import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer
import openai

# Try to import and potentially replace the system sqlite3 with pysqlite3
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    print("✅ Successfully replaced system sqlite3 with pysqlite3")
except ImportError:
    print("⚠️ pysqlite3 not found, using system sqlite3")

# === OpenRouter API Setup ===
openai.api_key = st.secrets.get("OPENAI_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

# === Initialize ChromaDB client ===
try:
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="institutions")
    print("✅ ChromaDB client initialized and collection accessed.")
except Exception as e:
    st.error(f"❌ Error initializing ChromaDB: {e}")
    st.stop()

# === Load embedding model ===
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ SentenceTransformer model loaded.")
except Exception as e:
    st.error(f"❌ Error loading SentenceTransformer model: {e}")
    st.stop()

# === Helper: Get top k chunks ===
def get_chunks(query, k=4):
    try:
        query_embedding = embedder.encode(query).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=k)
        if results and results["documents"] and len(results["documents"]) > 0:
            return results["documents"][0]
        else:
            return []
    except Exception as e:
        st.error(f"❌ Error retrieving chunks from ChromaDB: {e}")
        return []

# === Helper: Ask Mistral LLM ===
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
            temperature=0.2
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"❌ Error communicating with Mistral LLM: {e}")
        return "Sorry, I couldn't get an answer from the AI model."

# === Streamlit UI ===
st.set_page_config(page_title="Institution Chatbot", page_icon="🎓")
st.title("🎓 Institution Info Chatbot")
st.markdown("Ask me anything about colleges, courses, placement, etc!")

user_input = st.text_input("🔎 Ask a question:")

if user_input:
    if not openai.api_key:
        st.error("⚠️ Please set the OPENAI_API_KEY in Streamlit secrets.")
    elif not collection:
        st.error("⚠️ ChromaDB collection not initialized.")
    elif not embedder:
        st.error("⚠️ Embedding model not loaded.")
    else:
        with st.spinner("Fetching answer..."):
            context_chunks = get_chunks(user_input, k=4)
            if context_chunks:
                combined_context = "\n\n".join(context_chunks)
                answer = ask_mistral(user_input, combined_context)

                st.markdown("### ✅ Answer")
                st.write(answer)

                with st.expander("📄 Source Context"):
                    for i, chunk in enumerate(context_chunks):
                        st.markdown(f"**Chunk {i+1}:**\n{chunk}")
            else:
                st.warning("No relevant information found in the knowledge base.")
