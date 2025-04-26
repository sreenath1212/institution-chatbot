# === Imports ===
import os
import re
import sys
import streamlit as st
import chromadb
import openai
from sentence_transformers import SentenceTransformer

# === Setup sqlite3 compatibility for ChromaDB ===
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    print("✅ Using pysqlite3 for sqlite3.")
except ImportError:
    print("⚠️ pysqlite3 not found, using system sqlite3.")

# === Constants ===
CHROMA_COLLECTION_NAME = "institutions"
CHROMA_PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DATA_FILE = "institution_descriptions.txt"

# === Initialize Clients ===
openai.api_key = st.secrets.get("OPENAI_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

try:
    chroma_client = chromadb.Client(
        settings=chromadb.Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=CHROMA_PERSIST_DIR,
        )
    )
except Exception as e:
    st.error(f"❌ Error initializing ChromaDB: {e}")
    st.stop()

try:
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"✅ Loaded embedding model: {EMBEDDING_MODEL_NAME}")
except Exception as e:
    st.error(f"❌ Error loading embedding model: {e}")
    st.stop()

# === Functions ===

def load_or_create_collection():
    """Load existing collection or create new one if missing."""
    if not embedder:
        st.error("❌ Embedding model not available.")
        return None
    
    try:
        collection = chroma_client.get_collection(CHROMA_COLLECTION_NAME)
        if collection.count() > 0:
            print(f"✅ Collection '{CHROMA_COLLECTION_NAME}' loaded with {collection.count()} documents.")
            return collection
    except Exception:
        print(f"⚠️ Collection '{CHROMA_COLLECTION_NAME}' not found. Creating new one.")
    
    # Load and split data
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            raw_text = file.read()
    except FileNotFoundError:
        st.error(f"❌ File '{DATA_FILE}' not found.")
        return None

    chunks = [chunk.strip() for chunk in re.split(r"-{5,}", raw_text) if len(chunk.strip()) > 50]
    documents = [chunk for chunk in chunks]
    metadatas = [{"source": f"chunk_{i}"} for i in range(len(chunks))]
    ids = [f"inst_{i}" for i in range(len(chunks))]
    embeddings = embedder.encode(documents).tolist()

    try:
        collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
        collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
        chroma_client.persist()  # Save to disk
        print(f"✅ Stored {len(documents)} documents in ChromaDB.")
        return collection
    except Exception as e:
        st.error(f"❌ Failed to create collection: {e}")
        return None

def retrieve_relevant_chunks(query, k=4):
    """Retrieve top-k relevant documents from ChromaDB."""
    try:
        query_embedding = embedder.encode(query).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=k)
        return results["documents"][0] if results and results["documents"] else []
    except Exception as e:
        st.error(f"❌ Retrieval error: {e}")
        return []

def stream_mistral_llm(question, context):
    """Stream response from Mistral LLM using OpenRouter."""
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
            stream=True,  # 👈 Enable streaming
        )

        full_response = ""
        for chunk in response:
            if "choices" in chunk:
                delta = chunk["choices"][0]["delta"]
                content = delta.get("content", "")
                full_response += content
                yield full_response  # 👈 Yield the growing text piece by piece
    except Exception as e:
        st.error(f"❌ LLM streaming error: {e}")
        yield "⚠️ Sorry, no answer available."

# === Initialize collection ===
collection = load_or_create_collection()

# === Streamlit UI ===
st.set_page_config(page_title="🎓 Institution Chatbot", page_icon="🎓")
st.title("🎓 Institution Info Chatbot")
st.markdown("Ask me anything about colleges, courses, placements, and more!")

# === Session state: for memory ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === Chat input ===
user_query = st.text_input("🔎 Ask your question:", key="user_input")

if user_query:
    if not openai.api_key:
        st.error("⚠️ OPENAI_API_KEY missing.")
    elif not collection:
        st.error("⚠️ ChromaDB collection unavailable.")
    else:
        with st.spinner("Retrieving answer..."):
            context_chunks = retrieve_relevant_chunks(user_query, k=4)

            if context_chunks:
                combined_context = "\n\n".join(context_chunks)
                # Placeholder to update streaming text
                response_placeholder = st.empty()
                partial_answer = ""

                # Stream the response
                for partial in stream_mistral_llm(user_query, combined_context):
                    partial_answer = partial
                    response_placeholder.markdown("### ✅ Answer\n" + partial_answer)

                # Save to chat history
                st.session_state.chat_history.append({
                    "question": user_query,
                    "answer": partial_answer
                })

                with st.expander("📄 Source Context"):
                    for idx, chunk in enumerate(context_chunks):
                        st.markdown(f"**Chunk {idx+1}:**\n{chunk}")

            else:
                st.warning("⚠️ No relevant information found.")

# === Show chat history ===
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("## 🕑 Previous Conversations")
    for idx, chat in enumerate(reversed(st.session_state.chat_history), 1):
        st.markdown(f"**{idx}. Question:** {chat['question']}")
        st.markdown(f"**Answer:** {chat['answer']}")
