import streamlit as st
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Load secrets from Streamlit
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Initialize OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY,
)

# === Set up page ===
st.set_page_config(page_title="Institution Chatbot", page_icon="🎓")
st.title("🎓 Institution Info Chatbot")
st.markdown("Ask me anything about colleges, courses, placements, and more!")

# === Load FAISS index and documents ===
@st.cache_resource

def load_faiss_data():
    index = faiss.read_index("faiss_index.idx")
    with open("documents.pkl", "rb") as f:
        documents = pickle.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, documents, model

index, documents, model = load_faiss_data()

# === Search function ===
def search(query, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]

# === Ask Mistral ===
def ask_mistral(question, context):
    try:
        prompt = f"""Use the following context to answer the question accurately.

Context:
{context}

Question:
{question}

Answer:"""

        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"❌ Error communicating with Mistral LLM: {e}")
        return "Sorry, I couldn't get an answer from the AI model."

# === Chat Memory ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === User Input ===
user_query = st.text_input("\U0001F50D Ask a question:")
if user_query:
    context_docs = search(user_query)
    context = "\n".join(context_docs)
    answer = ask_mistral(user_query, context)
    
    st.session_state.chat_history.append((user_query, answer))

# === Display Chat History ===
if st.session_state.chat_history:
    st.markdown("## 💬 Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.markdown(f"**Q{i+1}:** {q}")
        st.markdown(f"**A{i+1}:** {a}")
