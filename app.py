import streamlit as st
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# ---- Settings ----
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
TEMPERATURE = 0.3
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TEXT_FILE_PATH = "institution_descriptions.txt"

# ---- Helper Functions ----
@st.cache_resource
def load_text_file(filepath: str) -> str:
    """Loads and decodes a text file, handling potential errors."""
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        st.error(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

@st.cache_resource
def create_vectorstore_from_text(text: str):
    """Creates a FAISS vectorstore from the given text."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def openrouter_chat(messages: list):
    """Sends messages to the OpenRouter chat API and returns the response."""

    api_key = st.secrets["OPENROUTER_API_KEY"]
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None
    except (KeyError, ValueError) as e:
        st.error(f"Invalid API response: {e}")
        return None

def ask_mistral(question: str, context: str):
    """Formats the prompt and gets an answer from the chat API."""

    prompt = f"""Answer the question in a friendly and helpful manner, using only the provided context. 
    Avoid technical jargon and do not mention the data flow or retrieval process. 
    If the question cannot be answered from the context, respond politely that you cannot provide an answer.

    {context}

    Question: {question}
    Answer:"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions about institutions."},
        {"role": "user", "content": prompt},
    ]
    return openrouter_chat(messages)

def correct_text(text: str) -> str:
    """Enhanced text correction using fuzzy matching and abbreviation handling."""

    corrections = {
        "instution": "institution",
        "addres": "address",
        "phone no": "phone number",
        "phone number": "phone number",
        "ph.no": "phone number",
        "pincode": "pin code",
        "pin code": "pin code",
        "email id": "email",
        "email address": "email",
        "websitee": "website",
        "web site": "website",
        "principal sir": "principal",
        "principal mam": "principal",
        "co-ordinator": "coordinator",
        "coordinator": "coordinator",
        "facebook page": "facebook",
        "facebook": "facebook",
        "insta": "instagram",
        "instagram": "instagram",
        "youtube channel": "youtube",
        "youtube": "youtube",
        "fund details": "funding details",
        "funding details": "funding details",
        "mous": "memorandum of understanding",
        "memorandum of understanding": "memorandum of understanding",
        "course offered": "courses offered",
        "courses offered": "courses offered",
        "how many seat": "intake",
        "seat intake": "intake",
        "admission details": "admission",
        "admission process": "admission",
        "placement record": "placement",
        "placement details": "placement",
        "higher study": "higher studies",
        "higher studies": "higher studies",
        "b.tech": "bachelor of technology",
        "bachelor of technology": "bachelor of technology",
        "m.tech": "master of technology",
        "master of technology": "master of technology",
        "bca": "bachelor of computer applications",
        "bachelor of computer application": "bachelor of computer applications",
        "mca": "master of computer applications",
        "master of computer application": "master of computer applications",
        "bsc cs": "bachelor of science in computer science",
        "bachelor of science computer science": "bachelor of science in computer science",
        "msc cs": "master of science in computer science",
        "master of science computer science": "master of science in computer science",
        "contact no": "contact number",
        "contact number": "contact number"
    }

    words = text.split()
    corrected_words = []

    for word in words:
        word_lower = word.lower()
        if word_lower in corrections:
            corrected_words.append(corrections[word_lower])
        else:
            best_match = process.extractOne(word, corrections.keys(), scorer=fuzz.token_set_ratio)
            if best_match and best_match[1] > 80:
                corrected_words.append(corrections[best_match[0]])
            else:
                corrected_words.append(word)

    return " ".join(corrected_words)

# ---- Streamlit App ----
def main():
    st.set_page_config(
        page_title="🎓 Institution Chatbot (RAG)", page_icon="📚", layout="wide"
    )
    st.title("🎓 Institution Information Chatbot")
    st.markdown("Ask questions about the institutions.")

    # Load data and create vectorstore
    text_data = load_text_file(TEXT_FILE_PATH)
    if text_data:
        vectorstore = create_vectorstore_from_text(text_data)
    else:
        vectorstore = None
        st.error("🚫 Failed to load institution data. Please check the file path.")
        return

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input
    user_query = st.chat_input("Type your question here...")

    # Show past chat
    if not st.session_state.chat_history:
        st.info("Start the conversation by asking a question!")
    else:
        for chat in st.session_state.chat_history:
            with st.chat_message(chat["role"]):
                st.markdown(chat["content"])

    # New user query
    if user_query and vectorstore:
        corrected_query = correct_text(user_query)
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Search relevant chunks
        with st.spinner("Searching for relevant information..."):
            retriever = vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 4}
            )
            relevant_docs = retriever.get_relevant_documents(corrected_query)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Ask Mistral
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = ask_mistral(corrected_query, context)
                    if answer:
                        st.markdown(answer)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": answer}
                        )
                    else:
                        st.warning(
                            "🚫 Assistant could not find an answer. Please rephrase your query."
                        )
                except Exception as e:
                    st.error(f"Error: {e}")

    elif user_query and not vectorstore:
        st.error("🚫 Data not loaded. Please check the file.")

if __name__ == "__main__":
    main()
