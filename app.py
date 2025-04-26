# chatbot_pro_secrets.py

import streamlit as st
import requests
from typing import List
import tiktoken
import json 

# ---- Settings ----
MODEL_NAME = "mistralai/mistral-7b-instruct"
TEMPERATURE = 0.3
MAX_TOKENS = 8000  # Safe limit for Mistral 7B

# ---- Helper Functions ----

@st.cache_resource
def load_text_file(file) -> str:
    return file.read().decode('utf-8')

def ask_openrouter(messages: List[dict], stream=False):
    import json  # Safe parsing
    
    api_key = st.secrets["OPENROUTER_API_KEY"]
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "stream": stream
    }

    response = requests.post(url, headers=headers, json=data, stream=stream)
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    if not stream:
        return response.json()['choices'][0]['message']['content'].strip()

    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8').replace('data: ', '')
            if decoded_line == '[DONE]':
                break
            content = json.loads(decoded_line)['choices'][0]['delta'].get('content', '')
            if content:
                yield content


def split_text_into_chunks(text, max_tokens=3000):
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    words = text.split()
    chunks = []
    current_chunk = []

    current_tokens = 0
    for word in words:
        token_count = len(tokenizer.encode(word))
        if current_tokens + token_count > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_tokens = 0
        current_chunk.append(word)
        current_tokens += token_count

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# ---- Streamlit App ----

def main():
    st.set_page_config(page_title="🚀 Institution Chatbot", page_icon="🎓", layout="wide")

    st.title("🎓 Institution Information Chatbot (Streamlit Cloud)")
    st.markdown("Ask anything about the uploaded institution descriptions file!")

    # Sidebar
    with st.sidebar:
        st.title("Settings ⚙️")
        uploaded_file = st.file_uploader("📄 Upload Institution Descriptions (.txt)", type="txt")
        if uploaded_file:
            context_text = load_text_file(uploaded_file)
            st.success("✅ File uploaded successfully!")
        else:
            context_text = None
        st.divider()

    # Initialize session states
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input
    user_query = st.chat_input("Type your question here...")

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    # If user asks something
    if user_query and context_text:
        with st.chat_message("user"):
            st.markdown(user_query)
        
        full_prompt = f"""You must answer based ONLY on the following text context:
        
{context_text}

Question: {user_query}
"""

        st.session_state.chat_history.append({"role": "user", "content": user_query})
        assistant_message = ""

        with st.chat_message("assistant"):
            placeholder = st.empty()
            messages = [
                {"role": "system", "content": "You are a helpful assistant answering ONLY based on the provided institution context."},
                {"role": "user", "content": full_prompt}
            ]

            try:
                for chunk in ask_openrouter(messages, stream=True):
                    assistant_message += chunk
                    placeholder.markdown(assistant_message + "▌")
                placeholder.markdown(assistant_message)
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_message})
            except Exception as e:
                st.error(f"Error: {e}")

    elif user_query and not context_text:
        st.error("🚫 Please upload a .txt file first!")

if __name__ == "__main__":
    main()
