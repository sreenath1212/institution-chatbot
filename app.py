import requests
import json
import streamlit as st
import os
import sqlite3  # For SQLite database

# 1. Connect to the Database
def connect_to_db(db_path):
    """
    Connects to the SQLite database.

    Args:
        db_path (str): The path to the SQLite database file.

    Returns:
        sqlite3.Connection: A connection object, or None on error.
    """
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        st.error(f"Error connecting to database: {e}")
        return None

# 2. Fetch Knowledge from Database
def get_knowledge_from_db(conn, query):
    """
    Fetches relevant knowledge from the database based on the user query.
    This function now attempts to find more relevant information.

    Args:
        conn (sqlite3.Connection): The database connection object.
        query (str): The user's query.

    Returns:
        str: The fetched knowledge, or None if not found.
    """
    try:
        cursor = conn.cursor()
        # Refined query to search for the query in the description
        cursor.execute("SELECT description FROM knowledge WHERE description LIKE ?", ('%' + query + '%',)) #changed table name
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            return None
    except sqlite3.Error as e:
        st.error(f"Error fetching data from database: {e}")
        return None

# 3. Prepare Prompt with Context
def prepare_prompt(knowledge_base, query):
    """
    Prepares the prompt for the language model, including the relevant knowledge.

    Args:
        knowledge_base (str): The knowledge base text.
        query (str): The user's query.

    Returns:
        str: The formatted prompt.
    """
    prompt = f"""
    You are a helpful chatbot.  Respond to the user's question using only the information provided below.
    Do not provide any information that is not in the knowledge base.
    If you cannot answer the question from the knowledge base, say you don't have enough information.

    Knowledge Base:
    {knowledge_base}

    User Question:
    {query}

    Your Response:
    """
    return prompt

# 4. Send Request to OpenRouter API
def get_response_from_openrouter(prompt, api_key, model="mistralai/mistral-7b-instruct"):
    """
    Sends a request to the OpenRouter API to get a response from the Mistral model.

    Args:
        prompt (str): The prompt to send to the model.
        api_key (str): Your OpenRouter API key.
        model (str, optional): The model to use. Defaults to "mistralai/mistral-7b-instruct".

    Returns:
        str: The model's response, or None on error.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,  # Lower temperature for more focused responses
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        response_json = response.json()
        if "choices" in response_json and len(response_json["choices"]) > 0:
            return response_json["choices"][0]["message"]["content"].strip()
        else:
            st.error("Error: No response from the model.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error making API request: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Error: Could not decode JSON response from API.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# 5. Main Chatbot Function with Streamlit Interface
def chatbot():
    """
    Main chatbot function that loads the knowledge base, gets user input from
    the Streamlit UI, generates a prompt, and gets a response from the
    language model.  It also handles API key input and storage.
    """
    st.title("Chatbot with Knowledge Base")

    # Use Streamlit's secrets management
    if "OPENROUTER_API_KEY" in st.secrets:
        api_key = st.secrets["OPENROUTER_API_KEY"]
    else:
        api_key = st.text_input("Enter your OpenRouter API Key:", type="password")
        if api_key:
            st.warning("Please save your API key to the Streamlit secrets using the GUI (top right menu -> Settings -> Secrets).")
        else:
            st.stop()

    db_path = "knowledge_base.db"
    conn = connect_to_db(db_path)
    if conn is None:
        st.error("Chatbot cannot start without a database connection.")
        return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about the knowledge base:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        knowledge_base = get_knowledge_from_db(conn, prompt)
        if not knowledge_base:
            knowledge_base = "I'm sorry, I don't have enough information to answer your question." #changed
        response = prepare_prompt(knowledge_base, prompt)
        full_response = get_response_from_openrouter(response, api_key)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(full_response)
        if not full_response:
            st.error("Failed to get a response from the chatbot.")
    conn.close()

if __name__ == "__main__":
    chatbot()
