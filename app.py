import requests
import json
import streamlit as st
import os
import time  # Import the time module
from openai import OpenAI

# 1. Load Knowledge Base from Text File
def load_knowledge_base(file_path):
    """
    Loads the knowledge base from a text file.

    Args:
        file_path (str): The path to the text file.

    Returns:
        str: The content of the text file, or None on error.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            knowledge_base = file.read()
        return knowledge_base
    except FileNotFoundError:
        st.error(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# 2. Prepare Prompt with Context
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
    You are a helpful chatbot. Use the following information to answer the user's question.
    If you don't know the answer, just say "I don't know".

    Knowledge Base:
    {knowledge_base}

    User Question:
    {query}

    Your Response:
    """
    return prompt

# 3. Send Request to OpenRouter API
def get_response_from_openrouter(prompt, api_key, model="mistralai/mistral-7b-instruct", max_retries=3, retry_delay=2):
    """
    Sends a request to the OpenRouter API to get a response from the Mistral model, with retry logic.

    Args:
        prompt (str): The prompt to send to the model.
        api_key (str): Your OpenRouter API key.
        model (str, optional): The model to use. Defaults to "mistralai/mistral-7b-instruct".
        max_retries (int, optional): Maximum number of times to retry the request. Defaults to 3.
        retry_delay (int, optional): Delay in seconds between retries. Defaults to 2.

    Returns:
        str: The model's response, or None on error.
    """
    if "gpt" in model: # Use the openai client for gpt models
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "http://localhost",  #  Replace with your site URL if you have one
                        "X-Title": "My Streamlit App",  # Replace with your site title
                    },
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                return completion.choices[0].message.content
            except Exception as e:
                st.error(f"Error with OpenAI client: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return None
        return None

    else: # Use the requests library for other models
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

        for attempt in range(max_retries):
            try:
                response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
                response.raise_for_status()
                st.write(f"Response status code: {response.status_code}")
                response_json = response.json()
                st.write(f"Raw JSON response: {response_json}")
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    return response_json["choices"][0]["message"]["content"].strip()
                else:
                    st.error("Error: No response from the model.")
                    return None
            except requests.exceptions.RequestException as e:
                st.error(f"Error making API request: {e}")
                if attempt < max_retries - 1 and "502" in str(e):  # Retry only for 502 errors
                    st.write(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    return None
            except json.JSONDecodeError:
                st.error("Error: Could not decode JSON response from API.")
                return None
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                return None
        return None  # Return None if all retries fail

# 4. Main Chatbot Function with Streamlit Interface
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

    knowledge_base_path = "institution_descriptions.txt" # Renamed to institution_descriptions.txt
    knowledge_base = load_knowledge_base(knowledge_base_path)
    if knowledge_base is None:
        st.error("Chatbot cannot start without a valid knowledge base.")
        return

     # Create a dummy knowledge base file if it doesn't exist
    if not os.path.exists(knowledge_base_path):
        with open(knowledge_base_path, "w", encoding="utf-8") as f:
            f.write("This is a sample knowledge base.  It can contain information about anything.")
        st.info(f"Created a dummy knowledge base file at {knowledge_base_path}.  Please edit this file with your desired knowledge.")

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

        response = prepare_prompt(knowledge_base, prompt)
        full_response = get_response_from_openrouter(response, api_key, model="google/gemma-3-1b-it:free") # added model
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(full_response)
        if not full_response:
            st.error("Failed to get a response from the chatbot.")

if __name__ == "__main__":
    # This is the original chatbot function.  I'm adding a new main block
    # to run your image analysis code.
    # chatbot()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=st.secrets["OPENROUTER_API_KEY"], # Use the API key from Streamlit secrets
    )

    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "http://localhost",  # Replace with your site URL
            "X-Title": "My Image Analysis App",  # Replace with your site title
        },
        model="google/gemma-3-1b-it:free",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                        }
                    }
                ]
            }
        ]
    )
    print(completion.choices[0].message.content)
    # st.write(completion.choices[0].message.content) #display in streamlit
