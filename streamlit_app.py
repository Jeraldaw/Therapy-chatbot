import streamlit as st
import torch
import transformers
from huggingface_hub import login # Needed for programmatic login if not using env var directly

# --- Configuration ---
# Set your MODEL constant. Consider "meta-llama/Meta-Llama-3-8B-Instruct" for better performance
# on Streamlit Cloud compared to larger models like 70B.
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# --- Hugging Face Token (Security Best Practice) ---
# Access the token from Streamlit Secrets.
# You MUST set this in your Streamlit Cloud app settings under "Secrets".
# For local testing, create a .streamlit/secrets.toml file with HF_TOKEN = "your_token_here".
if "HF_TOKEN" in st.secrets:
    # CORRECTED LINE: Access the secret using the key "HF_TOKEN"
    HF_TOKEN = st.secrets["HF_TOKEN"]
    # Programmatic login can be useful, but transformers often picks up the token
    # if it's set as an environment variable or via st.secrets.
    # login(token=HF_TOKEN, add_to_git_credential=True)
else:
    st.error("HF_TOKEN not found in Streamlit secrets. Please add it for model access.")
    st.stop() # Stop the app if the token is missing

# --- Streamlit App Title ---
st.title("My Therapy Chatbot") # Updated title for therapy chatbot

# --- Initial Rule-Based Question (adapted for Streamlit UI) ---
st.write("Hello! I'm here to listen and support you.") # Updated greeting
user_input_eat = st.radio(
    "Are you feeling physically well today? (This can impact your mood)", # Rephrased question for therapy context
    ("Yes, I feel good", "No, I'm feeling a bit off", "I'm not sure"), # Rephrased options
    key="eat_question" # Unique key for the radio button
)

if user_input_eat == "Yes, I feel good":
    st.write("That's good to hear! Physical well-being can be a great foundation for mental well-being.")
elif user_input_eat == "No, I'm feeling a bit off":
    st.write("I understand. Sometimes our physical state can influence our feelings. Take a moment to check in with yourself.")
else:
    st.write("It's okay to not be sure. Let's explore your thoughts and feelings together.")

st.write("\nNow, let's talk. You can share anything on your mind, and I'll do my best to offer support.") # Updated intro
st.write("Type 'end session' to conclude our conversation.") # Updated exit command

# --- Part 2: Setting up the AI (LLM) as a chat model ---
# Use st.cache_resource to load the model only once across app runs.
# This prevents reloading the model every time the user interacts with the app.
@st.cache_resource
def load_llm_model(model_name, hf_token):
    """
    Loads the Large Language Model using Hugging Face transformers pipeline.
    """
    try:
        # Determine the device (GPU if available, otherwise CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.write(f"Attempting to load {model_name} on {device}...")

        # Initialize the text generation pipeline
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            # Changed to torch.float16 for broader compatibility and memory efficiency
            model_kwargs={"torch_dtype": torch.float16},
            device_map="auto", # Automatically maps model layers to available devices
            # Pass the Hugging Face token for authentication, especially for gated models
            token=hf_token
        )
        st.write(f"AI chat model {model_name} loaded successfully on {device}.")
        return pipeline
    except Exception as e:
        st.error(f"Error loading AI chat model: {e}")
        st.warning("Falling back to a simpler response mechanism for open-ended questions.")
        return None

# Load the pipeline
pipeline = load_llm_model(MODEL, HF_TOKEN)

# --- Initialize chat history using st.session_state ---
# st.session_state persists variables across reruns of the Streamlit app.
if "messages" not in st.session_state:
    # The first message is the system prompt that defines the chatbot's persona.
    # Changed to a therapy chatbot persona.
    st.session_state.messages = [
        {"role": "system", "content": "You are a supportive, empathetic, and non-judgmental therapy chatbot assistant. Focus on active listening, validating feelings, and encouraging self-reflection. Do not give medical advice or diagnose. Always remind the user that you are an AI and not a substitute for professional human therapy."},
    ]

# --- Display chat messages from history on app rerun ---
# Iterate through the messages in the session state and display them.
for message in st.session_state.messages:
    # Skip displaying the system prompt directly in the chat UI.
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input from User ---
# st.chat_input provides a chat-like input box at the bottom of the app.
user_question = st.chat_input("You:")

if user_question:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_question})
    # Display user message in the chat UI
    with st.chat_message("user"):
        st.markdown(user_question)

    # Handle 'end session' command
    if user_question.lower() in ["end session", "quit", "exit"]: # Added more exit options
        with st.chat_message("assistant"):
            st.markdown("Thank you for sharing. Remember, I am an AI and not a substitute for professional human therapy. Please seek help from a qualified mental health professional if you need further support. Take care.")
        st.session_state.messages = [
            {"role": "system", "content": "You are a supportive, empathetic, and non-judgmental therapy chatbot assistant. Focus on active listening, validating feelings, and encouraging self-reflection. Do not give medical advice or diagnose. Always remind the user that you are an AI and not a substitute for professional human therapy."},
        ] # Reset history but keep the system prompt
    elif pipeline:
        try:
            # Show a spinner while the chatbot is generating a response
            with st.spinner("I'm listening..."): # Updated spinner text
                # Generate response using the LLM pipeline
                # Pass the entire conversation history for context
                outputs = pipeline(
                    st.session_state.messages,
                    max_new_tokens=8192, # Maximum tokens for the new response
                    do_sample=True,      # Enable sampling for more varied responses
                    temperature=0.7,     # Controls randomness (higher = more random)
                    top_p=0.9,           # Nucleus sampling parameter
                    # The pipeline for chat models often returns a list of dictionaries,
                    # where the last dictionary is the new assistant message.
                )

                chatbot_response = ""
                if outputs and outputs[0] and outputs[0].get("generated_text"):
                    # The 'generated_text' from the pipeline for chat models is typically
                    # the full list of messages, including the new assistant response.
                    # We want the content of the *last* message, which should be the assistant's.
                    last_message_dict = outputs[0]["generated_text"][-1]
                    if last_message_dict and last_message_dict.get("role") == "assistant":
                        chatbot_response = last_message_dict["content"]
                    else:
                        # Fallback if the structure is unexpected or role isn't 'assistant'
                        chatbot_response = "I apologize, I seem to have gotten a bit confused. Could you please rephrase what you're trying to express?" # Therapy-aligned fallback
                else:
                    chatbot_response = "I apologize, I seem to have gotten a bit lost. Could you please repeat what you said?" # Therapy-aligned fallback

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": chatbot_response})
                # Display assistant response in the chat UI
                with st.chat_message("assistant"):
                    st.markdown(chatbot_response)

        except Exception as e:
            # Handle errors during response generation
            with st.chat_message("assistant"):
                st.markdown(f"I apologize! I encountered a technical difficulty while processing your thoughts ({e}). Please try sharing something else.") # Therapy-aligned error message
            # Optionally, remove the last user message from history if it caused an error
            # st.session_state.messages.pop()

    else:
        # Fallback if the model failed to load initially
        with st.chat_message("assistant"):
            st.markdown("I apologize, my support functions are currently unavailable. Please feel free to share something simpler, or consider reaching out to a human professional for immediate support.")
