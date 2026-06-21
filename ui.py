import streamlit as st
import pandas as pd
import random

def render_ui(embedder, tokenizer, model, get_best_answer, conversation_memory, unknown_questions):
    # --- WhatsApp-style CSS ---
    chat_css = """
    <style>
    body, .stApp {
        background-color: #f0f0f0;
        color: #000000;
        font-family: Arial, sans-serif;
    }
    body.dark, .stApp.dark {
        background-color: #121212;
        color: #E0E0E0;
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
        color: #000000;
    }
    body.dark .stTextInput > div > div > input {
        background-color: #1E1E1E;
        color: #E0E0E0;
    }
    .stButton>button {
        background-color: #075E54;
        color: #FFFFFF;
        border-radius: 20px;
        padding: 8px 16px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #128C7E; /* greenish hover */
        color: #FFFFFF;
    }
    .user-bubble {
        text-align: right;
        background-color: #DCF8C6;
        padding: 10px;
        border-radius: 15px;
        margin: 5px;
        max-width: 70%;
        display: inline-block;
    }
    .assistant-bubble {
        text-align: left;
        background-color: #FFFFFF;
        color: #000000;
        padding: 10px;
        border-radius: 15px;
        margin: 5px;
        max-width: 70%;
        display: inline-block;
    }
    body.dark .assistant-bubble {
        background-color: #2A2A2A;
        color: #FFFFFF;
    }
    .suggestion-bubble {
        display: inline-block;
        background-color: #ECE5DD;
        color: #000000;
        padding: 6px 12px;
        border-radius: 15px;
        margin: 4px;
        font-size: 0.9em;
        cursor: pointer;
    }
    .suggestion-bubble:hover {
        background-color: #DCF8C6;
    }
    </style>
    """
    st.markdown(chat_css, unsafe_allow_html=True)

    # --- Sidebar Controls ---
    st.sidebar.title("📚 Chat Memory & Controls")
    temperature = st.sidebar.slider("Response Temperature", 0.1, 1.5, 1.0, 0.1)

    st.sidebar.write("Questions Asked:")
    for i, (q, _) in enumerate(conversation_memory):
        if st.sidebar.button(q, key=f"sidebar_q_{i}"):
            answer = get_best_answer(q, st.session_state['df'], st.session_state['embeddings'],
                                     embedder, tokenizer, model, temperature=temperature)
            if (q, answer) not in conversation_memory:
                conversation_memory.append((q, answer))

    st.sidebar.write("Unanswered Questions:")
    for uq in unknown_questions:
        st.sidebar.markdown(f"- {uq}")

    # --- Main Chat UI ---
    st.title("💬 NUST Q&A Assistant")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df
        st.success(f"Loaded file: {uploaded_file.name}")
        st.session_state['embeddings'] = embedder.encode(df['Question'].tolist(), convert_to_tensor=True)

        # Input + Send button inline
        col1, col2 = st.columns([8,1])
        with col1:
            user_question = st.text_input("💬 Ask a question:", value=st.session_state.get("user_question", ""), placeholder="Type here...")
        with col2:
            send = st.button("Send", key="send_btn")

        # Handle send
        if send and user_question.strip() != "":
            answer = get_best_answer(user_question, df, st.session_state['embeddings'],
                                     embedder, tokenizer, model, temperature=temperature)
            if (user_question, answer) not in conversation_memory:
                conversation_memory.append((user_question, answer))
            st.session_state["user_question"] = ""  # clear after sending

        # Suggestions (auto-answer on click)
        st.write("Suggestions:")
        if 'last_question' in st.session_state:
            suggestions = random.sample(df['Question'].tolist(), min(4, len(df)))
        else:
            suggestions = df['Question'].head(4).tolist()

        cols = st.columns(len(suggestions))
        for i, q in enumerate(suggestions):
            if cols[i].button(q, key=f"suggest_{i}"):
                answer = get_best_answer(q, df, st.session_state['embeddings'],
                                         embedder, tokenizer, model, temperature=temperature)
                if (q, answer) not in conversation_memory:
                    conversation_memory.append((q, answer))
                st.session_state['last_question'] = q
                st.session_state["user_question"] = ""  # clear input

        # Display full conversation history
        for i, (q, a) in enumerate(conversation_memory):
            st.markdown(f"<div class='user-bubble'>You: {q}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='assistant-bubble'>Assistant: {a}</div>", unsafe_allow_html=True)
