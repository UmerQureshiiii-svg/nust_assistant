import streamlit as st
import pandas as pd
from backend import load_models, get_best_answer, conversation_memory, unknown_questions
from ui import render_ui

def main():
    embedder, tokenizer, model = load_models()
    render_ui(embedder, tokenizer, model, get_best_answer, conversation_memory, unknown_questions)

if __name__ == "__main__":
    main()
