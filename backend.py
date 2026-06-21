import re
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Conversation memory ---
conversation_memory = []
unknown_questions = []

# --- Normalize text ---
def normalize(text):
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove punctuation
    return text

# --- Load models once ---
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return embedder, tokenizer, model

# --- Answer retrieval ---
def get_best_answer(user_question, df, question_embeddings, embedder, tokenizer, model,
                    threshold=0.85, top_k=3, temperature=1.0):
    normalized_question = normalize(user_question)
    q_emb = embedder.encode(normalized_question, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, question_embeddings)[0]

    top_scores, top_indices = torch.topk(scores, k=top_k)
    best_score = top_scores[0].item()

    if best_score >= threshold:
        best_idx = top_indices[0].item()
        raw_answer = df['Answer'].iloc[best_idx]
        return raw_answer
    else:
        unknown_questions.append(user_question)
        return "Sorry, I couldn’t find a matching answer in the handbook."
