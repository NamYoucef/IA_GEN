import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import streamlit as st

# Charger les modèles
@st.cache_resource
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

gpt2_model, gpt2_tokenizer = load_model_and_tokenizer("gpt2")
gpt_neo_model, gpt_neo_tokenizer = load_model_and_tokenizer("EleutherAI/gpt-neo-1.3B")

# Fonction pour générer un texte
def generate_poem(prompt, model, tokenizer, max_length=100):
    poetic_prompt = f"Écris en français un poème sur le thème suivant : {prompt}\n\n"
    inputs = tokenizer.encode(poetic_prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Fonction pour calculer les scores BLEU et ROUGE
def evaluate_similarity(reference, generated):
    bleu_score = sentence_bleu([reference.split()], generated.split())
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, generated)
    return bleu_score, rouge_scores

# Fonction pour enregistrer les poèmes dans un fichier
def save_poem(poem, filename):
    with open(filename, "w") as file:
        file.write(poem)

# Interface utilisateur avec Streamlit
st.title("Générateur de Poèmes avec GPT-2 et GPT-Neo")

# Entrée utilisateur
keyword = st.text_input("Entrez une phrase ou un mot-clé pour générer un poème :")

if st.button("Générer"):
    if keyword:
        with st.spinner("Génération des poèmes..."):
            # Générer les poèmes
            poem_gpt2 = generate_poem(keyword, gpt2_model, gpt2_tokenizer)
            poem_gpt_neo = generate_poem(keyword, gpt_neo_model, gpt_neo_tokenizer)

            # Enregistrer les poèmes
            save_poem(poem_gpt2, "poem_gpt2.txt")
            save_poem(poem_gpt_neo, "poem_gpt_neo.txt")

            # Afficher les poèmes
            st.subheader("Poème généré par GPT-2")
            st.write(poem_gpt2)

            st.subheader("Poème généré par GPT-Neo")
            st.write(poem_gpt_neo)

            # Évaluer les scores
            bleu_gpt2, rouge_gpt2 = evaluate_similarity(keyword, poem_gpt2)
            bleu_gpt_neo, rouge_gpt_neo = evaluate_similarity(keyword, poem_gpt_neo)

            # Afficher les scores
            st.subheader("Scores BLEU et ROUGE")
            st.write(f"GPT-2: BLEU={bleu_gpt2:.2f}, ROUGE-1={rouge_gpt2['rouge1'].fmeasure:.2f}")
            st.write(f"GPT-Neo: BLEU={bleu_gpt_neo:.2f}, ROUGE-1={rouge_gpt_neo['rouge1'].fmeasure:.2f}")

            # Ajouter un bouton pour télécharger les poèmes
            st.download_button("Télécharger le poème GPT-2", data=poem_gpt2, file_name="poem_gpt2.txt", mime="text/plain")
            st.download_button("Télécharger le poème GPT-Neo", data=poem_gpt_neo, file_name="poem_gpt_neo.txt", mime="text/plain")

    else:
        st.warning("Veuillez entrer une phrase ou un mot-clé.")

st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f4;
        color: #333;
        font-family: 'Helvetica', sans-serif;
    }
    .stButton>button {
        background-color: #6c63ff;
        color: white;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)