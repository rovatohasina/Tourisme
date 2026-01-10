import streamlit as st
import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from datetime import datetime
from embedding.query_embeddings import query_pinecone
from script import get_data

# Charger les variables d'environnement
load_dotenv()
st.set_page_config(page_title="Prévision du PIB", layout="wide", initial_sidebar_state="expanded")

def create_prompt(question, wbdata_context):
    """
    Crée un prompt structuré pour répondre à une question sur un indicateur économique spécifique
    en n'utilisant que les données disponibles pour cet indicateur et cette année avec les données économiques (wbdata) et les explications (pdf),
    sans utiliser de numérotation dans la réponse.
    """
    wbdata_part = wbdata_context.strip()

    if not wbdata_part :
        return f"""
Tu es un assistant économique et touristique. Voici la question de l'utilisateur :

{question}

Réponds de manière fluide, claire et structurée. Si aucune information n’est disponible, réponds simplement :
"Je ne dispose pas de cette information."
"""
    else:
        prompt = f"""
Tu es un assistant économique et touristique intelligent. Réponds à la question suivante avec clarté et structure.

**Question :** {question}

**Instructions :**
- Si la question est juste une salutation (ex : bonjour, salut, hello), réponds uniquement par une salutation naturelle suivie d'une proposition d'aide,sans aucune donnée économique et touristique.
- Sinon,Réponds uniquement à **l'indicateur économique demandé** dans la question.
- Ne mentionne **aucun autre indicateur**, même si des données sont disponibles.
- Ne parle que de l’année explicitement demandée. Ignore les autres années.
- Si l’information est absente, écris : "Je ne dispose pas de cette information."
{wbdata_part}

- Ne devine rien : si une information n'est pas présente, indique-le simplement ("Donnée non disponible").
- Ne répète pas la question. Concentre-toi sur une réponse directe, fluide et professionnelle.
"""

        return prompt


def ask_gemini(prompt):
    """
    Envoie le prompt à Gemini et retourne la réponse générée avec une bonne structure de phrase.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response

def save_conversation(question, response, file_path="logs/conversation_log.json"):
    if not os.path.exists("logs"):
        os.makedirs("logs")

    response_text = response.content if hasattr(response, "content") else str(response)

    conversation = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "response": response_text
    }

    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    else:
        data = []

    data.append(conversation)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    # --- Initialisation de l'historique ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- Récupération des données filtrées ---
    filtered_data, forecast_arrivee, forecast_solde, selected_Années, selected_specific_years, selected_specific_trimestre = get_data()

    # --- Affichage des messages existants ---
    for msg in st.session_state.chat_history:
        role = msg["role"]
        st.chat_message(role).write(msg["content"])

    # --- Input utilisateur ---
    if question := st.chat_input("Posez une question"):
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        # --- Vérification que question n'est pas vide avant Pinecone ---
        if question.strip() != "":
            with st.spinner("Analyse en cours..."):
                wbdata_context = []
                try:
                    wbdata_context = query_pinecone(
                        question,
                        selected_Années or selected_specific_years or selected_specific_trimestre
                    )
                except Exception as e:
                    st.error(f"Erreur Pinecone : {e}")

                prompt = create_prompt(question, wbdata_context)

                response_text = ""
                response = None
                try:
                    response = ask_gemini(prompt)
                    response_text = response.content
                except Exception as e:
                    response_text = f"Erreur Gemini : {e}"

                # --- Ajouter la réponse à l'historique ---
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                st.chat_message("assistant").write(response_text)

                # --- Sauvegarde ---
                save_conversation(question, response)

if __name__ == "__main__":
    main()
