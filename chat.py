from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

def create_chatbot():

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Template du prompt
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="""
Tu es un assistant spécialisé dans l'analyse des ventes.
Réponds de manière claire et précise.

QUESTION :
{question}

RÉPONSE :
"""
    )

    # On retourne un objet "qa" qui peut exécuter les prompts
    def ask(question):
        final_prompt = prompt_template.format(question=question)
        response = llm.invoke([HumanMessage(content=final_prompt)])
        return response.content

    return ask
