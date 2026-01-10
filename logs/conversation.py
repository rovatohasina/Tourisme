from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
from rag.retriever import create_retriever
import pinecone
def get_response_from_chatbot(question):
    try:
        # Logique pour récupérer les documents et générer la réponse
        print("Traitement de la question:", question)
        retriever = create_retriever()
        response = retriever.get_response(question)
        print("Réponse générée:", response)
        return response
    except Exception as e:
        print(f"Erreur lors de la génération de la réponse : {e}")
        return "Une erreur s'est produite. Veuillez réessayer."
