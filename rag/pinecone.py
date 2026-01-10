from dotenv import load_dotenv

import os
from pinecone import Pinecone

load_dotenv()

def init_pinecone():
    # Charger la clé API et l'environnement depuis les variables d'environnement
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("❌ Clé API non trouvée.")
    # Initialiser le client Pinecone avec la clé API
    pc = Pinecone(api_key=api_key)
    index_name = "nouvel"
    # Accéder à l'index créé ou existant
    index = pc.Index(index_name)
    return pc, index




