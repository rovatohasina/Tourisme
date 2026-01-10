import os
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Initialiser Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))

# Définir l'index Pinecone
index_name = "nouvel"
index = pc.Index(index_name)

# Initialiser le modèle d'embedding
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def vectorize_question(question: str):
    return embed_model.embed_query(question)

def query_pinecone(question: str,
                   selected_Années=None,
                   selected_specific_years=None,
                   selected_specific_trimestre=None,
                   top_k=None):
    # Vectorisation question
    question_embedding = vectorize_question(question)

    filtres = []
    total_annees = 0
    # Données global
    if selected_Années:
        filtres.append({
            "$and": [
                {"Année": {"$gte": selected_Années[0], "$lte": selected_Années[1]}},
                {"type": "global"}
            ]
        })
        total_annees += selected_Années[1] - selected_Années[0] + 1

    # Données selectionnés en année
    if selected_specific_years:
        filtres.append({
            "$and": [
                {"Année": {"$gte": selected_specific_years[0], "$lte": selected_specific_years[1]}},
                {"type": "Prévision arrivée"}  
            ]
        })
        total_annees += selected_specific_years[1] - selected_specific_years[0] + 1
    # Données selectionnés en trimestre
    if selected_specific_trimestre:
        filtres.append({
            "$and": [
                {"Année": {"$gte": selected_specific_trimestre[0], "$lte": selected_specific_trimestre[1]}},
                {"type": "Prévision arrivée"}  
            ]
        })
        total_annees += selected_specific_trimestre[1] - selected_specific_trimestre[0] + 1
        total_annees += 1
    top_k = total_annees if total_annees > 0 else 20
    pinecone_filter = {
    "$and": [
        {"source": "wbdata"},
        {"$or": filtres} 
    ]
}
    # Requête vers les données WBData (valeur)
    wbdata_results = index.query(
        vector=question_embedding,
        top_k=top_k,
        include_metadata=True,
        include_values=False,
        filter=pinecone_filter
    )
            # Résultats WBData 
    wbdata_contexts = []
    if wbdata_results and wbdata_results.get("matches"):
        for match in wbdata_results.get("matches", []):
            texte = match.get("metadata", {}).get("text", "").strip()
            if texte:
                wbdata_contexts.append(f"WBData : {texte}")
    else:
        print("Recherche WBData NON OK - Aucun résultat trouvé.")  
            # Fusion
    wbdata_context = ("\n".join(wbdata_contexts))

    return wbdata_context


