import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from script import get_data
import pandas as pd
# Charger les variables d'environnement
load_dotenv()

def get_data_final():
    filtered_data,forecast_arrivee,forecast_solde= get_data()

    dfs = [
        ("global",filtered_data),
        ("Prévision arrivée", forecast_arrivee),
        ("Prévision solde", forecast_solde),   
    ]
    # Ajoute une colonne "type" dans chaque DataFrame
    for nom_df, df in dfs:
        df['type'] = nom_df.lower()

    # Concatène tous les DataFrames en un seul
    df_wbdata = pd.concat([df for _, df in dfs], ignore_index=True)
    wbdata_docs = []

    for nom_df, df in dfs:
        for _, row in df.iterrows():
            try:
                année = int(row['Année']) if 'Année' in row else "Inconnue"
            except:
                année = "Inconnue"

            # parts = [f"[{nom_df}] En {année} :"]
            doc_type = row['type']  # récupère le type ici

            parts = [f"[{nom_df}] En {année} :"]

            for col in df.columns:
                if col != 'Année' and pd.notnull(row[col]):
                    try:
                        value = round(row[col], 2)
                        parts.append(f"{col} était de {value}")
                    except:
                        parts.append(f"{col} : {row[col]}")

            text = ". ".join(parts) + "."
            wbdata_docs.append(
                {"Année": année,
                 "type":doc_type,
                 "text": text}
                )
    return wbdata_docs


# Embedding des documents
def embed_data_final_and_check(df):
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # WBData
    wbdata_texts = df['text'].tolist()
    wbdata_embeddings = embed_model.embed_documents(wbdata_texts)
    wb_vectors = [(text, vec, "wbdata") for text, vec in zip(wbdata_texts, wbdata_embeddings)]
    # 5. Affichage pour vérification

    for i, emb in enumerate(wbdata_embeddings[:3]):
        print(f"WBData Embedding {i} : {len(emb)} dimensions")

    # Combiner les deux
    return wb_vectors

