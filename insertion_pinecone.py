# from pinecone import Pinecone
# from rag.pinecone import init_pinecone
# from embedding.doc_embeddings import embed_data_final_and_check, get_data_final
# import pandas as pd
# def insert_vectors():
#     # Charger WBData
#     data_final_docs = get_data_final()
#     df_final = pd.DataFrame(data_final_docs)
    
#     # Calculer embeddings
#     vectors = embed_data_final_and_check(df_final)
    
#     pc, index = init_pinecone()
#     inserted_ids = []
    
#     # Insérer WBData avec année
#     for i in range(len(df_final)):
#         text, embedding, source = vectors[i]
#         vector_id = f"{source}-{i}"
#         annee = df_final.loc[i, "Année"]
#         doc_type = df_final.loc[i, "type"]
#         index.upsert([{
#             "id": vector_id,
#             "values": embedding,
#             "metadata": {"source": source, "text": text, "Année": annee, "type": doc_type}
#         }])
#         inserted_ids.append(vector_id)
#     print("Vecteurs insérés avec succès dans Pinecone.")
#     for vid in inserted_ids:
#         print(vid)

# if __name__ == "__main__":
#     insert_vectors()

from pinecone import Pinecone
from rag.pinecone import init_pinecone
from embedding.doc_embeddings import embed_data_final_and_check, get_data_final
import pandas as pd

def insert_vectors():
    # Charger WBData
    data_final_docs = get_data_final()
    df_final = pd.DataFrame(data_final_docs)
    
    # Calculer embeddings
    vectors = embed_data_final_and_check(df_final)
    
    pc, index = init_pinecone()
    inserted_ids = []
    
    # Insérer WBData avec année
    for i in range(len(df_final)):
        text, embedding, source = vectors[i]
        vector_id = f"{source}-{i}"
        
        # Convertir metadata en types acceptés par Pinecone
        annee = float(df_final.loc[i, "Année"])  # int64 -> float
        doc_type = str(df_final.loc[i, "type"])  # convertir en str
        
        index.upsert([{
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "source": str(source),
                "text": str(text),
                "Année": annee,
                "type": doc_type
            }
        }])
        inserted_ids.append(vector_id)
    
    print("Vecteurs insérés avec succès dans Pinecone.")
    for vid in inserted_ids:
        print(vid)

if __name__ == "__main__":
    insert_vectors()

