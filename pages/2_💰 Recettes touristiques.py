import streamlit as st
import wbdata
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from chat import create_chatbot
from data_loader import get_data
from main import run_chatbot
df_monthly = get_data()
with st.sidebar:
        # ---- Filtre  : Plage d'années  ----
        min_Année, max_Année = df_monthly['Année'].min(), df_monthly['Année'].max()
        selected_Années = st.sidebar.slider("Sélectionner une plage d'années", min_Année, max_Année, (min_Année, max_Année), key="slider_Années")
        filtered_data = df_monthly[(df_monthly['Année'] >= selected_Années[0]) & (df_monthly['Année'] <= selected_Années[1])]
        filtered_data["Solde"] = filtered_data["recettes actuel"] - filtered_data["dépenses actuel"]
        filtered_data.dropna(inplace=True)
        selected_specific_years = []          # Liste vide par défaut
        selected_specific_trimestre = []
        # ---- Filtre 2 : années spécifiques écrites manuellement ----
        years_input = st.sidebar.text_input(
            "Entrer les années (ex : 2012, 2013)",
            placeholder="2012, 2013",
            key="textinput_Années"
        )
        error_message = ""  

        # ---- Transformation du texte ----
        if years_input.strip() != "":
            try:
                # Séparer par virgule / espace
                selected_specific_years = [
                    int(x) for x in years_input.replace(",", " ").split()
                ]

                # Vérifier si les années existent dans la plage filtrée
                years_available = filtered_data["Année"].unique().tolist()

                invalid_years = [y for y in selected_specific_years if y not in years_available]

                if len(invalid_years) > 0:
                    # Année invalide
                    error_message = f"Année(s) invalide(s) ou hors plage : {invalid_years}"
                    filtered_data_specific = filtered_data.copy()  # garder dataset original
                else:
                    # Filtrer normalement
                    filtered_data_specific = filtered_data[
                        filtered_data["Année"].isin(selected_specific_years)
                    ]

            except:
                error_message = "Format invalide. Exemple : 2012, 2013"
                filtered_data_specific = filtered_data.copy()

        else:
            # Aucun input → garder filtered_data
            filtered_data_specific = filtered_data.copy()

        # ---- Affichage de l'erreur (en bas de l’input) ----
        if error_message:
            st.sidebar.error(error_message)
            
            # ---- Filtre 3 : trimestre spécifiques écrites manuellement ----
        trimestre_input = st.sidebar.text_input(
            "Entrer les trimestres (ex : 1, 2)",
            placeholder="1, 2",
            key="textinput_Trimestre"
        )

        error_message = ""   # Pour afficher l’erreur proprement

        # ---- Transformation du texte ----
        if trimestre_input.strip() != "":
            try:
                # Séparer par virgule / espace
                selected_specific_trimestre = [
                    int(x) for x in trimestre_input.replace(",", " ").split()
                ]

                # Vérifier si les années existent dans la plage filtrée
                trimestre_available = filtered_data_specific["Trimestre"].unique().tolist()

                invalid_trimestre = [y for y in selected_specific_trimestre if y not in trimestre_available]

                if len(invalid_trimestre) > 0:
                    # Trimestre invalide
                    error_message = f"Trimestres(s) invalide(s) ou hors plage : {invalid_trimestre}"
                    filtered_data_trimestre = filtered_data_specific.copy()  # garder dataset original
                else:
                    # Filtrer normalement
                    filtered_data_trimestre = filtered_data_specific[
                        filtered_data_specific["Trimestre"].isin(selected_specific_trimestre)
                    ]

            except:
                error_message = "Format invalide. Exemple : 1, 2"
                filtered_data_trimestre = filtered_data_specific.copy()

        else:
            # Aucun input → garder filtered_data
            filtered_data_trimestre = filtered_data_specific.copy()

        # ---- Affichage de l'erreur (en bas de l’input) ----
        if error_message:
            st.sidebar.error(error_message)
            
        # ====== Calcul des KPI ======
# Dernière année disponible dans filtered_data
last_year = filtered_data["Année"].max()
# Données seulement pour cette dernière année
last_year_data = filtered_data[filtered_data["Année"] == last_year] 
total_recettes = last_year_data["recettes actuel"].sum() if "recettes actuel" in last_year_data else 0

col1, col2, col3, col4 = st.columns(4)
card_style = """
        <div style="
            border: 2px solid #00BCD4;  /* contour bleu clair */
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            background-color: #1E1E1E;  /* fond sombre */
            color: #FFFFFF;              /* texte blanc */
            box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
            font-size: 24px;
            font-weight: bold;
        ">
            {label}<br>{value}
        </div>
    """

with col1:
    st.markdown(card_style.format(label="💰 Recettes", value=f"{total_recettes:,.0f}"), unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
        somme_annuelle_recettes1 = filtered_data_trimestre.groupby("Année")["recettes pour les articles de transport"].sum().reset_index()
        somme_trimestrielle_recettes1 = filtered_data_trimestre.groupby(["Année", "Trimestre"])["recettes pour les articles de transport"].sum().reset_index()   
    # Fusionner par année
        st.subheader("Recettes pour les articles de transport par année et par trimestre")
        df_merge = pd.merge(somme_trimestrielle_recettes1, somme_annuelle_recettes1[["Année", "recettes pour les articles de transport"]], left_on="Année", right_on="Année", how="inner")
        fig = px.bar(
        df_merge,
        x="Année",
        y="recettes pour les articles de transport_x",
        color="Trimestre",    
        barmode="group",      
        # title="Arrivées touristiques par année et par trimestre",
        )

        fig.update_layout(
        xaxis_title="Année",
        yaxis_title="Recettes",
        legend_title="Trimestre",
        template="plotly_white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02           
        )
        )
        st.plotly_chart(fig, use_container_width=True)
with col2:
        somme_annuelle_recettes2 = filtered_data_trimestre.groupby("Année")["recettes pour les articles de voyage"].sum().reset_index()
        somme_trimestrielle_recettes2 = filtered_data_trimestre.groupby(["Année", "Trimestre"])["recettes pour les articles de voyage"].sum().reset_index()   
    # Fusionner par année
        st.subheader("Recettes pour les articles de voyage par année et par trimestre")
        df_merge = pd.merge(somme_trimestrielle_recettes2, somme_annuelle_recettes2[["Année", "recettes pour les articles de voyage"]], left_on="Année", right_on="Année", how="inner")
        fig = px.bar(
        df_merge,
        x="Année",
        y="recettes pour les articles de voyage_x",
        color="Trimestre",    
        barmode="group",      
        # title="Arrivées touristiques par année et par trimestre",
        )

        fig.update_layout(
        xaxis_title="Année",
        yaxis_title="Recettes",
        legend_title="Trimestre",
        template="plotly_white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02           
        )
        )
        st.plotly_chart(fig, use_container_width=True)
        
# Sélecteur de type d'indicateur
df_yearly = filtered_data.groupby("Année", as_index=False)[
        ["recettes pour les articles de transport", "recettes pour les articles de voyage"]
    ].sum()

    # Onglets horizontaux avec soulignement rouge
tab_options = ["recettes pour les articles de transport", "recettes pour les articles de voyage", "Les deux"]
    # Utilisation de st.tabs pour simuler l'effet onglets
tab1, tab2, tab3 = st.tabs(tab_options)

with tab1:
        fig = px.area(
            df_yearly,
            x="Année",
            y="recettes pour les articles de transport",
            title="Recettes pour les articles de transport",
            line_shape='spline'
        )
        fig.update_layout(
            xaxis_title="Année",
            yaxis_title="Valeurs",
            legend_title="Indicateur",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
        fig = px.area(
            df_yearly,
            x="Année",
            y="recettes pour les articles de voyage",
            title="Recettes pour les articles de voyage",
            line_shape='spline'
        )
        fig.update_layout(
            xaxis_title="Année",
            yaxis_title="Valeurs",
            legend_title="Indicateur",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
        fig = px.area(
            df_yearly,
            x="Année",
            y=["recettes pour les articles de transport", "recettes pour les articles de voyage"],
            title="Recettes pour les articles de transport et pour les articles de voyage",
            line_shape='spline'
        )
        fig.update_layout(
            xaxis_title="Année",
            yaxis_title="Valeurs",
            legend_title="Indicateur",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

# Ajouter le chatbot
st.subheader("Assistant économique et touristique")
run_chatbot(selected_Années, selected_specific_years, selected_specific_trimestre)