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
data = get_data()
with st.sidebar:
        # ---- Filtre  : Plage d'années  ----
        min_Année, max_Année = data['Année'].min(), data['Année'].max()
        selected_Années = st.sidebar.slider("Sélectionner une plage d'années", min_Année, max_Année, (min_Année, max_Année), key="slider_Années")
        filtered_data = data[(data['Année'] >= selected_Années[0]) & (data['Année'] <= selected_Années[1])]
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
        somme_trimestrielle = filtered_data_trimestre.groupby(["Année", "Trimestre"])["Arrivees"].sum().reset_index()

st.markdown("""
<style>

/* HEADER FIXE */
.custom-header {
    position: fixed;
    top: 0;
    left: 250px;
    right: 0;
    
    width: 100%;
    height: 70px;

    display: flex;
    align-items: center;

    font-size: 24px;
    font-weight: 600;

    padding: 0 20px;

    background: white;
    border-bottom: 2px solid #ddd;

    box-shadow: 0px 4px 6px rgba(0,0,0,0.1);

    z-index: 999999; /* 🔥 très élevé pour passer au-dessus des graphes */
}

/* CONTENU PRINCIPAL */
.main .block-container {
    padding-top: 90px; /* espace pour éviter que le header cache */
}

/* OPTION : éviter que plotly passe au-dessus */
.js-plotly-plot {
    z-index: 1 !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="custom-header">🌿 Tableau de bord écotourisme à Madagascar</div>',
    unsafe_allow_html=True
)
# ====== SVG ICONS ======
svg_arrivees = """
<svg width="35" height="35" viewBox="0 0 24 24" fill="#FFFFFF">
    <path d="M2.5 19l19-7-19-7v5l13 2-13 2z"/>
</svg>
"""

svg_recettes = """
<svg width="35" height="35" viewBox="0 0 24 24" fill="#FFFFFF">
    <path d="M12 21V3M19 14l-7-7-7 7"/>
</svg>
"""

svg_depenses = """
<svg width="35" height="35" viewBox="0 0 24 24" fill="#FFFFFF">
    <path d="M12 1v22M5 6h9a4 4 0 010 8H7a4 4 0 000 8h12"/>
</svg>
"""

svg_solde = """
<svg width="35" height="35" viewBox="0 0 24 24" fill="#FFFFFF">
    <path d="M3 12h18M12 3v18"/>
</svg>
"""

# ====== STYLE CARD ======
card_style = """
<div style="
    border: 2px solid #00BCD4;
    border-radius: 12px;
    padding: 20px;
    background-color: #1E1E1E;
    color: #FFFFFF;
    box-shadow: 2px 2px 12px rgba(0,0,0,0.6);
    display: flex;
    align-items: center;
    gap: 15px;
">
    <div>{icon}</div>
    <div>
        <div style="font-size:14px; color:#B0BEC5;">{label}</div>
        <div style="font-size:22px; font-weight:bold;">{value}</div>
    </div>
</div>
"""

# ====== CALCUL KPI ======
last_year = filtered_data["Année"].max()
last_year_data = filtered_data[filtered_data["Année"] == last_year]

total_arrivees = last_year_data["Arrivees"].sum() if "Arrivees" in last_year_data else 0
total_recettes = last_year_data["recettes actuel"].sum() if "recettes actuel" in last_year_data else 0
total_depenses = last_year_data["dépenses actuel"].sum() if "dépenses actuel" in last_year_data else 0
total_soldes = last_year_data["Solde"].sum() if "Solde" in last_year_data else 0

# ====== DISPLAY ======
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(card_style.format(
        icon=svg_arrivees,
        label="Nb Arrivées",
        value=f"{total_arrivees:,.0f}".replace(",", " ")
    ), unsafe_allow_html=True)

with col2:
    st.markdown(card_style.format(
        icon=svg_recettes,
        label="Recettes",
        value=f"{total_recettes:,.0f}".replace(",", " ")
    ), unsafe_allow_html=True)

with col3:
    st.markdown(card_style.format(
        icon=svg_depenses,
        label="Dépenses",
        value=f"{total_depenses:,.0f}".replace(",", " ")
    ), unsafe_allow_html=True)

with col4:
    st.markdown(card_style.format(
        icon=svg_solde,
        label="Solde",
        value=f"{total_soldes:,.0f}".replace(",", " ")
    ), unsafe_allow_html=True)
      
    # Sélecteur de type d'indicateur
df_yearly = filtered_data.groupby("Année", as_index=False)[
        ["Arrivees","recettes actuel","dépenses actuel","Solde"]
].sum()

    # Onglets horizontaux avec soulignement rouge
tab_options = ["Arrivees","recettes actuel","dépenses actuel","Solde", "Tous"]

    # Utilisation de st.tabs pour simuler l'effet onglets
tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_options)

with tab1:
    fig = px.area(
        df_yearly,
        x="Année",
        y="Arrivees",
        title="Arrivées touristiques",
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
            y="recettes actuel",
            title="Recettes actuel touristiques",
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
            y="dépenses actuel",
            title="Dépenses actuel touristiques",
            line_shape='spline'
        )
    fig.update_layout(
            xaxis_title="Année",
            yaxis_title="Valeurs",
            legend_title="Indicateur",
            template="plotly_white"
        )
    st.plotly_chart(fig, use_container_width=True)
        
with tab4:
    fig = px.area(
            df_yearly,
            x="Année",
            y="Solde",
            title="Soldes touristiques",
            line_shape='spline'
        )
    fig.update_layout(
            xaxis_title="Année",
            yaxis_title="Valeurs",
            legend_title="Indicateur",
            template="plotly_white"
        )
    st.plotly_chart(fig, use_container_width=True)
with tab5:
    fig = px.area(
            df_yearly,
            x="Année",
            y=["Arrivees","recettes actuel","dépenses actuel","Solde"],
            title="Evolution du tourisme",
            line_shape='spline'
        )
    fig.update_layout(
            xaxis_title="Année",
            yaxis_title="Valeurs",
            legend_title="Indicateur",
            template="plotly_white"
        )
    st.plotly_chart(fig, use_container_width=True)
    
        # Mise en forme des données pour le graphique
somme_annuelle_dep_transport = filtered_data_trimestre.groupby("Année")["dépenses pour le transport"].sum().reset_index()
somme_annuelle_dep_article_voy = filtered_data_trimestre.groupby("Année")["dépenses pour les articles de voyage"].sum().reset_index()
# Fusionner par année
df_merge = pd.merge(somme_annuelle_dep_transport[["Année", "dépenses pour le transport"]],somme_annuelle_dep_article_voy[["Année", "dépenses pour les articles de voyage"]], left_on="Année", right_on="Année", how="inner")
    
df_melted = df_merge.melt(
id_vars="Année",
value_vars=["dépenses pour le transport", "dépenses pour les articles de voyage"],
var_name="Type de dépense",
value_name="Montant (USD)"
)

fig = px.bar(
    df_melted,
    x="Année",
    y="Montant (USD)",
    color="Type de dépense",
    barmode="group",
    title="Comparaison des dépenses touristiques par type"
)
st.plotly_chart(fig, use_container_width=True, key="depenses_bar_chart")
    
somme_annuelle_rec_transport = filtered_data_trimestre.groupby("Année")["recettes pour les articles de transport"].sum().reset_index()
somme_annuelle_rec_article_voy = filtered_data_trimestre.groupby("Année")["recettes pour les articles de voyage"].sum().reset_index()
# Fusionner par année
df_merge = pd.merge(somme_annuelle_rec_transport[["Année", "recettes pour les articles de transport"]],somme_annuelle_rec_article_voy[["Année", "recettes pour les articles de voyage"]], left_on="Année", right_on="Année", how="inner")
df_melted = df_merge.melt(
id_vars="Année",
value_vars=["recettes pour les articles de transport", "recettes pour les articles de voyage"],
var_name="Type de recette",
value_name="Montant (USD)"
)

fig = px.bar(
    df_melted,
    x="Année",
    y="Montant (USD)",
    color="Type de recette",
    barmode="group",
    title="Comparaison des recettes touristiques par type"
)
st.plotly_chart(fig, use_container_width=True)

# Ajouter le chatbot
st.subheader("Assistant économique et touristique")
run_chatbot(selected_Années, selected_specific_years, selected_specific_trimestre)