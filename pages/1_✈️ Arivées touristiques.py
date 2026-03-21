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
# with st.sidebar:
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

svg_arrivees = """
<svg width="35" height="35" viewBox="0 0 24 24" fill="#FFFFFF">
    <path d="M2.5 19l19-7-19-7v5l13 2-13 2z"/>
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

# ====== DISPLAY ======
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(card_style.format(
        icon=svg_arrivees,
        label="Nb Arrivées",
        value=f"{total_arrivees:,.0f}".replace(",", " ")
    ), unsafe_allow_html=True)


col1, col2 = st.columns(2)
with col1:
        somme_annuelle_arrivees = filtered_data_trimestre.groupby("Année")["Arrivees"].sum().reset_index()
        somme_trimestrielle = filtered_data_trimestre.groupby(["Année", "Trimestre"])["Arrivees"].sum().reset_index()

    # Fusionner par année
        st.subheader("Arrivées touristiques par année et par trimestre")
        df_merge = pd.merge(somme_trimestrielle, somme_annuelle_arrivees[["Année", "Arrivees"]], left_on="Année", right_on="Année", how="inner")
        fig = px.bar(
        df_merge,
        x="Année",
        y="Arrivees_x",
        color="Trimestre",    
        barmode="group",      
        # title="Arrivées touristiques par année et par trimestre",
        )

        fig.update_layout(
        xaxis_title="Année",
        yaxis_title="Arrivées",
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
    
    # Prevision arrivees touristiques 
        # Sélection des colonnes utiles
with col2:
        # Somme annuelle des arrivées
        df_yearly = filtered_data.groupby("Année", as_index=False)["Arrivees"].sum()
        X = df_yearly[["Arrivees"]]
        y = df_yearly['Arrivees'].fillna(df_yearly['Arrivees'].mean())
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modèle Random Forest
        rf_reg = RandomForestRegressor(n_estimators=400, max_depth=5, random_state=42)
        rf_reg.fit(X_train, y_train)

    # Prédictions
        df_yearly['Prévision Arrivées'] = rf_reg.predict(X)
    # Sélection de la dernière année disponible
        latest_Année = df_yearly['Année'].max()
        latest_data = df_yearly[df_yearly['Année'] == latest_Année]

    # Prévions futures
        future_Années = np.array(range(max_Année + 1, max_Année + 5)).reshape(-1, 1)

        def forecast_trend(variable):
            """Prévoit la tendance d'une variable en utilisant une régression linéaire."""
            Années = df_yearly['Année'].values.reshape(-1, 1)
            values = df_yearly[variable].values
            if np.isnan(values).any():
                raise ValueError(f"Des valeurs manquantes existent dans {variable}")
            model_trend = LinearRegression()
            model_trend.fit(Années, values)
            future_values = model_trend.predict(future_Années)
            noise = np.random.uniform(-0.5, 0.5, size=future_values.shape)
            return future_values + noise
        variables = ['Arrivees']
        future_exog = pd.DataFrame({var: forecast_trend(var) for var in variables})
        future_forecast = rf_reg.predict(future_exog)
        forecast_arrivee = pd.DataFrame({
            'Année': list(df_yearly['Année']) + list(future_Années.flatten()),
            'Arrivées': list(df_yearly['Arrivees']) + [np.nan] * len(future_Années), 
            'Prévision Arrivées': list(df_yearly['Prévision Arrivées'])+ list(future_forecast)
        })
        st.subheader("Évolution des arrivées touristiques et sa prévision de 4 ans")
        fig = px.line(
            forecast_arrivee,
            x="Année",
            y="Prévision Arrivées",
            # title="Évolution des arrivées touristiques et sa prévision de 4 ans",
            line_shape='spline',      
            color_discrete_sequence=['#1f77b4'] 
            )
        st.plotly_chart(fig, use_container_width=True)
    # Heatmap pour les arrivées touristiques
    
    # Pivot pour la heatmap
heatmap_data = filtered_data.pivot_table(
        index='Année', 
        columns='Mois', 
        values='Arrivees',
        aggfunc='sum'
    )

    # Pour Plotly, on "dé-pivot" pour avoir long format
heatmap_long = heatmap_data.reset_index().melt(id_vars='Année', var_name='Mois', value_name='Arrivees')

    # Création de la heatmap interactive
st.subheader("Heatmap interactive des arrivées touristiques")
fig = px.imshow(
        heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale='YlGnBu',
        text_auto=True 
    )

fig.update_layout(
        # title="Heatmap interactive des arrivées touristiques",
        xaxis_title="Mois",
        yaxis_title="Année"
    )

st.plotly_chart(fig, use_container_width=True)

# Ajouter le chatbot
st.subheader("Assistant économique et touristique")
run_chatbot(selected_Années, selected_specific_years, selected_specific_trimestre)
