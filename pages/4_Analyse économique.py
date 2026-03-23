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

col1, col2 = st.columns(2)
#     with col1:
#             # --- Création du nuage de points ---
#     # Renommer pour éviter conflits
somme_annuelle_recettes = filtered_data.groupby("Année")["recettes actuel"].sum().reset_index()
df_wb_renamed = somme_annuelle_recettes.rename(columns={"recettes actuel": "Recettes"}).copy()
# # Fusionner par année
somme_annuelle_arrivvees = data.groupby("Année")["Arrivees"].sum().reset_index()

df_merge = pd.merge(somme_annuelle_arrivvees, df_wb_renamed[["Année", "Recettes"]], left_on="Année", right_on="Année", how="inner")
with col1:
        # --- Corrélation Pearson ---
        correlation = df_merge["Arrivees"].corr(df_merge["Recettes"])

        # --- Régression linéaire ---

        X = df_merge[["Arrivees"]]      # variable explicative
        y = df_merge["Recettes"]        # variable cible

        model = LinearRegression()
        model.fit(X, y)

        # prédiction pour tracer la ligne
        df_merge["Prediction"] = model.predict(X)

        # ============================
        # --- NUAGE DE POINTS PLOTLY ---
        # ============================
        st.write("")
        fig = px.scatter(
            data_frame=df_merge,
            x="Arrivees",
            y="Recettes",
            color="Année",
            size="Recettes",
            hover_name="Année",
            title="Relation entre les recettes et le nombre d'arrivées",
            labels={
                "Arrivees": "Nombre d'arrivées",
                "Recettes": "Recettes"
            }
        )

        # # --- Ajouter la ligne de régression ---
        # fig.add_traces(
        #     px.line(
        #         df_merge, 
        #         x="Arrivees", 
        #         y="Prediction"
        #     ).data
        # )

        st.plotly_chart(fig, use_container_width=True)

with col2:
        st.write("")
        st.write("")
        # =================================
        # --- AFFICHAGE DES RÉSULTATS ---
        # ================================
        
        # Detection d'anomalie
        if "Arrivees" in df_merge.columns and "Recettes" in df_merge.columns:

            df = df_merge.copy()

            # Régression linéaire simple
            x = df["Arrivees"]
            y = df["Recettes"]

            m, b = np.polyfit(x, y, 1)

            # Valeur prédite
            df["recettes_predites"] = m * df["Arrivees"] + b

            # Ecart entre réel et prédit
            df["écart"] = df["Recettes"] - df["recettes_predites"]

            # Détection des anomalies (seuil = 1 écart-type)
            seuil = df["écart"].std()

            df["type anomalie"] = "Normal"

            df.loc[df["écart"] < -seuil, "type anomalie"] = "🟥 Beaucoup d'arrivées mais peu de recettes"
            df.loc[df["écart"] > seuil, "type anomalie"] = "🟩 Peu d'arrivées mais beaucoup de recettes"

            # Affichage dans le dashboard
            anomalies = df[df["type anomalie"] != "Normal"]

            if anomalies.empty:
                st.success("Aucune anomalie détectée dans la période sélectionnée.")
            else:
                st.warning("Anomalies détectées :")
                st.dataframe(anomalies[["Année","Arrivees", "Recettes", "écart", "type anomalie"]])

        else:
            st.error("Colonnes 'Arrivees' et 'Recettes' manquantes.")
col1, col2 = st.columns(2)
with col1:
        # --- Calcul des valeurs annuelles ---
        df_yearly = filtered_data.groupby("Année", as_index=False)["recettes % des exportations"].sum()

        # --- Calcul de la variation en % par rapport à l'année précédente ---
        df_yearly['pct_change'] = df_yearly['recettes % des exportations'].pct_change() * 100

        # --- Définir la couleur selon la variation ---
        df_yearly['color'] = df_yearly['pct_change'].apply(lambda x: 'green' if x > 0 else ('red' if x < 0 else 'blue'))

        # --- Créer le graphique ---
        fig = px.line(
            df_yearly,
            x="Année",
            y="recettes % des exportations",
            title="Parts du tourisme dans les exportations",
            markers=True,  # Affiche les points
            color_discrete_sequence=['#1f77b4']  # couleur personnalisée

        )

        # --- Ajouter les annotations (pourcentage sur chaque point) ---
        for i, row in df_yearly.iterrows():
            if pd.notna(row['pct_change']):  # Ignorer la première année (pas de variation)
                fig.add_annotation(
                    x=row['Année'],
                    y=row['recettes % des exportations'],
                    text=f"{row['pct_change']:.1f}%",
                    showarrow=False,
                    arrowhead=1,
                    arrowcolor=row['color'],
                    font=dict(color=row['color']),
                    yshift=15
                )

        st.plotly_chart(fig, use_container_width=True)

with col2:
                # Somme annuelle des arrivées
        df_yearly = filtered_data.groupby("Année", as_index=False)["dépenses % des importations"].sum()
        # --- Calcul de la variation en % par rapport à l'année précédente ---
        df_yearly['pct_change'] = df_yearly['dépenses % des importations'].pct_change() * 100

        # --- Définir la couleur selon la variation ---
        df_yearly['color'] = df_yearly['pct_change'].apply(lambda x: 'green' if x > 0 else ('red' if x < 0 else 'blue'))

        # Création du graphique stylé
        fig = px.line(
            df_yearly,
            x="Année",
            y="dépenses % des importations",
            title="Dépenses touristique par rapport aux importations",
            markers=True,
                   # affiche des points sur la ligne
            # line_shape='spline',          # ligne lisse
            color_discrete_sequence=['#1f77b4']  # couleur personnalisée
            )
        # --- Ajouter les annotations (pourcentage sur chaque point) ---
        for i, row in df_yearly.iterrows():
            if pd.notna(row['pct_change']):  # Ignorer la première année (pas de variation)
                fig.add_annotation(
                    x=row['Année'],
                    y=row['dépenses % des importations'],
                    text=f"{row['pct_change']:.1f}%",
                    showarrow=False,
                    arrowhead=1,
                    arrowcolor=row['color'],
                    font=dict(color=row['color']),
                    yshift=15
                )
        st.plotly_chart(fig, use_container_width=True)
#Recettes vs dépenses actuels
if "recettes actuel" in filtered_data_trimestre.columns and "dépenses actuel" in filtered_data_trimestre.columns:
        somme_annuelle_rec = filtered_data_trimestre.groupby("Année")["recettes actuel"].sum().reset_index()
        somme_annuelle_dep = filtered_data_trimestre.groupby("Année")["dépenses actuel"].sum().reset_index()
# Fusionner par année
        df_merge = pd.merge(somme_annuelle_rec[["Année", "recettes actuel"]],somme_annuelle_dep[["Année", "dépenses actuel"]], left_on="Année", right_on="Année", how="inner")
       # Calcul du solde annuel
        df_merge["Solde"] = df_merge["recettes actuel"] - df_merge["dépenses actuel"]
    # Créer un DataFrame long pour le diagramme empilé
        df_dep_rec = df_merge.melt(
        id_vars=["Année"],
        value_vars=["recettes actuel", "dépenses actuel"],
        var_name="Type",
        value_name="Valeurs"
        )

    # Graphique empilé avec hover
        fig = px.bar(
        df_dep_rec,
        x="Année",
        y="Valeurs",
        color="Type",
        labels={"Valeurs": "Valeurs", "Année": "Année", "Type": "Indicateur"},
        title="Recettes et dépenses actuelles du tourisme",
        color_discrete_map={"recettes actuel": "#1f77b4", "dépenses actuel": "#063970"}
        )

    # Améliorer l'affichage
        # fig.update_traces(texttemplate='%{text:,.0f}', textposition='inside', hovertemplate='%{x}<br>%{y:,.0f} USD')
        fig.update_layout(barmode='stack', template='plotly_white')
    # Ajouter la ligne du solde
        fig.add_trace(
        go.Scatter(
            x=df_merge["Année"],
            y=df_merge["Solde"],
            mode='lines+markers+text',
            name='Solde',
            line=dict(color='green', width=3),
            marker=dict(size=8)
            )
    )
        st.plotly_chart(fig, use_container_width=True)
else:
        st.warning("Les colonnes nécessaires ne sont pas disponibles dans les données.")
        
# --- Calcul du solde
df_solde = filtered_data.groupby("Année", as_index=False)[["recettes actuel","dépenses actuel"]].sum()
df_solde["Solde"] = df_solde["recettes actuel"] - df_solde["dépenses actuel"]
    # Sélection des colonnes utiles
X = df_solde[['recettes actuel','dépenses actuel']]
y = df_solde['Solde']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle Random Forest
rf_reg = RandomForestRegressor(n_estimators=400, max_depth=5, random_state=42)
rf_reg.fit(X_train, y_train)

# Prédictions
df_solde['Prévision Solde'] = rf_reg.predict(X)
# Sélection de la dernière année disponible
latest_Année = df_solde['Année'].max()
latest_data = df_solde[df_solde['Année'] == latest_Année]

# Prévions futures
future_Années = np.array(range(max_Année + 1, max_Année + 5)).reshape(-1, 1)

def forecast_trend(variable):
        """Prévoit la tendance d'une variable en utilisant une régression linéaire."""
        Années = df_solde['Année'].values.reshape(-1, 1)
        values = df_solde[variable].values
        if np.isnan(values).any():
            raise ValueError(f"Des valeurs manquantes existent dans {variable}")
        model_trend = LinearRegression()
        model_trend.fit(Années, values)
        future_values = model_trend.predict(future_Années)
        noise = np.random.uniform(-0.5, 0.5, size=future_values.shape)
        return future_values + noise
variables = ['recettes actuel','dépenses actuel']
future_exog = pd.DataFrame({var: forecast_trend(var) for var in variables})
future_forecast = rf_reg.predict(future_exog)
forecast_solde = pd.DataFrame({
        'Année': list(df_solde['Année']) + list(future_Années.flatten()),
        'Solde': list(df_solde['Solde']) + [np.nan] * len(future_Années), 
        'Valeur': list(df_solde['Prévision Solde'])+ list(future_forecast)
    })
fig = px.line(
        forecast_solde,
    x="Année",
    y="Valeur",
    title="Évolution du solde et sa prévision de 4 ans",
    line_shape='spline',      
    color_discrete_sequence=['#1f77b4'] 
        )
st.plotly_chart(fig, use_container_width=True)

# Ajouter le chatbot
st.subheader("Assistant économique et touristique")
run_chatbot(selected_Années, selected_specific_years, selected_specific_trimestre)