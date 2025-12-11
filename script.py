import streamlit as st
import wbdata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from chat import create_chatbot

def get_data():
    indicateurs = {
        "ST.INT.RCPT.CD":"recettes actuel",
        "ST.INT.RCPT.XP.ZS":"recettes % des exportations",
        "ST.INT.TRNR.CD":"recettes pour les articles de transport",
        "ST.INT.TVLR.CD":"recettes pour les articles de voyage",
        "ST.INT.TRNX.CD":"dépenses pour le transport",
        "ST.INT.TVLX.CD":"dépenses pour les articles de voyage",
        "ST.INT.XPND.CD":"dépenses actuel",
        "ST.INT.XPND.MP.ZS":"dépenses % des importations"
    }
    # recuperation des données
    df = wbdata.get_dataframe(indicateurs , country = "MDG")
    df.reset_index(inplace = True)
    df.rename(columns = {"date":"Année"}, inplace = True)
    df = df.dropna(subset=["recettes actuel","recettes % des exportations",
                           "recettes pour les articles de transport","recettes pour les articles de voyage",
                           "dépenses pour le transport","dépenses pour les articles de voyage",
                           "dépenses actuel","dépenses % des importations"], how='all')
    df["Année"] = df["Année"].apply(lambda x: int(x.replace("YR","")) if isinstance(x, str) else x)
    # afficher les données

# 1️⃣ Charger le fichier Excel
    data = pd.read_excel("data.xlsx", engine="openpyxl", sheet_name="Data")

# 2️⃣ Identifier les colonnes qui sont des mois (ex: "janv.2005")
    mois_fr = ["janv", "févr", "mars", "avr", "mai", "juin", 
           "juil", "août", "sept", "oct", "nov", "déc"]

# Colonnes contenant un nom de mois, peu importe le format
    colonnes_mois = [c for c in data.columns if any(m in c.lower() for m in mois_fr)]

# 3️⃣ Transformer en format long
    df_long = data.melt(
    id_vars=[c for c in data.columns if c not in colonnes_mois],
    var_name="Mois",
    value_name="Arrivees"
    )

# Nettoyer la colonne Mois (supprimer les points et espaces)
    df_long["Mois_clean"] = df_long["Mois"].str.replace(r"\.", "", regex=True).str.strip()

# Nettoyer la colonne Arrivees
    df_long["Arrivees"] = df_long["Arrivees"].astype(str).str.replace(" ", "").str.replace(",", ".")
    df_long["Arrivees"] = pd.to_numeric(df_long["Arrivees"], errors="coerce")

# Remplacer mois français par anglais
    mois_fr_en = {
    "janv": "Jan", "févr": "Feb", "mars": "Mar", "avr": "Apr", "mai": "May", "juin": "Jun",
    "juil": "Jul", "août": "Aug", "sept": "Sep", "oct": "Oct", "nov": "Nov", "déc": "Dec"
    }

    for fr, en in mois_fr_en.items():
        df_long["Mois_clean"] = df_long["Mois_clean"].str.replace(fr, en, case=False)

# Convertir en datetime
    df_long["Date"] = pd.to_datetime(df_long["Mois_clean"], format="%b %Y", errors="coerce")

# Supprimer les lignes où la date n'a pas pu être convertie
    df_long = df_long.dropna(subset=["Date"])

# Extraire Année et Trimestre
    df_long["Année"] = df_long["Date"].dt.year
    df_long["Trimestre"] = df_long["Date"].dt.quarter
    df_long["Mois"] = df_long["Date"].dt.strftime("%b")
    
    # --- Somme annuelle ---
    somme_annuelle_arrivvees = df_long.groupby("Année")["Arrivees"].sum().reset_index()
    
    # --- 3️⃣ Calcul des poids mensuels selon les arrivées ---
    df_long["Total_Année"] = df_long.groupby("Année")["Arrivees"].transform("sum")
    df_long["Poids"] = df_long["Arrivees"] / df_long["Total_Année"]
    # --- 4️⃣ Appliquer les poids aux indicateurs annuels ---
    indicateurs_annuels = ["recettes actuel", "recettes % des exportations",
                           "recettes pour les articles de transport", "recettes pour les articles de voyage",
                           "dépenses pour le transport", "dépenses pour les articles de voyage",
                           "dépenses actuel", "dépenses % des importations"]

    for col in indicateurs_annuels:
        df_long[col] = df_long["Année"].map(df.set_index("Année")[col]) * df_long["Poids"]

    # --- 5️⃣ Résultats mensuels prêts à l’usage ---
    df_monthly = df_long[["Année","Trimestre","Mois"] + indicateurs_annuels + ["Arrivees"]].copy()

    with st.sidebar:
        # ---- Filtre  : Plage d'années  ----
        min_Année, max_Année = df_monthly['Année'].min(), df_monthly['Année'].max()
        selected_Années = st.sidebar.slider("Sélectionner une plage d'années", min_Année, max_Année, (min_Année, max_Année), key="slider_Années")
        filtered_data = df_monthly[(df_monthly['Année'] >= selected_Années[0]) & (df_monthly['Année'] <= selected_Années[1])]
        filtered_data["Solde"] = filtered_data["recettes actuel"] - filtered_data["dépenses actuel"]
        filtered_data.dropna(inplace=True)
        # ---- Filtre 2 : années spécifiques écrites manuellement ----
        years_input = st.sidebar.text_input(
            "Entrer les années (ex : 2012, 2013)",
            placeholder="2012, 2013"
        )

        error_message = ""   # Pour afficher l’erreur proprement

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
            placeholder="1, 2"
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
        # ====== Calcul des KPI ======
    total_arrivees = filtered_data_specific["Arrivees"].sum() if "Arrivees" in filtered_data_specific else 0
    total_recettes = filtered_data_specific["recettes actuel"].sum() if "recettes actuel" in filtered_data_specific else 0
    total_depenses = filtered_data_specific["dépenses actuel"].sum() if "dépenses actuel" in filtered_data_specific else 0
    total_soldes = filtered_data_specific["Solde"].sum() if "Solde" in filtered_data_specific else 0

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
        st.markdown(card_style.format(label="✈️ Nb Arrivées", value=f"{total_arrivees:,.0f}"), unsafe_allow_html=True)

    with col2:
        st.markdown(card_style.format(label="💰 Recettes", value=f"{total_recettes:,.0f}"), unsafe_allow_html=True)

    with col3:
        st.markdown(card_style.format(label="📉 Dépenses", value=f"{total_depenses:,.0f}"), unsafe_allow_html=True)
        
    with col4:
        st.markdown(card_style.format(label="⚖️ Soldes", value=f"{total_soldes:,.0f}"), unsafe_allow_html=True)
        
    col1, col2 = st.columns(2)
    with col1:
            # --- Création du nuage de points ---
    # Renommer pour éviter conflits
        somme_annuelle_recettes = filtered_data.groupby("Année")["recettes actuel"].sum().reset_index()
        df_wb_renamed = somme_annuelle_recettes.rename(columns={"recettes actuel": "Recettes"}).copy()
        df_merge = pd.merge(somme_annuelle_arrivvees, df_wb_renamed[["Année", "Recettes"]], left_on="Année", right_on="Année", how="inner")

        # ============================
        # --- ANALYSE STATISTIQUE ---
        # ============================

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

        fig = px.scatter(
            data_frame=df_merge,
            x="Arrivees",
            y="Recettes",
            color="Année",
            size="Recettes",
            hover_name="Année",
            title="Relation entre les recettes et le nombre d'arrivées par année",
            labels={
                "Arrivees": "Nombre d'arrivées",
                "Recettes": "Recettes"
            }
        )

        # --- Ajouter la ligne de régression ---
        fig.add_traces(
            px.line(
                df_merge, 
                x="Arrivees", 
                y="Prediction"
            ).data
        )

        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.write("")

        # =================================
        # --- AFFICHAGE DES RÉSULTATS ---
        # =================================
        st.subheader("Analyse statistique")
        st.markdown(f"""
        **Coefficient de corrélation (Pearson)** : **{correlation:.3f}**
        **Modèle de régression linéaire :**
        - Formule : `Recettes = {model.coef_[0]:.2f} × Arrivées + {model.intercept_:.2f}`
        **Analyse automatique :**  
Chaque arrivée supplémentaire entraîne une variation de **{model.coef_[0]:.2f}** dans les recettes.  
La valeur de base des recettes, lorsque le nombre d’arrivées est nul,est de **{model.intercept_:.2f}**.  
Celà montre comment les arrivées influencent l’évolution globale des recettes.
        """)
        
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
            df["ecart"] = df["Recettes"] - df["recettes_predites"]

            # Détection des anomalies (seuil = 1 écart-type)
            seuil = df["ecart"].std()

            df["type_anomalie"] = "Normal"

            df.loc[df["ecart"] < -seuil, "type_anomalie"] = "🟥 Beaucoup d'arrivées mais peu de recettes"
            df.loc[df["ecart"] > seuil, "type_anomalie"] = "🟩 Peu d'arrivées mais beaucoup de recettes"

            # Affichage dans le dashboard
            st.subheader("🔍 Détection des anomalies Arrivées ↔ Recettes")

            anomalies = df[df["type_anomalie"] != "Normal"]

            if anomalies.empty:
                st.success("Aucune anomalie détectée dans la période sélectionnée.")
            else:
                st.warning("Anomalies détectées :")
                st.dataframe(anomalies[["Année","Arrivees", "Recettes", "ecart", "type_anomalie"]])

        else:
            st.error("Colonnes 'Arrivees' et 'Recettes' manquantes.")
# Somme annuelle des arrivées
    df_yearly = filtered_data.groupby("Année", as_index=False)["Arrivees"].sum()

# Création du graphique stylé
    fig = px.line(
        df_yearly,
        x="Année",
        y="Arrivees",
        title="Évolution des arrivées touristiques",
        line_shape='spline',          # ligne lisse
        color_discrete_sequence=['#1f77b4']  # couleur personnalisée
            )

# Mise en forme du layout
    fig.update_layout(
        xaxis_title="Année",
        yaxis_title="Nombre d'arrivées",
        title_font_size=20,
        xaxis=dict(tickmode='linear'),
        yaxis=dict(tickformat=','),
        template="plotly_white",
        font=dict(family="Arial", size=12)
            )
    
    somme_annuelle_arrivees = filtered_data_trimestre.groupby("Année")["Arrivees"].sum().reset_index()
# Fusionner par année
    df_merge = pd.merge(somme_trimestrielle, somme_annuelle_arrivees[["Année", "Arrivees"]], left_on="Année", right_on="Année", how="inner")
    fig = px.bar(
        df_merge,
        x="Année",
        y="Arrivees_x",
        color="Trimestre",    
        barmode="group",      
        title="Arrivées touristiques par année et par trimestre",
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
    fig = px.imshow(
        heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale='YlGnBu',
        text_auto=True 
    )

    fig.update_layout(
        title="Heatmap interactive des arrivées touristiques",
        xaxis_title="Mois",
        yaxis_title="Année"
    )

    st.plotly_chart(fig, use_container_width=True)
# Affichage
    st.plotly_chart(fig, use_container_width=True)
    if "recettes actuel" in filtered_data_trimestre.columns and "dépenses actuel" in filtered_data_trimestre.columns:
        somme_annuelle_rec = filtered_data_trimestre.groupby("Année")["recettes actuel"].sum().reset_index()
        somme_annuelle_dep = filtered_data_trimestre.groupby("Année")["dépenses actuel"].sum().reset_index()
# Fusionner par année
        df_merge = pd.merge(somme_annuelle_rec[["Année", "recettes actuel"]],somme_annuelle_dep[["Année", "dépenses actuel"]], left_on="Année", right_on="Année", how="inner")
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
        text="Valeurs",               # affiche la valeur sur la barre
        labels={"Valeurs": "Valeurs", "Année": "Année", "Type": "Indicateur"},
        title="Recettes et dépenses actuelles du tourisme",
        color_discrete_map={"recettes actuel": "#1f77b4", "dépenses actuel": "#063970"}
        )

    # Améliorer l'affichage
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='inside', hovertemplate='%{x}<br>%{y:,.0f} USD')
        fig.update_layout(barmode='stack', template='plotly_white')

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Les colonnes nécessaires ne sont pas disponibles dans les données.")


# Sélecteur de type d'indicateur
    df_yearly = filtered_data.groupby("Année", as_index=False)[["recettes pour les articles de transport","recettes pour les articles de voyage"]].sum()
    options = ["recettes pour les articles de transport","recettes pour les articles de voyage", "Les deux"]
    choix = st.radio("Choisissez les indicateurs à afficher :", options)
# Filtrage selon le choix
    if choix == "recettes pour les articles de transport":
        fig = px.line(df_yearly, x="Année", y="recettes pour les articles de transport",line_shape='spline', title="Recettes pour les articles de transport")
    elif choix == "recettes pour les articles de voyage":
        fig = px.line(df_yearly, x="Année", y="recettes pour les articles de voyage",line_shape='spline', title="Recettes pour les articles de voyage")
    else:
        fig = px.line(df_yearly, x="Année", y=["recettes pour les articles de transport","recettes pour les articles de voyage"],
            line_shape='spline',title="Recettes pour les articles de transport et pour les articles de voyage")  
        # Mise en forme du layout
    fig.update_layout(
    xaxis_title="Année",
    yaxis_title="Valeurs",
    legend_title="Indicateur",
    title_font_size=20,
    xaxis=dict(tickmode='linear'),
    yaxis=dict(tickformat=','),
    template="plotly_white",
    font=dict(family="Arial", size=12)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    df_yearly = filtered_data.groupby("Année", as_index=False)[["dépenses pour le transport","dépenses pour les articles de voyage"]].sum() 
    options = ["dépenses pour le transport","dépenses pour les articles de voyage", "Les deux"]
    choix = st.radio("Choisissez les indicateurs à afficher :", options)
# Filtrage selon le choix
    if choix == "dépenses pour le transport":
        fig = px.line(df_yearly, x="Année", y="dépenses pour le transport",line_shape='spline', title="Dépenses pour le transport")
    elif choix == "dépenses pour les articles de voyage":
        fig = px.line(df_yearly, x="Année", y="dépenses pour les articles de voyage",line_shape='spline', title="Dépenses pour les articles de voyage")
    else:
        fig = px.line(df_yearly, x="Année", y=["dépenses pour le transport","dépenses pour les articles de voyage"],line_shape='spline',
                title="Dépenses pour le transport et pour les articles de voyage")  
        # Mise en forme du layout
    fig.update_layout(
    xaxis_title="Année",
    yaxis_title="Valeurs",            
    legend_title="Indicateur",
    title_font_size=20,
    xaxis=dict(tickmode='linear'),
    yaxis=dict(tickformat=','),
    template="plotly_white",
    font=dict(family="Arial", size=12)
    )
    st.plotly_chart(fig, use_container_width=True)
        
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
        title="Évolution des dépenses touristiques par type"
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
        title="Évolution des recettes touristiques par type"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.title("Solde budgétaire")
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
    forecast_df = pd.DataFrame({
        'Année': list(df_solde['Année']) + list(future_Années.flatten()),
        'Solde': list(df_solde['Solde']) + [np.nan] * len(future_Années), 
        'Prévision Solde': list(df_solde['Prévision Solde'])+ list(future_forecast)
    })
    fig = px.line(
        forecast_df,
    x="Année",
    y="Prévision Solde",
    title="Évolution du solde de Madagascar et sa prévision de 10 ans",
    line_shape='spline',      
    color_discrete_sequence=['#1f77b4'] 
        )
    st.plotly_chart(fig, use_container_width=True)

# creation du chatbot
    qa = create_chatbot()
    data_prompt1 = "\n".join([f"{row['Année']}: {row['Trimestre']}:{row['Mois']}: {row['recettes actuel']}: {row['recettes % des exportations']}: {row['recettes pour les articles de transport']}: {row['recettes pour les articles de voyage']}: {row['dépenses pour le transport']}: {row['dépenses pour les articles de voyage']}: {row['dépenses actuel']}: {row['dépenses % des importations']}" for _, row in filtered_data.iterrows()])
    data_prompt2 = "\n".join([f"{row['Solde']}: {row['Prévision Solde']}" for _, row in df_solde.iterrows()])

# Prompt complet 
    prompt = f"""
    Voici les données de notre tableau de bord :

    données touristiques et economiques :
    {data_prompt1},
    {data_prompt2}
    """

# Chatbot interactif 
    st.title("Chatbot Analyse des données")

    if "message" not in st.session_state:
        st.session_state.message = []

    for msg in st.session_state.message:
        st.chat_message(msg["role"]).write(msg["content"])

    if question := st.chat_input("Posez une question"):
        st.session_state.message.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        full_prompt = f"{prompt}\n\nQuestion de l'utilisateur : {question}"
        response = qa(full_prompt)

        st.session_state.message.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
data = get_data()