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
    df["Solde"] = df["recettes actuel"] - df["dépenses actuel"]

    # 1️⃣ Charger le fichier Excel
    data = pd.read_excel("data.xlsx", engine="openpyxl", sheet_name="Data")

    # 2️⃣ Charger le CSV scrappé
    data1 = pd.read_csv("statistiques_tourisme_mta_long.csv")
    data1 = data1.rename(columns={"Valeur": "Arrivees"})

    # 3️⃣ Convertir les mois français en anglais
    mois_fr_en = {
        "Janvier": "Jan", "Février": "Feb", "Mars": "Mar", "Avril": "Apr",
        "Mai": "May", "Juin": "Jun", "Juillet": "Jul", "Août": "Aug",
        "Septembre": "Sep", "Octobre": "Oct", "Novembre": "Nov", "Décembre": "Dec"
    }
    data1["Mois_clean"] = data1["Mois"].map(mois_fr_en)

    # 4️⃣ Créer la colonne Date
    data1["Date"] = pd.to_datetime(data1["Mois_clean"] + " " + data1["Année"].astype(str), format="%b %Y", errors="coerce")
    data1 = data1.dropna(subset=["Date"])

    # 5️⃣ Extraire Année, Trimestre, Mois
    data1["Année"] = data1["Date"].dt.year
    data1["Trimestre"] = data1["Date"].dt.quarter
    data1["Mois"] = data1["Date"].dt.strftime("%b")

    # 6️⃣ Calculer Total_Année et Poids
    data1["Total_Année"] = data1.groupby("Année")["Arrivees"].transform("sum")
    data1["Poids"] = data1["Arrivees"] / data1["Total_Année"]

    # Colonnes finales
    cols = ["Année", "Trimestre", "Mois", "Arrivees", "Total_Année", "Poids", "Date"]

    # 7️⃣ Transformer le fichier Excel en format long
    mois_fr = ["janv", "févr", "mars", "avr", "mai", "juin", 
            "juil", "août", "sept", "oct", "nov", "déc"]
    colonnes_mois = [c for c in data.columns if any(m in c.lower() for m in mois_fr)]

    df_long = data.melt(
        id_vars=[c for c in data.columns if c not in colonnes_mois],
        var_name="Mois",
        value_name="Arrivees"
    )
    df_long["Mois_clean"] = df_long["Mois"].str.replace(r"\.", "", regex=True).str.strip()
    df_long["Arrivees"] = df_long["Arrivees"].astype(str).str.replace(" ", "").str.replace(",", ".")
    df_long["Arrivees"] = pd.to_numeric(df_long["Arrivees"], errors="coerce")

    mois_fr_en2 = {
        "janv": "Jan", "févr": "Feb", "mars": "Mar", "avr": "Apr", "mai": "May", "juin": "Jun",
        "juil": "Jul", "août": "Aug", "sept": "Sep", "oct": "Oct", "nov": "Nov", "déc": "Dec"
    }
    for fr, en in mois_fr_en2.items():
        df_long["Mois_clean"] = df_long["Mois_clean"].str.replace(fr, en, case=False)

    df_long["Date"] = pd.to_datetime(df_long["Mois_clean"], format="%b %Y", errors="coerce")
    df_long = df_long.dropna(subset=["Date"])
    df_long["Année"] = df_long["Date"].dt.year
    df_long["Trimestre"] = df_long["Date"].dt.quarter
    df_long["Mois"] = df_long["Date"].dt.strftime("%b")

    df_long["Total_Année"] = df_long.groupby("Année")["Arrivees"].transform("sum")
    df_long["Poids"] = df_long["Arrivees"] / df_long["Total_Année"]

    # Fusionner les deux sources
    df_final = pd.concat([df_long[cols], data1[cols]], ignore_index=True)
    df_final = df_final.sort_values(["Année", "Date"]).reset_index(drop=True)

    # 8️⃣ Nowcasting ANNUEL
    df_annuel = df_final.groupby("Année", as_index=False)["Arrivees"].sum().sort_values("Année")
    df_annuel["lag1"] = df_annuel["Arrivees"].shift(1)
    df_annuel["lag2"] = df_annuel["Arrivees"].shift(2)
    df_annuel["lag3"] = df_annuel["Arrivees"].shift(3)

    train = df_annuel[(df_annuel["Année"] >= 2008) & (df_annuel["Année"] <= 2019)].dropna()
    X_train = train[["lag1", "lag2", "lag3"]]
    y_train = train["Arrivees"]

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    for annee in [2020, 2021, 2022]:
        last = df_annuel.iloc[-1]
        X_pred = pd.DataFrame([{"lag1": last["Arrivees"], "lag2": last["lag1"], "lag3": last["lag2"]}])
        pred = model.predict(X_pred)[0]
        df_annuel = pd.concat([df_annuel, pd.DataFrame([{"Année": annee, "Arrivees": pred}])], ignore_index=True)
        df_annuel["lag1"] = df_annuel["Arrivees"].shift(1)
        df_annuel["lag2"] = df_annuel["Arrivees"].shift(2)
        df_annuel["lag3"] = df_annuel["Arrivees"].shift(3)

    # 9️⃣ Reconstruire les mois et trimestres pour 2020-2022
    mois = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    rows = []
    for annee in [2020, 2021, 2022]:
        for i, m in enumerate(mois, start=1):
            rows.append({"Année": annee, "Mois": m, "Trimestre": (i-1)//3 + 1, "Date": pd.to_datetime(f"{m} {annee}")})

    df_mois_manquant = pd.DataFrame(rows)

    # 10️⃣ Calculer les poids moyens historiques
    poids_moyens = df_final[df_final["Année"] <= 2019].groupby("Mois")["Poids"].mean().to_dict()
    df_mois_manquant["Poids"] = df_mois_manquant["Mois"].map(poids_moyens)

    # 11️⃣ Fusion avec les totaux annuels et calcul Arrivees mensuelles
    df_mois_manquant = df_mois_manquant.merge(df_annuel[["Année","Arrivees"]], on="Année", how="left")
    df_mois_manquant["Arrivees"] = df_mois_manquant["Arrivees"] * df_mois_manquant["Poids"]
    df_mois_manquant["Total_Année"] = df_mois_manquant.groupby("Année")["Arrivees"].transform("sum")

    # 12️⃣ Fusion finale
    df_final_corrige = pd.concat([df_final, df_mois_manquant], ignore_index=True)
    df_final_corrige = df_final_corrige.sort_values(["Année","Date"]).reset_index(drop=True)

    # --- 4️⃣ Appliquer les poids aux indicateurs annuels ---
    indicateurs_annuels = ["recettes actuel", "recettes % des exportations",
                           "recettes pour les articles de transport", "recettes pour les articles de voyage",
                           "dépenses pour le transport", "dépenses pour les articles de voyage",
                           "dépenses actuel", "dépenses % des importations"]

    for col in indicateurs_annuels:
        df_final_corrige[col] = df_final_corrige["Année"].map(df.set_index("Année")[col]) * df_final_corrige["Poids"]

    # --- 5️⃣ Résultats mensuels prêts à l’usage ---
    df_monthly = df_final_corrige[["Année","Trimestre","Mois"] + indicateurs_annuels + ["Arrivees"]].copy()
    # Sélectionner les années avec données WB existantes
    df_wb = df_monthly[df_monthly["Année"] <= 2020].copy()

    # Variables explicatives (ex: Arrivées et éventuellement poids mensuels)
    X = df_wb[["Arrivees"]]
    indicateurs = ["recettes actuel", "recettes % des exportations",
               "recettes pour les articles de transport", "recettes pour les articles de voyage",
               "dépenses pour le transport", "dépenses pour les articles de voyage",
               "dépenses actuel", "dépenses % des importations"]

    for ind in indicateurs:
        y = df_wb[ind]

        # Créer le modèle
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Prédire pour les années 2021-2025
        df_future = df_monthly[df_monthly["Année"] > 2020]
        X_future = df_future[["Arrivees"]]
        df_monthly.loc[df_monthly["Année"] > 2020, ind] = model.predict(X_future)

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
        somme_trimestrielle = filtered_data_trimestre.groupby(["Année", "Trimestre"])["Arrivees"].sum().reset_index()   
    st.dataframe(filtered_data)
 
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
#     with col1:
#             # --- Création du nuage de points ---
#     # Renommer pour éviter conflits
    somme_annuelle_recettes = filtered_data.groupby("Année")["recettes actuel"].sum().reset_index()
    df_wb_renamed = somme_annuelle_recettes.rename(columns={"recettes actuel": "Recettes"}).copy()
# # Fusionner par année
    somme_annuelle_arrivvees = df_final.groupby("Année")["Arrivees"].sum().reset_index()

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
        st.subheader("Relation entre les recettes et le nombre d'arrivées")
        fig = px.scatter(
            data_frame=df_merge,
            x="Arrivees",
            y="Recettes",
            color="Année",
            size="Recettes",
            hover_name="Année",
            # title="Relation entre les recettes et le nombre d'arrivées par année",
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

        # =================================
        # --- AFFICHAGE DES RÉSULTATS ---
        # =================================
        st.subheader("Analyse statistique et détection des anomalies")
        st.markdown(f"""
        Coefficient de corrélation: **{correlation:.3f}**
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
        somme_annuelle_arrivees = filtered_data_trimestre.groupby("Année")["Arrivees"].sum().reset_index()
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
        st.subheader("Recettes et dépenses actuelles du tourisme")

    # Graphique empilé avec hover
        fig = px.bar(
        df_dep_rec,
        x="Année",
        y="Valeurs",
        color="Type",
        labels={"Valeurs": "Valeurs", "Année": "Année", "Type": "Indicateur"},
        # title="Recettes et dépenses actuelles du tourisme",
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
    st.subheader("Évolution du solde de Madagascar et sa prévision de 4 ans")
    fig = px.line(
        forecast_solde,
    x="Année",
    y="Valeur",
    # title="Évolution du solde de Madagascar et sa prévision de 5 ans",
    line_shape='spline',      
    color_discrete_sequence=['#1f77b4'] 
        )
    st.plotly_chart(fig, use_container_width=True)
    return filtered_data, forecast_arrivee, forecast_solde, selected_Années, selected_specific_years, selected_specific_trimestre
data = get_data()
