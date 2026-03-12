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

    return df_monthly