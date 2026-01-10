# from selenium import webdriver
# from selenium.webdriver.common.action_chains import ActionChains
# from selenium.webdriver.common.by import By
# from time import sleep

# # Lance le navigateur
# driver = webdriver.Chrome()
# driver.get("https://www.mta.gov.mg/statistiques/")

# sleep(5)  # attendre que la page charge

# actions = ActionChains(driver)

# # Trouve les points de données du graphique (CSS à adapter)
# points = driver.find_elements(By.CSS_SELECTOR, "svg .highcharts-point")

# data = []
# for p in points:
#     # Survole le point
#     actions.move_to_element(p).perform()
#     sleep(0.3)

#     # Cherche le tooltip qui contient la valeur
#     tooltip = driver.find_element(By.CSS_SELECTOR, ".highcharts-tooltip text")
#     data.append(tooltip.text)

# print(data)
# driver.quit()



import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time

# --- Configurer Selenium ---
options = Options()
options.add_argument("--headless")  # mode headless
driver = webdriver.Chrome(options=options)

driver.get("https://www.mta.gov.mg/statistiques/")

# --- Attendre dynamiquement que le graphique se charge ---
max_wait = 15  # secondes
js = ""
for i in range(max_wait):
    scripts = driver.find_elements("tag name", "script")
    js = ""
    for s in scripts:
        content = s.get_attribute("innerHTML")
        if content and ("Chart" in content or "myChart" in content):
            js += content
    # On vérifie si au moins 4 datasets (les 4 années) sont présents
    if js.count("data:") >= 4:
        break
    time.sleep(1)

driver.quit()

# --- Récupérer les labels (mois) ---
labels_match = re.search(r"labels\s*:\s*\[([^\]]+)\]", js)
if not labels_match:
    print("❌ Labels non trouvés")
    exit()
labels = [x.strip().replace('"','').replace("'","") for x in labels_match.group(1).split(",")]

# --- Récupérer tous les datasets ---
# Cette regex capture label et data même si l'ordre est différent
dataset_matches = re.findall(r"{[^}]*label\s*:\s*['\"]?([^'\"]*)['\"]?[^}]*data\s*:\s*\[([^\]]+)\]", js)

# Si aucun label trouvé, fallback avec juste data
if not dataset_matches:
    dataset_matches = re.findall(r"data\s*:\s*\[([^\]]+)\]", js)
    dataset_matches = [(f"Dataset_{i+1}", data) for i, data in enumerate(dataset_matches)]

records = []
for year_label, data_str in dataset_matches:
    values = []
    for x in data_str.split(","):
        x = x.strip()
        if x in ["", "null", "undefined"]:
            values.append(None)
        else:
            try:
                values.append(float(x))
            except:
                values.append(None)
    # Aligner avec labels
    if len(values) < len(labels):
        values += [None]*(len(labels)-len(values))
    elif len(values) > len(labels):
        values = values[:len(labels)]
    # Créer les lignes
    for month, val in zip(labels, values):
        records.append({"Année": year_label, "Mois": month, "Valeur": val})

df = pd.DataFrame(records)
df.to_csv("statistiques_tourisme_mta_long.csv", index=False, encoding="utf-8")

print(df)
print("\n✅ Fichier créé : statistiques_tourisme_mta_long.csv")
