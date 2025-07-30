import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder




# 📁 Dossier contenant les CSV (relatif à ton projet VS Code)
DOSSIER_DONNEES = "donnee"

# 🔁 Lire et fusionner tous les fichiers CSV
def charger_donnees(dossier):
    fichiers_csv = [f for f in os.listdir(dossier) if f.endswith(".csv")]
    df_liste = []
    for f in fichiers_csv:
        chemin = os.path.join(dossier, f)
        df = pd.read_csv(chemin)
        df_liste.append(df)
    df_combine = pd.concat(df_liste, ignore_index=True)
    return df_combine

# 🧼 Nettoyage et préparation des données
def preparer_donnees(df):
    # Supprimer les valeurs manquantes s'il y en a
    df = df.dropna()

    # Supprimer les colonnes inutiles
    if "id_client" in df.columns:
        df = df.drop(columns=["id_client"])

    # Encoder les variables catégorielles avec LabelEncoder
    encodeurs = {}
    colonnes_categor = df.select_dtypes(include="object").columns
    for col in colonnes_categor:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encodeurs[col] = le  # On stocke l'encodeur pour réutilisation éventuelle

    return df, encodeurs

# 🚀 Point d'entrée
if __name__ == "__main__":
    print("Chargement des données...")
    df_brut = charger_donnees(DOSSIER_DONNEES)
    print(f"{len(df_brut)} lignes chargées.")

    print("Préparation des données...")
    df_prete, encodeurs = preparer_donnees(df_brut)
    print("Données prêtes pour l'entraînement !")

    # 📦 Sauvegarder les données préparées
    df_prete.to_csv("donnees_preparees.csv", index=False)
    print("Fichier sauvegardé : donnees_preparees.csv")
