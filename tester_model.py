import pandas as pd
import joblib  # pour charger le modèle sauvegardé
from sklearn.metrics import accuracy_score, classification_report

# Chargement des données de test
df_test = pd.read_csv('donnees_test.csv')

# Séparer les caractéristiques (X) et la cible (y)
X_test = df_test.drop(columns=['statut_defaut'])  # colonne cible à adapter si besoin
y_test = df_test['statut_defaut']

# Charger le modèle entraîné
modele = joblib.load('modele.pkl')

# Faire les prédictions sur les données de test
y_pred = modele.predict(X_test)

# Évaluer les performances
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nRapport de classification :\n", classification_report(y_test, y_pred))
