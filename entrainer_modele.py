import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Charger les données préparées
df = pd.read_csv('donnees_preparees.csv')

# Séparer les features (X) et la cible (y)
X = df.drop(columns=['statut_defaut'])
y = df['statut_defaut']

# Séparer en données d'entraînement et de test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer le modèle
modele = RandomForestClassifier(random_state=42)

# Entraîner le modèle
modele.fit(X_train, y_train)

# Prédire sur les données de test
y_pred = modele.predict(X_test)

# Évaluer le modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Matrice de confusion:")
print(confusion_matrix(y_test, y_pred))

# Sauvegarder le modèle
joblib.dump(modele, 'modele.pkl')
print("\nModèle sauvegardé dans modele.pkl")
