
import pandas as pd
from sklearn.model_selection import train_test_split

# Charger les données préparées (attention au bon encodage)
df = pd.read_csv('donnees_preparees.csv', encoding='utf-8')

# Séparer en train/test 80%/20%
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Sauvegarder test en CSV propre
df_test.to_csv('donnees_test.csv', index=False, sep=',', encoding='utf-8')

print(f"donnees_test.csv créé avec {len(df_test)} lignes.")
