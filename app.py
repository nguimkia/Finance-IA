from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

modele = joblib.load('modele.pkl')

# Colonnes exactes utilisées pour l'entraînement
colonnes = ['age', 'sexe', 'revenu_mensuel', 'profession', 'nb_enfants',
            'montant_pret_demande', 'duree_pret_mois', 'historique_remboursement', 'zone']

# Encodage des variables catégorielles utilisées
encodage = {
    'sexe': {'Homme': 0, 'Femme': 1},
    'profession': {'Salarié': 0, 'Commerçant': 1, 'Fonctionnaire': 2, 'Autre': 3},
    'historique_remboursement': {'Bon': 0, 'Moyen': 1, 'Mauvais': 2},
    'zone': {'Urbaine': 0, 'Rurale': 1}
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            input_data = {
                'age': int(request.form['age']),
                'sexe': encodage['sexe'][request.form['sexe']],
                'revenu_mensuel': float(request.form['revenu_mensuel']),
                'profession': encodage['profession'][request.form['profession']],
                'nb_enfants': int(request.form['nb_enfants']),
                'montant_pret_demande': float(request.form['montant_pret_demande']),
                'duree_pret_mois': int(request.form['duree_pret_mois']),
                'historique_remboursement': encodage['historique_remboursement'][request.form['historique_remboursement']],
                'zone': encodage['zone'][request.form['zone']],
            }

            df = pd.DataFrame([input_data], columns=colonnes)

            pred = modele.predict(df)[0]
            prediction = "Accepté" if pred == 0 else "Refusé"
        except Exception as e:
            prediction = f"Erreur lors de la prédiction : {str(e)}"

    return render_template('formulaire.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
