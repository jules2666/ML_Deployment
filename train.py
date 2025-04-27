
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Charger les données
data = load_iris()
X, y = data.data, data.target

# Entraîner le modèle
model = RandomForestClassifier()
model.fit(X, y)

# Sauvegarder le modèle
with open('app/model.pkl', 'wb') as f:
    pickle.dump(model, f)
