# %%
# Importation des bibliothèques
import pandas as pd
import numpy as np
# Charger le dataset
url = "https://raw.githubusercontent.com/SoukainaZAHTI/Exercice_KNN/master/Datasets/cardio_train.csv" 
df = pd.read_csv(url)
# Aperçu des données
print(df.head())
# Dimensions et informations
print(df.shape)
print(df.info())
# Vérification des valeurs manquantes
print(df.isnull().sum())
# Statistiques descriptives
print(df.describe())

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Encodage des variables catégoriques
df['Sex'] = df['Sex'].apply(lambda x: 1 if x == "Male" else 0)
# Variables explicatives (X) et cible (y)
X = df.drop('Risk', axis=1)
y = df['Risk']
# Normalisation des données continues
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
# Distribution du risque
sns.countplot(df['Risk'])
plt.title("Distribution du risque cardiovasculaire")
plt.show()
# Analyse par âge
sns.boxplot(x='Risk', y='Age', data=df)
plt.title("Âge et risque")
plt.show()
# Matrice de corrélation
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Matrice de corrélation")
plt.show()

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
# Modèle de régression logistique
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
# Prédictions
y_pred_log = log_model.predict(X_test)
# Évaluation
print(classification_report(y_test, y_pred_log))
print("AUC-ROC :", roc_auc_score(y_test, y_pred_log))

# %%
from sklearn.ensemble import RandomForestClassifier
# Modèle de forêt aléatoire
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
# Prédictions
y_pred_rf = rf_model.predict(X_test)
# Évaluation
print(classification_report(y_test, y_pred_rf))
print("AUC-ROC :", roc_auc_score(y_test, y_pred_rf))

# %%
from sklearn.svm import SVC
# Modèle SVM
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)
# Prédictions
y_pred_svm = svm_model.predict(X_test)
# Évaluation
print(classification_report(y_test, y_pred_svm))
print("AUC-ROC :", roc_auc_score(y_test, y_pred_svm))

# %%
from sklearn.model_selection import GridSearchCV
# Paramètres à optimiser
param_grid = {
'n_estimators': [50, 100, 200],
'max_depth': [5, 10, 20]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid,
cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)
print("Meilleurs paramètres :", grid_search.best_params_)
print("Meilleur score AUC-ROC :", grid_search.best_score_)


