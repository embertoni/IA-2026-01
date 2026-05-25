# código feito com auxílio de IA (Gemini)
# dataset importado seguindo instruções de: https://archive.ics.uci.edu/dataset/109/wine

from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

wine = fetch_ucirepo(id=109) 
  
X = wine.data.features 
y = wine.data.targets.values.ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

total_test = len(y_test)
print(f"Resultados do Teste (Total de amostras avaliadas: {total_test})")
print("-" * 60)

svm_model = SVC(kernel='rbf', C=1.0, random_state=42) # SVM com kernel RBF (Radial Basis Function) e regularização C=1.0
svm_model.fit(X_train_scaled, y_train)
svm_preds = svm_model.predict(X_test_scaled)

svm_acc = accuracy_score(y_test, svm_preds)
svm_hits = accuracy_score(y_test, svm_preds, normalize=False)
print(f" SVM        | Acurácia: {svm_acc * 100:.2f}% | Número de acertos: {svm_hits}/{total_test}")

perceptron_model = Perceptron(max_iter=1000, tol=1e-3, random_state=42) # Perceptron com máximo de 1000 iterações e tolerância de 0.001 para convergência
perceptron_model.fit(X_train_scaled, y_train)
perceptron_preds = perceptron_model.predict(X_test_scaled)

perceptron_acc = accuracy_score(y_test, perceptron_preds)
perceptron_hits = accuracy_score(y_test, perceptron_preds, normalize=False)
print(f" Perceptron | Acurácia: {perceptron_acc * 100:.2f}% | Número de acertos: {perceptron_hits}/{total_test}")

mlp_model = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42) # MLP com camadas ocultas de 50 e 25 neurônios, e máximo de 1000 iterações
mlp_model.fit(X_train_scaled, y_train)
mlp_preds = mlp_model.predict(X_test_scaled)

mlp_acc = accuracy_score(y_test, mlp_preds)
mlp_hits = accuracy_score(y_test, mlp_preds, normalize=False)
print(f" MLP        | Acurácia: {mlp_acc * 100:.2f}% | Número de acertos: {mlp_hits}/{total_test}")
print("-" * 60)