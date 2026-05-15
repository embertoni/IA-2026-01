# código feito com auxílio de IA (Gemini)
# dataset importado seguindo instruções de: https://archive.ics.uci.edu/dataset/109/wine

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

wine = fetch_ucirepo(id=109)

X = wine.data.features
y = wine.data.targets.values.ravel() 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
total_testes = len(y_test)

print(f"Total de amostras reservadas para teste: {total_testes}\n")
print("-" * 40)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)
acertos_knn = accuracy_score(y_test, pred_knn, normalize=False) 
acuracia_knn = accuracy_score(y_test, pred_knn)

print(f"Resultado - KNN:")
print(f" -> Número de acertos: {acertos_knn} de {total_testes}")
print(f" -> Acurácia: {acuracia_knn:.2%}\n")


nb = GaussianNB()
nb.fit(X_train, y_train)
pred_nb = nb.predict(X_test)
acertos_nb = accuracy_score(y_test, pred_nb, normalize=False)
acuracia_nb = accuracy_score(y_test, pred_nb)

print(f"Resultado - Naive Bayes:")
print(f" -> Número de acertos: {acertos_nb} de {total_testes}")
print(f" -> Acurácia: {acuracia_nb:.2%}\n")

dt = DecisionTreeClassifier(random_state=42) 
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)
acertos_dt = accuracy_score(y_test, pred_dt, normalize=False)
acuracia_dt = accuracy_score(y_test, pred_dt)

print(f"Resultado - Decision Tree:")
print(f" -> Número de acertos: {acertos_dt} de {total_testes}")
print(f" -> Acurácia: {acuracia_dt:.2%}")
print("-" * 40)