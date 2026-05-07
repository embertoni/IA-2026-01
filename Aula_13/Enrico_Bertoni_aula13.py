# código feito com auxílio de IA (Gemini)
# dataset importados seguindo o formato explicado pelo site do link de download

from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

wine = fetch_ucirepo(id=109) 

X = wine.data.features 
y = wine.data.targets

df = X.copy()
df['class'] = y.values 

print("Tamanho do dataset (Features + Class):", df.shape)

plt.figure(figsize=(14, 6))
sns.boxplot(data=X)
plt.xticks(rotation=45, ha='right')
plt.title('a) Boxplot das Features Originais (Wine Dataset)')
plt.tight_layout()
plt.show()

print("\n--- Estatísticas Descritivas (Dataset Original) ---")
print(X.describe().T[['mean', 'std']])

scaler = StandardScaler()
features_scaled = scaler.fit_transform(X)

df_scaled = pd.DataFrame(features_scaled, columns=X.columns)

plt.figure(figsize=(14, 6))
sns.boxplot(data=df_scaled)
plt.xticks(rotation=45, ha='right')
plt.title('b) Boxplot das Features Normalizadas (StandardScaler)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x='class', data=df, palette='viridis')
plt.title('c) Distribuição de Elementos em Cada Classe')
plt.xlabel('Classe')
plt.ylabel('Quantidade de Instâncias')
plt.show()

plt.figure(figsize=(12, 8))
correlation_matrix = X.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('d) Matriz de Correlação entre as Features')
plt.tight_layout()
plt.show()

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)

df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
df_pca['class'] = y.values

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='class', palette='viridis', data=df_pca, s=80, alpha=0.8)
plt.title('e) Gráfico 2D aplicando PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Classe')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()