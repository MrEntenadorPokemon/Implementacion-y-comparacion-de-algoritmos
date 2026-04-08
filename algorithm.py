import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def run_kmeans_iris():
    print("--- Ejecutando K-Means en el Dataset Iris ---")
    
    # 1. Cargar el dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    
    # 2. Preprocesamiento (Escalado de datos)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Determinar el número de clusters (K)
    # Aunque sabemos que son 3 especies, usaremos el método del codo (Elbow Method) para ilustrar el criterio
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    # Visualización del Método del Codo
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Método del Codo (Elbow Method)')
    plt.xlabel('Número de Clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.savefig('elbow_method.png')
    print("Gráfica del método del codo guardada como 'elbow_method.png'.")

    # 4. Aplicar K-Means con K=3 (basado en el dataset real)
    k = 3
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # 5. Visualización 2D (usando las dos primeras características: Sepal length y Sepal width)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=clusters, palette='viridis', s=100, style=clusters)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroides')
    plt.title(f'K-Means Clustering (K={k}) - Iris Dataset')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend()
    plt.grid(True)
    plt.savefig('kmeans_iris_2d.png')
    print("Visualización 2D guardada como 'kmeans_iris_2d.png'.")
    
    # 6. Interpretación de métricas
    score = silhouette_score(X_scaled, clusters)
    print(f"\nNúmero de clusters: {k}")
    print(f"Silhouette Score: {score:.4f}")
    
    # Comparación básica con etiquetas reales
    df = pd.DataFrame({'Actual': y, 'Cluster': clusters})
    ct = pd.crosstab(df['Actual'], df['Cluster'])
    print("\nMatriz de Contingencia (Real vs Cluster):")
    print(ct)
    print("\nInterpretación:")
    print("Cluster 0 parece coincidir fuertemente con una de las especies (Setosa es usualmente muy separable).")
    print("Clusters 1 y 2 muestran cierto solapamiento entre Versicolor y Virginica, lo cual es esperado debido a su similitud morfológica.")

if __name__ == "__main__":
    run_kmeans_iris()
