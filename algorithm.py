import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, accuracy_score, confusion_matrix, 
                             classification_report, precision_score, recall_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def evaluate_classification(model, X_train, X_test, y_train, y_test, model_name, dataset_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n--- {model_name} en {dataset_name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (weighted): {prec:.4f}")
    print(f"Recall (weighted): {rec:.4f}")
    print("Matriz de Confusión:")
    print(cm)
    return acc

def run_clustering(X_scaled, y, dataset_name):
    print(f"\n--- K-Means Clustering en {dataset_name} ---")
    k = len(np.unique(y))
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    score = silhouette_score(X_scaled, clusters)
    print(f"Número de clusters (K): {k}")
    print(f"Silhouette Score: {score:.4f}")
    
    df = pd.DataFrame({'Actual': y, 'Cluster': clusters})
    ct = pd.crosstab(df['Actual'], df['Cluster'])
    print("Matriz de Contingencia (Real vs Cluster):")
    print(ct)
    df.to_csv(f'{dataset_name}_resultados_clustering.csv', index = False)
        
    # Visualización 2D (primeras dos componentes)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=clusters, palette='viridis', s=80)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X')
    plt.title(f'K-Means (K={k}) - {dataset_name}')
    plt.savefig(f'kmeans_{dataset_name.lower()}_2d.png')
    plt.close()

def run_experiment():
    datasets = {
        "Iris": load_iris(),
        "Wine": load_wine()
    }
    
    for name, data in datasets.items():
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Clasificación: Naive Bayes
        evaluate_classification(GaussianNB(), X_train, X_test, y_train, y_test, "Naive Bayes", name)
        
        # Clasificación: 1R (Decision Tree de profundidad 1)
        evaluate_classification(DecisionTreeClassifier(max_depth=1), X_train, X_test, y_train, y_test, "1R (One Rule)", name)
        
        # Clustering: K-Means
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        run_clustering(X_scaled, y, name)

if __name__ == "__main__":
    run_experiment()
