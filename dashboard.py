import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="Dashboard - Minería de Datos", layout="wide")

# Estilo personalizado para las métricas
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; }
    </style>
    """, unsafe_allow_html=True)

# --- NAVEGACIÓN ---
st.sidebar.title("Act.4 Final")
st.sidebar.info("Andre Siqueiros Perez\n\nGael Alejando Nevarez \n\n Luis Morales Medina \n\nProfesor: Dr. Meza Ibarra")
pagina = st.sidebar.radio("Selecciona una sección:", 
                          ["Comparativa General", "Breast Cancer (Salud)", "Wine (Química)", "Iris (Agrupamiento)"])

# --- DATOS GLOBALES (Métricas del Reporte) ---
metricas_data = {
    'Dataset': ['Iris', 'Breast Cancer', 'Wine'],
    'Algoritmo': ['Naive Bayes', 'Naive Bayes', 'OneR'],
    'Accuracy': [0.96, 0.94, 0.8314],
    'Precision': [0.96, 0.9737, 0.8153],
    'Recall': [0.97, 0.9024, 0.7464]
}
df_metricas = pd.DataFrame(metricas_data)

# --- SECCIONES ---

if pagina == "Comparativa General":
    st.title("📊 Comparativo de Rendimiento - Act. 4")
    st.write("Resumen de métricas obtenidas en los tres conjuntos de datos analizados.")
    
    # Gráfico comparativo interactivo
    fig = go.Figure()
    for metric in ['Accuracy', 'Precision', 'Recall']:
        fig.add_trace(go.Bar(name=metric, x=df_metricas['Dataset'], y=df_metricas[metric]))
    
    fig.update_layout(barmode='group', title="Comparación de Métricas por Dataset")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Tabla Resumen")
    st.table(df_metricas)

elif pagina == "Breast Cancer (Salud)":
    st.title("🩺 Análisis: Breast Cancer Wisconsin")
    st.write("Modelo: Naive Bayes Gaussiano")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Accuracy", "94%")
        st.metric("F1-Score", "93.67%")
        st.metric("Recall (Maligno)", "90.24%")
        
        st.error("**Análisis Crítico:** Se detectaron 4 falsos negativos (malignos clasificados como benignos).")
        st.info("En el sector salud, los falsos negativos tienen un mayor costo real.")

    with c2:
        st.subheader("Matriz de Confusión")
        # Datos de la matriz del reporte
        data_bc = [[72, 1], [4, 37]]
        df_bc = pd.DataFrame(data_bc, columns=['Benigno (Pred)', 'Maligno (Pred)'], index=['Benigno (Real)', 'Maligno (Real)'])
        st.dataframe(df_bc.style.background_gradient(cmap='Blues'), use_container_width=True)

elif pagina == "Wine (Química)":
    st.title("🍷 Análisis: Dataset Wine")
    st.write("Modelo: OneR (Basado en Flavonoids)")
    
    c1, col_img = st.columns(2)
    with c1:
        st.metric("Exactitud (Accuracy)", "83.14%")
        st.warning("**Atributo Relevante:** Flavonoids")
        st.write("Este fue el rasgo químico más discriminante para construir la regla de clasificación.")
        
        # Cargar resultados del CSV generado por Andre
        try:
            df_w = pd.read_csv('Wine_resultados_clustering.csv')
            ct_w = pd.crosstab(df_w['Actual'], df_w['Cluster'])
            st.subheader("Matriz de Contingencia (Clustering)")
            st.dataframe(ct_w, use_container_width=True)
        except:
            st.write("Carga el archivo 'Wine_resultados_clustering.csv' para ver la matriz.")

    with col_img:
        try:
            img_w = Image.open('kmeans_wine_2d.png')
            st.image(img_w, caption="Visualización de Clusters en Wine")
        except:
            st.write("Imagen 'kmeans_wine_2d.png' no encontrada.")

elif pagina == "Iris (Agrupamiento)":
    st.title("🌸 Análisis: Dataset Iris")
    st.write("Modelo: K-Means (K=3)")
    
    col_mat, col_viz = st.columns(2)
    with col_mat:
        st.subheader("Resultados de Agrupamiento")
        try:
            df_i = pd.read_csv('Iris_resultados_clustering.csv')
            ct_i = pd.crosstab(df_i['Actual'], df_i['Cluster'])
            st.dataframe(ct_i.style.highlight_max(axis=0), use_container_width=True)
        except:
            st.error("Archivo CSV no encontrado.")
            
        st.metric("Silhouette Score", "0.45 - 0.55")
        st.write("**Patrón identificado:** El grupo Setosa es perfectamente separable linealmente.")

    with col_viz:
        try:
            img_i = Image.open('kmeans_iris_2d.png')
            st.image(img_i, caption="Separación física de las especies")
        except:
            st.write("Imagen 'kmeans_iris_2d.png' no encontrada.")