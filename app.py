import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

st.set_page_config(page_title="Clustering Algorithm Explorer")
st.title("Interactive Clustering Algorithm Exploration And Analysis")
with st.sidebar:
    st.header("Synthetic Dataset Configuration")
    dataset_type = st.selectbox("Select Dataset Shape", ['Blobs', 'Moons'])
    n_samples = st.slider("Number of Data Points", 500, 2000, 1000)
    noise = st.slider("Noise Level", 0.05, 0.5, 0.1)

    st.markdown("---")
    st.header("Scaling Techniques")
    scaler_choice = st.selectbox("Choose a Scaler", ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'None'])

    st.markdown("---")
    st.header("Clustering Algorithm")
    algorithm_choice = st.selectbox("Choose an Algorithm", ['K-Means', 'DBSCAN', 'Agglomerative', 'HDBSCAN'])
if dataset_type=='Blobs' :
    X, y = make_blobs(n_samples=n_samples, random_state=42)
else:
    X, y = make_moons(n_samples=n_samples,noise=noise, random_state=42)

if scaler_choice == 'StandardScaler':
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
elif scaler_choice == 'MinMaxScaler':
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
elif scaler_choice == 'RobustScaler':
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
else:
    X_scaled = X

    
if algorithm_choice == 'K-Means':
    k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 4)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)

elif algorithm_choice == 'DBSCAN':
    eps = st.sidebar.slider("Epsilon (eps)", 0.1, 2.0, 0.5)
    min_samples = st.sidebar.slider("Minimum Samples", 1, 20, 5)
    model = DBSCAN(eps=eps, min_samples=min_samples)

elif algorithm_choice == 'Agglomerative':
    n_clusters_agg = st.sidebar.slider("Number of Clusters", 2, 10, 4)
    model = AgglomerativeClustering(n_clusters=n_clusters_agg)

elif algorithm_choice == 'HDBSCAN':
    min_cluster_size = st.sidebar.slider("Min Cluster Size", 2, 30, 5)
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)


try:
    clusters = model.fit_predict(X_scaled)

    fig = go.Figure(data=go.Scatter(x=X_scaled[:, 0],y=X_scaled[:, 1],mode='markers',marker=dict(color=clusters,colorscale='Viridis',showscale=True),text=[f'Cluster: {c}' for c in clusters]))
    fig.update_layout(title=f'Clustering Results for {algorithm_choice}',xaxis_title='Feature 1',yaxis_title='Feature 2',height=600)
    st.plotly_chart(fig, use_container_width=True)

    if len(set(clusters)) > 1:

        st.subheader("Performance Metrics")

        col1, col2, col3 = st.columns(3)

        silhouette = silhouette_score(X_scaled, clusters)
        col1.metric(label="Silhouette Score", value=f"{silhouette:.3f}")

        davies_bouldin = davies_bouldin_score(X_scaled, clusters)
        col2.metric(label="Davies-Bouldin Index", value=f"{davies_bouldin:.3f}")

        calinski_harabasz = calinski_harabasz_score(X_scaled, clusters)
        col3.metric(label="Calinski-Harabasz Score", value=f"{calinski_harabasz:.3f}")

    else:
        st.warning("Only one cluster found. Performance metrics are not applicable.")

except Exception as e:
    st.error(f"An error occurred: {e}")










