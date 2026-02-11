# ===============================
# Customer Segmentation Streamlit App
# ===============================

import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("📊 Customer Segmentation using Clustering Algorithms")

# ===============================
# File Upload
# ===============================
uploaded_file = st.file_uploader("Upload your customer data CSV file", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    # ===============================
    # Basic Data Exploration
    # ===============================
    st.subheader("Dataset Preview")
    st.write(data.head())

    st.subheader("Dataset Information")
    st.write("Shape:", data.shape)
    st.write("Size:", data.size)

    st.write("Columns:", data.columns.tolist())

    st.write("Missing Values:")
    st.write(data.isnull().sum())

    st.write("Duplicate Values:", data.duplicated().sum())

    st.subheader("Statistical Summary")
    st.write(data.describe())

    # ===============================
    # Feature Selection (same as your code)
    # ===============================
    st.subheader("Selected Features (Column 3 and 4)")

    X = data.iloc[:, [3, 4]]
    st.write(X.head())

    # ===============================
    # Scaling
    # ===============================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ===============================
    # Elbow Method
    # ===============================
    st.subheader("Elbow Method")

    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    fig1, ax1 = plt.subplots()
    ax1.plot(range(1, 11), wcss)
    ax1.set_title("Elbow Method")
    ax1.set_xlabel("Number of Clusters")
    ax1.set_ylabel("WCSS")
    st.pyplot(fig1)

    # ===============================
    # KMeans Clustering
    # ===============================
    st.subheader("KMeans Clustering")

    k = st.slider("Select number of clusters", 2, 10, 5)

    kmeans = KMeans(n_clusters=k, random_state=42)
    y_kmeans = kmeans.fit_predict(X_scaled)

    # Silhouette Score
    score = silhouette_score(X_scaled, y_kmeans)
    st.write("Silhouette Score:", round(score, 3))

    # Add Cluster Column
    data['Cluster'] = y_kmeans

    st.subheader("Dataset with Cluster Labels")
    st.write(data.head())

    # ===============================
    # Visualization
    # ===============================
    st.subheader("Cluster Visualization")

    fig2, ax2 = plt.subplots()

    for i in range(k):
        ax2.scatter(
            X_scaled[y_kmeans == i, 0],
            X_scaled[y_kmeans == i, 1],
            label=f'Cluster {i+1}'
        )

    ax2.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=200,
        c='black',
        label='Centroids'
    )

    ax2.set_title("Customer Segmentation")
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")
    ax2.legend()

    st.pyplot(fig2)

else:
    st.info("Please upload a CSV file to start the analysis.")
