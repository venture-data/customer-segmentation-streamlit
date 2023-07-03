# Importing the necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.cm as cm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Creating the Streamlit app
def main():
    st.title("Customer Segmentation App")
    #st.write("This app performs customer segmentation using K-means clustering.")

    # Generate dummy data
    num_samples = 500
    num_features = 4
    centers = 4
    cluster_std = 1.0
    random_state = 42
    X, _ = make_blobs(
        n_samples=num_samples,
        n_features=num_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    df = pd.DataFrame(X, columns=[f"Feature {i+1}" for i in range(num_features)])

    # Perform data preprocessing
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # User input for number of clusters
    num_clusters = st.slider("Select the number of clusters", min_value=2, max_value=10)

    # Perform clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(scaled_data)
    df['Cluster'] = kmeans.labels_

    # Display the results
    st.write("Customer Segmentation Results:")
    st.dataframe(df.style.highlight_max(axis=0))

    # Generate distinct colors for each cluster
    colors = cm.get_cmap('tab10', num_clusters)

    # Scatter plot
    fig = px.scatter(
        df, x="Feature 1", y="Feature 2", color="Cluster", color_discrete_sequence=colors.colors,
        title="Customer Segmentation"
    )
    st.plotly_chart(fig)

    # Box plot
    fig = px.box(df, x="Cluster", y="Feature 3", title="Feature 3 Distribution by Cluster")
    st.plotly_chart(fig)

    # 3D Scatter plot
    fig = px.scatter_3d(
        df,
        x="Feature 1",
        y="Feature 2",
        z="Feature 3",
        color="Cluster",
        color_discrete_sequence=colors.colors,
        title="Customer Segmentation (3D)",
    )
    fig.update_layout(scene=dict(xaxis_title="Feature 1", yaxis_title="Feature 2", zaxis_title="Feature 3"))
    st.plotly_chart(fig)

    # Bar chart
    cluster_counts = df["Cluster"].value_counts().reset_index()
    cluster_counts.columns = ["Cluster", "Count"]
    fig = px.bar(cluster_counts, x="Cluster", y="Count", title="Cluster Size")
    st.plotly_chart(fig)

    # Radar chart
    feature_means = df.groupby("Cluster").mean().reset_index()
    fig = go.Figure()
    for cluster in feature_means["Cluster"]:
        fig.add_trace(go.Scatterpolar(
            r=feature_means.loc[feature_means["Cluster"] == cluster, df.columns[:-1]].values.flatten(),
            theta=df.columns[:-1],
            fill="toself",
            name=f"Cluster {cluster}",
            line_color=colors(cluster),
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[-2, 2]),
        ),
        showlegend=True,
        title="Cluster Feature Comparison"
    )
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
