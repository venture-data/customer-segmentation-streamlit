# Importing the necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Creating the Streamlit app
def main():
    st.title("Customer Segmentation App")
    st.write("This app performs customer segmentation using K-means clustering.")

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
    st.dataframe(df)

    # Scatter plot
    fig = px.scatter(
        df, x="Feature 1", y="Feature 2", color="Cluster", title="Customer Segmentation"
    )
    st.plotly_chart(fig)

    # Box plot
    fig = px.box(df, x="Cluster", y="Feature 3", title="Feature 3 Distribution by Cluster")
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
