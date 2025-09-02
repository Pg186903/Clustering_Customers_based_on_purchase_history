# Project: Customer Segmentation using Clustering
### Overview

This project applies unsupervised machine learning to segment customers based on their purchasing behavior. Using the Mall_Customers.csv dataset, customers are grouped into clusters with similar characteristics using the K-Means algorithm. The results provide actionable insights for targeted marketing and personalized strategies.

# 

### Steps in the Notebook
#### 1. Environment Setup & Data Loading

* Used Google Colab to upload the dataset.

      from google.colab import files
      uploaded = files.upload()

* Libraries used:

  * numpy, pandas → Data manipulation
  
  * matplotlib, seaborn → Visualization
  
  * warnings → Suppressing warnings

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import warnings
        warnings.filterwarnings("ignore")
        
#### 2. Data Exploration

* Dropped irrelevant columns: CustomerID (not useful for clustering).

* Renamed column Genre to Gender for clarity.

* Converted categorical gender values into numeric form.

#### 3. Preprocessing

* Retained the most relevant features for clustering:

  * Age
  
  * Annual Income (k$)
  
  * Spending Score (1-100)

* Ensured data was numeric and suitable for clustering.

#### 4. Exploratory Data Analysis (EDA)

* Visualized the distribution of features such as age, income, and spending score.

* Used scatter plots and pair plots to observe customer patterns.

* Checked correlations to understand feature relationships.

      sns.pairplot(data, diag_kind="kde")
      plt.show()

#### 5. Model Building (K-Means Clustering)

* Applied the Elbow Method to determine the optimal number of clusters.

* Chose the best value of k and fitted the K-Means model.

      from sklearn.cluster import KMeans
      
      wcss = []  # Within Cluster Sum of Squares
      for i in range(1, 11):
          kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
          kmeans.fit(data[['Annual Income (k$)', 'Spending Score (1-100)']])
          wcss.append(kmeans.inertia_)
      
      plt.plot(range(1, 11), wcss, marker="o")
      plt.xlabel("Number of clusters")
      plt.ylabel("WCSS")
      plt.title("Elbow Method")
      plt.show()

      # Fitting KMeans with optimal clusters (say k=5)
      kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
      data['Cluster'] = kmeans.fit_predict(data[['Annual Income (k$)', 'Spending Score (1-100)']])

#### 6. Model Evaluation & Visualization

* Plotted clusters to visualize segmentation.

* Each cluster shows a distinct type of customer group.

      plt.figure(figsize=(8,6))
      sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", hue="Cluster", data=data, palette="tab10")
      plt.title("Customer Segments")
      plt.show()

# 

### Results

* Customers were successfully segmented into groups such as:

  * High income, high spending (premium customers)
  
  * Low income, low spending (budget customers)
  
  * High income, low spending (savers)
  
  * Moderate income, high spending (potential loyal customers)

* These clusters can be used for targeted marketing campaigns, loyalty programs, and personalized recommendations.

# 

### Key Features of the Project

* Simple yet effective customer segmentation using K-Means clustering.

* Used Elbow Method to determine the number of clusters.

* Clear visualizations of customer groups.

* Can be extended with:

  * More features (purchase frequency, product categories).
  
  * Other algorithms (Hierarchical Clustering, DBSCAN).
  
  * Real-world deployment for marketing analytics.
