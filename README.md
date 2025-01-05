# Unsupervised Machine Learning: Clustering Analysis

## Project Overview
This project explores the application of unsupervised machine learning algorithms to segment and analyze a trade-related dataset. The main objective is to identify meaningful clusters within the data using various clustering techniques, such as K-Means, DBSCAN, and Birch. The analysis focuses on determining the optimal number of clusters, analyzing the characteristics of each cluster, and deriving actionable business insights from the findings.

## Problem Statement
The goal of this project is to segment the dataset into distinct clusters based on various trade-related features. This segmentation will help in understanding hidden patterns, relationships, and trends within the data, which can be used for strategic business decision-making. The key objectives are:

- Identify homogeneous groups (clusters) within the dataset.
- Analyze and interpret the distinguishing features of each cluster.
- Provide actionable insights and recommendations based on the clustering results.

## Key Features
- **Clustering Algorithms**: Implementation of K-Means, DBSCAN, and Birch clustering algorithms.
- **Cluster Evaluation**: Use of performance metrics like Silhouette Coefficient and Davies-Bouldin Index to assess clustering quality.
- **Optimal Cluster Selection**: Identification of the optimal number of clusters through algorithm evaluation and hyperparameter tuning.
- **Cluster Characteristics**: In-depth analysis of cluster profiles and patterns for actionable business insights.
- **Data Preprocessing**: Handling of categorical and numerical data, with feature engineering for enhanced analysis.

## Dataset
The dataset used in this project contains a total of 15,000 trade transactions, with a sample size of 5,001 rows selected for analysis. Key variables include:

- **Categorical Variables**: Country, Product, Import/Export, Category, Port, Shipping Method, Supplier, Customer, Payment Terms (nominal and ordinal data).
- **Non-Categorical Variables**: Quantity, Value, Weight (numerical data).

The data is stored in CSV format and is structured to reflect international trade transactions over time.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Environment**: Google Colab
- **Version Control**: GitHub (for version tracking and collaboration)

## Methodology
# Data Preprocessing and Analysis

## **Data Preprocessing**  
1. **Data Cleaning:**  
   - No missing data identified; no treatment required.  

2. **Encoding:**  
   - Categorical variables encoded using **Ordinal Encoder**.  

3. **Scaling:**  
   - Numerical features scaled using **Min-Max Scaling** to normalize data within a range.  


## **Classification Models**  
1. **Dataset Split:**  
   - Data split into training (70%) and testing (30%) sets for model evaluation.  

2. **Evaluation Metrics:**  
   - Models assessed based on **test set performance** and **cross-validation results**.  


## **Performance Metrics**  
1. **Metrics Used:**  
   - **Confusion Matrix** for classification performance visualization.  
   - **Precision**, **Recall**, **F1-Score**, and **AUC** for quantitative evaluation.  

2. **Runtime Analysis:**  
   - **Training time** and **memory usage** analyzed to assess computational efficiency.  


## **Feature Analysis**  
1. **Feature Importance:**  
   - Significant features identified using **Random Forest** and **Logistic Regression** models.  

2. **Threshold Derivation:**  
   - Key features like **Value**, **Weight**, and **Shipping_Method** analyzed to derive optimal thresholds.  


## Insights and Applications
# Model Insights and Business Applications  

## **Model Insights**  
1. **Logistic Regression:**  
   - Excelled in **runtime efficiency** and **interpretability**.  
   - Struggled with **class imbalance**, impacting overall performance.  

2. **Random Forest:**  
   - Provided **robust feature importance insights**, aiding feature selection.  
   - Resource-intensive in terms of **memory** and **runtime**.  

3. **Support Vector Machine (SVM) and Stochastic Gradient Descent (SGD):**  
   - Focused heavily on **dominant classes**, revealing the need for a more balanced dataset to improve generalization.  

---

## **Business Applications**  
1. **Targeted Marketing:**  
   - Leverage insights from **key features and clusters** to craft targeted marketing strategies that address specific customer needs.  

2. **Inventory Management:**  
   - Focus on trade-specific patterns, such as differentiating **bulk** from **lightweight goods**, to optimize stock and supply chains.  

3. **Supplier-Customer Relations:**  
   - Enhance trade efficiency by optimizing **Payment Terms** and **Shipping Methods** based on identified patterns and customer preferences.  

## Contributors
- **Mohit Agarwal**
