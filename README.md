# Supervised Machine Learning: Classification & Regression Analysis  

## **Project Overview**  
This project applies supervised machine learning algorithms to analyze and classify trade-related data. The main objective is to evaluate the performance of various classification models, identify significant features, and derive actionable insights. Models such as Logistic Regression, Decision Trees, and others are implemented and compared based on their performance metrics.  

## **Problem Statement**  
The goal of this project is to classify trade transactions into distinct categories based on various trade-related attributes. This classification helps uncover patterns and trends that can drive strategic business decisions. The key objectives include:  

- Classifying trade transactions using supervised learning models.  
- Identifying important features and determining thresholds for classification.  
- Comparing models to determine the most suitable one for the dataset.  

## **Key Features**  

### **Classification Models:**  
- Logistic Regression (LR)  
- Support Vector Machines (SVM)  
- Stochastic Gradient Descent (SGD)  
- Decision Trees (DT)  
- K-Nearest Neighbors (KNN)  
- Random Forest (RF)  
- Naive Bayes (NB)  
- XGBoost (Extreme Gradient Boosting)  

### **Performance Evaluation:**  
- Accuracy, Precision, Recall, F1-Score, AUC  
- Cross-Validation (K-Fold) for robust model evaluation  

### **Run Statistics:**  
- Training time and memory usage for all models  

### **Feature Importance:**  
- Identification of significant features like Country, Product, and Value  
- Thresholds and business implications of key features  

---

## **Data Preprocessing**  
- **Data Cleaning:** No missing data; no treatment required.  
- **Encoding:** Ordinal encoding applied for categorical variables.  
- **Scaling:** Min-Max Scaling for numerical data.  

---

## **Dataset**  
- **Source:** Kaggle Import-Export Dataset  
- **Sample Size:** 5,001 rows selected from 15,000 records  

### **Key Variables:**  
- **Categorical:** Country, Product, Import_Export, Category, Port, Shipping_Method, Supplier, Customer, Payment_Terms  
- **Numerical:** Quantity, Value, Weight  
- **Index Variables:** Transaction_ID, Invoice_Number  

---

## **Technologies Used**  
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Environment:** Google Colab  
- **Version Control:** GitHub (for version tracking and collaboration)  

---

## **Methodology**  

### **Data Preprocessing:**  
- **Data Cleaning:** No missing data identified, no further treatment required.  
- **Encoding:** Ordinal encoding applied for categorical data.  
- **Scaling:** Min-Max Scaling applied for numerical data normalization.  

### **Classification Models:**  
- **Training and Testing Split:** 70% training, 30% testing to evaluate model performance.  
- **Comparison:** Models compared based on test set performance and cross-validation results.  

### **Performance Metrics:**  
- **Confusion Matrix**  
- **Precision**, **Recall**, **F1-Score**, **AUC**  
- **Training Time** and **Memory Usage** for runtime analysis  

### **Feature Analysis:**  
- Significant features identified using **Random Forest** and **Logistic Regression**.  
- Thresholds derived for key features like **Value**, **Weight**, and **Shipping_Method**.  

---

## **Insights and Applications**  

### **Model Insights:**  
- **Logistic Regression:** Performed well in **runtime** and **interpretability** but struggled with **class imbalance**.  
- **Random Forest:** Provided **robust feature importance** insights but was **resource-intensive**.  
- **SVM** and **SGD:** Focused heavily on **dominant classes**, highlighting the need for a more **balanced dataset** for better performance.  

### **Business Applications:**  
- **Targeted Marketing:** Use insights from key features and clusters to create tailored marketing strategies.  
- **Inventory Management:** Focus on trade-specific patterns, such as differentiating **bulk** and **lightweight** goods.  
- **Supplier-Customer Relations:** Optimize **Payment Terms** and **Shipping Methods** to enhance trade efficiency.  

---

## **Contributors**  
- Mohit Agarwal
