# **Supervised Learning Models on Import-Export Dataset using Python**



---



## **PROJECT CONTENTS:**
- Project Information
- Description of Data
- Data Sampling
- Project Objectives | Problem Statements
- Analysis of Data
- Observations | Findings
- Managerial Insights | Recommendations

---

### **1. Project Information**
- **Title**: Data Exploration with Python using Pandas & Numpy Libraries
- **Students**:Mohit Agarwal (055024)
- **Group Number**: 2

---

### **2. Description of Data**
- **Data Source**: https://www.kaggle.com/datasets/chakilamvishwas/imports-exports-15000

  A live link is used in the dataset

- **Data Columns Description**:
  - **Transaction_ID**: Unique identifier for each trade transaction.
  - **Country**: Country of origin or destination for the trade.
  - **Product**: Product being traded.
  - **Import_Export**: Indicates whether the transaction is an import or export.
  - **Quantity**: Amount of the product traded.
  - **Value**: Monetary value of the product in USD.
  - **Date**: Date of the transaction.
  - **Category**: Category of the product (e.g., Electronics, Clothing, Machinery).
  - **Port**: Port of entry or departure.
  - **Customs_Code**: Customs or HS code for product classification.
  - **Weight**: Weight of the product in kilograms.
  - **Shipping_Method**: Method used for shipping (e.g., Air, Sea, Land).
  - **Supplier**: Name of the supplier or manufacturer.
  - **Customer**: Name of the customer or recipient.
  - **Invoice_Number**: Unique invoice number for the transaction.
  - **Payment_Terms**: Terms of payment (e.g., Net 30, Net 60, Cash on Delivery).

- **Data Type**: Since the dataset contains multiple entities (countries) and records data over time, this is an example of Panel Data (also called longitudinal data).

- **Data Variables**:
  - **All non-null Variables**
    - **Numbers**:
      - Integer Variables: 3 (Quantity, Customs_Code, Invoice_Number)
      - Float (Decimal) Variables: 2 (Value, Weight)
    - **Text**: 9 (Country, Product, Import_Export, Category, Port, Shipping_Method, Supplier, Customer, Payment_Terms)
    - **DateTime**: 1 (Date)

---

### **3. Data Sampling**
From the dataset containing 15,000 values, a sample of 5001 entries was taken. The dataset sample (now referred to as `ma24_sample`) was taken into account for further exploration.

- **Data Variables**:
  - **Index Variables**: 'Transaction_ID', 'Invoice_Number'
  - **Categorical Variables**:
    - Nominal Variables: Country, Product, Import_Export, Category, Port, Shipping_Method, Supplier, Customs_Code, Customer
    - Ordinal Variable: Payment_Terms
  - **Non-Categorical Variables**: Quantity, Value, and Weight

---

### **4. Project Objectives**
- Classification of Dataset into {Segments | Clusters | Classes} using Supervised Learning Classification Algorithms
- Identification of {Important | Contributing | Significant} Variables or Features and their Thresholds for Classification
- Determination of an appropriate Classification Model based on Performance Metrics


---

### **5. Exploratory Data Analysis**
#### **5.1. Data Preprocessing:**
- The data has no missing values, hence no missing data treatment was performed.
- For Encoding, **Ordinal Encoder** was used.
- For Scaling, **Min-Max Scaler** was used.

#### **5.2. Descriptive Statistics**
- **Non-Categorical Variables**:
  - Measures of Central Tendency: Minimum, Maximum, Mean, Median, Mode, Percentile
  - Measures of Dispersion: Range, Standard Deviation, Skewness, Kurtosis, Correlation (Matrix)
  - Composite Measure: Coefficient of Variation, Confidence Interval

- **Categorical Variables**:
  - Count, Frequency, Proportion, Minimum, Maximum, Mode, Rank

#### **5.3. Data Visualization**
- Various subplots were used such as Bar, Heatmaps, Histograms, and Correlation Matrices.

#### **5.4. Inferential Statistics**
- **Categorical Variable** (Nominal | Ordinal):
  - Test of Homogeneity (Chi-sq)
- **Non-Categorical Variable**:
  - Test of Normality (Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling, Jarque-Bera)
  - Test of Correlation (t)

#### **5.5. Machine Learning Models**

- **Logistic Regression (LR)**:
  Logistic regression is a classification algorithm that models the probability of a binary outcome based on one or more predictor variables using a logistic function.
  
- **Support Vector Machines (SVM)**:
  SVM finds a hyperplane that best separates the data into classes by maximizing the margin between the closest points of the classes (support vectors).

- **Stochastic Gradient Descent (SGD)**:
  An iterative optimization algorithm used for minimizing loss functions in linear classifiers and regressors, particularly useful for large datasets.

- **Decision Trees**:
  A tree-based algorithm where each internal node represents a decision on a feature, each leaf node represents an outcome, and paths from root to leaf represent classification rules.

- **K-Nearest Neighbors (KNN)**:
  A lazy learning algorithm that classifies a data point based on the majority class of its k-nearest neighbors in the feature space.
  
- **Naive Bayes (NB)**:
  A probabilistic classifier based on Bayes' theorem, assuming strong independence between features.

- **Bagging (Bootstrap Aggregating)**:
  Combines predictions from multiple models (e.g., Random Forest) to reduce variance and improve accuracy.

- **Boosting**:
  Combines weak learners sequentially to form a strong learner by minimizing errors iteratively (e.g., Extreme Gradient Boosting - XGBoost).

#### **5.6. Model Performance Metrics**

- **Confusion Matrix**:
  - Sensitivity (Recall): True Positive Rate
  - Specificity: True Negative Rate
  - Accuracy: Overall correctness
  - Precision: Positive Predictive Value
  - F1-Score: Harmonic mean of Precision and Recall
  - AUC: Area Under the ROC Curve

- **K-Fold Cross-Validation**: Splits data into k subsets; trains on k-1 and tests on the remaining one iteratively to ensure robust performance evaluation.

- **Model Run Statistics**:
  - Time Taken
  - Memory Used
  - Inherent Complexity
---


### **6. Observations and Findings**

#### **6.1. Nature of Data**
- **Missing Data**: The dataset contained no missing values.  
- **Non-Numeric Categorical Data**: Categorical variables (e.g., Country, Product) were encoded using ordinal encoding.  
- **Scales and Outliers**:  
  - Non-categorical numeric data (Quantity, Value, Weight) were scaled using Min-Max scaling.  
  - Box plots revealed potential outliers, but they were not removed as they might hold valuable information.  

#### **6.2. Supervised Machine Learning**




1. **Logistic Regression (LR) vs. Support Vector Machine (SVM)** 

   - *Performance Metrics*:

     - LR Accuracy: 33.71%, SVM Accuracy: 33.84%

     - Both models showed poor performance overall, with SVM slightly outperforming LR in accuracy.  

   - *Run Statistics*:

     - Logistic Regression was faster (0.02 seconds) compared to SVM (0.97 seconds).  

     - SVM had slightly lower memory usage (0.27 MB vs. 0.34 MB for LR).  

   - *Conclusion*:  

     While SVM has a marginal accuracy advantage, Logistic Regression is preferred for its faster runtime and simplicity, especially for datasets where interpretability and time are critical.



2. **Decision Tree (DT) vs. K-Nearest Neighbors (KNN)** 

   - *Performance Metrics*:

     - DT Accuracy: 34.64%, KNN Accuracy: 34.11%.  

     - DT slightly outperformed KNN in accuracy and consistency across classes.  

   - *Run Statistics*:

     - KNN was significantly faster (0.0072 seconds) compared to DT (0.0415 seconds).  

     - Memory usage was comparable for both models.  

   - *Conclusion*:  

     Decision Tree is preferred for its higher accuracy and interpretability, while KNN may be suitable for scenarios prioritizing runtime.



3. **Logistic Regression (LR) vs. Decision Tree (DT)** 

   - *Performance Metrics*:

     - LR Accuracy: 33.71%, DT Accuracy: 34.64%.  

     - DT showed slightly better accuracy and more balanced performance across classes.  

   - *Run Statistics*:

     - LR was faster (0.02 seconds) compared to DT (0.0415 seconds).  

     - Both models had similar memory usage.  

   - *Conclusion*:  

     Decision Tree is favored due to better accuracy, while Logistic Regression remains a viable option for faster computations and interpretability.



4. **Decision Tree (DT) vs. Random Forest (RF)**  

   - *Performance Metrics*:

     - DT Accuracy: 34.64%, RF Accuracy: 34.57%.  

     - Both models performed similarly, with DT showing marginally higher accuracy.  

   - *Run Statistics*:

     - DT was significantly faster (0.0415 seconds) compared to RF (1.49 seconds).  

     - RF consumed more memory (0.46 MB vs. 0.23 MB for DT).  

   - *Conclusion*:  

     Decision Tree is preferred for faster training, while Random Forest may be more effective for larger, more complex datasets.



---



#### **Cross-Validation of Results and Performance**



1. **Logistic Regression**:

   - Mean Cross-Validation Accuracy: 33.03%  

   - Test Set Accuracy: 36.44%  

   - Observations:

     - Struggles with class imbalance, especially for Class 1 (0% recall).  

     - Shows good recall for Class 2 (77%), indicating potential in capturing patterns for specific classes.



2. **Support Vector Machine (SVM)**:

   - Mean Cross-Validation Accuracy: 33.09%  

   - Test Set Accuracy: 34.24%  

   - Observations:

     - Strong bias towards predicting Class 2, achieving perfect recall (100%) but failing completely for Classes 0 and 1.  



3. **Decision Tree (DT)**:

   - Mean Cross-Validation Accuracy: 34.64%  

   - Test Set Accuracy: 35.00%  

   - Observations:

     - Consistent performance across classes, with slight improvements in precision, recall, and F1-score for Class 2.  

     - Handles non-linear relationships better than LR and SVM.  



4. **Random Forest (RF)**:

   - Mean Cross-Validation Accuracy: 34.57%  

   - Test Set Accuracy: 34.58%  

   - Observations:

     - Similar performance to DT but higher memory usage and runtime, making it less practical for this dataset.  

### **7. Managerial Insights and Recommendations**

#### **7.1. Preprocessing of Data**
- **Missing Data**: No specific treatment is necessary due to the absence of missing data.  
- **Encoding**: Ordinal encoding proved effective for categorical variables.  
- **Scaling**: Min-Max scaling was appropriate to address potential scale differences.  
- **Data Split**: A **70:30 split** for training and testing sets would be appropriate for future supervised learning tasks.  

#### **7.2. Supervised Machine Learning**


#### **Inference regarding the Purpose of Classification**  
The purpose of the classification is to identify patterns in the data and accurately assign instances to predefined classes. However, all models struggled with poor overall accuracy, suggesting challenges such as class imbalance, insufficient data quality, or lack of relevant features. Improving these areas is critical for achieving meaningful insights.  

---

#### **Logistic Regression (LR)**  
#### **Key Features and Thresholds**  
- Focus on highly correlated features for interpretability, e.g., categorical variables or numeric predictors with strong linear relationships.  
- **Threshold:** Evaluate multiple decision thresholds (e.g., 0.5, 0.6) to balance precision and recall, particularly for underrepresented classes.  

#### **Insight on Uniqueness, Usability, and Suitability**  
- **Uniqueness:** Simple and interpretable; suitable for datasets with linearly separable classes.  
- **Usability:** Best for quick preliminary analysis and insights, especially when time and interpretability are critical.  
- **Suitability:** Limited by poor performance in non-linear relationships and class imbalance. More effective with balanced and linearly distributed datasets.  

---

#### **Support Vector Machine (SVM)**  
#### **Key Features and Thresholds**  
- Focus on features that enhance separability between classes in a high-dimensional space.  
- **Threshold:** Use a soft margin for hyperparameter tuning to balance precision and recall for minority classes.  

#### **Insight on Uniqueness, Usability, and Suitability**  
- **Uniqueness:** Excellent at capturing complex decision boundaries, especially with kernel tricks for non-linear relationships.  
- **Usability:** Less interpretable and computationally expensive; more suited to small, high-dimensional datasets.  
- **Suitability:** Performs poorly on imbalanced datasets. Best for situations where capturing subtle patterns is essential.  

---

#### **Decision Tree (DT)**  
#### **Key Features and Thresholds**  
- Select features with high information gain or Gini index for splitting nodes.  
- **Threshold:** Adjust tree depth to avoid overfitting while maintaining interpretability (e.g., max depth of 5).  

#### **Insight on Uniqueness, Usability, and Suitability**  
- **Uniqueness:** Provides interpretable, tree-structured decisions, capturing non-linear relationships effectively.  
- **Usability:** Easy to visualize and explain, making it useful for decision-making.  
- **Suitability:** Suitable for moderately complex datasets but prone to overfitting on small datasets. Effective for balanced and moderately imbalanced data.  

---

#### **Random Forest (RF)**  
#### **Key Features and Thresholds**  
- Use ensemble averaging to improve predictive accuracy and reduce variance.  
- **Threshold:** Focus on selecting the optimal number of trees (e.g., 50–100) and features per split for performance without excessive resource consumption.  

#### **Insight on Uniqueness, Usability, and Suitability**  
- **Uniqueness:** Combines multiple decision trees to improve stability and generalization, reducing overfitting compared to DT.  
- **Usability:** Higher computational cost and memory usage compared to DT but handles noisy and imbalanced data better.  
- **Suitability:** Ideal for larger datasets with complex relationships, where interpretability is less critical than predictive accuracy.  

---

### **Recommendations for Managerial Actions**  
1. **Data Quality and Balance:**  
   - Address class imbalance and improve feature engineering to enhance model performance.  
   - Consider oversampling, undersampling, or using class-weight adjustments.  

2. **Model Selection:**  
   - **Logistic Regression** for fast, interpretable results where linear relationships dominate.  
   - **Decision Tree** for interpretable, non-linear modeling of moderately complex data.  
   - **Support Vector Machine** when high-dimensional data and non-linear patterns are present but computational efficiency is secondary.  
   - **Random Forest** for robust predictions in large, complex datasets.  

3. **Operational Use:**  
   - Select models based on the trade-off between accuracy, runtime, and interpretability, aligning with project objectives and data constraints.  


---
### **Identification of Important Variables or Features and Their Thresholds**

#### **1. Key Features Identified from Multiple Models**
Analyzing outputs from Decision Tree, Random Forest, XGBoost, KNN, Naive Bayes, and Logistic Regression reveals crucial insights into features influencing classification or clustering. Despite moderate accuracy across models, certain variables consistently emerge as significant:

- **Country**  
  - Strongly correlates with trade patterns and economic agreements. Specific countries often dominate imports or exports in particular categories.  
  - **Thresholds**: Major trade partners (e.g., based on transaction counts or values) often form distinct groups influencing cluster formation.

- **Product**  
  - Plays a pivotal role due to trade specialization, such as Electronics and Machinery versus Clothing.  
  - **Thresholds**: Products with high weights or values tend to indicate distinct classes or clusters.

- **Import_Export**  
  - Differentiates transaction types and highlights trade flow distinctions.  
  - **Thresholds**: Exports are generally associated with higher-value products compared to imports in this dataset.

- **Quantity, Value, and Total_Value**  
  - High numerical correlation with trade significance, particularly in models like Decision Tree and Random Forest.  
  - **Thresholds**:  
    - **Quantity**: Higher quantities are linked to lower-value items like raw materials or bulk goods.  
    - **Value**: Higher-value products cluster separately due to their economic significance.

- **Weight**  
  - Critical for clustering based on shipping and logistics costs.  
  - **Thresholds**: Lightweight, high-value items (e.g., electronics) cluster distinctly from heavy, low-value items (e.g., raw metals).

- **Shipping_Method**  
  - Reflects trade-specific preferences, such as air for lightweight, high-value items versus sea or land for bulk or low-value goods.  
  - **Thresholds**: Specific clusters align with shipping modes, such as air freight for electronics.

- **Port**  
  - Ports often specialize in handling certain categories or trade flows.  
  - **Thresholds**: High-traffic ports typically align with distinct clusters.

- **Customs_Code**  
  - Provides a categorical classification of goods, significantly impacting model performance.  
  - **Thresholds**: HS codes differentiate goods by economic sector or trade priority.

---

#### **2. Model-Specific Observations and Insights**  

#### Decision Tree
- **Accuracy**: 34.6% (low).  
- **Key Variables**: Country, Product, Quantity, and Value consistently influence tree splits.  
- **Misclassifications**:  
  - Significant confusion between Cluster 0 and Cluster 1.  
  - Slightly better precision, recall, and F1-score for Class 2.0, indicating stronger performance in distinguishing specific clusters.  
- **Thresholds**:  
  - Products with higher Value and lower Weight are more accurately identified.  
  - Quantity thresholds help differentiate bulk from specialized goods.

#### Random Forest
- **Key Variables**: Similar to Decision Tree, with enhanced handling of non-linear relationships.  
- **Insights**:  
  - Features like Port and Customs_Code significantly influence predictions due to ensemble averaging.  
  - Suggested thresholds include grouping Ports by transaction volume.

#### XGBoost
- **Accuracy**: 33.24% (low).  
- **Key Variables**: Product, Country, and Value are prominent based on feature importance scores.  
- **Insights**:  
  - Performance suffers from imbalanced data. Adjusting thresholds for major features like Value or using sampling techniques could improve results.

#### KNN
- **Accuracy**: 34.11% (low).  
- **Key Variables**: Quantity and Weight, as KNN relies heavily on distance-based relationships.  
- **Insights**:  
  - Normalizing data could enhance clustering based on these variables.

#### Naive Bayes
- **Accuracy**: 33.77% (low).  
- **Key Variables**: Import_Export, Customs_Code, and Shipping_Method.  
- **Insights**:  
  - Assumes feature independence, yet categorical variables like Import_Export and Customs_Code drive predictions.

#### Logistic Regression
- **Accuracy**: 33.71% (low).  
- **Key Variables**: Country, Product, and Shipping_Method.  
- **Insights**:  
  - Struggles significantly with Cluster 1 due to class imbalance or overlapping data.  
  - Refining thresholds for categorical features like Shipping_Method may help separate clusters.
