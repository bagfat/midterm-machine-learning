# Machine Learning Midterm Projects

## Project Overview

This repository contains three comprehensive machine learning projects demonstrating different types of machine learning tasks: Classification, Regression, and Clustering. Each project includes data preprocessing, model training, hyperparameter tuning, and detailed evaluation.

---

## Identification

**Name:** Bagus Fatkhurrohman  
**Class:** Machine Learning
**NIM:** 1103223195
**Date:** 5 December 2025

---

## Repository Structure

```
midterm-machine-learning/
│
├── images/                                      # All visualizations
│   ├── Fraud_Detection_Classification/
│   │   └── model-comparison-fraud.png
│   │
│   ├── Song_Release_Year_Regression/
│   │   ├── feature-importance-song.png
│   │   ├── model-comparison-song.png
│   │   └── target-distribution-song.png
│   │
│   └── Customer_Clustering/
│       ├── cluster-profile-heatmap-customer.png
│       ├── determining-optimal-number-of-cluster.png
│       ├── silhouette-analysis-customer.png
│       └── visualizing-clusters-customer.png
│
├── notebooks/                                   # Jupyter notebooks
│       └── submissions/                            # Auto-generated results
│       ├── fraud_detection_submission.csv
│       ├── regression_submission.csv
│       ├── clustering_results.csv
│       └── cluster_profiles.csv
│   ├── 1_Fraud_Detection_Classification.ipynb
│   ├── 2_Song_Release_Year_Regression.ipynb
│   └── 3_Customer_Clustering.ipynb
│
└── README.md                                    # This file
```

---

## Project Descriptions

### Project 1: Fraud Detection (Classification)

**Objective:** Predict whether an online transaction is fraudulent or not.

**Task Type:** Binary Classification  
**Target Variable:** `isFraud` (0 = Not Fraud, 1 = Fraud)  
**Datasets:** `train_transaction.csv`, `test_transaction.csv`

**Workflow:**
1. Load and explore transaction data
2. Handle missing values and class imbalance
3. Feature preprocessing and scaling
4. Train multiple classification models:
   - Logistic Regression
   - Random Forest Classifier
   - XGBoost Classifier
5. Evaluate using appropriate metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
6. Compare models and select the best performer
7. Generate predictions on test data

**Key Metrics:**
- **Accuracy:** Overall correctness of predictions
- **Precision:** Proportion of predicted frauds that are actually frauds
- **Recall:** Proportion of actual frauds caught by the model
- **F1-Score:** Harmonic mean of Precision and Recall
- **ROC-AUC:** Area under the ROC curve for classification threshold evaluation

**Expected Results:**
- Multiple models trained and compared
- Best model identified based on ROC-AUC score
- Submission file with fraud probability predictions

---

### Project 2: Song Release Year Prediction (Regression)

**Objective:** Predict the release year of a song based on audio features.

**Task Type:** Regression (Continuous Value Prediction)  
**Target Variable:** Release Year (numeric value)  
**Dataset:** `midterm-regresi-dataset.csv`

**Workflow:**
1. Load and explore audio feature data
2. Handle missing values and outliers
3. Analyze feature correlations with target
4. Feature preprocessing and scaling
5. Train multiple regression models:
   - Linear Regression
   - Ridge Regression
   - Lasso Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - XGBoost Regressor
6. Evaluate using regression metrics (MSE, RMSE, MAE, R²)
7. Compare models and select the best performer
8. Analyze feature importance

**Key Metrics:**
- **MSE (Mean Squared Error):** Average squared difference between actual and predicted values
- **RMSE (Root Mean Squared Error):** Square root of MSE, in same units as target
- **MAE (Mean Absolute Error):** Average absolute difference between actual and predicted values
- **R² Score:** Proportion of variance explained by the model (0-1, higher is better)

**Expected Results:**
- Multiple regression models trained and evaluated
- Best model identified based on R² score
- Visualization of actual vs. predicted values
- Feature importance analysis (for tree-based models)

---

### Project 3: Customer Clustering (Unsupervised Learning)

**Objective:** Segment customers based on credit card usage and payment behavior.

**Task Type:** Unsupervised Learning (Clustering)  
**Target Variable:** None (Unsupervised)  
**Dataset:** `clusteringmidterm.csv`

**Workflow:**
1. Load and explore customer behavior data
2. Handle missing values and outliers
3. Feature preprocessing and scaling
4. Determine optimal number of clusters using:
   - Elbow Method
   - Silhouette Score
   - Davies-Bouldin Index
5. Train multiple clustering algorithms:
   - K-Means Clustering
   - Hierarchical Clustering
   - DBSCAN
6. Evaluate clusters using Silhouette Score
7. Interpret cluster characteristics
8. Visualize clusters using PCA reduction

**Key Metrics:**
- **Silhouette Score:** Measure of how similar objects are to their cluster vs. other clusters
- **Davies-Bouldin Index:** Ratio of intra-cluster to inter-cluster distances (lower is better)
- **Inertia:** Sum of squared distances from cluster centers

**Expected Results:**
- Optimal number of clusters determined
- Multiple clustering algorithms compared
- Cluster characteristics and profiles documented
- Customer assignments to clusters
- Business insights from cluster interpretation

---

## Model Performance Summary

### Classification (Fraud Detection)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | [Value] | [Value] | [Value] | [Value] | [Value] |
| Random Forest | [Value] | [Value] | [Value] | [Value] | [Value] |
| XGBoost | [Value] | [Value] | [Value] | [Value] | [Value] |
| **Best Model** | **[Model Name]** | **[Metrics]** | | | |

### Regression (Song Year Prediction)
| Model | MSE | RMSE | MAE | R² Score |
|-------|-----|------|-----|----------|
| Linear Regression | [Value] | [Value] | [Value] | [Value] |
| Ridge Regression | [Value] | [Value] | [Value] | [Value] |
| Lasso Regression | [Value] | [Value] | [Value] | [Value] |
| Random Forest | [Value] | [Value] | [Value] | [Value] |
| Gradient Boosting | [Value] | [Value] | [Value] | [Value] |
| XGBoost | [Value] | [Value] | [Value] | [Value] |
| **Best Model** | **[Model Name]** | **[Metrics]** | | |

### Clustering (Customer Segmentation)
| Model | Silhouette Score | Number of Clusters | Davies-Bouldin Index |
|-------|------------------|-------------------|----------------------|
| K-Means | [Value] | [Value] | [Value] |
| Hierarchical | [Value] | [Value] | [Value] |
| DBSCAN | [Value] | [Value] | [Value] |
| **Best Model** | **[Model Name]** | **[Value]** | **[Value]** |

---

## How to Navigate the Repository

### 1. **For Fraud Detection Project:**
   - Open: `notebooks/1_Fraud_Detection_Classification.ipynb`
   - Data: `data/train_transaction.csv`, `data/test_transaction.csv`
   - Results: `data/submissions/fraud_detection_submission.csv`

### 2. **For Regression Project:**
   - Open: `notebooks/2_Song_Release_Year_Regression.ipynb`
   - Data: `data/midterm-regresi-dataset.csv`
   - Results: `data/submissions/regression_submission.csv`

### 3. **For Clustering Project:**
   - Open: `notebooks/3_Customer_Clustering.ipynb`
   - Data: `data/clusteringmidterm.csv`
   - Results: `data/submissions/clustering_results.csv`

---

## Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- Libraries: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, scipy

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bagfat/midterm-machine-learning.git
   cd midterm-machine-learning
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate # On Windows 
   ```

3. **Install required libraries:**
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy
   ```

4. **Download datasets:**
   - Place `train_transaction.csv` and `test_transaction.csv` in `data/` folder
   - Place `midterm-regresi-dataset.csv` in `data/` folder
   - Place `clusteringmidterm.csv` in `data/` folder

5. **Run Jupyter Notebooks:**
   ```bash
   jupyter notebook
   ```
   Then navigate to `notebooks/` and open each notebook.

---

## Notebook Structure

Each Jupyter notebook follows this structure:

1. **Import Libraries** - Import all necessary Python packages
2. **Load and Explore Data** - Load data and perform initial exploration
3. **Data Preprocessing** - Handle missing values, scaling, encoding
4. **Feature Analysis** - Analyze feature relationships and importance
5. **Model Training** - Train multiple models
6. **Evaluation** - Evaluate model performance
7. **Comparison** - Compare all models
8. **Results** - Generate predictions or cluster assignments
9. **Conclusions** - Summarize findings and insights

---

## Key Insights

### Classification (Fraud Detection)
- Class imbalance handling is crucial for fraud detection
- Tree-based models (Random Forest, XGBoost) often outperform linear models
- ROC-AUC is a better metric than accuracy for imbalanced datasets

### Regression (Song Year Prediction)
- Audio features contain temporal information
- Tree-based models capture non-linear relationships better
- Cross-validation helps in robust model evaluation

### Clustering (Customer Segmentation)
- Multiple clustering algorithms should be tested
- Silhouette score helps in determining optimal cluster count
- Cluster profiles provide actionable business insights

---

## References

- Scikit-learn Documentation: https://scikit-learn.org/
- XGBoost Documentation: https://xgboost.readthedocs.io/
- Pandas Documentation: https://pandas.pydata.org/
- Matplotlib & Seaborn: https://matplotlib.org/, https://seaborn.pydata.org/

---

## Contact

For questions or issues, please contact:
- **Email:** bagussukses0b@gmail.com
- **GitHub:** bagfat (https://github.com/bagfat)

---

## License

This project is submitted as part of the Machine Learning course assignment.

---

**Last Updated:** 5/12/2025 10.34 AM WIB
**Version:** 1.0





# Machine Learning Midterm Projects

## 📋 Project Overview

This repository contains three comprehensive machine learning projects demonstrating different types of machine learning tasks: **Classification**, **Regression**, and **Clustering**. Each project includes data preprocessing, model training, hyperparameter tuning, and detailed evaluation.

---

## 👤 Identification

**Name:** Bagus Fatkhurrohman  
**Class:** Machine Learning
**NIM :** 1103223195
**Date:** 5-12-2025

---

## 📁 Repository Structure

```
midterm-machine-learning/
│
├── images/                                      # All visualizations
│   ├── Fraud_Detection_Classification/
│   │   └── model-comparison-fraud.png
│   │
│   ├── Song_Release_Year_Regression/
│   │   ├── feature-importance-song.png
│   │   ├── model-comparison-song.png
│   │   └── target-distribution-song.png
│   │
│   └── Customer_Clustering/
│       ├── cluster-profile-heatmap-customer.png
│       ├── determining-optimal-number-of-cluster.png
│       ├── silhouette-analysis-customer.png
│       └── visualizing-clusters-customer.png
│
├── notebooks/                                   # Jupyter notebooks
│       └── submissions/                            # Auto-generated results
│       ├── fraud_detection_submission.csv
│       ├── regression_submission.csv
│       ├── clustering_results.csv
│       └── cluster_profiles.csv
│   ├── 1_Fraud_Detection_Classification.ipynb
│   ├── 2_Song_Release_Year_Regression.ipynb
│   └── 3_Customer_Clustering.ipynb
│
└── README.md                                    # This file
```

---

## Project Descriptions

### Project 1: Fraud Detection (Classification)

**Objective:** Predict whether an online transaction is fraudulent or not.

**Task Type:** Binary Classification  
**Target Variable:** `isFraud` (0 = Not Fraud, 1 = Fraud)  
**Datasets:** `train_transaction.csv`, `test_transaction.csv`

**Workflow:**
1. Load and explore transaction data
2. Handle missing values and class imbalance
3. Feature preprocessing and scaling
4. Train multiple classification models:
   - Logistic Regression
   - Random Forest Classifier
   - XGBoost Classifier
5. Evaluate using appropriate metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
6. Compare models and select the best performer
7. Generate predictions on test data

**Key Metrics:**
- **Accuracy:** Overall correctness of predictions
- **Precision:** Proportion of predicted frauds that are actually frauds
- **Recall:** Proportion of actual frauds caught by the model
- **F1-Score:** Harmonic mean of Precision and Recall
- **ROC-AUC:** Area under the ROC curve for classification threshold evaluation

**Expected Results:**
- Multiple models trained and compared
- Best model identified based on ROC-AUC score
- Submission file with fraud probability predictions

---

### Project 2: Song Release Year Prediction (Regression)

**Objective:** Predict the release year of a song based on audio features.

**Task Type:** Regression (Continuous Value Prediction)  
**Target Variable:** Release Year (numeric value)  
**Dataset:** `midterm-regresi-dataset.csv`

**Workflow:**
1. Load and explore audio feature data
2. Handle missing values and outliers
3. Analyze feature correlations with target
4. Feature preprocessing and scaling
5. Train multiple regression models:
   - Linear Regression
   - Ridge Regression
   - Lasso Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - XGBoost Regressor
6. Evaluate using regression metrics (MSE, RMSE, MAE, R²)
7. Compare models and select the best performer
8. Analyze feature importance

**Key Metrics:**
- **MSE (Mean Squared Error):** Average squared difference between actual and predicted values
- **RMSE (Root Mean Squared Error):** Square root of MSE, in same units as target
- **MAE (Mean Absolute Error):** Average absolute difference between actual and predicted values
- **R² Score:** Proportion of variance explained by the model (0-1, higher is better)

**Expected Results:**
- Multiple regression models trained and evaluated
- Best model identified based on R² score
- Visualization of actual vs. predicted values
- Feature importance analysis (for tree-based models)

---

### Project 3: Customer Clustering (Unsupervised Learning)

**Objective:** Segment customers based on credit card usage and payment behavior.

**Task Type:** Unsupervised Learning (Clustering)  
**Target Variable:** None (Unsupervised)  
**Dataset:** `clusteringmidterm.csv`

**Workflow:**
1. Load and explore customer behavior data
2. Handle missing values and outliers
3. Feature preprocessing and scaling
4. Determine optimal number of clusters using:
   - Elbow Method
   - Silhouette Score
   - Davies-Bouldin Index
5. Train multiple clustering algorithms:
   - K-Means Clustering
   - Hierarchical Clustering
   - DBSCAN
6. Evaluate clusters using Silhouette Score
7. Interpret cluster characteristics
8. Visualize clusters using PCA reduction

**Key Metrics:**
- **Silhouette Score:** Measure of how similar objects are to their cluster vs. other clusters
- **Davies-Bouldin Index:** Ratio of intra-cluster to inter-cluster distances (lower is better)
- **Inertia:** Sum of squared distances from cluster centers

**Expected Results:**
- Optimal number of clusters determined
- Multiple clustering algorithms compared
- Cluster characteristics and profiles documented
- Customer assignments to clusters
- Business insights from cluster interpretation

---

## 📊 Model Performance Summary

### Classification (Fraud Detection)

**Metrics Table:**
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | [Value] | [Value] | [Value] | [Value] | [Value] |
| Random Forest | [Value] | [Value] | [Value] | [Value] | [Value] |
| XGBoost | [Value] | [Value] | [Value] | [Value] | [Value] |
| **Best Model** | **[Model Name]** | **[Metrics]** | | | |

**Performance Visualization:**
![Classification Model Comparison](path/to/classification_comparison.png)

**ROC Curves:**
![ROC Curves](path/to/roc_curves.png)

---

### Regression (Song Year Prediction)

**Metrics Table:**
| Model | MSE | RMSE | MAE | R² Score |
|-------|-----|------|-----|----------|
| Linear Regression | [Value] | [Value] | [Value] | [Value] |
| Ridge Regression | [Value] | [Value] | [Value] | [Value] |
| Lasso Regression | [Value] | [Value] | [Value] | [Value] |
| Random Forest | [Value] | [Value] | [Value] | [Value] |
| Gradient Boosting | [Value] | [Value] | [Value] | [Value] |
| XGBoost | [Value] | [Value] | [Value] | [Value] |
| **Best Model** | **[Model Name]** | **[Metrics]** | | |

**Performance Visualization:**
![Regression Model Comparison](path/to/regression_comparison.png)

**Actual vs Predicted:**
![Actual vs Predicted](path/to/actual_vs_predicted.png)

---

### Clustering (Customer Segmentation)

**Metrics Table:**
| Model | Silhouette Score | Number of Clusters | Davies-Bouldin Index |
|-------|------------------|-------------------|----------------------|
| K-Means | [Value] | [Value] | [Value] |
| Hierarchical | [Value] | [Value] | [Value] |
| DBSCAN | [Value] | [Value] | [Value] |
| **Best Model** | **[Model Name]** | **[Value]** | **[Value]** |

**Elbow Method & Silhouette Scores:**
![Elbow Method](path/to/elbow_method.png)

**Cluster Visualization (PCA):**
![Cluster Visualization](path/to/cluster_visualization.png)

**Cluster Profiles Heatmap:**
![Cluster Profiles](path/to/cluster_profiles_heatmap.png)

---

## 🗺️ How to Navigate the Repository

### Quick Links to Each Project

#### 1. **Fraud Detection (Classification)**
- 📓 **Notebook:** `notebooks/1_Fraud_Detection_Classification.ipynb`
- 📊 **Visualizations:** `images/Fraud_Detection_Classification/`
- 📥 **Data:** `dataset/train_transaction.csv`, `dataset/test_transaction.csv`
- 📤 **Results:** `dataset/submissions/fraud_detection_submission.csv`

#### 2. **Song Release Year (Regression)**
- 📓 **Notebook:** `notebooks/2_Song_Release_Year_Regression.ipynb`
- 📊 **Visualizations:** `images/Song_Release_Year_Regression/`
- 📥 **Data:** `dataset/midterm-regresi-dataset.csv`
- 📤 **Results:** `dataset/submissions/regression_submission.csv`

#### 3. **Customer Clustering (Unsupervised Learning)**
- 📓 **Notebook:** `notebooks/3_Customer_Clustering.ipynb`
- 📊 **Visualizations:** `images/Customer_Clustering/`
- 📥 **Data:** `dataset/clusteringmidterm.csv`
- 📤 **Results:** `dataset/submissions/clustering_results.csv`, `dataset/submissions/cluster_profiles.csv`

### Recommended Reading Order:
1. Start with **README.md** (this file)
2. Review **Model Performance Summary** below
3. Open notebooks in order: 1 → 2 → 3
4. Check visualizations in `images/` folder

---

## 📥 Dataset

⚠️ **IMPORTANT:** Datasets are not included in this repository due to size constraints. You must download them separately and place them in the `dataset/` folder.

### Dataset Download Links

Please download the following datasets from Google Drive:

| Dataset | File Size | Location |
|---------|-----------|----------|
| **train_transaction.csv** | ~100 MB | [Download](https://drive.google.com/YOUR_FILE_ID) |
| **test_transaction.csv** | ~50 MB | [Download](https://drive.google.com/YOUR_FILE_ID) |
| **midterm-regresi-dataset.csv** | ~30 MB | [Download](https://drive.google.com/YOUR_FILE_ID) |
| **clusteringmidterm.csv** | ~20 MB | [Download](https://drive.google.com/YOUR_FILE_ID) |

### Setup Instructions:

1. **Download all CSV files** from the links above
2. **Create `dataset/` folder** (if not exists):
   ```bash
   mkdir dataset
   mkdir dataset/submissions
   ```
3. **Place CSV files** in the `dataset/` folder:
   ```
   dataset/
   ├── train_transaction.csv
   ├── test_transaction.csv
   ├── midterm-regresi-dataset.csv
   └── clusteringmidterm.csv
   ```
4. **Proceed with running notebooks** (see Installation section below)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- Libraries: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, scipy
- Downloaded datasets (see section above)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/[your-username]/midterm-machine-learning.git
   cd midterm-machine-learning
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required libraries:**
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy
   ```

4. **Download datasets:**
   - Place `train_transaction.csv` and `test_transaction.csv` in `data/` folder
   - Place `midterm-regresi-dataset.csv` in `data/` folder
   - Place `clusteringmidterm.csv` in `data/` folder

5. **Run Jupyter Notebooks:**
   ```bash
   jupyter notebook
   ```
   Then navigate to `notebooks/` and open each notebook.

---

## 📝 Notebook Structure

Each Jupyter notebook follows this structure:

1. **Import Libraries** - Import all necessary Python packages
2. **Load and Explore Data** - Load data and perform initial exploration
3. **Data Preprocessing** - Handle missing values, scaling, encoding
4. **Feature Analysis** - Analyze feature relationships and importance
5. **Model Training** - Train multiple models
6. **Evaluation** - Evaluate model performance
7. **Comparison** - Compare all models
8. **Results** - Generate predictions or cluster assignments
9. **Conclusions** - Summarize findings and insights

---

## 💡 Key Insights

### Classification (Fraud Detection)
- Class imbalance handling is crucial for fraud detection
- Tree-based models (Random Forest, XGBoost) often outperform linear models
- ROC-AUC is a better metric than accuracy for imbalanced datasets

### Regression (Song Year Prediction)
- Audio features contain temporal information
- Tree-based models capture non-linear relationships better
- Cross-validation helps in robust model evaluation

### Clustering (Customer Segmentation)
- Multiple clustering algorithms should be tested
- Silhouette score helps in determining optimal cluster count
- Cluster profiles provide actionable business insights

---

## 📚 References

- Scikit-learn Documentation: https://scikit-learn.org/
- XGBoost Documentation: https://xgboost.readthedocs.io/
- Pandas Documentation: https://pandas.pydata.org/
- Matplotlib & Seaborn: https://matplotlib.org/, https://seaborn.pydata.org/

---

## 📧 Contact

For questions or issues, please contact:
- **Email:** [your.email@university.edu]
- **GitHub:** [@your-github-username](https://github.com/your-username)

---

## 📄 License

This project is submitted as part of the Machine Learning course assignment.

---

**Last Updated:** [Date]  
**Version:** 1.0
