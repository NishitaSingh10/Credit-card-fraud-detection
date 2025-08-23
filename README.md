# Credit-card-fraud-detection
Machine Learning models for credit card fraud detection with SMOTE &amp; tuning.

# Credit Card Fraud Detection

## Project Overview
Credit card fraud poses a major challenge in the financial industry, where fraudulent transactions are rare but extremely costly. This project applies machine learning models to detect fraudulent transactions in an imbalanced dataset. Since fraud cases make up only a tiny fraction of the total, techniques like SMOTE (Synthetic Minority Oversampling Technique) were used to balance the training data.

We compared multiple models - Logistic Regression, Decision Tree, Random Forest, and Naive Bayes - using cross-validation with F1 score as the key metric (since accuracy alone is misleading in imbalanced data). Among these, Random Forest consistently outperformed the others, achieving the best mean F1 score.

To further improve detection, we performed threshold tuning on predicted probabilities instead of relying on the default 0.5 cutoff. This allows a better balance between precision (minimizing false alarms) and recall (catching more frauds), which is critical in real-world fraud detection.

Finally, we evaluated the chosen model with Confusion Matrix, ROC Curve, and Precision-Recall Curve, ensuring robustness and interpretability of the results.



## Dataset
- The dataset contains **284,807 transactions** with only **492 fraud cases** (~0.17%).  
- Features are anonymized (V1–V28) with additional fields: `Time`, `Amount`, and `Class` (target).  
- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)



## Methodology

### 1. Data Preprocessing
- Split into **training** and **untouched test set**.  
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** on the training set to handle class imbalance.  
- Standardized numerical features for better model performance.  

### 2. Modeling
Trained and compared the following models using **5-fold cross-validation**:
- **Logistic Regression**  
- **Decision Tree**  
- **Random Forest**  
- **Naive Bayes**  

### 3. Evaluation Metrics
- Confusion Matrix  
- ROC Curve (AUC)  
- Precision-Recall Curve (PR-AUC)  
- Precision, Recall, and F1-Score  



## Results

### Cross-Validation F1 Scores

| Model              | Mean F1 Score |
|---------------------|---------------|
| Logistic Regression | **0.11** |
| Decision Tree       | **0.51** |
| Random Forest       | **0.84** |
| Naive Bayes         | **0.11** |

 **Random Forest** clearly outperformed other models and was chosen for the final evaluation.



### Final Test Results (Random Forest)

On the **untouched test set** (with 95 fraud cases):  

| Metric | Score |
|--------|-------|
| Precision | ~0.78 |
| Recall | ~0.78 |
| F1-Score | ~0.78 |
| ROC-AUC | ~0.98 |
| PR-AUC (Average Precision) | ~0.78 |



## Visualizations
The notebook includes:
- Confusion Matrix Heatmap (Random Forest)  
- ROC Curve with AUC  
- Precision-Recall Curve  
- Threshold tuning to balance precision and recall  



## Next Steps
- Implement **cost-sensitive learning** to penalize fraud misclassification more heavily.  
- Explore **stacking/ensemble methods** for improved recall.  
- Deploy model with **real-time streaming data** and **periodic retraining**.  



## References
- [Kaggle Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- Scikit-learn, Seaborn, Matplotlib  


