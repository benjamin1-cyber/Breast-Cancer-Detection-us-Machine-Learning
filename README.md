---

<div align="center">

# 🩺 Breast Cancer Detection Using Machine Learning

🚀 *A supervised machine learning approach for early breast cancer diagnosis*

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

</div>

---

## 📖 Overview

Breast cancer is one of the most prevalent cancers globally, and **early detection is critical** for improving survival rates.
This project applies **machine learning techniques** to classify breast tumors as **benign** or **malignant** using diagnostic features extracted from medical images.

✔ Model optimization using **Grid Search**
✔ Robust evaluation with **K-Fold Cross Validation**
✔ Emphasis on **clinical relevance** (low false negatives)

---

## 📂 Repository Structure

```bash
📦 Breast-Cancer-Detection-ML
 ┣ 📓 Breast_Cancer_Detection_using_ML.ipynb
 ┗ 📄 README.md
```

---

## 📊 Dataset

### 🔹 Dataset Name

**Breast Cancer Wisconsin (Diagnostic) Dataset**

### 🔹 Source

* 🏛 **UCI Machine Learning Repository**
  [https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
* 📦 Also available via:

  ```python
  sklearn.datasets.load_breast_cancer
  ```

### 🔹 Dataset Summary

| Attribute      | Value                     |
| -------------- | ------------------------- |
| Total Samples  | **569**                   |
| Features       | **30 numeric features**   |
| Classes        | Malignant (0), Benign (1) |
| Missing Values | ❌ None                    |

### 🔹 Feature Categories

The dataset includes measurements of:

* Mean tumor characteristics
* Standard error values
* Worst (largest) values

**Examples:**

* Mean Radius
* Mean Texture
* Mean Area
* Worst Concave Points
* Worst Symmetry

---

## ⚙️ Methodology

### 🧠 Workflow

1. **Data Loading & Exploration**
2. **Feature–Target Separation**
3. **Train–Test Split**

   * Training set: **455 samples**
   * Test set: **114 samples**
4. **Model Selection**

   * Random Forest Classifier
5. **Hyperparameter Tuning**

   * `GridSearchCV`
6. **Model Validation**

   * 5-Fold Cross Validation
7. **Final Evaluation**

   * Accuracy, Precision, Recall, F1-score

---

## 🔧 Model Optimization Results

### 🏆 Best Hyperparameters

```text
n_estimators: 200
max_depth: None
min_samples_split: 2
min_samples_leaf: 1
```

### 📈 Cross-Validation Performance

* CV Scores:
  `[0.9341, 0.9670, 0.9670, 0.9890, 0.9341]`
* **Mean CV Accuracy:** **95.82%**

---

## 📊 Final Model Evaluation

### ✅ Test Set Accuracy

🎯 **95.61%**

### 📄 Classification Report

```text
              precision    recall  f1-score   support

Malignant (0)     0.95      0.93      0.94        42
Benign (1)        0.96      0.97      0.97        72

Accuracy                              0.96       114
Macro Avg         0.96      0.95      0.95       114
Weighted Avg      0.96      0.96      0.96       114
```

### 🧮 Confusion Matrix

```text
[[39  3]
 [ 2 70]]
```

📌 **Key Insight:**
The model achieves **high recall for malignant cases**, minimizing false negatives — a critical requirement in medical diagnostics.

---

## 🛠️ Tech Stack

| Tool             | Purpose                 |
| ---------------- | ----------------------- |
| Python           | Core programming        |
| Jupyter Notebook | Development environment |
| NumPy            | Numerical computation   |
| Pandas           | Data manipulation       |
| Scikit-learn     | Machine learning        |



---

## 🔮 Future Enhancements

✨ Potential improvements include:

* Comparing with **SVM, Logistic Regression, XGBoost**
* Feature importance and **SHAP analysis**
* Web deployment using **Streamlit or Flask**
* Integration with real-world clinical datasets

---

## 📜 License

📚 This project is intended for **educational and research purposes only**.



