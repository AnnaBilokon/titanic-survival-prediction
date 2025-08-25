# Titanic Survival Prediction 🚢

This project uses Logistic Regression to predict whether a passenger survived the Titanic disaster.

## 📊 Dataset

- Source: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)
- Features: Age, Sex, PClass, LogFare, FamilySize, hasCabin, etc.
- Target: Survived (0 = No, 1 = Yes)

## ⚙️ Steps

1. Data cleaning & preprocessing (handled missing values, encoded categorical variables).
2. Model training with Logistic Regression (scikit-learn).
3. Evaluation using Accuracy
4. Interpreted feature importance (Sex, Pclass, Age were key predictors).

## 📈 Results

This project analyzes the Titanic dataset to predict passenger survival using machine learning. Key factors were gender, class, and age, reflecting the “women and children first” evacuation rule. After preprocessing and feature engineering, baseline models achieved ~0.78–0.82 validation accuracy.

## 🚀 How to Run

```bash
pip install -r requirements.txt
jupyter /titanic_project.ipynb
```
