Titanic Survival Prediction 🚢

This project predicts whether a passenger survived the Titanic disaster using exploratory data analysis (EDA), feature engineering, and multiple machine learning models. It also includes a Streamlit web app so anyone can try predictions live.

📊 Dataset

Source: Kaggle Titanic Dataset

Core features used: Age, Sex, Pclass, LogFare, FamilySize, HasCabin, plus one-hot dummies for Pclass (Pclass_1/2/3)

Target: Survived (0 = No, 1 = Yes)

⚙️ Steps

EDA to understand class imbalance and key drivers of survival.

Data cleaning & preprocessing

Imputed Age via Title-based medians, Embarked via mode, and Fare via median.

Engineered features: FamilySize, HasCabin, Title, LogFare.

One-hot encoded Pclass.

Modeling with a compact pipeline: Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting.

Evaluation with Stratified K-Fold cross-validation (Accuracy, ROC AUC, Precision, Recall, F1).

Deployment of the selected model via Streamlit.


🔎 EDA Highlights

Survival is imbalanced (~62% died, ~38% survived).

Gender and class are dominant signals: females and 1st-class passengers had much higher survival rates.

Age matters (children fared better); Fare (log-scaled) adds signal.

🤖 Models Compared

Implemented in pipeline.py with simple constructors and a shared evaluate_models helper:

Logistic Regression (baseline, interpretable; scaled)

K-Nearest Neighbors (KNN) (distance-based; scaled)

Decision Tree (interpretable splits)

Random Forest (bagged trees; robust, strong)

Gradient Boosting (sequential trees; high accuracy)

Cross-Validation Results (5-fold)
Model	Accuracy (mean)	ROC AUC (mean)	F1 (mean)
Random Forest	0.829	0.882	0.766
Gradient Boosting	0.838	0.872	0.776
Logistic Reg.	0.804	0.860	0.737
KNN (k=5)	0.807	0.860	0.738
Decision Tree	0.806	0.852	0.743

Takeaway: Ensemble models performed best. Gradient Boosting led in accuracy/F1, while Random Forest achieved the highest ROC AUC.

✅ Conclusion

In this project, I explored the Titanic dataset to predict passenger survival.

EDA confirmed class imbalance and highlighted Sex and Pclass as key drivers, with Age also important.

Missing data were handled with sensible imputations; engineered features (FamilySize, HasCabin, LogFare, Title) improved signal.

Among five models, ensemble methods (Random Forest, Gradient Boosting) performed best, capturing non-linear relationships in the data.

For deployment, I selected the top model and served it via a Streamlit app for interactive predictions.


🧩 Project Structure
.
├─ data/
│  ├─ train.csv
│  └─ test.csv
├─ features.py          # feature engineering utilities
├─ pipeline.py          # model builders + CV evaluation
├─ app.py               # Streamlit app (live demo)
├─ titanic_project.ipynb
├─ titanic_model.pkl    # saved trained model (or artifact dict)
├─ requirements.txt
└─ images/
   ├─ eda_gender_class.png
   ├─ leaderboard.png
   └─ app_screenshot.png

🧪 Reproduce / Run Locally
pip install -r requirements.txt
jupyter lab titanic_project.ipynb   


🌐 Streamlit App (Live Demo)

https://titanic-prediction-project.streamlit.app/



🖥️ Run Locally
Prerequisites

Python 3.10 or 3.11

Git

(Optional) venv or conda for an isolated environment

1) Clone the repo
git clone https://github.com/<your-username>/titanic-ml.git
cd titanic-ml

2) Create & activate a virtual env

macOS / Linux

python3 -m venv .venv
source .venv/bin/activate


Windows (PowerShell)

py -3 -m venv .venv
.venv\Scripts\Activate.ps1

3) Install dependencies
pip install -r requirements.txt

4) Get the data

Download the Kaggle Titanic dataset and place files like this:

data/
├─ train.csv
└─ test.csv


Dataset: https://www.kaggle.com/c/titanic

(You can also rename or add paths in the notebook if needed.)

5) (Optional) Explore & train in Jupyter

Open the notebook for EDA and model training:

jupyter lab titanic_project.ipynb
# or
jupyter notebook titanic_project.ipynb


At the end of the notebook, save the trained model as an artifact dict (recommended):

import joblib
FEATURES = ["Sex","Age","LogFare","FamilySize","HasCabin","Pclass_1","Pclass_2","Pclass_3"]
artifact = {"model": best_model, "features": FEATURES}  # best_model fitted on all data
joblib.dump(artifact, "titanic_model.pkl")


You should now have:

titanic_model.pkl

6) Run the Streamlit app
streamlit run app.py


Then open the local URL Streamlit prints (usually http://localhost:8501
).