# 💳 Credit Card Fraud Detection using Python (Advanced)

This project is an **advanced machine learning solution** for detecting fraudulent credit card transactions using Python. It uses a real-world dataset, handles **imbalanced data** with **SMOTE**, and implements **XGBoost** for high-performance fraud classification. A **Streamlit web interface** is also provided for interactive predictions.

---

## 📂 Project Structure

```

credit-card-fraud-detection/
├── creditcard.csv                 # Dataset
├── CreditCardFraudDetection.ipynb # Jupyter Notebook
├── app.py                         # Streamlit App
├── models\best_model.pkl          # Trained XGBoost Model
└── requirements.txt               # Dependencies

````

---

## 📊 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Contains **284,807 transactions**, including **492 frauds**
- Features: `Time`, `Amount`, anonymized PCA features (`V1` to `V28`), and `Class` (fraud=1, normal=0)

---

## 🚀 Features

- Full **exploratory data analysis (EDA)**
- **Feature scaling** using `StandardScaler`
- **Imbalanced data handling** with **SMOTE**
- Model training using **XGBoost**
- Performance evaluation with **confusion matrix**, **classification report**, **ROC-AUC**
- **Model saving** with `joblib`
- **Streamlit UI** for real-time fraud predictions

---

## 🛠 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
````

2. Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## 🧪 Run the Jupyter Notebook

Open and execute the notebook to:

* Load and preprocess the dataset
* Handle class imbalance
* Train the model using XGBoost
* Evaluate the model
* Save the model to `best_model.pkl`

```bash
jupyter notebook CreditCardFraudDetection.ipynb
```

---

## 🌐 Run the Streamlit App

After training and saving the model:

```bash
streamlit run app.py
```

This will launch an interactive interface where you can:

* Upload transaction data
* Get predictions on whether a transaction is **fraudulent or not**

---


---

## 🧠 Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* Imbalanced-learn (SMOTE)
* XGBoost
* Streamlit
* Jupyter Notebook

---

## 🙋‍♂️ Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you would like to change.

---

## 📫 Contact

Created by Dhruv Pratap Singh - [Github](https://github.com/iamdpsingh), [LinkedIn](https://www.linkedin.com/in/dhruv-pratap-singh-088442253/)  — feel free to reach out!

```

