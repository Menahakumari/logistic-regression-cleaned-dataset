# logistic-regression-cleaned-dataset
# 🧠 Logistic Regression on Cleaned Dataset

A complete end-to-end machine learning pipeline for **binary classification** using **Logistic Regression**. This project demonstrates how to handle real-world data with missing values and categorical features, train a logistic model, and evaluate it using key metrics like precision, recall, and ROC-AUC.

---

## 📁 Dataset

- The dataset is assumed to be in CSV format.
- Dateset-kaggle:(https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- Includes a mix of **numerical** and **categorical** features.
- Contains **missing values** handled via:
  - Mean imputation for numeric columns
  - Mode imputation for categorical columns

---

## 🔧 Steps Implemented

1. Load and inspect the dataset
2. Clean missing values
3. Encode categorical variables using one-hot encoding
4. Split the data into training and testing sets
5. Standardize the features using `StandardScaler`
6. Train a Logistic Regression model
7. Evaluate using:
   - Confusion Matrix
   - Precision & Recall
   - ROC Curve & AUC score
8. Tune threshold for classification
9. Explain sigmoid function

---

## 📊 Output Metrics

- Confusion Matrix
- Precision
- Recall
- ROC-AUC Score
- ROC Curve Plot
- Confusion matrix with custom threshold (e.g., 0.4)

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/your-username/logistic-regression-cleaned-dataset.git
cd logistic-regression-cleaned-dataset

# Install dependencies
pip install -r requirements.txt

# Run the pipeline script
python logistic_regression_pipeline.py
🧪 Dependencies
pandas

numpy

scikit-learn

matplotlib

Install them with:

bash
Copy
Edit
pip install -r requirements.txt
🧾 License
This project is licensed under the MIT License.

yaml
Copy
Edit

---

### 📄 `requirements.txt`

Create a file named `requirements.txt`:

```txt
pandas
numpy
scikit-learn
matplotlib
📄 logistic_regression_pipeline.py
Use the full working Python code I gave you earlier (the one that handled missing values, encoding, training, evaluation, ROC curve, and threshold tuning).

You can save it in your repo as:

bash
Copy
Edit
logistic_regression_pipeline.py
