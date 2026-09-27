# 🧬 Clinical Trial Disease Classifier

An end-to-end Natural Language Processing (NLP) and Machine Learning
project that classifies clinical trial brief summaries into disease
categories.

The project uses clinical trial data to build a text classification
system that can process both existing clinical trial summaries and
new external summaries.

## 📊 Dataset

- Total Records: 60,337
- Total Columns: 16
- Text Feature: `brief_summary`
- Target: `source_condition_query`
- Data Source: Clinical trial dataset

## 🧠 Machine Learning Approach

The project follows a complete machine learning workflow:

1. Data Loading
2. Data Cleaning
3. Basic Exploratory Data Analysis
4. Text Preprocessing
5. TF-IDF Feature Extraction
6. Logistic Regression Baseline
7. Model Comparison
8. Model Evaluation
9. Best Model Selection
10. Model & Vectorizer Saving
11. Streamlit Deployment

## 🤖 Models Evaluated

The following classification models are compared:

- Logistic Regression
- Random Forest
- Linear SVM

The final model is selected based on the **highest Weighted F1-score**.

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | Weighted F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | YOUR SCORE | YOUR SCORE | YOUR SCORE | YOUR SCORE |
| Random Forest | YOUR SCORE | YOUR SCORE | YOUR SCORE | YOUR SCORE |
| Linear SVM | YOUR SCORE | YOUR SCORE | YOUR SCORE | YOUR SCORE |

### Best Model

The model with the highest Weighted F1-score is saved as:

`model_files/best_model.pkl`

The trained TF-IDF vectorizer is saved as:

`model_files/tfidf_vectorizer.pkl`

## 🔤 NLP Processing

The `brief_summary` text goes through:

- Lowercasing
- Punctuation removal
- Special-character cleanup
- Tokenization
- Stop-word removal
- Lemmatization
- TF-IDF vectorization

## 🚀 Streamlit Application

The Streamlit application accepts a Brief Summary and predicts the
Source Condition.

### Internal Clinical Trial Summary

If the entered normalized Brief Summary exists in the internal dataset,
the application displays the corresponding:

- NCT ID
- Title
- Conditions
- Interventions
- Status
- Phase

### External Clinical Trial Summary

If the entered summary is not present in the internal dataset, the
application displays:

- Predicted Source Condition
- NCT ID: `--`
- Title: `--`
- Conditions: `--`
- Interventions: `--`
- Status: `--`
- Phase: `--`

This prevents unrelated clinical-trial records from being displayed
for new external text.

## 📁 Project Structure

```text
Clinical-Trial-Disease-Classifier/
│
├── P05_Step_01_TF-IDF_and_Logistic_Regression.ipynb
├── P05_Step_02_Model_Comparison_and_Saving.ipynb
├── app.py
├── clinical_trial_disease_dataset.csv
├── requirements.txt
├── README.md
│
└── model_files/
    ├── best_model.pkl
    └── tfidf_vectorizer.pkl


