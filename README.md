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

| **Model** | **Accuracy** | **Precision** | **Recall** | **Weighted F1** |
|---|---:|---:|---:|---:|
| **Linear SVM** | 0.9548 | 0.9553 | 0.9548 | **0.9548** |
| Logistic Regression | 0.9453 | 0.9463 | 0.9453 | 0.9452 |
| Random Forest | 0.9354 | 0.9371 | 0.9354 | 0.9351 |

### 🏆 Best Model

**Linear SVM**

- Accuracy: **95.48%**
- Precision: **95.53%**
- Recall: **95.48%**
- Weighted F1: **95.48%**

The **Linear SVM** achieved the highest Weighted F1-score among the three evaluated models and was selected as the Best Model.

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
├── step01_processed_data.csv
│
└── model_files/
    ├── best_model.pkl
    └── tfidf_vectorizer.pkl


