import os
import re
import pickle
import string

import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ------------------------------------------------------------
# Project 05 - Clinical Trial Disease Category Classification
# Streamlit App - Update 2
# ------------------------------------------------------------

DATA_PATH = "clinical_trial_disease_dataset.csv"
MODEL_PATH = os.path.join("model_files", "best_model.pkl")
VECTORIZER_PATH = os.path.join("model_files", "tfidf_vectorizer.pkl")

DISPLAY_COLUMNS = [
    "nct_id",
    "title",
    "conditions",
    "interventions",
    "overall_status",
    "phase",
]


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Clinical Trial Disease Classifier",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Clinical Trial Disease Category Classification")
st.write("Enter a clinical trial brief summary to predict the source condition.")


# ------------------------------------------------------------
# NLTK resources
# ------------------------------------------------------------

@st.cache_resource
def load_nltk_resources():
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    for resource_path, package in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package, quiet=True)

    return set(stopwords.words("english")), WordNetLemmatizer()


STOP_WORDS, LEMMATIZER = load_nltk_resources()


# ------------------------------------------------------------
# Same preprocessing used in Step 01
# ------------------------------------------------------------

def preprocess_text(text):
    """
    Apply the same text preprocessing used in Step 01:
    lowercase -> punctuation removal -> special-character cleanup
    -> tokenization -> stop-word removal -> lemmatization
    """
    text = "" if pd.isna(text) else str(text)

    text = text.lower()

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [word for word in tokens if word not in STOP_WORDS]
    tokens = [LEMMATIZER.lemmatize(word) for word in tokens]

    return " ".join(tokens)


# ------------------------------------------------------------
# Load trained model and TF-IDF vectorizer
# ------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    with open(VECTORIZER_PATH, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return model, vectorizer


# ------------------------------------------------------------
# Load internal clinical-trial CSV
# ------------------------------------------------------------

@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "brief_summary",
        "source_condition_query",
        *DISPLAY_COLUMNS,
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in CSV: {missing_columns}"
        )

    df["brief_summary"] = df["brief_summary"].fillna("")
    df["source_condition_query"] = (
        df["source_condition_query"].fillna("Unknown")
    )

    # Create the normalized text once.
    # This is used only to identify an exact internal sentence.
    df["brief_summary_normalized"] = df["brief_summary"].apply(
        preprocess_text
    )

    return df


# ------------------------------------------------------------
# Load files
# ------------------------------------------------------------

try:
    model, tfidf_vectorizer = load_artifacts()
    df = load_dataset()
except Exception as error:
    st.error(f"Could not load required files: {error}")
    st.stop()


# ------------------------------------------------------------
# Input
# ------------------------------------------------------------

brief_summary = st.text_area(
    "Enter Brief Summary",
    height=180,
    placeholder=(
        "Example: This clinical trial evaluates a new treatment "
        "for patients with breast cancer..."
    ),
)


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

if st.button("Predict Source Condition", type="primary"):

    if not brief_summary.strip():
        st.warning("Please enter a Brief Summary.")

    else:
        # Step 1: Apply the same preprocessing used during training.
        cleaned_text = preprocess_text(brief_summary)

        # Step 2: Convert the new text using the saved TF-IDF vectorizer.
        input_vector = tfidf_vectorizer.transform([cleaned_text])

        # Step 3: Predict the source condition using the saved Best Model.
        prediction = model.predict(input_vector)[0]

        st.success("Prediction completed.")

        st.subheader("Predicted Source Condition")
        st.write(f"**{prediction}**")

        # ----------------------------------------------------
        # Internal vs External sentence detection
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # We do NOT use the predicted condition to retrieve
        # trial details.
        #
        # We only check whether the normalized input sentence
        # exactly matches a normalized Brief Summary in the CSV.
        #
        # No cosine similarity.
        # No top-similar trial retrieval.
        # ----------------------------------------------------

        matches = df[
            df["brief_summary_normalized"] == cleaned_text
        ].copy()

        st.subheader("Clinical Trial Details")

        if not matches.empty:

            st.caption(
                "Internal sentence detected — details are taken "
                "from the matching CSV record."
            )

            result = matches[DISPLAY_COLUMNS].copy()

            # Keep the requested six columns in the requested order.
            result = result[
                [
                    "nct_id",
                    "title",
                    "conditions",
                    "interventions",
                    "overall_status",
                    "phase",
                ]
            ]

            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.caption(
                "External sentence detected — the sentence was not "
                "found in the internal CSV. Only the model prediction "
                "is available."
            )

            external_result = pd.DataFrame(
                [{
                    "nct_id": "--",
                    "title": "--",
                    "conditions": "--",
                    "interventions": "--",
                    "overall_status": "--",
                    "phase": "--",
                }]
            )

            st.dataframe(
                external_result,
                use_container_width=True,
                hide_index=True,
            )
