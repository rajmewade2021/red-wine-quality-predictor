import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

st.title("Wine Quality Prediction App 🍷")

st.write("Enter the chemical properties of wine to predict its quality.")

# Feature Inputs
fixed_acidity = st.number_input("Fixed Acidity")
volatile_acidity = st.number_input("Volatile Acidity")
citric_acid = st.number_input("Citric Acid")
residual_sugar = st.number_input("Residual Sugar")
chlorides = st.number_input("Chlorides")
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide")
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide")
density = st.number_input("Density")
ph = st.number_input("pH")
sulphates = st.number_input("Sulphates")
alcohol = st.number_input("Alcohol")

# Prediction button
if st.button("Predict Wine Quality 🚀"):

    features = np.array([[fixed_acidity,
                          volatile_acidity,
                          citric_acid,
                          residual_sugar,
                          chlorides,
                          free_sulfur_dioxide,
                          total_sulfur_dioxide,
                          density,
                          ph,
                          sulphates,
                          alcohol]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success("🍷 Good Quality Wine")
    else:
        st.error("⚠️ Bad Quality Wine")

    # ---------- FOOTER ----------

st.markdown(
    """
    <style>
    /* Add bottom padding so footer doesn't overlap chat input */
    .block-container {
        padding-bottom: 80px;
    }

    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(0,0,0,0);
        color: #B3B3B3;
        text-align: center;
        font-size: 16px;
        padding: 10px;
        z-index: 100;
    }

    /* Blue links */
    .footer a {
        color: #1DA1F2;   /* Blue */
        text-decoration: none;
        margin: 0 8px;
        font-weight: 500;
    }

    .footer a:hover {
        color: #0A66C2;   /* Darker blue on hover */
        text-decoration: underline;
    }
    </style>

    <div class="footer">
        © 2025 <b>Developed by Shreeyansh Asati</b> |
        <a href="https://www.linkedin.com/in/shreeyansh-asati-18shreey/" target="_blank">
            🔗 LinkedIn
        </a> |
        <a href="https://github.com/SHREEYANSHGIT" target="_blank">
            💻 GitHub
        </a>
    </div>
    """,
    unsafe_allow_html=True
)


