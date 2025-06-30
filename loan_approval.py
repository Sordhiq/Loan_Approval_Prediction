import pandas as pd
import numpy as np
import streamlit as st
import pickle

# -----------------------------------------
# Page Configuration
# -----------------------------------------
st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="💰",
    layout="centered"
)

# -----------------------------------------
# Load Trained Model
# -----------------------------------------
@st.cache_resource
def load_model():
    try:
        with open('loan_prediction_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("The model file does not exist. Please check the filename and location.")
        return None

model = load_model()

# -----------------------------------------
# Prediction Function
# -----------------------------------------
def predict_loan_status(Age, Rewards_Points, Loan_Amount, Interest_Rate, Account_Balance, Credit_Card_Balance, Transaction_Amount, Credit_Limit):

    # Feature Engineering
    Spending_Rate = Transaction_Amount / (Account_Balance + 1e-5)
    Loan_to_Credit_Ratio = Loan_Amount / (Credit_Limit + 1e-5)
    Credit_Utilization = Credit_Card_Balance / (Credit_Limit + 1e-5)

    # Combine into feature array
    features = np.array([[Age, Rewards_Points, Loan_Amount, Interest_Rate, Account_Balance, Credit_Card_Balance, Transaction_Amount, Spending_Rate, Credit_Limit, Loan_to_Credit_Ratio, Credit_Utilization]])
    
    prediction = model.predict(features)
    return prediction

# -----------------------------------------
# Streamlit UI
# -----------------------------------------
def main():
    st.title("Loan Approval Prediction App 🚀")

    st.markdown("""
    <div style="background-color:teal;padding:10px">
        <h1 style="color:white;text-align:center;">Byte x Brains 💻🧠</h1>
    </div>
    """, unsafe_allow_html=True)

    st.write("This app predicts whether a loan will be approved, rejected, or closed based on customer financial behavior.")

    Name = st.text_input('Kindly enter your name')
    Age = st.slider('How old are you?', 18, 70)
    Account_Balance = st.number_input('Enter your current account balance', min_value=0.0, max_value=1_000_000.0)
    Credit_Card_Balance = st.number_input('Enter your credit card balance', min_value=0.0, max_value=1_000_000.0)
    Loan_Amount = st.number_input('Loan amount requested', min_value=0.0, max_value=1_000_000.0)
    Rewards_Points = st.slider('Accumulated Reward Points on your credit card', 0, 10000)
    Credit_Limit = st.number_input('Maximum credit allowed on your card', min_value=1.0, max_value=1_000_000.0)
    Transaction_Amount = st.number_input('Last transaction amount', min_value=0.0, max_value=1_000_000.0)
    Interest_Rate = st.number_input('Interest accumulated', min_value=0.0, max_value=100.0)

    # -------------------------------
    if st.button("Predict"): 
        if model is None:
            st.error("⚠️ Unable to load the model. Prediction cannot proceed.")
        else:
            prediction = predict_loan_status(Age, Rewards_Points, Loan_Amount, Interest_Rate, Account_Balance, Credit_Card_Balance, Transaction_Amount, Credit_Limit)
            if prediction[0] == 0:
                st.success(f"🎉 Congratulations {Name}, your loan request is Approved!")
            elif prediction[0] == 2:
                st.warning(f"😞 Sorry {Name}, your loan request is Rejected.")
            else:
                st.info(f"ℹ️ Dear {Name}, your loan request is currently Closed.")

    with st.expander("▶️ About the App!"):
        st.write("This app was developed by Team Byte x Brains 💻🧠 for the TDI Hackathon.")

# -----------------------------------------
# Run the App
# -----------------------------------------
if __name__ == '__main__':
    main()
