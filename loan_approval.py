import pandas as pd
import numpy as np
import sklearn
import streamlit as st
import pickle
# import google.generativeai as genai  # 🔁 REMOVE if unused

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
    with open("loan_prediction_model.pkl", "rb") as file:
      mod = pickle.load(file)
    return mod
    
  except FileNotFoundError:
    st.error("You have attempted to load a wrong pickle file")
    return None
    
model = load_model()

# Streamlit UI
# -----------------------------------------
def main():
    st.title("Loan Approval Prediction App 🚀")

    html_temp = """
    <div style="background-color:teal;padding:10px">
        <h1 style="color:white;text-align:center;">Byte x Brains 💻🧠</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.write("This loan prediction application provides real-time suggestions on approval or rejection for loan applicants based on their provided details.")

    # -----------------------
    # User Inputs
    # -----------------------
    Name = st.text_input('Kindly enter your name')
    Age = st.slider('How old are you?', 18, 70)
    Account_Balance = st.number_input('Enter your current account balance', min_value=0.0, max_value=1_000_000.0)
    Credit_Card_Balance = st.number_input('Enter your credit card balance', min_value=0.0, max_value=1_000_000.0)
    Loan_Amount = st.number_input('Loan amount requested', min_value=0.0, max_value=1_000_000.0)
    Rewards_Points = st.slider('Accumulated Reward Points on your credit card', 0, 10000)
    Credit_Limit = st.slider('Maximum credit allowed on your card', min_value=1, max_value=1_000_000)
    Transaction_Amount = st.slider('Last transaction amount', 0, 1_000_000)
    Interest_Rate = st.number_input('Interest accumulated', 0.0, 100.0)

    # -----------------------
    # Derived Features
    # -----------------------
    Spending_Rate = Transaction_Amount / (Account_Balance + 1e-5)
    Loan_to_Credit_Ratio = Loan_Amount / (Credit_Limit + 1e-5)
    Credit_Utilization = Credit_Card_Balance / (Credit_Limit + 1e-5)

    def predict_loan_status(Age, Rewards_Points, Loan_Amount, Interest_Rate, Account_Balance, Credit_Card_Balance, Transaction_Amount, Credit_Limit):

        features = np.array([[Age, Rewards_Points, Loan_Amount, Interest_Rate, Account_Balance, Credit_Card_Balance, Transaction_Amount, Spending_Rate, Credit_Limit, Loan_to_Credit_Ratio, Credit_Utilization]])
        prediction = model.predict(features)
        return prediction
    # -----------------------
    # Prediction Trigger
    # -----------------------
    if st.button("Predict"): 
        prediction = predict_loan_status(Age, Rewards_Points, Loan_Amount, Interest_Rate, Account_Balance, Credit_Card_Balance, Transaction_Amount, Credit_Limit)
        if prediction[0] == 0:
            st.success(f"🎉 Congratulations {Name}, your loan request is Approved!")
        elif prediction[0] == 2:
            st.warning(f"😞 Sorry {Name}, your loan request is Rejected.")
        else:
            st.info(f"ℹ️ Dear {Name}, your loan request is currently Closed.")

    # -----------------------
    # App Footer
    # -----------------------
    with st.expander("▶️ About the App!"):
        st.write("""This loan prediction application is proudly developed by Team Byte x Brains 💻🧠 for the TDI Hackathon project.""")


# -----------------------------------------
# Prediction Function
# -----------------------------------------


# -----------------------------------------

# -----------------------------------------
if __name__ == '__main__':
    main()
