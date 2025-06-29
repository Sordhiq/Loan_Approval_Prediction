import pandas as pd
import numpy as np
import sklearn
import streamlit as st
import pickle
import google.generativeai as genai 

# Set page config
st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="💰",
    layout="centered"
)

@st.cache_resource
def load_model():
    try:
        with open("loan_prediction_model.pkl", "rb") as file:
            mod = pickle.load(file)
        return mod
    except FileNotFoundError:
        return None

# Instantiate the model
model = load_model()

def predic(Age, Rewards_Points, Loan_Amount, Interest_Rate, Account_Balance, Credit_Card_Balance, Transaction_Amount, Spending_Rate, Credit_Limit, Loan_to_Credit_Ratio, Credit_Utilization):

    # Features array
    features = np.array([[Age, Rewards_Points, Loan_Amount, Interest_Rate, Account_Balance, Credit_Card_Balance, Transaction_Amount, Spending_Rate, Credit_Limit, Loan_to_Credit_Ratio, Credit_Utilization]])
                
    prediction = model.predict(features)
    return prediction

def main():
    st.title("Loan Approval Prediction App🚀")
    html_temp = """
    <div style="background-color:teal;padding:10px">
        <h1 style="color:white;text-align:center;">Bytes x Brains 💻🧠</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    "This loan prediction application is developed to be able to give realtime suggestions on approval or rejection for\
              loan applicants, given their provided details."

    # Inputs
    Name = st.text_input('Kindly enter your name')
    Age = st.slider('How old are you?', 18, 70)
    Account_Balance = st.number_input('Enter your current account balance', min_value=0.0, max_value=1_000_000.0)
    Credit_Card_Balance = st.number_input('Enter your credit card balance', min_value=0.0, max_value=1_000_000.0)
    Loan_Amount = st.number_input('Loan amount requested', min_value=0.0, max_value=1_000_000.0)
    Rewards_Points = st.slider('Accumulated Reward Points on your credit card', 0, 1000)
    Credit_Limit = st.slider('Maximum credit allowed on your card', min_value=1, max_value=1000)
    Transaction_Amount = st.slider('Last transaction amount', 0, 1_000_000)

    # Derived features
    Spending_Rate = Transaction_Amount / (Account_Balance + 1e-5)  # prevent divide-by-zero
    Loan_to_Credit_Ratio = Loan_Amount / Credit_Limit
    Credit_Utilization = Credit_Card_Balance / Credit_Limit
    
    if st.button("Predict"):
        predictions = predic(Age, Rewards_Points, Loan_Amount, Interest_Rate, Account_Balance, Credit_Card_Balance, Transaction_Amount, Spending_Rate, Credit_Limit, Loan_to_Credit_Ratio, Credit_Utilization)
        
        if predictions[0] == 0:
            st.success(f"Congratulations {Name}, your loan request is Approved!")
        elif predictions[0] == 2:
            st.warning(f"Sorry {Name}, your loan request is hereby Rejected!")
        else:
            st.info(f"Dear {Name}, your loan request is currently Closed.")
    
    with st.expander("▶️ About the App!"):
        st.write("""This loan prediction application is proudly developed by Team Bytes x Brains💻🧠 for the TDI Hackathon project.""")

if __name__=='__main__':
    main()
