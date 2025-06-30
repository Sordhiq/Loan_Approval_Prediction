import pandas as pd
import numpy as np
import sklearn
import streamlit as st
import pickle
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables (including your API key)
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize the Gemini model
# You can choose different models like 'gemini-pro', 'gemini-1.5-flash', etc.
# 'gemini-pro' is a good general-purpose model for text generation.
gemini_model = genai.GenerativeModel('gemini-pro')

# -----------------------------------------
# Page Configuration
# -----------------------------------------
st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="💰",
    layout="centered"
)

@st.cache_resource
def load_model():
    try:
        with open('loan_prediction_model.pkl', 'rb') as file:
            mod = pickle.load(file)
        return mod
    except FileNotFoundError:
        st.error("The file you have attempted to load does not exist in the file directory.\
                    Kindly recheck please.")
        return None
        
model = load_model()

def predict_loan_status(Age, Rewards_Points, Loan_Amount, Interest_Rate, Account_Balance, Credit_Card_Balance, Transaction_Amount, Spending_Rate, Credit_Limit, Loan_to_Credit_Ratio, Credit_Utilization):
    features = np.array([[Age, Rewards_Points, Loan_Amount, Interest_Rate, Account_Balance, Credit_Card_Balance, Transaction_Amount, Spending_Rate, Credit_Limit, Loan_to_Credit_Ratio, Credit_Utilization]])
    prediction = model.predict(features)
    return prediction

# Function to get AI-tailored response
def get_ai_response(name, status, loan_amount):
    prompt = ""
    if status == 0:  # Approved
        prompt = f"The loan for {name} for an amount of {loan_amount} has been approved. Provide a short, encouraging, and helpful message with next steps. Start with 'Fantastic news, {name}!'"
    elif status == 2:  # Rejected
        prompt = f"The loan for {name} for an amount of {loan_amount} has been rejected. Provide a short, polite, and constructive message suggesting general ways to improve their chances for future loan applications (e.g., improve credit score, reduce debt, increase income). Start with 'Dear {name},'"
    else:  # Closed (assuming this means neither approved nor rejected, perhaps withdrawn or incomplete)
        prompt = f"The loan request for {name} for an amount of {loan_amount} is currently closed. Provide a short, informative message about what 'closed' might mean and what they could do next. Start with 'Hello {name},'"

    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating AI response: {e}")
        return "Could not generate an AI-tailored response at this time."


def main():
    st.title("Loan Approval Prediction App 🚀")

    html_temp = """
    <div style="background-color:teal;padding:10px">
        <h1 style="color:white;text-align:center;">Byte x Brains 💻🧠</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.write("This loan prediction application provides real-time suggestions on approval or rejection for loan applicants based on their provided details.")

    Name = st.text_input('Kindly enter your name')
    Age = st.slider('How old are you?', 18, 70)
    Account_Balance = st.slider('Enter your current account balance', min_value=0.0, max_value=1_000_000.0)
    Credit_Card_Balance = st.slider('Enter your credit card balance', min_value=0.0, max_value=1_000_000.0)
    Loan_Amount = st.slider('Loan amount requested', min_value=0.0, max_value=1_000_000.0)
    Rewards_Points = st.slider('Accumulated Reward Points on your credit card', 0, 10000)
    Credit_Limit = st.slider('Maximum credit allowed on your card', min_value=1, max_value=1_000_000)
    Transaction_Amount = st.slider('Last transaction amount', 0, 1_000_000)
    Interest_Rate = st.number_input('Interest accumulated', 0.0, 100.0)
      
    # Calculate the derived features before calling the prediction function
    Spending_Rate = Transaction_Amount / (Account_Balance + 1e-5)
    Loan_to_Credit_Ratio = Loan_Amount / (Credit_Limit + 1e-5)
    Credit_Utilization = Credit_Card_Balance / (Credit_Limit + 1e-5)

    # -------------------------------
    if st.button("Predict"): 
        if model is not None: # Ensure model is loaded before predicting
            prediction = predict_loan_status(Age, Rewards_Points, Loan_Amount, Interest_Rate, Account_Balance, Credit_Card_Balance, Transaction_Amount, Spending_Rate, Credit_Limit, Loan_to_Credit_Ratio, Credit_Utilization)
            loan_status = prediction[0]

            if loan_status == 0:
                st.success(f"🎉 Congratulations {Name}, your loan request is Approved!")
            elif loan_status == 2:
                st.warning(f"😞 Sorry {Name}, your loan request is Rejected.")
            else:
                st.info(f"ℹ️ Dear {Name}, your loan request is currently Closed.")

            # Get and display AI-tailored response
            st.subheader("AI Assistant's Advice:")
            with st.spinner("Generating personalized advice..."):
                ai_response = get_ai_response(Name, loan_status, Loan_Amount)
                st.write(ai_response)
        else:
            st.error("Model could not be loaded. Please check the 'loan_prediction_model.pkl' file.")

    with st.expander("▶️ About the App!"):
        st.write("""This loan prediction application is proudly developed by Team Byte x Brains 💻🧠 for the TDI Hackathon project.""")
    # ------------------------

if __name__=='__main__':
    main()
