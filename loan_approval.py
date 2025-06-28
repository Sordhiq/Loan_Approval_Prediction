import pandas as pd
import numpy as np
import sklearn
import streamlit as st
import pickle

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

def main():
    st.title("Loan Approval Prediction App")

    html_temp = """
    <div style="background-color:teal;padding:10px">
        <h1 style="color:white;text-align:center;">Bytes x Brains 💻🧠</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.subheader("Loaner Pred App")
# App Header
st.header('Loan Approval Prediction App 🚀')

# Inputs
Name = st.text_input('Kindly enter your name')
Age = st.slider('How old are you?', 18, 70)
Account_Balance = st.number_input('Enter your current account balance', min_value=0.0, max_value=1_000_000.0)
Credit_Card_Balance = st.number_input('Enter your credit card balance', min_value=0.0, max_value=1_000_000.0)
Loan_Amount = st.number_input('Loan amount requested', min_value=0.0, max_value=1_000_000.0)
Loan_Type = st.selectbox('What is this loan for?', ['Personal', 'Mortgage', 'Auto'])
Loan_Term = st.slider('Loan term (months)', min_value=1, max_value=60)
Transaction_Amount = st.slider('Last transaction amount', 0, 1_000_000)

# Derived features
Spending_Rate = Transaction_Amount / (Account_Balance + 1e-5)  # prevent divide-by-zero
Loan_to_Credit_Ratio = Loan_Amount / 5550

def predict(Spending_Rate, Credit_Card_Balance, Account_Balance, Loan_Amount, Age,
            Loan_to_Credit_Ratio, Loan_Term, Transaction_Amount, Loan_Type):

    loan_type_map = {"Auto": 0, "Personal": 1, "Mortgage": 2}
    Loan_Type = loan_type_map.get(Loan_Type, -1)

    # Features array
    features = np.array([[Spending_Rate, Credit_Card_Balance, Account_Balance,
                          Loan_Amount, Age, Loan_to_Credit_Ratio,
                          Loan_Term, Transaction_Amount, Loan_Type]])

    prediction = model.predict(features)
    return prediction


if st.button("Predict"):
    if model:
        predictions = predict(Spending_Rate, Credit_Card_Balance, Account_Balance,
                                  Loan_Amount, Age, Loan_to_Credit_Ratio,
                                  Loan_Term, Transaction_Amount, Loan_Type)
        if predictions[0] == 0:
            st.success(f"Congratulations {Name}, your loan request is Approved!")
        elif predictions[0] == 2:
            st.warning(f"Sorry {Name}, your loan request is hereby Rejected!")
        else:
            st.info(f"Dear {Name}, your loan request is currently Closed.")
    else:
        st.error("Model not found. Please check your model file path.")

    with st.expander("▶️ About the App!"):
        st.write("""This loan prediction application is proudly developed by Team Bytes x Brains💻🧠 for the TDI Hackathon project.""")

if __name__ == '__main__':
    main()
