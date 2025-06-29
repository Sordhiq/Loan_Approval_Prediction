import pandas as pd
import numpy as np
import sklearn 
import streamlit as st
import pickle 
from sklearn.preprocessing import OrdinalEncoder


# Set page config
st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="💰",
    layout="centered"
)
@st.cache_resource
def load_model():
  try:
    with open("loan_prediction_model.pkl.pkl", "rb") as file:
      mod = pickle.load(file)
    return mod

    except FileNotFoundError:
    return None
# Instantiating the model

model = load_model()

st.header('Loan Approval Predction Appppppp')
Name = st.text_input('Kindly enter your name')
Age = st.slider('How old are you', 18, 70)
Account_Balance = st.number_input('Kindly enter your current account balance', min_value=0.0, max_value=1000000)
Credit_Card_Balance = st.number_input('Kindly enter your credit card balance', min_value=0.0, max_value=1000000)
Loan_Amount = st.number_input('How much Loan are you requesting', min_value=0.0, max_value=1000000)
Loan_Type = st.selectbox('What is this loan for?', ['Personal', 'Mortgage', 'Auto'])
Loan_Term = st.slider('Duration of loan in months', min_value=1, max_value=60)
Transaction_Amount = st.slider('Last transaction amount', 0, 10000000)
Spending_Rate = Transaction_Amount/Account_Balance
Loan_to_Credit_Ratio = Loan_Amount/5550
def predict(Spending_Rate, Credit_Card_Balance, Account_Balance, Loan_Amount, Age, Loan_to_Credit_Ratio, Loan_Term, Transaction_Amount, Loan_Type):
    #encoder = OrdinalEncoder() Loan_Type = encoder.fit_transform(Loan_Type).astype(int)#
    Loan_Type = Loan_Type.map({"Auto": 0, "Personal": 1, "Mortgage":2})    
    #independent variables
    features = np.array([['Spending_Rate', 'Credit_Card_Balance', 'Account_Balance', 'Loan_Amount', 'Age', 'Loan_to_Credit_Ratio', 'Loan_Term', 'Transaction_Amount', 'Loan_Type']])
    prediction = model.predict(features)
    return prediction
def main():
    st.title("Loan Approval Prediction App")
    html_temp = """
    <div style="background-color:teal;padding:10px">
    <h1 style="color:white;text-align:center;">Loan Prediction App</h1>
    </div>
    """
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 

    """This is a platform where you could enter Applicant's details and get prediction about their eligibility for Loan requests"""  
    st.subheader("Bytes x Brains💻🧠")
    if st.button("Predict"):
        predictions = predict(Spending_Rate, Credit_Card_Balance, Account_Balance, Loan_Amount, Age, Loan_to_Credit_Ratio, Loan_Term, Transaction_Amount, Loan_Type)
        if predictions[0] == 0:
            st.success(f"Congratulations {Name}, your loan request is Approved!")
            print(Loan_Amount)
        elif prediction[0] == 2:
            st.warning(f"Sorry {Name}, your loan request is hereby Rejected!")
        else:
            st.success(f"Dear {Name}, your loan request is currently Closed!")

    with st.expander("▶️ About the App!"):
        st.write("""This loan prediction application is proudly developed by Team Bytes x Brains💻🧠 for the TDI Hackathon project""")
if name=='main':
    main()
