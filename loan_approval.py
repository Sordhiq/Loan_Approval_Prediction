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
Account Balance = st.number_input('Kindly enter your current account balance', min=0.0, max=1_000_000)
Credit Card Balance = st.number_input('Kindly enter your credit card balance', min=0.0, max=1_000_000)
Loan Amount = st.number_input('How much Loan are you requesting', min=0.0, max=1_000_000)
Loan Type = st.selectbox('What is this loan for?', ['Personal', 'Mortgage', 'Auto'])
Loan Term (Months) = st.slider('Duration of loan in months', 1, 60)
Transaction Amount = st.slider('Last transaction amount', 0, 10000000)
Spending Rate = Transaction_Amount/Account_Balance
Loan-to-Credit Ratio = Loan_Amount/5550


def predict(Spending Rate, Credit Card Balance, Account Balance, Loan Amount, Age, Loan-to-Credit Ratio, Loan Term, Transaction Amount, Loan Type):
    #encoder = OrdinalEncoder() Loan_Type = encoder.fit_transform(Loan_Type).astype(int)#
    Loan_Type = Loan_Type.map({"Auto": 0, "Personal": 1, "Mortgage":2})    
    #independent variables
    features = np.array([['Spending Rate', 'Credit Card Balance', 'Account Balance', 'Loan Amount', 'Age', 'Loan-to-Credit Ratio', 'Loan Term', 'Transaction Amount', 'Loan Type']])
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
    predictions = predict(Spending Rate, Credit Card Balance, Account Balance, Loan Amount, Age, Loan-to-Credit Ratio, Loan Term, Transaction Amount, Loan Type)
    if predictions[0] == 0:
        st.success(f"Congratulations {Name}, your loan request is Approved!")
        print(Loan_Amount)
    elif prediction[0] == 2:
        st.warning(f"Sorry {Name}, your loan request is hereby Rejected!")
    else:
        st.success(f"Dear {Name}, your loan request is currently Closed!")
             
    with st.expander("▶️ About the App!"):
        st.write("""This loan prediction application is proudly developed by Team Bytes x Brains💻🧠 for the TDI Hackathon project""")

if __name__=='__main__':
    main()
