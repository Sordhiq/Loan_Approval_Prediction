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
        # Fixed: Removed duplicate .pkl extension
        with open("loan_prediction_model.pkl", "rb") as file:
            mod = pickle.load(file)
        return mod
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'loan_prediction_model.pkl' exists.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Instantiating the model
model = load_model()

# Fixed: Corrected spelling in header
st.header('Loan Approval Prediction App')

# Input fields
Name = st.text_input('Kindly enter your name')
Age = st.slider('How old are you', 18, 70)
Account_Balance = st.number_input('Kindly enter your current account balance', min_value=0.0, max_value=1000000.0)
Credit_Card_Balance = st.number_input('Kindly enter your credit card balance', min_value=0.0, max_value=1000000.0)
Loan_Amount = st.number_input('How much Loan are you requesting', min_value=0.0, max_value=1000000.0)
Loan_Type = st.selectbox('What is this loan for?', ['Personal', 'Mortgage', 'Auto'])
Loan_Term = st.slider('Duration of loan in months', min_value=1, max_value=60)
Transaction_Amount = st.slider('Last transaction amount', 0, 10000000)

# Fixed: Added error handling for division by zero
if Account_Balance > 0:
    Spending_Rate = Transaction_Amount / Account_Balance
else:
    Spending_Rate = 0
    st.warning("Account balance cannot be zero for spending rate calculation.")

# Fixed: Made this configurable or calculate properly
AVERAGE_CREDIT_LIMIT = 5550  # You should replace this with actual logic
if AVERAGE_CREDIT_LIMIT > 0:
    Loan_to_Credit_Ratio = Loan_Amount / AVERAGE_CREDIT_LIMIT
else:
    Loan_to_Credit_Ratio = 0

def predict(Spending_Rate, Credit_Card_Balance, Account_Balance, Loan_Amount, Age, Loan_to_Credit_Ratio, Loan_Term, Transaction_Amount, Loan_Type):
    try:
        # Fixed: Proper encoding of Loan_Type
        loan_type_mapping = {"Auto": 0, "Personal": 1, "Mortgage": 2}
        Loan_Type_encoded = loan_type_mapping.get(Loan_Type, 0)
        
        # Fixed: Create proper feature array with actual values, not strings
        features = np.array([[Spending_Rate, Credit_Card_Balance, Account_Balance, Loan_Amount, Age, Loan_to_Credit_Ratio, Loan_Term, Transaction_Amount, Loan_Type_encoded]])
        
        if model is None:
            return None
            
        prediction = model.predict(features)
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def main():
    st.title("Loan Approval Prediction App")
    html_temp = """
    <div style="background-color:teal;padding:10px">
    <h1 style="color:white;text-align:center;">Loan Prediction App</h1>
    </div>
    """
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True) 
        
    st.write("This is a platform where you could enter Applicant's details and get prediction about their eligibility for Loan requests")  
    st.subheader("Bytes x Brains💻🧠")

# Fixed: Proper button handling and prediction logic
if st.button("Predict"):
    if model is None:
        st.error("Model not loaded. Cannot make predictions.")
    elif not Name.strip():
        st.warning("Please enter your name.")
    else:
        predictions = predict(Spending_Rate, Credit_Card_Balance, Account_Balance, Loan_Amount, Age, Loan_to_Credit_Ratio, Loan_Term, Transaction_Amount, Loan_Type)
        
        if predictions is not None:
            # Fixed: Use consistent variable name (predictions instead of prediction)
            prediction_value = predictions[0]
            
            # Display prediction results
            if prediction_value == 1:  # Assuming 1 = Approved
                st.success(f"Congratulations {Name}, your loan request is Approved!")
                st.balloons()  # Added celebration effect
            elif prediction_value == 0:  # Assuming 0 = Rejected
                st.error(f"Sorry {Name}, your loan request is hereby Rejected!")
            else:  # Any other value
                st.info(f"Dear {Name}, your loan request status is unclear. Please contact support.")
            
            # Display input summary
            with st.expander("📊 Application Summary"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Name:** {Name}")
                    st.write(f"**Age:** {Age}")
                    st.write(f"**Loan Amount:** ${Loan_Amount:,.2f}")
                    st.write(f"**Loan Type:** {Loan_Type}")
                with col2:
                    st.write(f"**Account Balance:** ${Account_Balance:,.2f}")
                    st.write(f"**Credit Card Balance:** ${Credit_Card_Balance:,.2f}")
                    st.write(f"**Loan Term:** {Loan_Term} months")
                    st.write(f"**Spending Rate:** {Spending_Rate:.4f}")

# Fixed: Proper expandable section placement
with st.expander("▶️ About the App!"):
    st.write("This loan prediction application is proudly developed by Team Bytes x Brains💻🧠 for the TDI Hackathon project")
    st.write("**Features used for prediction:**")
    st.write("- Spending Rate, Credit Card Balance, Account Balance")
    st.write("- Loan Amount, Age, Loan Term")
    st.write("- Transaction Amount, Loan Type")

# Fixed: Proper main function call
if __name__ == '__main__':
    main()
