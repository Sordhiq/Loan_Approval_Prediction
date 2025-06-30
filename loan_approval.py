import pandas as pd
import numpy as np
import sklearn
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

@st.cache_resource
def load_model():
    try:
        with open('loan_prediction_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("The file you have attempted to load does not exist in the file directory. Kindly recheck please.")
        return None
        
model = load_model()

# 🔧 FIXED: Simplified function signature - calculate derived features inside
def predict_loan_status(Age, Rewards_Points, Loan_Amount, Interest_Rate, Account_Balance, Credit_Card_Balance, Transaction_Amount, Credit_Limit):
    try:
        # Calculate derived features inside the function
        Spending_Rate = Transaction_Amount / (Account_Balance + 1e-5)
        Loan_to_Credit_Ratio = Loan_Amount / (Credit_Limit + 1e-5)
        Credit_Utilization = Credit_Card_Balance / (Credit_Limit + 1e-5)
        
        # Create feature array with all 11 features
        features = np.array([[Age, Rewards_Points, Loan_Amount, Interest_Rate, Account_Balance, 
                            Credit_Card_Balance, Transaction_Amount, Spending_Rate, 
                            Credit_Limit, Loan_to_Credit_Ratio, Credit_Utilization]])
        
        if model is None:
            return None
            
        prediction = model.predict(features)
        return prediction
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def main():
    st.title("Loan Approval Prediction App 🚀")

    html_temp = """
    <div style="background-color:teal;padding:13px">
        <h1 style="color:white;text-align:center;">Byte x Brains 💻🧠</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.write("This loan prediction application provides real-time suggestions on approval or rejection for loan applicants based on their provided details.")

    # Input fields
    Name = st.text_input('Kindly enter your name')
    Age = st.slider('How old are you?', 18, 70)
    Account_Balance = st.slider('Enter your current account balance', min_value=0.0, max_value=1_000_000.0)
    Credit_Card_Balance = st.slider('Enter your credit card balance', min_value=0.0, max_value=1_000_000.0)
    Loan_Amount = st.slider('Loan amount requested', min_value=0.0, max_value=1_000_000.0)
    Rewards_Points = st.slider('Accumulated Reward Points on your credit card', 0, 10000)
    Credit_Limit = st.slider('Maximum credit allowed on your card', min_value=1, max_value=1_000_000)
    Transaction_Amount = st.slider('Last transaction amount', 0, 1_000_000)
    Interest_Rate = st.number_input('Interest accumulated', 0.0, 100.0)

    # 🔧 FIXED: Correct function call with proper parameters
    if st.button("Predict"): 
        if not Name.strip():
            st.warning("Please enter your name.")
        elif model is None:
            st.error("Model not loaded. Cannot make predictions.")
        else:
            # Call function with the 8 required parameters
            prediction = predict_loan_status(Age, Rewards_Points, Loan_Amount, Interest_Rate, 
                                           Account_Balance, Credit_Card_Balance, Transaction_Amount, Credit_Limit)
            
            if prediction is not None:
                if prediction[0] == 0:
                    st.success(f"🎉 Congratulations {Name}, your loan request is Approved!")
                    st.balloons()
                elif prediction[0] == 2:
                    st.warning(f"😞 Sorry {Name}, your loan request is Rejected.")
                else:
                    st.info(f"ℹ️ Dear {Name}, your loan request is currently Closed.")
                
                # 🆕 ADDED: Show calculated metrics
                with st.expander("📊 Calculated Metrics"):
                    spending_rate = Transaction_Amount / (Account_Balance + 1e-5)
                    loan_to_credit = Loan_Amount / (Credit_Limit + 1e-5)
                    credit_util = Credit_Card_Balance / (Credit_Limit + 1e-5)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Spending Rate", f"{spending_rate:.4f}")
                    with col2:
                        st.metric("Loan-to-Credit Ratio", f"{loan_to_credit:.4f}")
                    with col3:
                        st.metric("Credit Utilization", f"{credit_util:.4f}")

    with st.expander("▶️ About the App!"):
        st.write("""This loan prediction application is proudly developed by Team Byte x Brains 💻🧠 for the TDI Hackathon project.""")
        st.write("**Features used for prediction:**")
        st.write("- Age, Reward Points, Loan Amount, Interest Rate")
        st.write("- Account Balance, Credit Card Balance, Transaction Amount")
        st.write("- Credit Limit, and calculated ratios")

if __name__=='__main__':
    main()
