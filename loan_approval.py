import pandas as pd  
import numpy as np
import sklearn
import streamlit as st
import pickle
import google.generativeai as genai
import os
from typing import Optional

# -----------------------------------------
# Page Configuration
# -----------------------------------------
st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="💰",
    layout="centered"
) 

# -----------------------------------------
# Gemini API Configuration
# -----------------------------------------
def configure_gemini():
    """Configure Gemini API with API key from Streamlit secrets"""
    try:
        # Get API key from Streamlit secrets
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        return True
    except KeyError:
        st.error("🔑 API key not found in secrets.toml file.")
        return False
    except Exception as e:
        st.error(f"Error configuring API: {str(e)}")
        return False

def generate_ai_recommendation(
    name: str,
    prediction_outcome: int,
    age: int,
    loan_amount: float,
    account_balance: float,
    credit_utilization: float,
    spending_rate: float,
    loan_to_credit_ratio: float,
    interest_rate: float
) -> Optional[str]:
    """Generate AI-powered recommendations based on loan outcome and user profile"""
    
    if not configure_gemini():
        return None
    
    try:
        # Map prediction outcome to status
        status_map = {0: "Approved", 1: "Closed/Pending", 2: "Rejected"}
        status = status_map.get(prediction_outcome, "Unknown")
        
        # Create context-rich prompt
        prompt = f"""
        You are a financial advisor providing personalized recommendations for a loan applicant.
        
        **Applicant Profile:**
        - Name: {name}
        - Age: {age}
        - Loan Amount Requested: ₦{loan_amount:,.2f}
        - Account Balance: ₦{account_balance:,.2f}
        - Interest Rate: {interest_rate}%
        
        **Financial Metrics:**
        - Credit Utilization: {credit_utilization:.2%}
        - Spending Rate: {spending_rate:.4f}
        - Loan-to-Credit Ratio: {loan_to_credit_ratio:.4f}
        
        **Loan Status: {status}**
        
        Please provide personalized, actionable financial advice based on the loan outcome:
        
        If APPROVED:
        - Congratulate them and provide tips for responsible loan management
        - Suggest ways to maintain good financial health
        - Recommend strategies for timely repayment
        
        If REJECTED:
        - Provide constructive feedback on areas for improvement
        - Suggest specific steps to enhance their financial profile
        - Recommend timeframes for reapplication
        - Offer alternative financing options
        
        If CLOSED/PENDING:
        - Explain possible reasons for the pending status
        - Suggest steps to strengthen their application
        - Provide guidance on what documents or improvements might be needed
        
        Make the response contextualized for a Nigerian loan applicant, keep it under 300 words, make it slightly funny and hilarious, friendly, relatable, and actionable. 
        Use bullet points for clarity. All amounts entered are in Nigerian Naira, so make the responses also in Naira.
        """
        
        # Generate response using Gemini (Free tier available)
        # Try different model names in order of preference
        model_names = ['gemini-1.5-flash-latest', 'gemini-2.5-flash', 'gemini-pro']
        
        response = None
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                break  # If successful, exit loop
            except Exception as model_error:
                continue  # Try next model
        
        if response is None:
            raise Exception("No available Gemini models found") 
        
        return response.text
        
    except Exception as e:
        st.error(f"Error generating AI recommendation: {str(e)}")
        return None

@st.cache_resource
def load_model():
    try:
        with open('loan_prediction_model.pkl', 'rb') as file:
            mod = pickle.load(file)
        return mod
    except FileNotFoundError:
        st.error("The file you have attempted to load does not exist in the file directory. Kindly recheck please.")
        return None
        
model = load_model()

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
            return None, None, None, None
            
        prediction = model.predict(features)
        return prediction, Spending_Rate, Loan_to_Credit_Ratio, Credit_Utilization
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None, None

def main():
    st.title("Loan Approval Prediction App 🚀")

    html_temp = """
    <div style="background-color:teal;padding:13px">
        <h1 style="color:white;text-align:center;">Byte x Brains 💻🧠</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.write("Welcome to the Byte x Brains' AI-powered Loan Prediction App.")
    st.write("This intelligent system provides real-time loan decision, actionable financial insights along with a tailored financial advise based on financial history.")
    
    # Test Gemini API connection on startup
    if configure_gemini():
        st.success("🤖 AI Recommendations Ready!")
    else:
        st.warning("⚠️ AI Recommendations unavailable - API configuration issue")

    # Input fields
    Name = st.text_input('Kindly enter your name')
    Age = st.slider('How old are you?', 18, 70)
    Account_Balance = st.number_input('Enter your current account balance (Naira)', min_value=0.0, max_value=1_000_000.0)
    Credit_Card_Balance = st.number_input('Enter your credit card balance', min_value=0.0, max_value=1_000_000.0)
    Loan_Amount = st.number_input('Loan amount requested', min_value=0.0, max_value=1_000_000.0)
    Rewards_Points = st.number_input('Accumulated Reward Points on your credit card', 0, 10000)
    Credit_Limit = st.number_input('Maximum credit allowed on your card', min_value=1, max_value=1_000_000)
    Transaction_Amount = st.number_input('Last transaction amount', 0, 1_000_000)
    Interest_Rate = st.number_input('Interest accumulated', 0.0, 100.0)

    if st.button("Predict"): 
        if not Name.strip():
            st.warning("Please enter your name.")
        elif model is None:
            st.error("Model not loaded. Cannot make predictions.")
        else:
            # Call function with the 8 required parameters
            prediction, spending_rate, loan_to_credit, credit_util = predict_loan_status(
                Age, Rewards_Points, Loan_Amount, Interest_Rate, 
                Account_Balance, Credit_Card_Balance, Transaction_Amount, Credit_Limit
            )
            
            if prediction is not None:
                # Display prediction result
                prediction_value = prediction[0]
                
                if prediction_value == 0:
                    st.success(f"🎉 Congratulations {Name}, your loan request is Approved!")
                    st.balloons()
                elif prediction_value == 2:
                    st.warning(f"😞 Sorry {Name}, your loan request is Rejected.")
                else:
                    st.info(f"ℹ️ Dear {Name}, your loan request is currently Closed.")
                
                # Show calculated metrics
                with st.expander("📊 Calculated Metrics"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Spending Rate", f"{spending_rate:.4f}")
                    with col2:
                        st.metric("Loan-to-Credit Ratio", f"{loan_to_credit:.4f}")
                    with col3:
                        st.metric("Credit Utilization", f"{credit_util:.4f}")
                
                # 🆕 AI-Powered Recommendations Section
                st.markdown("---")
                st.subheader("🤖 AI-Powered Financial Recommendations")
                
                with st.spinner("Generating personalized recommendations..."):
                    ai_recommendation = generate_ai_recommendation(
                        name=Name,
                        prediction_outcome=prediction_value,
                        age=Age,
                        loan_amount=Loan_Amount,
                        account_balance=Account_Balance,
                        credit_utilization=credit_util,
                        spending_rate=spending_rate,
                        loan_to_credit_ratio=loan_to_credit,
                        interest_rate=Interest_Rate
                    )
                
                if ai_recommendation:
                    # Display AI recommendation in a nice container
                    with st.container():
                        st.markdown(
                            f"""
                            <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
                                <h4 style="color: #2E8B57; margin-top: 0;">💡 Personalized Advice for {Name}</h4>
                                <p style="color: #333; line-height: 1.6;">{ai_recommendation}</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                else:
                    st.warning("⚠️ Unable to generate AI recommendations. Please check your API key configuration.")
                
                # Additional resources section
                with st.expander("📚 Additional Financial Resources"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Improve Your Credit Score:**")
                        st.write("• Pay bills on time consistently")
                        st.write("• Keep credit utilization below 30%")
                        st.write("• Don't close old credit accounts")
                        st.write("• Monitor your credit report regularly")
                    
                    with col2:
                        st.markdown("**Financial Planning Tips:**")
                        st.write("• Build an emergency fund (3-6 months expenses)")
                        st.write("• Create and stick to a budget")
                        st.write("• Consider debt consolidation if needed")
                        st.write("• Seek financial counseling if struggling")

    with st.expander("▶️ About the App!"):
        st.write("""This loan prediction application is proudly developed by Team Byte x Brains 💻🧠 at the TDI Hackathon project.""")
        st.write("**Features used for prediction:**")
        st.write("- Age, Reward Points, Loan Amount, Interest Rate")
        st.write("- Account Balance, Credit Card Balance, Transaction Amount")
        st.write("- Credit Limit, and calculated ratios")
        st.write("\n**AI Features:**")
        st.write("- Personalized financial recommendations powered by Google Gemini (Free tier)")
        st.write("- Context-aware advice based on your financial profile")
        st.write("- Actionable steps for improving loan approval chances")

if __name__=='__main__':
    main()
