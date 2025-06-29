# ------------------ IMPORTS ------------------
import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="💰",
    layout="centered"
)

# ------------------ CACHED MODEL LOADING & TRAINING ------------------
@st.cache_resource
def load_model():
    data = pd.read_csv('dataset.csv')

    # Clean column names
    data.columns = data.columns.str.strip().str.replace('-', '_').str.replace(' ', '_')

    # Select and prepare features
    features = ['Age', 'Rewards_Points', 'Loan_Amount', 'Interest_Rate', 'Account_Balance',
                'Credit_Card_Balance', 'Transaction_Amount', 'Spending_Rate',
                'Credit_Limit', 'Loan_to_Credit_Ratio', 'Credit_Utilization', 'Loan_Status']

    if not set(features).issubset(data.columns):
        raise ValueError("Required columns missing in the dataset")

    final_data = data[features]

    # Split data
    X = final_data.drop('Loan_Status', axis=1)
    y = final_data['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

    # Train model
    model = AdaBoostClassifier(n_estimators=200, random_state=11)
    model.fit(X_train, y_train)

    return model

model = load_model()

# ------------------ MAIN FUNCTION ------------------
def main():
    st.title("Loan Approval Prediction App 🚀")

    html_temp = """
    <div style="background-color:teal;padding:10px">
        <h1 style="color:white;text-align:center;">Byte x Brains 💻🧠</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.write("This app provides real-time suggestions on loan approval or rejection based on applicant details.")

    # ------------------ USER INPUTS ------------------
    Name = st.text_input('Kindly enter your name')
    Age = st.slider('How old are you?', 18, 70)
    Account_Balance = st.number_input('Enter your current account balance', min_value=0.0, max_value=1_000_000.0)
    Credit_Card_Balance = st.slider('Enter your credit card balance', min_value=0.0, max_value=1_000_000.0)
    Loan_Amount = st.slider('Loan amount requested', min_value=0.0, max_value=1_000_000.0)
    Rewards_Points = st.slider('Accumulated Reward Points on your credit card', 0, 10000)
    Credit_Limit = st.slider('Maximum credit allowed on your card', min_value=1, max_value=1_000_000)
    Transaction_Amount = st.slider('Last transaction amount', 0, 1_000_000)
    Interest_Rate = st.slider('Interest accumulated', 0.0, 100.0)

    # ------------------ FEATURE ENGINEERING ------------------
    Spending_Rate = Transaction_Amount / (Account_Balance + 1e-5)
    Loan_to_Credit_Ratio = Loan_Amount / (Credit_Limit + 1e-5)
    Credit_Utilization = Credit_Card_Balance / (Credit_Limit + 1e-5)

    # ------------------ PREDICTION ------------------
    def predict_loan_status():
        features = np.array([[Age, Rewards_Points, Loan_Amount, Interest_Rate,
                              Account_Balance, Credit_Card_Balance, Transaction_Amount,
                              Spending_Rate, Credit_Limit, Loan_to_Credit_Ratio, Credit_Utilization]])
        prediction = model.predict(features)
        return prediction

    if st.button("Predict"):
        prediction = predict_loan_status()

        if prediction[0] == 0:
            st.success(f"🎉 Congratulations {Name}, your loan request is Approved!")
        elif prediction[0] == 2:
            st.warning(f"😞 Sorry {Name}, your loan request is Rejected.")
        else:
            st.info(f"ℹ️ Dear {Name}, your loan request is currently Closed.")

    # ------------------ ABOUT SECTION ------------------
    with st.expander("▶️ About the App!"):
        st.write("""This loan prediction application is proudly developed by Team Byte x Brains 💻🧠 for the TDI Hackathon project.""")

# ------------------ RUN APP ------------------
if __name__ == '__main__':
    main()
