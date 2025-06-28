import pandas as pd
import sklearn as sk 
import streamlit as st
import pickle as pk

model = pk.load(open('model.pkl','rb'))

# Set page config
st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="💰",
    layout="centered"
)

st.header('Loan Approval Predction App')

Name = st.text_input('Kindly enter your name')
Age = st.slider('How old are you', 18, 70)
Account_Balance = st.number_input('Kindly enter your current account balance', min=0.0, max=1_000_000)
Credit_Card_Balance = st.number_input('Kindly enter your credit card balance', min=0.0, max=1_000_000)
Loan_Amount = st.number_input('How much Loan are you requesting', min=0.0, max=1_000_000)
Loan_Type = st.selectbox('What is this loan for?', ['Personal', 'Mortgage', 'Auto'])
Loan_Term = st.slider('Duration of loan in months', 12, 60)
Transaction_Amount = st.slider('Last transaction amount', 0, 10000000)
Spending_Rate = Transaction_Amount/Account_Balance
Loan-to-Credit Ratio = Loan_Amount/5550


#independent variables
features = ['Spending_Rate', 'Credit_Card_Balance', 'Account_Balance', 'Loan_Amount', 
            'Age, 'Loan-to-Credit Ratio', 'Loan_Term', 'Transaction_Amount', 'Loan_Type']
X = data[features]
#dependent variables
dependent = 'Loan_Approved'
y = data[dependent]
X.shape, y.shape

#split dataset into train (80%) and test (20%), shuffle observations
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 10, shuffle = True)

#build random forest model, limit max depth to avoid overfitting
forest = RandomForestClassifier(max_depth=4, random_state = 10, n_estimators = 100, min_samples_leaf=5) 
model = forest.fit(x_train, y_train)


@st.cache()

# defining the function which will make the prediction using the data which the user inputs 
def prediction(Gender, Married, TotalApplicantIncome, LoanAmount, Credit_History):   
    
    if Gender == "Male":
        Gender = 0
    else:
        Gender = 1
 
    if Married == "Unmarried":
        Married = 0
    else:
        Married = 1
 
    if Credit_History == "No Credit History":
        Credit_History = 0
    else:
        Credit_History = 1  
 
    LoanAmount = LoanAmount / 1000
 
    # Making predictions 
    pred_inputs = model.predict(pd.DataFrame([[Gender, Married, TotalApplicantIncome, LoanAmount, Credit_History]]))
        
    if pred_inputs[0] == 0:
        pred = 'I am sorry, you have been rejected for the loan.'
    elif pred_inputs[0] == 1:
        pred = 'Congrats! You have been approved for the loan!'
    else:
        pred = 'Error'
    return pred

def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:teal;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    Gender = st.selectbox('Gender',("Male","Female"))
    Married = st.selectbox('Marital Status',("Unmarried","Married")) 
    ApplicantIncome = st.number_input("Total Monthly Income, (Include Coborrower if Applicable)") 
    LoanAmount = st.number_input("Loan Amount (ex. 125000)")
    Credit_History = st.selectbox('Credit History',("Has Credit History","No Credit History"))
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History) 
        st.success('Final Decision: {}'.format(result))
        print(LoanAmount)
     
if __name__=='__main__': 
    main()



