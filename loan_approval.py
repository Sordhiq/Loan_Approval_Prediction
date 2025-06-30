eimport pandas as pd
import numpy as np
import pickle
import streamlit as st

# Set page config
st.set_page_config(
    page_title="Malaria Prediction App",
    page_icon="🔖",
    layout="centered"
)

@st.cache_resource
def load_model():
  try:
    with open("loan_prediction_model.pkl", "rb") as file:
      mod = pickle.load(file)
    return mod
    
  except FileNotFoundError:
    st.error("You have attempted to load a wrong pickle file")
    return None

# Instantiating model
model = load_model()

def predi(rainfall_rolling, rainfall_lag, cumulative_rainfall):
  features = np.array([[rainfall_rolling, rainfall_lag, cumulative_rainfall]])
  prediction = model.predict(features)
  return prediction

def main():
    st.title("Malaria Prediction App")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Malaria Prediction App</h2>
    </div>
    """
        
    """Proudly... Group 2, Climate Accelerator Network (CAN) Training, 2024!"""  
    st.subheader("This web application uses climatic variables in predicting malaria prevalences")
   
    rainfall_rolling = st.number_input("Rainfall Rolling Average", value=17.0, min_value=0.0, max_value=263.0)
    rainfall_lag = st.number_input("Ranfall Lag 3", value=50.0, min_value=0.0, max_value=400.0)
    cumulative_rainfall = st.number_input("Cumulative Rainfall", value=4500.0, min_value=1000.0, max_value=8000.0)
    
    if st.button("Predict"):
        predictions = predi(rainfall_rolling, rainfall_lag, cumulative_rainfall)
        st.success(f"The Predicted Malaria Case is: {int(predictions)}")
        
    with st.expander("▶️ About this App!"):
        st.write("""This machine learning application is proudly developed by Group 2 members of the CAN Data Science Fellowship.\
                The model uses climatic variables like Rainfall and Temperatures in predicting malaria prevalence.""")

if __name__=='__main__':
    main()
