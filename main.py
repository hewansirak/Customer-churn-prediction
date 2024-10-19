import streamlit as st
import pandas as pd 
import pickle
import numpy as np
import os
from openai import OpenAI

st.title("Customer Churn Prediction")

df = pd.read_csv("churn.csv")

customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
  selected_customer_id = int(selected_customer_option.split(" - ")[0])
  print ("Selected Customer ID", selected_customer_id)

  selected_surname = selected_customer_option.split(" - ")[1]
  print ("Surname", selected_surname)

  selected_customer = df.loc[df["CustomerId"] == selected_customer_id].iloc[0]
  print("Selected Customer", selected_customer)

  col1, col2 = st.columns(2)
  
  with col1:
  
    credit_score = st.number_input(
      "Credit Score", 
      min_value=300,
      max_value=850,
      value=int(selected_customer["CreditScore"]))

    location = st.selectbox(
      "Location", ["Spain","France", "Germany"],
      index=["Spain","France", "Germany"].index(
        selected_customer["Geography"]))

    gender = st.radio("Gender", ["Male", "Female"], 
                      index=0 if selected_customer['Gender'] == 'Male' 
                      else 1)

    age = st.number_input(
      "Age",
      min_value=18,
      max_value=100,
      value=int(selected_customer["Age"]))

    tenure = st.number_input(
      "Tenure (years)",
      min_value=0,
      max_value=24,
      value=int(selected_customer["Tenure"])
    )

  with col2:

    balance = st.number_input(
      "Balance",
      min_value=0.0,
      value=float(selected_customer["Balance"]))
    
    num_products = st.number_input(
      "Number of Products",
      min_value=1,
      max_value=10,
      value=int(selected_customer["NumOfProducts"]))

    has_credit_card = st.checkbox(
      "Has Credit Card",
      value=bool(selected_customer["HasCrCard"]))

    is_active_member = st.checkbox(
      "Is Active Member",
      value=bool(selected_customer["IsActiveMember"]))

    estimated_salary = st.number_input(
      "Estimated Salary",
      min_value=0.0,
      value=float(selected_customer["EstimatedSalary"]))


  input_df, input_dict = prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary)

  make_predictions(input_df, input_dict)

