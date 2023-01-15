import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# load your data into a pandas dataframe

data = pd.read_csv("https://raw.githubusercontent.com/giriprasathd/data/main/1651277648862_healthinsurance.csv")

# split the data into training and test sets
X = data[['no_of_dependents','smoker','bloodpressure','diabetes']] # select relevant features
y = data['claim']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# create a random forest regressor model
X = data[['no_of_dependents','smoker','bloodpressure','diabetes']] # select relevant features
reg1 = RandomForestRegressor(n_estimators=100, random_state=0)


# fit the model using the training data
reg1.fit(X_train, y_train)

# make predictions using the test data
y_pred = reg1.predict(X_test)

st.title("Health Care Insurance Tracker")
st.markdown(
    "Welcome to our Health Care Insurance Tracker! With this app, you can input your health information such as number of dependents,"
    "smoker status, blood pressure, and diabetes status and receive personalized recommendations insurance claim")
            
# Create input fields for each feature
no_of_dependents = st.number_input("Enter the number of dependents",1,10)
st.image('diabetics.png')
smoker = st.selectbox("Are you a smoker?", [0, 1])
st.write("""
The zero indicates "no" and one indicates "yes."
""")
bloodpressure = st.number_input("Enter your blood pressure",60, 160)

diabetes = st.selectbox("Do you have diabetes?", [0, 1])
st.write("""
The zero indicates "no" and one indicates "yes."
""")
st.checkbox('I agree to predict My insurance claim')
# Create a prediction button
if st.button('Predict'):
    new_input = [[no_of_dependents, smoker, bloodpressure, diabetes]]
    prediction = reg1.predict(new_input)
    st.write("The insurance amount to be paid is: ", int(prediction[0]))

st.write("Please make sure the input values are correct before clicking the Predict button.")

st.markdown(""" Developed by:
1. Giri Prasath
2. Aman Yadav
3. Anna Utkin""")


