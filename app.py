import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os

# Page config
st.set_page_config(
    page_title="Predict My Score",
    page_icon="🎓",
    layout="wide"
)

# Header
st.markdown("<h1 style='text-align:center; color: blue;'>🎓 Student Performance Predictor</h1>", unsafe_allow_html=True)
st.markdown("This ML-based system predicts your final exam score based on study hours, attendance, and previous score.")

# Load dataset
DATA_PATH = "student_data.csv"
if not os.path.exists(DATA_PATH):
    st.error(f"CSV file not found: {DATA_PATH}")
else:
    data = pd.read_csv(DATA_PATH)

# Sidebar Inputs
st.sidebar.header("Input Parameters")
hours = st.sidebar.number_input("Hours Studied", min_value=0, max_value=24, value=5)
attendance = st.sidebar.slider("Attendance (%)", min_value=0, max_value=100, value=75)
previous_score = st.sidebar.number_input("Previous Score", min_value=0, max_value=100, value=70)

# Tabs
tab1, tab2, tab3 = st.tabs(["Predict Score", "Dataset Overview", "Analytics"])

with tab1:
    st.subheader("Your Prediction")
    features = ['Hours_Studied', 'Attendance', 'Previous_Score']
    X = data[features]
    y = data['Final_Score']

    model = LinearRegression()
    model.fit(X, y)

    # Optional: save model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    if st.button("Predict"):
        input_data = [[hours, attendance, previous_score]]
        predicted_score = model.predict(input_data)[0]
        st.success(f"🎯 Predicted Final Score: {predicted_score:.2f}")
        st.progress(min(predicted_score/100, 1.0))

with tab2:
    st.subheader("Dataset Preview")
    st.dataframe(data.head(20))
    with st.expander("Show more"):
        st.dataframe(data)

with tab3:
    st.subheader("Analytics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Hours Studied vs Final Score")
        st.line_chart(data[['Hours_Studied','Final_Score']])
    with col2:
        st.markdown("### Attendance vs Final Score")
        st.bar_chart(data[['Attendance','Final_Score']])
