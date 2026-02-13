import streamlit as st
import requests
import pandas as pd

# ======================
# Page Config
# ======================
st.set_page_config(page_title="Doctor Recommendation Dashboard", layout="wide")

st.title("Smart Doctor Recommendation Dashboard")
st.write("Enter patient details to get Top-N doctor recommendations")

# ======================
# Patient Input Form
# ======================
with st.form(key="patient_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=35)
    gender = st.selectbox("Gender", options=["Female", "Male"])
    gender_code = 0 if gender == "Female" else 1
    location = st.text_input("Location", value="Lahore")
    chronic_conditions = st.text_input("Chronic Conditions", value="Diabetes")
    top_n = st.slider("Number of Top Doctors", min_value=1, max_value=10, value=3)
    
    submit_button = st.form_submit_button(label="Get Recommendations")

# ======================
# Call FastAPI if submitted
# ======================
if submit_button:
    payload = {
        "age": age,
        "gender": gender_code,
        "location": location,
        "chronic_conditions": chronic_conditions,
        "top_n": top_n
    }

    # FastAPI URL (change if deployed on server)
    api_url = "http://api:8000/recommend_top_n"
    try:
        response = requests.post(api_url, json=payload)
        data = response.json()
        
        if "top_doctors" in data:
            df = pd.DataFrame(data["top_doctors"])
            st.success(f"Top {top_n} recommended doctors")
            st.dataframe(df)
            
            # Optional: show bar chart of success rate
            st.bar_chart(df.set_index("doctor_name")["success_rate"])
        else:
            st.error(data.get("detail", "Error in API response"))

    except Exception as e:
        st.error(f"API call failed: {e}")
