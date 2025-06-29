import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("best_model_final.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Load dataset
df = pd.read_csv("Final dataset.csv")
df.drop(columns=['Name', 'Unnamed: 22'], inplace=True, errors='ignore')
target_columns = ['Predicted Career', 'Predicted Degree', 'Predicted University']
feature_columns = [col for col in df.columns if col not in target_columns]

# Helper: get top N predictions
def get_top_n_predictions(probabilities, encoder, n=2):
    top_indices = np.argsort(probabilities)[::-1][:n]
    return encoder.inverse_transform(top_indices)

# App Title
st.title("🎓 Career, Degree, and University Predictor - Pakistan")

# Name
st.markdown("<b style='font-size:18px;'>🧑 Your Name</b>", unsafe_allow_html=True)
user_name = st.text_input("", key="name_input")

# Age
st.markdown("<b style='font-size:18px;'>🎂 Your Age</b>", unsafe_allow_html=True)
age = st.slider("", min_value=16, max_value=22, value=18, key="age_slider")
st.markdown(f"<b style='font-size:16px;'>Selected Age: {age}</b>", unsafe_allow_html=True)

user_input = {'Age': age}

# Inputs Section
st.markdown("<hr style='margin-top:10px;margin-bottom:10px;'>", unsafe_allow_html=True)
st.header("📝 Additional Details")

for col in feature_columns:
    if col == 'Age':
        continue

    label = col
    if col.lower() == 'physical trait':
        label = 'Physical Activeness'
    elif col.lower() == 'personality trait':
        label = 'Personality Traits (Choose up to 3)'

    st.markdown(f"<b style='font-size:16px;'>{label}</b>", unsafe_allow_html=True)

    if df[col].dtype == object:
        if 'personality' in col.lower():
            predefined_traits = [
                "Adventurous",
                "Analytical",
                "Assertive",
                "Creative",
                "Curious",
                "Detail-Oriented",
                "Empathetic",
                "Organized",
                "Patient",
                "Practical"
            ]
            selected_traits = st.multiselect(
                label="",
                options=predefined_traits,
                max_selections=3,
                key=f"{col}_multi"
            )
            user_input[col] = ', '.join(selected_traits) if selected_traits else predefined_traits[0]
        else:
            options = sorted(df[col].dropna().unique().tolist())
            user_input[col] = st.selectbox("", options, key=f"{col}_select")
    else:
        min_val = int(df[col].min())
        max_val = int(df[col].max())
        mean_val = int(df[col].mean())
        user_input[col] = st.slider("", min_val, max_val, mean_val, key=f"{col}_slider")

    st.markdown("<div style='margin-top:-8px;margin-bottom:5px;'></div>", unsafe_allow_html=True)

# Predict button
if st.button("🔍 Predict Career Path"):
    input_df = pd.DataFrame([user_input])

    # Encode input
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    # Predict careers
    career_encoder = label_encoders['Predicted Career']
    career_model = model.estimators_[0]
    career_probs = career_model.predict_proba(input_df)[0]
    top_careers = get_top_n_predictions(career_probs, career_encoder, n=2)

    # Build results dictionary: career -> degree -> set(universities)
    rec = {
        "Careers": top_careers,
        "Career_Degree_University": {}
    }

    for career in top_careers:
        matches = df[df['Predicted Career'] == career]
        degree_uni_map = {}
        for _, row in matches.iterrows():
            degree = row['Predicted Degree']
            university = row['Predicted University']
            if pd.notna(degree) and pd.notna(university):
                degree_uni_map.setdefault(degree, set()).add(university)
        rec["Career_Degree_University"][career] = degree_uni_map

    # --- Results Display ---
    st.markdown(
        f"<h3 style='text-align: center;'>🎉 Hi <span style='font-size:22px; color:#004d40; font-weight:bold;'>{user_name if user_name else 'there'}</span>, here are your personalized predictions:</h3>",
        unsafe_allow_html=True
    )

    # Display Headers
    cols = st.columns([2, 2, 4])
    with cols[0]:
        st.markdown("<h4 style='color:#FF6F61;'>💼 Career</h4>", unsafe_allow_html=True)
    with cols[1]:
        st.markdown("<h4 style='color:#6A1B9A;'>🎓 Degree</h4>", unsafe_allow_html=True)
    with cols[2]:
        st.markdown("<h4 style='color:#00796B;'>🏫 Universities</h4>", unsafe_allow_html=True)

    # Flat row-wise display: career - degree - universities (comma separated)
    for career in rec["Careers"]:
        degree_uni_map = rec["Career_Degree_University"].get(career, {})
        for degree, universities in degree_uni_map.items():
            unis_str = ", ".join(sorted(universities))
            cols = st.columns([2, 2, 4])
            with cols[0]:
                st.markdown(f"<div style='font-weight:bold; color:black;'>{career}</div>", unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"<div style='color:black;'>{degree}</div>", unsafe_allow_html=True)
            with cols[2]:
                st.markdown(f"<div style='color:black;'>{unis_str}</div>", unsafe_allow_html=True)

    # Final Good Luck Message
    st.markdown(f"""
        <div style='
            margin-top: 30px;
            background-color: #e0f7fa;
            border-left: 6px solid #00796B;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-family: Arial, sans-serif;
        '>
            <h4 style='color: #004d40;'>🌟 Good luck, {user_name if user_name else "future star"}! 🌟</h4>
            <p style='color: #333; font-size: 16px;'>
                Your journey is just beginning — believe in yourself and pursue your dreams with confidence. 🚀
            </p>
        </div>
    """, unsafe_allow_html=True)