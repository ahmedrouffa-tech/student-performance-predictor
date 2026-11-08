import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Prédicteur de Performance", page_icon="🎓")
st.title("🎓 Prédicteur de Performance de l'Élève")

@st.cache_resource
def load_artifacts():
    model = joblib.load('best_linear_regression_model.joblib')
    scaler = joblib.load('robust_scaler.joblib')
    return model, scaler

try:
    model, scaler = load_artifacts()
    st.success("✅ Modèle chargé avec succès !")
except Exception as e:
    st.error(f"Erreur : {e}")

hours_studied = st.slider("Heures d'étude", 1, 9, 5)
previous_scores = st.slider("Scores précédents", 40, 99, 70)
extracurricular = st.selectbox("Activités extra-scolaires", ["Non", "Oui"])
sleep_hours = st.slider("Heures de sommeil", 4, 9, 7)
sample_papers = st.slider("Sujets d'examen pratiqués", 0, 9, 4)

if st.button("Estimer la performance 🚀"):
    ext_encoded = 1 if extracurricular == "Oui" else 0
    input_data = np.array([[hours_studied, previous_scores, ext_encoded, sleep_hours, sample_papers]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    st.metric("Indice de Performance Estimé", f"{np.clip(prediction, 10, 100):.1f} / 100")
