import streamlit as st
from src.inference import load_bilstm, predict_top3

st.set_page_config(page_title="Smart MCQ Solver", layout="centered")

# @st.cache_resource ensures the model only loads once when the app starts, 
# preventing lag every time the user clicks "Predict".
@st.cache_resource
def init_model():
    return load_bilstm("models/bilstm_model.pt")

st.title("🧠 Smart Science MCQ Solver")
st.markdown("Enter a scientific prompt and 5 options. The model will rank the top 3 most likely answers.")

with st.spinner("Loading BiLSTM Model Weights..."):
    model, tokenizer = init_model()

prompt = st.text_area("Scientific Question / Prompt:", height=100)

col1, col2 = st.columns(2)
with col1:
    opt_a = st.text_input("Option A:")
    opt_b = st.text_input("Option B:")
    opt_c = st.text_input("Option C:")
with col2:
    opt_d = st.text_input("Option D:")
    opt_e = st.text_input("Option E:")

if st.button("Predict Top 3 Answers", type="primary"):
    options = [opt_a, opt_b, opt_c, opt_d, opt_e]
    
    if not prompt or not all(options):
        st.warning("Please enter a prompt and fill out all 5 options.")
    else:
        with st.spinner("Analyzing sequence with Custom Attention..."):
            results = predict_top3(prompt, options, model, tokenizer)
            
            st.success("Analysis Complete!")
            
            st.markdown(f"### 🏆 1st Prediction: Option {results[0][0]}")
            st.write(f"*{results[0][1]}* (Confidence: {results[0][2]:.1%})")
            
            st.markdown(f"### 🥈 2nd Prediction: Option {results[1][0]}")
            st.write(f"*{results[1][1]}* (Confidence: {results[1][2]:.1%})")
            
            st.markdown(f"### 🥉 3rd Prediction: Option {results[2][0]}")
            st.write(f"*{results[2][1]}* (Confidence: {results[2][2]:.1%})")
