import streamlit as st
import pickle



# Fonction pour charger le modèle, le scaler et le PCA
@st.cache_resource
def load_models():
    with open('models/model_rbf.pkl', 'rb') as file:
        print("ls")
        model = pickle.load(file)
    with open('models/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('models/pca_model.pkl', 'rb') as file:
        pca = pickle.load(file)
    return model, scaler, pca


load_models()