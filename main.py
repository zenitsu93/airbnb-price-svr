import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Prix Airbnb",
    layout="wide"
)

# Fonction pour charger le modèle, le scaler et le PCA
@st.cache_resource
def load_models():
    try:
        model = joblib.load('models/model_rbf.pkl')
        scaler = joblib.load('models/scaler.pkl')
        pca = joblib.load('models/pca_model.pkl')
        return model, scaler, pca
    except Exception as e:
        st.error(f"⚠️ Erreur: {str(e)}")
        st.info("Assurez-vous que les fichiers .joblib sont présents dans le répertoire")
        return None, None, None

def main():
    st.title('💰 Prédicteur de Prix Airbnb')
    
    model, scaler, pca = load_models()
    if not model:
        return

    # Création de 3 colonnes pour une meilleure organisation
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📊 Informations sur l'hôte")
        host_response_rate = st.slider('Taux de réponse de l\'hôte (%)', 0, 100, 95)
        host_listings = st.number_input('Nombre total d\'annonces de l\'hôte', 1, 100, 1)

        st.subheader("📍 Localisation")
        latitude = st.number_input('Latitude', -90.0, 90.0, 48.8566, format="%.4f")
        longitude = st.number_input('Longitude', -180.0, 180.0, 2.3522, format="%.4f")

    with col2:
        st.subheader("🏠 Caractéristiques du logement")
        bathrooms = st.number_input('Nombre de salles de bain', 0.0, 10.0, 1.0, 0.5)
        bedrooms = st.number_input('Nombre de chambres', 0, 10, 1)
        beds = st.number_input('Nombre de lits', 0, 20, 1)
        
        st.subheader("💰 Tarification")
        security_deposit = st.number_input('Dépôt de garantie (€)', 0, 1000, 0)
        cleaning_fee = st.number_input('Frais de ménage (€)', 0, 500, 30)
        guests_included = st.number_input('Voyageurs inclus', 1, 16, 2)
        extra_people = st.number_input('Prix par personne supplémentaire (€)', 0, 100, 0)

    with col3:
        st.subheader("📅 Disponibilité et durée")
        minimum_nights = st.number_input('Nombre minimum de nuits', 1, 30, 1)
        maximum_nights = st.number_input('Nombre maximum de nuits', 1, 365, 30)
        availability_30 = st.slider('Disponibilité sur 30 jours', 0, 30, 15)
        availability_90 = st.slider('Disponibilité sur 90 jours', 0, 90, 45)

        st.subheader("⭐ Évaluations")
        number_reviews = st.number_input('Nombre d\'avis', 0, 500, 0)
        reviews_per_month = st.number_input('Avis par mois', 0.0, 20.0, 0.0, format="%.1f")
        
        with st.expander("Notes détaillées"):
            review_scores = {
                'Review Scores Rating': st.slider('Note globale', 0, 100, 90),
                'Review Scores Accuracy': st.slider('Précision', 0, 10, 9),
                'Review Scores Cleanliness': st.slider('Propreté', 0, 10, 9),
                'Review Scores Checkin': st.slider('Arrivée', 0, 10, 9),
                'Review Scores Communication': st.slider('Communication', 0, 10, 9),
                'Review Scores Location': st.slider('Emplacement', 0, 10, 9),
                'Review Scores Value': st.slider('Rapport qualité/prix', 0, 10, 9)
            }

    # Bouton pour prédire
    if st.button('Prédire le prix'):
        # Création du DataFrame avec les inputs
        input_data = pd.DataFrame({
            'Host Response Rate': [host_response_rate],
            'Host Total Listings Count': [host_listings],
            'Latitude': [latitude],
            'Longitude': [longitude],
            'Bathrooms': [bathrooms],
            'Bedrooms': [bedrooms],
            'Beds': [beds],
            'Security Deposit': [security_deposit],
            'Cleaning Fee': [cleaning_fee],
            'Guests Included': [guests_included],
            'Extra People': [extra_people],
            'Minimum Nights': [minimum_nights],
            'Maximum Nights': [maximum_nights],
            'Availability 30': [availability_30],
            'Availability 90': [availability_90],
            'Number of Reviews': [number_reviews],
            'Review Scores Rating': [review_scores['Review Scores Rating']],
            'Review Scores Accuracy': [review_scores['Review Scores Accuracy']],
            'Review Scores Cleanliness': [review_scores['Review Scores Cleanliness']],
            'Review Scores Checkin': [review_scores['Review Scores Checkin']],
            'Review Scores Communication': [review_scores['Review Scores Communication']],
            'Review Scores Location': [review_scores['Review Scores Location']],
            'Review Scores Value': [review_scores['Review Scores Value']],
            'Reviews per Month': [reviews_per_month]
        })

        try:
            # Standardisation des données
            input_scaled = scaler.transform(input_data)
            
            # Application du PCA
            input_pca = pca.transform(input_scaled)
            
            # Prédiction
            prediction = model.predict(input_pca)[0]
            
            # Animation du résultat
            st.markdown("""
            <style>
            @keyframes pulse {
                0% { transform: scale(1) rotate(0deg); }
                25% { transform: scale(1.1) rotate(-5deg); }
                50% { transform: scale(1.2) rotate(0deg); }
                75% { transform: scale(1.1) rotate(5deg); }
                100% { transform: scale(1) rotate(0deg); }
            }
            .price-container {
                background: linear-gradient(145deg, #f0f2f6, #e6e9ef);
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                margin-top: 20px;
                box-shadow: 5px 5px 15px rgba(0,0,0,0.1);
            }
            .price-display {
                font-size: 48px;
                font-weight: bold;
                color: #2e7d32;
                animation: pulse 2s infinite;
                display: inline-block;
            }
            .price-label {
                font-size: 18px;
                color: #666;
                margin-top: 10px;
                font-weight: 500;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class='price-container'>
                <div class='price-display'>
                    💰 {prediction:.2f}€
                </div>
                <p class='price-label'>Prix estimé par nuit</p>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Une erreur s'est produite lors de la prédiction: {str(e)}")
            st.write("Données d'entrée pour le débogage:", input_data)

if __name__ == '__main__':
    main()