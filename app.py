# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from PIL import Image

class AddPrefixToColumns(BaseEstimator, TransformerMixin):
    def __init__(self, prefix=""):
        self.prefix = prefix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X.columns = [f"{self.prefix}{col}" for col in X.columns]
        return X

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Rotación de Empleados",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y descripción
st.title("Sistema de Predicción de Rotación de Empleados")
st.markdown("""
Esta aplicación predice la probabilidad de que un empleado deje la empresa basándose en sus características.
""")

# Cargar los modelos (mejor hacerlo una vez al inicio)
@st.cache_resource
def load_models():
    try:
        # Cargar todo en un solo archivo
        model_objects = joblib.load('model_full.joblib')
        return model_objects['preprocessor'], model_objects['model']
        
        # O cargar por separado (descomenta si prefieres)
        # preprocessor = joblib.load('preprocessing_pipeline.joblib')
        # model = joblib.load('sta_model.joblib')
        # return preprocessor, model
        
    except Exception as e:
        st.error(f"Error cargando modelos: {str(e)}")
        return None, None

preprocessing_pipeline, model = load_models()

if preprocessing_pipeline is None or model is None:
    st.stop()  # Detener la app si no se cargan los modelos
    
# Función para obtener los inputs del usuario
def get_user_input():
    # Dividir en columnas para mejor organización
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        education = st.selectbox('Nivel Educativo', ['Bachelors', 'Masters', 'PHD'])
        joining_year = st.slider('Año de Ingreso', 2012, 2018, 2015)
        city = st.selectbox('Ciudad', ['Bangalore', 'New Delhi', 'Pune'])
        payment_tier = st.slider('Nivel de Pago (1-3)', 1, 3, 2)
        
    with col2:
        age = st.slider('Edad', 20, 60, 30)
        gender = st.selectbox('Género', ['Male', 'Female'])
        ever_benched = st.selectbox('Alguna vez en banquillo', ['No', 'Yes'])
        experience = st.slider('Experiencia en dominio actual (años)', 0, 10, 3)
    
    # Crear diccionario con los datos
    user_data = {
        'Education': education,
        'JoiningYear': joining_year,
        'City': city,
        'PaymentTier': payment_tier,
        'Age': age,
        'Gender': gender,
        'EverBenched': ever_benched,
        'ExperienceInCurrentDomain': experience
    }
    
    # Convertir a DataFrame
    features = pd.DataFrame(user_data, index=[0])
    return features

# Obtener input del usuario
user_input = get_user_input()

# Mostrar los datos ingresados
st.subheader("Datos del Empleado")
st.write(user_input)

# Preprocesamiento y predicción
if st.sidebar.button('Predecir Rotación'):
    try:
        # Preprocesar los datos
        processed_data = preprocessing_pipeline.transform(user_input)
        
        # Hacer la predicción
        prediction = model.predict(processed_data)
        prediction_proba = model.predict_proba(processed_data)
        
        # Mostrar resultados
        st.subheader("Resultado de la Predicción")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicción", 
                     "Dejará la empresa" if prediction[0] == 1 else "No dejará la empresa",
                     delta=f"{prediction_proba[0][1]*100:.2f}% de probabilidad" if prediction[0] == 1 else f"{prediction_proba[0][0]*100:.2f}% de probabilidad",
                     delta_color="inverse")
        
        with col2:
            # Gráfico de probabilidades
            proba_df = pd.DataFrame({
                'Probabilidad': [prediction_proba[0][0], prediction_proba[0][1]],
                'Clase': ['No Rotación', 'Rotación']
            })
            st.bar_chart(proba_df.set_index('Clase'))
        
        # Explicación adicional
        st.info("""
        **Interpretación:**
        - **No dejará la empresa (0):** Probabilidad alta de permanecer
        - **Dejará la empresa (1):** Probabilidad alta de rotación
        """)
        
    except Exception as e:
        st.error(f"Error al procesar la predicción: {str(e)}")

# Sección de información adicional
st.sidebar.markdown("---")
st.sidebar.info("""
**Nota:** 
Este modelo utiliza un ensamble STA (Stacking) entrenado con Random Forest, XGBoost y SVC para predecir la rotación de empleados.
""")

# Posible sección para cargar archivos CSV (opcional)
st.markdown("---")
st.subheader("Opcional: Predicción por lote (CSV)")

uploaded_file = st.file_uploader("Suba un archivo CSV con datos de empleados", type="csv")
if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.write("Datos cargados:", batch_data.head())
        
        if st.button('Predecir lote'):
            with st.spinner('Procesando...'):
                # Preprocesar
                processed_batch = preprocessing_pipeline.transform(batch_data)
                
                # Predecir
                batch_predictions = model.predict(processed_batch)
                batch_proba = model.predict_proba(processed_batch)
                
                # Añadir resultados al DataFrame
                results = batch_data.copy()
                results['Predicción'] = batch_predictions
                results['Probabilidad Rotación'] = batch_proba[:, 1]
                results['Probabilidad Permanencia'] = batch_proba[:, 0]
                
                st.success("Predicciones completadas!")
                st.dataframe(results)
                
                # Opción para descargar resultados
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Descargar resultados",
                    csv,
                    "resultados_prediccion.csv",
                    "text/csv",
                    key='download-csv'
                )
    except Exception as e:
        st.error(f"Error al procesar el archivo: {str(e)}")
