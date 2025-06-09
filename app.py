# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Funciones personalizadas necesarias para el pipeline
def eliminar_duplicados(data):
    data = data.copy()
    return data.drop_duplicates(keep='first')


def apply_log_age(df):
    df = df.copy()
    df['Age'] = np.log(df['Age'] + 1)
    return df

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

# Cargar modelos
@st.cache_resource
def load_models():
    try:
        model_objects = joblib.load('model_full.joblib')
        return model_objects['preprocessor'], model_objects['model']
    except Exception as e:
        st.error(f"Error cargando modelos: {str(e)}")
        return None, None

preprocessing_pipeline, model = load_models()

if preprocessing_pipeline is None or model is None:
    st.stop()

# Entrada de usuario
def get_user_input():
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

    columnas_modelo = [
        'Education',
        'JoiningYear',
        'City',
        'PaymentTier',
        'Age',
        'Gender',
        'EverBenched',
        'ExperienceInCurrentDomain'
    ]

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

    return pd.DataFrame([user_data], columns=columnas_modelo)

# Obtener input del usuario
user_input = get_user_input()

# Mostrar los datos ingresados
st.subheader("Datos del Empleado")
st.write(user_input)

# Botón de predicción
if st.sidebar.button('Predecir Rotación'):
    try:
        expected_cols = [
            'Education', 'JoiningYear', 'City', 'PaymentTier',
            'Age', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain'
        ]
        missing_cols = set(expected_cols) - set(user_input.columns)
        if missing_cols:
            st.error(f"Faltan columnas en el input: {missing_cols}")
            st.stop()

        # 🔍 Validación antes del transform
        st.subheader("Validación de columnas antes del transform")
        st.write("Columnas actuales del input:")
        st.write(user_input.columns.tolist())

        st.write("Tipos de datos:")
        st.write(user_input.dtypes)

        # (opcional) Forzar tipos esperados
        user_input = user_input.astype({
            'Education': 'object',
            'JoiningYear': 'int64',
            'City': 'object',
            'PaymentTier': 'int64',
            'Age': 'float64',
            'Gender': 'object',
            'EverBenched': 'object',
            'ExperienceInCurrentDomain': 'int64'
        })

        # Preprocesamiento y predicción
        processed_data = preprocessing_pipeline.transform(user_input)
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
            proba_df = pd.DataFrame({
                'Probabilidad': [prediction_proba[0][0], prediction_proba[0][1]],
                'Clase': ['No Rotación', 'Rotación']
            })
            st.bar_chart(proba_df.set_index('Clase'))

        st.info("""
        **Interpretación:**
        - **No dejará la empresa (0):** Probabilidad alta de permanecer
        - **Dejará la empresa (1):** Probabilidad alta de rotación
        """)
    except Exception as e:
        st.error(f"Error al procesar la predicción: {str(e)}")

# Información adicional
st.sidebar.markdown("---")
st.sidebar.info("""
**Nota:** 
Este modelo utiliza un ensamble STA (Stacking) entrenado con Random Forest, XGBoost y SVC para predecir la rotación de empleados.
""")

# Predicción por lote (CSV)
st.markdown("---")
st.subheader("Opcional: Predicción por lote (CSV)")

uploaded_file = st.file_uploader("Suba un archivo CSV con datos de empleados", type="csv")
if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)

        expected_cols = [
            'Education', 'JoiningYear', 'City', 'PaymentTier',
            'Age', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain'
        ]
        missing_cols = set(expected_cols) - set(batch_data.columns)
        if missing_cols:
            st.error(f"El archivo no contiene estas columnas requeridas: {missing_cols}")
            st.stop()

        st.write("Datos cargados:", batch_data.head())

        if st.button('Predecir lote'):
            with st.spinner('Procesando...'):
                batch_data = batch_data.astype({
                    'Education': 'object',
                    'JoiningYear': 'int64',
                    'City': 'object',
                    'PaymentTier': 'int64',
                    'Age': 'float64',
                    'Gender': 'object',
                    'EverBenched': 'object',
                    'ExperienceInCurrentDomain': 'int64'
                })

                processed_batch = preprocessing_pipeline.transform(batch_data)
                batch_predictions = model.predict(processed_batch)
                batch_proba = model.predict_proba(processed_batch)

                results = batch_data.copy()
                results['Predicción'] = batch_predictions
                results['Probabilidad Rotación'] = batch_proba[:, 1]
                results['Probabilidad Permanencia'] = batch_proba[:, 0]

                st.success("Predicciones completadas!")
                st.dataframe(results)

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
