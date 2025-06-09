import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Funciones necesarias para deserializar el modelo

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

st.title("Sistema de Predicción de Rotación de Empleados")
st.markdown("""
Esta aplicación predice la probabilidad de que un empleado deje la empresa basándose en sus características.
""")

@st.cache_resource
def load_model():
    try:
        model = joblib.load('sta_model.joblib')
        return model
    except Exception as e:
        st.error(f"Error cargando modelo: {str(e)}")
        return None

model = load_model()
if model is None:
    st.stop()

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

    return pd.DataFrame([user_data])

user_input = get_user_input()
st.subheader("Datos del Empleado")
st.write(user_input)

# ----------- PREDICCIÓN INDIVIDUAL -----------
if st.sidebar.button('Predecir Rotación'):
    try:
        prediction = model.predict(user_input)
        prediction_proba = model.predict_proba(user_input)

        st.subheader("Resultado de la Predicción")

        prob_no = prediction_proba[0][0] * 100
        prob_yes = prediction_proba[0][1] * 100

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicción",
                      "Dejará la empresa" if prediction[0] == 1 else "No dejará la empresa",
                      delta=f"{prob_yes:.2f}% prob. rotación" if prediction[0] == 1 else f"{prob_no:.2f}% prob. permanencia",
                      delta_color="inverse")

        with col2:
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob_yes,
                delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "gold"},
                        {'range': [70, 100], 'color': "tomato"}
                    ],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': prob_yes}
                },
                title={'text': "Probabilidad de Rotación (%)"}
            ))
            st.plotly_chart(gauge_fig)

        st.info("""
        **Interpretación:**
        - **No dejará la empresa (0):** Probabilidad alta de permanecer.
        - **Dejará la empresa (1):** Probabilidad alta de rotación.
        """)
    except Exception as e:
        st.error(f"Error al procesar la predicción: {str(e)}")

# ----------- PREDICCIÓN POR LOTE -----------
st.markdown("---")
st.subheader("Opcional: Predicción por lote (CSV)")

uploaded_file = st.file_uploader("Suba un archivo CSV con datos de empleados", type="csv")
if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.write("Datos cargados:", batch_data.head())

        if st.button('Predecir lote'):
            with st.spinner('Procesando...'):
                batch_predictions = model.predict(batch_data)
                batch_proba = model.predict_proba(batch_data)

                results = batch_data.copy()
                results['Predicción'] = batch_predictions
                results['Probabilidad Rotación (%)'] = (batch_proba[:, 1] * 100).round(2)
                results['Probabilidad Permanencia (%)'] = (batch_proba[:, 0] * 100).round(2)

                st.success("Predicciones completadas!")
                st.dataframe(results)

                hist_fig = px.histogram(results, x="Probabilidad Rotación (%)", nbins=10,
                                        title="Distribución de Probabilidades de Rotación",
                                        labels={"Probabilidad Rotación (%)": "Probabilidad de Rotación (%)"},
                                        color_discrete_sequence=["indianred"])
                hist_fig.update_layout(bargap=0.1)
                st.plotly_chart(hist_fig)

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
