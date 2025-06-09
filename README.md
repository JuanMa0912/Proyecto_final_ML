# Sistema de Predicción de Rotación de Empleados

Este proyecto es una aplicación web interactiva construida con **Streamlit** que predice la probabilidad de que un empleado deje la empresa, utilizando un modelo de Machine Learning basado en un ensamble tipo **Stacking**.

## 🚀 Tecnologías Utilizadas

* Python 3.10+
* Streamlit
* Scikit-learn
* Pandas, NumPy
* Plotly (gráficos interactivos)
* Joblib (serialización del modelo)

## 🧠 Modelo de Machine Learning

Se entrena un modelo de tipo **StackingClassifier** que combina:

* Random Forest
* XGBoost
* SVC (Support Vector Classifier)

Y se encapsula en un **pipeline de Scikit-learn** junto con un preprocesamiento que incluye:

* Encoding de variables categóricas
* Transformaciones numéricas (ej: `log(Age)`)
* Eliminación de duplicados

El modelo entrenado se guarda como `sta_model.joblib`.

## 📂 Estructura del Proyecto

```
.
├── app.py                  # Aplicación principal de Streamlit
├── sta_model.joblib        # Modelo entrenado con el pipeline
├── requirements.txt        # Dependencias necesarias
└── README.md               # Documentación del proyecto
```

## 🧪 Características de la Aplicación

### Predicción individual:

* Ingreso de datos por interfaz (edad, ciudad, experiencia, etc.)
* Predicción inmediata con gráfico tipo **gauge** (indicador visual)
* Resultado textual y porcentaje de probabilidad

### Predicción por lote:

* Carga de archivo CSV con datos de empleados
* Predicción masiva
* Descarga del archivo con resultados
* Visualización de histograma interactivo con distribución de probabilidades

## 📥 Ejemplo de CSV para carga por lote

```csv
Education,JoiningYear,City,PaymentTier,Age,Gender,EverBenched,ExperienceInCurrentDomain
Bachelors,2015,Bangalore,2,30,Male,No,3
Masters,2016,New Delhi,3,28,Female,Yes,2
```

## ▶️ Ejecución local

1. Clona el repositorio:

```bash
git clone https://github.com/tuusuario/rotacion-empleados-app.git
cd rotacion-empleados-app
```

2. Instala dependencias:

```bash
pip install -r requirements.txt
```

3. Ejecuta la app:

```bash
streamlit run app.py
```

## 📌 Notas importantes

* El archivo `sta_model.joblib` debe estar en el mismo directorio que `app.py`.
* El modelo requiere las funciones `eliminar_duplicados` y `apply_log_age` en el archivo donde se cargue (`app.py`).
* El pipeline fue entrenado con columnas:

```
['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain']
```

## 📄 Licencia

Este proyecto está bajo licencia MIT. Puedes usarlo, modificarlo y distribuirlo libremente.

