import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Predicción de Ventas Estimadas",
    page_icon="📊",
    layout="centered"
)

# Ruta del modelo entrenado
MODEL_PATH = "modelo_xgboost_ventas.json"

@st.cache_resource
def load_model():
    """Carga el modelo XGBoost utilizando st.cache_resource para optimizar el rendimiento."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el archivo del modelo en la ruta: {MODEL_PATH}")
    
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    return model

def main():
    st.title("📈 Predictor de Ventas Estimadas")
    st.markdown("""
    Esta aplicación utiliza un modelo de Machine Learning (**XGBoost**) para predecir las **Ventas Estimadas** 
    en función del precio unitario, el gasto en publicidad y el día de la semana.
    """)
    
    # Carga del modelo con manejo de errores robusto
    try:
        model = load_model()
    except Exception as e:
        st.error(f"⚠️ **Error al cargar el modelo:** {str(e)}")
        st.info(f"Por favor, asegúrate de que el archivo `{MODEL_PATH}` se encuentra en el directorio raíz del repositorio.")
        return

    # --- SECCIÓN LATERAL: EVALUACIÓN DE MAE ---
    st.sidebar.header("📊 Evaluación del Modelo (MAE)")
    st.sidebar.markdown("Sube un archivo CSV con datos de prueba para calcular el Error Absoluto Medio.")
    
    test_file = st.sidebar.file_uploader("Subir CSV de prueba", type=["csv"])
    
    if test_file is not None:
        try:
            df_test = pd.read_csv(test_file)
            st.sidebar.success("¡Archivo cargado con éxito!")
            
            # Selector de columna objetivo real
            target_col = st.sidebar.selectbox(
                "Selecciona la columna de Ventas Reales (Target)",
                options=df_test.columns
            )
            
            # Columnas esperadas por el modelo
            expected_features = ["Precio Unitario (USD)", "Gasto Publicidad (USD)", "Día Semana (0-6)"]
            
            # Validar que las columnas existan en el CSV
            missing_cols = [col for col in expected_features if col not in df_test.columns]
            
            if missing_cols:
                st.sidebar.error(f"⚠️ Faltan las siguientes columnas en el CSV: {missing_cols}")
            else:
                if st.sidebar.button("Calcular MAE"):
                    X_test = df_test[expected_features]
                    y_true = df_test[target_col]
                    
                    # Realizar predicciones en lote
                    dtest = xgb.DMatrix(X_test)
                    y_pred = model.predict(dtest)
                    
                    # Calcular MAE
                    mae_value = mean_absolute_error(y_true, y_pred)
                    
                    st.sidebar.metric(
                        label="MAE (Error Absoluto Medio)",
                        value=f"${mae_value:,.2f} USD"
                    )
        except Exception as e:
            st.sidebar.error(f"⚠️ Error al procesar el archivo: {str(e)}")

    st.sidebar.markdown("---")
    st.sidebar.header("Parámetros de Entrada Individual")

    with st.form("prediction_form"):
        # Formulario de entrada para las 3 variables exactas del modelo
        precio_unitario = st.number_input(
            "Precio Unitario (USD)",
            min_value=0.0,
            value=25.0,
            step=0.5,
            format="%.2f",
            help="Valor numérico en dólares del precio unitario."
        )
        
        gasto_publicidad = st.number_input(
            "Gasto Publicidad (USD)",
            min_value=0.0,
            value=100.0,
            step=1.0,
            format="%.2f",
            help="Valor numérico en dólares destinado a publicidad."
        )
        
        dia_semana = st.selectbox(
            "Día Semana (0-6)",
            options=list(range(7)),
            index=0,
            help="Día de la semana (0 = Primer día de la semana, 6 = Último día)."
        )
        
        submitted = st.form_submit_button("Predecir Ventas")

    if submitted:
        # Validación de datos en la interfaz
        if precio_unitario < 0 or gasto_publicidad < 0:
            st.error("⚠️ Los valores de Precio Unitario y Gasto Publicidad no pueden ser negativos.")
            return
            
        if not (0 <= dia_semana <= 6):
            st.error("⚠️ El Día de la Semana debe estar dentro del rango entero de 0 a 6.")
            return

        # Preparación de los datos usando un DataFrame de Pandas con los nombres exactos esperados
        input_df = pd.DataFrame(
            [[precio_unitario, gasto_publicidad, float(dia_semana)]],
            columns=["Precio Unitario (USD)", "Gasto Publicidad (USD)", "Día Semana (0-6)"]
        )
        
        try:
            dmatrix = xgb.DMatrix(input_df)
            prediction = model.predict(dmatrix)
            pred_value = float(prediction[0])
            
            st.success("¡Predicción realizada con éxito!")
            st.metric(
                label="Ventas Estimadas", 
                value=f"${pred_value:,.2f} USD"
            )
            
            with st.expander("Ver resumen de entradas"):
                st.write(f"- **Precio Unitario:** ${precio_unitario:,.2f} USD")
                st.write(f"- **Gasto Publicidad:** ${gasto_publicidad:,.2f} USD")
                st.write(f"- **Día de la Semana:** {dia_semana}")
                
        except Exception as e:
            st.error(f"⚠️ Ocurrió un error al ejecutar la predicción: {str(e)}")

if __name__ == "__main__":
    main()
