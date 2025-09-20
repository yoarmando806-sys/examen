import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# ------------------------------
# Mostrar imagen de portada
# ------------------------------
st.set_page_config(page_title="Predicción Gasolina", layout="centered")
st.image("precio_gas.jpg", use_column_width=True)
st.title("Predicción del Precio de la Gasolina ⛽")
st.write("Introduce los datos y obtén la predicción con el modelo de regresión lineal.")

# ------------------------------
# Cargar dataset limpio
# ------------------------------
df = pd.read_csv("dataset_limpio.csv")
df = df.dropna(subset=["Precio"])  # eliminar posibles NaN en Precio

# ------------------------------
# Preparar variables para scikit-learn
# ------------------------------
X = pd.get_dummies(df[["Año","Mes_num","ENTIDAD"]], drop_first=True)
y = df["Precio"]

# Entrenar modelo
model = LinearRegression()
model.fit(X, y)

# ------------------------------
# Entradas del usuario
# ------------------------------
anio = st.number_input("Año", min_value=2020, max_value=2030, value=2025, step=1)
mes = st.selectbox("Mes", 
                   ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
                    "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"])
entidad = st.selectbox("Entidad Federativa", df["ENTIDAD"].unique())

# Convertir mes a número
meses = {
    "Enero":1, "Febrero":2, "Marzo":3, "Abril":4, "Mayo":5, "Junio":6,
    "Julio":7, "Agosto":8, "Septiembre":9, "Octubre":10, "Noviembre":11, "Diciembre":12
}
mes_num = meses[mes]

# ------------------------------
# Predicción
# ------------------------------
if st.button("Predecir precio"):
    input_df = pd.DataFrame({"Año":[anio], "Mes_num":[mes_num], "ENTIDAD":[entidad]})
    input_X = pd.get_dummies(input_df, columns=["ENTIDAD"], drop_first=True)
    
    # Asegurarse que todas las columnas coincidan con el dataset original
    input_X = input_X.reindex(columns=X.columns, fill_value=0)
    
    pred = model.predict(input_X)[0]
    st.success(f"💰 El precio estimado es: {pred:.2f} MXN")
