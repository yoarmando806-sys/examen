import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline

# ------------------------------
# Mostrar imagen de portada
# ------------------------------
st.set_page_config(page_title="Predicción Gasolina", layout="centered")
st.image("precio_gas.jpg", use_column_width=True)

st.title("Predicción del Precio de la Gasolina ⛽")
st.write("Introduce los datos y obtén la predicción con el modelo de regresión lineal.")

# ------------------------------
# Configuración de Spark
# ------------------------------
spark = SparkSession.builder.appName("PrecioGasolinaApp").getOrCreate()

# ------------------------------
# Cargar dataset limpio
# ------------------------------
pdf = pd.read_csv("dataset_limpio.csv")
df_filtered = spark.createDataFrame(pdf)
df_filtered = df_filtered.dropna(subset=["Precio"])  # eliminar posibles NaN en Precio

# ------------------------------
# Preparar pipeline
# ------------------------------
indexer_entidad = StringIndexer(inputCol="ENTIDAD", outputCol="ENTIDAD_index")
encoder = OneHotEncoder(inputCols=["ENTIDAD_index"], outputCols=["ENTIDAD_vec"])
assembler = VectorAssembler(inputCols=["Año", "Mes_num", "ENTIDAD_vec"], outputCol="features")
lr = LinearRegression(featuresCol="features", labelCol="Precio")
pipeline = Pipeline(stages=[indexer_entidad, encoder, assembler, lr])

# Entrenar modelo
model = pipeline.fit(df_filtered)

# ------------------------------
# Entradas del usuario
# ------------------------------
anio = st.number_input("Año", min_value=2020, max_value=2030, value=2025, step=1)
mes = st.selectbox("Mes", 
                   ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
                    "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"])
entidad = st.selectbox("Entidad Federativa", 
                       df_filtered.select("ENTIDAD").distinct().rdd.flatMap(lambda x: x).collect())

meses = {
    "Enero":1, "Febrero":2, "Marzo":3, "Abril":4, "Mayo":5, "Junio":6,
    "Julio":7, "Agosto":8, "Septiembre":9, "Octubre":10, "Noviembre":11, "Diciembre":12
}
mes_num = meses[mes]

# ------------------------------
# Predicción
# ------------------------------
if st.button("Predecir precio"):
    input_data = spark.createDataFrame([(anio, mes_num, entidad)], ["Año","Mes_num","ENTIDAD"])
    pred = model.transform(input_data)
    resultado = pred.select("prediction").collect()[0][0]
    st.success(f"💰 El precio estimado es: {resultado:.2f} MXN")
