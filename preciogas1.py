import sys
# Instalar scikit-learn si no está presente
!{sys.executable} -m pip install scikit-learn --upgrade

import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# ------------------------------
# Imagen y título
# ------------------------------
st.set_page_config(page_title="Predicción Gasolina", layout="centered")
st.image("precio_gas.jpg", use_column_width=True)
st.write('''# Predicción del precio de la gasolina regular ⛽''')

# ------------------------------
# Entradas del usuario
# ------------------------------
st.header('Datos para predicción')

def user_input_features():
    anio = st.number_input('Año:', min_value=2020, max_value=2030, value=2025, step=1)
    mes = st.selectbox('Mes:', 
                       ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
                        "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"])
    
    # Cargar dataset temporalmente para obtener lista de entidades
    dataset = pd.read_csv('dataset_limpio.csv')
    entidad = st.selectbox('Entidad Federativa:', dataset['ENTIDAD'].unique())
    
    meses = {
        "Enero":1, "Febrero":2, "Marzo":3, "Abril":4, "Mayo":5, "Junio":6,
        "Julio":7, "Agosto":8, "Septiembre":9, "Octubre":10, "Noviembre":11, "Diciembre":12
    }
    mes_num = meses[mes]
    
    return pd.DataFrame({'Año':[anio], 'Mes_num':[mes_num], 'ENTIDAD':[entidad]})

df = user_input_features()

# ------------------------------
# Cargar dataset y entrenar modelo
# ------------------------------
dataset = pd.read_csv('dataset_limpio.csv')
dataset = dataset.dropna(subset=['Precio'])

X = pd.get_dummies(dataset[['Año','Mes_num','ENTIDAD']], drop_first=True)
y = dataset['Precio']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------
# Predicción
# ------------------------------
input_X = pd.get_dummies(df, columns=['ENTIDAD'], drop_first=True)
input_X = input_X.reindex(columns=X.columns, fill_value=0)

prediccion = model.predict(input_X)[0]

st.subheader('Precio estimado')
st.write(f'💰 El precio estimado de la gasolina regular es: {prediccion:.2f} MXN')
