import streamlit as st
import pandas as pd
import joblib

# Título e imagen
st.write("# Predicción del precio de gasolina por estado")
st.image("gasolina.jpg", caption="Precio estimado de gasolina por estado")  # Asegúrate de tener esta imagen en la misma carpeta

# Función para entrada de datos del usuario
def user_input_features():
    Estado = st.selectbox('Estado:', [
        'Aguascalientes','Baja California','Baja California Sur','Campeche','Coahuila',
        'Colima','Chiapas','Chihuahua','Ciudad de México','Durango','Guanajuato',
        'Guerrero','Hidalgo','Jalisco','México','Michoacán','Morelos','Nayarit',
        'Nuevo León','Oaxaca','Puebla','Querétaro','Quintana Roo','San Luis Potosí',
        'Sinaloa','Sonora','Tabasco','Tamaulipas','Tlaxcala','Veracruz','Yucatán','Zacatecas'
    ])
    Mes = st.number_input('Mes (1-12):', min_value=1, max_value=12, value=1, step=1)
    Año = st.number_input('Año:', min_value=2000, max_value=2100, value=2025, step=1)

    return pd.DataFrame({'Estado': [Estado], 'Mes': [Mes], 'Año': [Año]})

# Capturar datos del usuario
df = user_input_features()

# Cargar encoder y modelo preentrenados
encoder = joblib.load('encoder_gasolina.joblib')  # codificador de estados y variables
model = joblib.load('modelo_gasolina.joblib')    # modelo de regresión entrenado

# Transformar datos y predecir
X_encoded = encoder.transform(df)  # devuelve un array listo para el modelo
prediccion = model.predict(X_encoded)

# Mostrar resultado
st.subheader('Precio estimado de gasolina')
st.write('El precio estimado es:', round(float(prediccion), 2))
