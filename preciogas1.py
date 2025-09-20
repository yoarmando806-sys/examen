import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Título y imagen
st.write("# Predicción del precio de gasolina por estado")
st.image("gasolina.jpg", caption="Precio estimado de gasolina por estado")  # Asegúrate de tener esta imagen en la misma carpeta

# Función para entrada de datos
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

    data = {'Estado': Estado, 'Mes': Mes, 'Año': Año}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Cargar dataset
gasolina = pd.read_csv('Gasolina.csv', encoding='latin-1')
X = pd.get_dummies(gasolina[['Estado', 'Mes', 'Año']], drop_first=True)
y = gasolina['Precio']

# Convertir inputs del usuario a mismas columnas que X
df_encoded = pd.get_dummies(df, drop_first=True)
df_encoded = df_encoded.reindex(columns=X.columns, fill_value=0)

# Entrenamiento del modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)

# Predicción
prediccion = model.predict(df_encoded)

st.subheader('Precio estimado de gasolina')
st.write('El precio estimado es:', round(float(prediccion), 2))
