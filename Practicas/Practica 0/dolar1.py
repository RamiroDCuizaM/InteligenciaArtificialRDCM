import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# 1. Cargar datos
df = pd.read_csv(r"C:\Users\ASUS\Desktop\recuperado\Disco D\UNIVERSIDAD\UNIVERSIDAD-2025-2\INGENIERIA DE SISTEMAS\SIS420-INTELIGENCIA ARTIFICIAL I\InteligenciaArtificialIRDCM\DATABASE DOLAR BLUE EN BOLIVIA - Hoja1.csv")


# 2. Procesar fecha y ordenar
df['Fecha'] = pd.to_datetime(df['Fecha'])
df = df.sort_values('Fecha')

# 3. Crear índice numérico como variable X
df['Dias'] = (df['Fecha'] - df['Fecha'].min()).dt.days

# Variable dependiente (valor del dólar)
y = df['Dolar'].values
X = df[['Dias']].values

# 4. Entrenar modelo
modelo = LinearRegression()
modelo.fit(X, y)

# 5. Predecir para el siguiente día
proximo_dia = np.array([[df['Dias'].max() + 1]])
prediccion = modelo.predict(proximo_dia)

print(f"Predicción del dólar para el siguiente día: {prediccion[0]:.2f} Bs")
