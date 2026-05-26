from sklearn.linear_model import LinearRegression
import numpy as np

# Datos históricos ficticios (día, valor del dólar)
# Día 1 al 10 y su respectivo valor
dias = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
valores_dolar = np.array([13.10, 13.15, 13.12, 13.20, 13.22, 13.25, 13.30, 13.32, 13.33, 13.35])

# Crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(dias, valores_dolar)

# Predecir el valor del dólar para el día 11
dia_siguiente = np.array([[11]])
prediccion = modelo.predict(dia_siguiente)

print(f"Predicción del valor del dólar para el día {dia_siguiente[0][0]}: ${prediccion[0]:.2f}")
