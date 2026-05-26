import pandas as pd

serie = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(serie)

import pandas as pd

datos = {
    'Nombre': ['Ana', 'Luis', 'Pedro'],
    'Edad': [23, 34, 29],
    'Ciudad': ['La Paz', 'Cochabamba', 'Santa Cruz']
}

df = pd.DataFrame(datos)
print(df)

# Mostrar solo personas mayores de 30 años
mayores = df[df['Edad'] > 30]
print(mayores)

print("Promedio de edad:", df['Edad'].mean())
print("Edad máxima:", df['Edad'].max())

grupo = df.groupby('Ciudad')['Edad'].mean()
print(grupo)

