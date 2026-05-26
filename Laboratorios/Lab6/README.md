# Census-Income-Prediction

Proyecto de clasificacion sobre el dataset Adult Census Income. El objetivo principal es predecir si el ingreso anual de una persona es `<=50K` o `>50K` usando variables demograficas, educativas y laborales.

Este repositorio reune notebooks de preprocesamiento, entrenamiento clasico y modelado con PyTorch. El notebook central de esta version es [Census_PyTorch_Regularizacion_Optimizacion_Metricas.ipynb](Census_PyTorch_Regularizacion_Optimizacion_Metricas.ipynb), donde se comparan estrategias de regularizacion, optimizacion y metricas para el problema binario de ingresos.

## Que se quiere predecir

La variable objetivo del dataset original es `income`:

- `<=50K`
- `>50K`

Esto convierte el problema en una clasificacion binaria. El modelo aprende a estimar si una persona tiene ingresos altos o bajos a partir de sus caracteristicas del censo.

## Sobre el dataset

El dataset Adult Census Income contiene informacion de personas adultas recogida del censo. Es un dataset tabular clasico porque mezcla columnas numericas y categoricas, tiene valores faltantes y presenta desbalance entre clases.

### Variables disponibles

| Column ID | Column Name | Type | Descripcion |
|---:|---|---|---|
| 0 | age | numerica | Edad de la persona |
| 1 | workclass | categorica | Tipo de trabajo o sector |
| 2 | fnlwgt | numerica | Peso muestral final |
| 3 | education | categorica | Nivel educativo |
| 4 | education.num | numerica | Anos de educacion |
| 5 | marital.status | categorica | Estado civil |
| 6 | occupation | categorica | Ocupacion |
| 7 | relationship | categorica | Relacion familiar |
| 8 | race | categorica | Raza |
| 9 | sex | categorica | Sexo |
| 10 | capital.gain | numerica | Ganancia de capital |
| 11 | capital.loss | numerica | Perdida de capital |
| 12 | hours.per.week | numerica | Horas trabajadas por semana |
| 13 | native.country | categorica | Pais de origen |
| 14 | income | categorica | Variable objetivo |

## Flujo de trabajo

### 1. Limpieza de datos

- Se eliminan espacios extras en columnas y valores de texto.
- Se reemplazan los `?` por valores nulos.
- Se separa `income` como target y el resto como variables de entrada.

### 2. Preprocesamiento

- Las variables numericas se imputan con la mediana y se escalan con `StandardScaler`.
- Las variables categoricas se imputan con la moda y se codifican con `OneHotEncoder`.
- El preprocesamiento se ajusta solo con train para evitar fuga de informacion.

### 3. Entrenamiento con PyTorch

El notebook [Census_PyTorch_Regularizacion_Optimizacion_Metricas.ipynb](Census_PyTorch_Regularizacion_Optimizacion_Metricas.ipynb) organiza el experimento en tres partes:

- regularizacion con `Dropout`, `BatchNorm` y `weight_decay`;
- optimizacion con `SGD`, `Momentum`, `Adam` y scheduler;
- evaluacion con accuracy, precision, recall, F1, matriz de confusion y ROC-AUC.

La red usada es una MLP tabular que toma todas las variables procesadas y produce una salida binaria.

### 4. Evaluacion

Se calculan las siguientes metricas:

- accuracy;
- precision;
- recall;
- F1-score;
- matriz de confusion;
- ROC-AUC;
- reporte de clasificacion.

## Que aporta cada parte del trabajo

- `Regularizacion` ayuda a reducir sobreajuste y mejora la generalizacion.
- `Optimizacion` controla la velocidad y estabilidad del aprendizaje.
- `Metricas` permiten evaluar el modelo mas alla de la accuracy, que puede ser enganosa cuando hay desbalance.

## Otros notebooks del proyecto

Este repositorio tambien contiene otros experimentos complementarios:

- [05 regularization.ipynb](05%20regularization.ipynb)
- [06 optimization.ipynb](06%20optimization.ipynb)
- [08_metricas_clasificacion.ipynb](08_metricas_clasificacion.ipynb)
- [CLasificacionCensus_Preprocesamiento.ipynb](CLasificacionCensus_Preprocesamiento.ipynb)
- [CLasificaionCensus.ipynb](CLasificaionCensus.ipynb)
- [Census_PyTorch_Multivariable.ipynb](Census_PyTorch_Multivariable.ipynb)
- [Census_PyTorch_Multivariable_system.ipynb](Census_PyTorch_Multivariable_system.ipynb)

## Aplicacion web

El proyecto incluye una pequena app Flask para probar el modelo en una interfaz web.

- App: [Flask/app.py](Flask/app.py)
- Modelo: [Flask/model.py](Flask/model.py)
- Interfaz: [Flask/templates/index.html](Flask/templates/index.html)
- Estilos: [Flask/static/styles.css](Flask/static/styles.css)

## Como ejecutar

### Notebook

1. Abrir [Census_PyTorch_Regularizacion_Optimizacion_Metricas.ipynb](Census_PyTorch_Regularizacion_Optimizacion_Metricas.ipynb).
2. Instalar las dependencias necesarias si hace falta.
3. Ejecutar las celdas en orden.

### Aplicacion Flask

1. Entrar en la carpeta `Flask`.
2. Instalar dependencias con `pip install -r requirements.txt`.
3. Ejecutar `python app.py`.

## Estructura del repositorio

- [adult.csv](adult.csv)
- [README.md](README.md)
- [Flask/](Flask)
- [Readme/](Readme)

## Posibles mejoras

- probar arquitecturas mas profundas o con embeddings para variables categoricas;
- comparar contra modelos como LightGBM o CatBoost;
- ajustar hiperparametros con validacion cruzada;
- probar tecnicas adicionales de balanceo si la clase positiva sigue siendo dificil de detectar.

## Conclusiones

El dataset es util para estudiar clasificacion supervisada en datos tabulares reales. En este proyecto se trabajo todo el flujo: limpieza, preprocesamiento, entrenamiento, regularizacion, optimizacion y evaluacion, con el objetivo de construir un modelo que prediga correctamente el nivel de ingreso de cada persona.
