#!/usr/bin/env python3
"""Generador de dataset: edad (`x`) vs sueldo en dólares (`y`) para regresión.

Este script crea un CSV con `n` muestras donde el sueldo está fuertemente
relacionado (creciente o decreciente) con la edad. Incluye opciones para ruido,
semilla y guardado de CSV. También puede mostrar una gráfica.
"""

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_dataset(
	n: int = 10000,
	relation: str = "increasing",
	noise_std: float = 20000.0,
	seed: Optional[int] = 42,
	age_min: int = 16,
	age_max: int = 40,
	slope: Optional[float] = None,
	intercept: Optional[float] = None,
	filename: str = "dataset_players.csv",
	show_plot: bool = True,
) -> pd.DataFrame:
	"""Genera y guarda un dataset de `n` jugadores.

	Args:
		n: número de filas a generar.
		relation: 'increasing' o 'decreasing'.
		noise_std: desviación estándar del ruido (USD).
		seed: semilla para reproducibilidad.
		age_min, age_max: rango entero de edades.
		slope, intercept: si se indican, fuerzan la recta base y = intercept + slope*age.
		filename: ruta de salida CSV.
		show_plot: si True, muestra una dispersión y la recta de regresión.

	Returns:
		pd.DataFrame generado con columnas `age` y `salary`.
	"""

	rng = np.random.default_rng(seed)
	ages = rng.integers(age_min, age_max + 1, size=n)

	# Pendiente por defecto grande para asegurar alta correlación relativa al ruido
	if slope is None:
		slope = 50000.0 if relation == "increasing" else -50000.0
	if intercept is None:
		intercept = 200000.0 if relation == "increasing" else 2000000.0

	noise = rng.normal(0.0, noise_std, size=n)
	salaries = intercept + slope * ages + noise
	salaries = np.clip(salaries, a_min=0.0, a_max=None)  # no salarios negativos

	df = pd.DataFrame({"age": ages, "salary": salaries})
	df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # mezclar

	df.to_csv(filename, index=False)

	if show_plot:
		coeffs = np.polyfit(df["age"], df["salary"], 1)
		x_line = np.linspace(age_min, age_max, 100)
		y_line = coeffs[0] * x_line + coeffs[1]

		plt.figure(figsize=(8, 5))
		# muestreamos hasta 1000 puntos para no sobrecargar la gráfica
		sample = df.sample(min(1000, len(df)), random_state=seed)
		plt.scatter(sample["age"], sample["salary"], s=6, alpha=0.5)
		plt.plot(x_line, y_line, color="red", lw=2)
		plt.xlabel("Edad")
		plt.ylabel("Sueldo (USD)")
		plt.title(f"Dataset: relación {relation} (n={n}) — pendiente≈{coeffs[0]:.1f}")
		plt.grid(True)
		plt.tight_layout()
		plt.show()

	return df


if __name__ == "__main__":
	# Genera el dataset por defecto (10k filas) y lo guarda en CSV sin mostrar la gráfica
	df = generate_dataset(n=10000, relation="increasing", noise_std=20000.0, seed=42, filename="dataset_players.csv", show_plot=False)
	print("Dataset creado:", df.shape)
	print(df.head())

