"""
Script para visualizar e inspeccionar el contenido de la tabla Q entrenada (q_table.pkl).
Muestra el estado discretizado del entorno a la izquierda y el valor de las acciones a la derecha.
"""

import pickle
import os
import numpy as np


ACCIONES = ["Frenar", "Mantener", "Acelerar", "Cambiar Carril"]


N_BINS_NOMBRES = {
    0: ["Muy Cerca", "Cerca", "Media-Cerca", "Media", "Media-Lejos", "Lejos", "Muy Lejos", "Fuera de Rango (Min)", "Fuera de Rango (Max)"],
    5: ["Derecho", "Izquierdo"]
}

def main():
    ruta_tabla = "q_table.pkl"

    if not os.path.exists(ruta_tabla):
        print(f"Error: No se encontró el archivo '{ruta_tabla}'.")
        print("Por favor, ejecuta primero el entrenamiento con: python train.py")
        return

    print(f"Cargando la tabla Q desde: {ruta_tabla} ...")
    with open(ruta_tabla, "rb") as f:
        data = pickle.load(f)
    
    Q = data["Q"]
    epsilon = data.get("epsilon", 0.0)

   
    shape = Q.shape
    print("=" * 70)
    print(f"  INFORMACIÓN DE LA TABLA Q")
    print("=" * 70)
    print(f"  Dimensiones del tensor Q : {shape}")
    print(f"  Epsilon actual guardado  : {epsilon:.4f}")
    

    indices_visitados = np.argwhere(np.any(Q != 0.0, axis=-1))
    total_visitados = len(indices_visitados)
    total_estados = np.prod(shape[:-1])
    porcentaje = (total_visitados / total_estados) * 100

    print(f"  Estados totales posibles : {total_estados}")
    print(f"  Estados visitados/útiles : {total_visitados} ({porcentaje:.2f}%)")
    print("=" * 70)

    if total_visitados == 0:
        print("La tabla Q está vacía (todos los valores son cero). Entrena al agente primero.")
        return

    max_filas_a_mostrar = 40
    print(f"Mostrando los primeros {max_filas_a_mostrar} estados entrenados:\n")

    header = f"{'ESTADO DEL ENTORNO [i0][i1][i2][i3][i4][i5]':<45} | {'VALORES Q POR ACCIÓN':<55} | {'MEJOR ACCIÓN'}"
    print(header)
    print("-" * len(header))

    count = 0
    for idx in indices_visitados:
        state_tuple = tuple(idx)
        q_values = Q[state_tuple]
        
        mejor_accion_idx = np.argmax(q_values)
        mejor_accion_nombre = ACCIONES[mejor_accion_idx]
        mejor_q_val = q_values[mejor_accion_idx]

        estado_str = "".join(f"[{i}]" for i in state_tuple)

        acciones_valores = []
        for i, val in enumerate(q_values):

            prefijo = "*" if i == mejor_accion_idx else ""
            acciones_valores.append(f"{prefijo}{ACCIONES[i][:4]}:{val:>+6.1f}")
        
        acciones_str = ", ".join(acciones_valores)

        print(f"Estado {estado_str:<38} | [{acciones_str}] | {mejor_accion_nombre} ({mejor_q_val:+.1f})")
        
        count += 1
        if count >= max_filas_a_mostrar:
            break

    print("-" * len(header))
    print(f"Leyenda del Estado: [DistLento][VelAgente][VelLento][DistContrario][VelContrario][Carril]")
    print(f"(*indica la mejor acción elegida por la política aprendida)")

if __name__ == "__main__":
    main()
