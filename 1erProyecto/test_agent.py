"""
Prueba del agente Q-Learning usando una tabla Q ya entrenada.

Ejemplo:
    python test_agent.py
    python test_agent.py --tabla q_table.pkl --episodios 10 --render
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.overtaking_env import OvertakingEnv
from agent.q_agent import QLearningAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Prueba un agente con una tabla Q entrenada.")
    parser.add_argument("--tabla",      type=str,   default="q_table.pkl", help="Ruta de la tabla Q guardada")
    parser.add_argument("--episodios",  type=int,   default=100,           help="Número de episodios de prueba")
    parser.add_argument("--render",                 action="store_true", help="Renderiza el entorno con Pygame")
    return parser.parse_args()


def evaluar_agente(ruta_tabla: str, episodios: int, render: bool):
    env   = OvertakingEnv(render_mode="human" if render else None)
    agent = QLearningAgent(epsilon=0.0)
    agent.cargar(ruta_tabla)
    agent.epsilon = 0.0

    recompensas = []
    colisiones  = []
    exitos      = []
    adelantados = []

    print("\n[TEST] Ejecutando agente con tabla Q cargada desde:", ruta_tabla)
    print(f"      Episodios: {episodios}")
    print(f"      Render: {'SÍ' if render else 'NO'}\n")

    for ep in range(1, episodios + 1):
        obs, _ = env.reset()
        recompensa_ep = 0.0
        terminado = False
        truncado  = False

        while not (terminado or truncado):
            accion = agent.elegir_accion(obs, explorar=False)
            obs, reward, terminado, truncado, info = env.step(accion)
            recompensa_ep += reward

        resultado = "META" if info["exito"] else ("CHOQUE" if info["colision"] else "TIEMPO")
        recompensas.append(recompensa_ep)
        colisiones.append(int(info["colision"]))
        exitos.append(int(info["exito"]))
        adelantados.append(info["adelantamientos"])

        print(
            f"Episodio {ep:>2}: R={recompensa_ep:+7.1f}  "
            f"Pos={info['pos_agente']:.0f}m  "
            f"Adv={info['adelantamientos']:>2}  "
            f"[{resultado}]"
        )

    print("\n[TEST] Resumen final")
    print("----------------------")
    print(f"Recompensa media: {np.mean(recompensas):+.1f}")
    print(f"Tasa de colisión: {np.mean(colisiones) * 100:.1f}%")
    print(f"Tasa de éxito:    {np.mean(exitos) * 100:.1f}%")
    print(f"Adelantamientos por episodio: {np.mean(adelantados):.2f}")

    env.close()


if __name__ == "__main__":
    args = parse_args()
    evaluar_agente(args.tabla, args.episodios, args.render)
