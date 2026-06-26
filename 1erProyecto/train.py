"""
Entrenamiento Q-Learning + visualización Pygame opcional.

Uso:
    python train.py                  # entrenamiento silencioso
    python train.py --render         # con Pygame periódico (NO bloquea)
    python train.py --episodios 500  # número de episodios
    python train.py --cargar         # continúa entrenamiento guardado
"""

import sys
import os
import argparse
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.overtaking_env import OvertakingEnv
from agent.q_agent import QLearningAgent

def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento Q-Learning – Adelantamiento")
    p.add_argument("--episodios",  type=int,   default=300,    help="Episodios de entrenamiento")
    p.add_argument("--render",                 action="store_true",         help="Mostrar Pygame")
    p.add_argument("--render_cada",            type=int,   default=50,     help="Render cada N episodios")
    p.add_argument("--render_cada_segundos",   type=float, default=2.0,    help="Render cada N segundos de entrenamiento real (0=desactivado)")
    p.add_argument("--cargar",                 action="store_true",         help="Cargar tabla Q existente")
    p.add_argument("--guardar_cada",type=int,  default=100,    help="Guardar tabla cada N episodios")
    p.add_argument("--tabla",      type=str,   default="q_table.pkl", help="Ruta de la tabla Q")
    return p.parse_args()


def entrenar():
    args = parse_args()

    env   = OvertakingEnv(render_mode=None)
    agent = QLearningAgent()

    if args.cargar:
        agent.cargar(args.tabla)

    historial_recompensas = []
    historial_pasos       = []
    historial_colisiones  = []
    historial_adelant     = []
    t_inicio              = time.time()
    ultimo_render_t       = time.perf_counter()

    usar_render = args.render or args.render_cada_segundos > 0

    print("=" * 60)
    print("  ENTRENAMIENTO Q-LEARNING — ADELANTAMIENTO")
    print("=" * 60)
    print(f"  Episodios : {args.episodios}")
    render_desc = "NO"
    if args.render_cada_segundos > 0:
        render_desc = f"SÍ (cada {args.render_cada_segundos:.1f}s)"
    elif args.render:
        render_desc = f"SÍ (cada {args.render_cada} ep.)"
    print(f"  Render    : {render_desc}")
    print(f"  Tabla Q   : {args.tabla}")
    print("=" * 60)

    for ep in range(1, args.episodios + 1):

        render_este_ep = False
        if usar_render:
            ahora = time.perf_counter()
            if args.render_cada_segundos > 0:
                if (ahora - ultimo_render_t) >= args.render_cada_segundos:
                    render_este_ep = True
            elif args.render and ep % args.render_cada == 0:
                render_este_ep = True

        env.render_mode = "human" if render_este_ep else None

        if render_este_ep and env._renderer is not None:
            env._renderer.reopen()

        obs, _ = env.reset()
        recompensa_ep = 0.0
        terminado     = False
        truncado      = False

        while not (terminado or truncado):
            accion = agent.elegir_accion(obs, explorar=True)
            obs_nuevo, reward, terminado, truncado, info = env.step(accion)

            agent.actualizar(obs, accion, reward, obs_nuevo, terminado or truncado)
            obs = obs_nuevo
            recompensa_ep += reward

        if render_este_ep:
            ultimo_render_t = time.perf_counter()

        agent.decaer_epsilon()

        historial_recompensas.append(recompensa_ep)
        historial_pasos.append(info["pos_agente"])
        historial_colisiones.append(int(info["colision"]))
        historial_adelant.append(info["adelantamientos"])

        if ep % 10 == 0 or ep == 1:
            ult = min(ep, 20)
            media_r   = np.mean(historial_recompensas[-ult:])
            media_p   = np.mean(historial_pasos[-ult:])
            tasa_col  = np.mean(historial_colisiones[-ult:]) * 100
            media_adv = np.mean(historial_adelant[-ult:])
            elapsed   = time.time() - t_inicio

            exito_str = "✓ META" if info["exito"] else ("✗ CHOQUE" if info["colision"] else "⏱ TIEMPO")
            print(
                f"Ep {ep:>4}/{args.episodios}  "
                f"R:{recompensa_ep:>+8.1f}  "
                f"R_med:{media_r:>+8.1f}  "
                f"Pos:{media_p:>6.0f}m  "
                f"Col:{tasa_col:>5.1f}%  "
                f"Adv:{media_adv:.1f}  "
                f"ε:{agent.epsilon:.3f}  "
                f"[{exito_str}]  "
                f"{elapsed:.0f}s"
            )

        if ep % args.guardar_cada == 0:
            agent.guardar(args.tabla)

    env.render_mode = None

    agent.guardar(args.tabla)

    print("\n" + "=" * 60)
    print("  RESUMEN FINAL")
    print("=" * 60)
    print(f"  Recompensa media (últimos 50 ep): {np.mean(historial_recompensas[-50:]):+.1f}")
    print(f"  Tasa de colisiones (últimos 50):  {np.mean(historial_colisiones[-50:])*100:.1f}%")
    print(f"  Adelantamientos medio:            {np.mean(historial_adelant[-50:]):.2f}")
    print(f"  Tiempo total:                     {time.time() - t_inicio:.1f}s")
    print("=" * 60)

    env.close()

    return historial_recompensas, historial_pasos


def demo(ruta_tabla: str = "q_table.pkl", episodios: int = 5):
    """Ejecuta el agente entrenado con visualización completa."""
    env   = OvertakingEnv(render_mode="human")
    agent = QLearningAgent(epsilon=0.0)  
    agent.cargar(ruta_tabla)

    print(f"\n[DEMO] Ejecutando {episodios} episodios con política aprendida…")

    for ep in range(1, episodios + 1):
        obs, _        = env.reset()
        recompensa_ep = 0.0
        terminado     = False
        truncado      = False

        while not (terminado or truncado):
            accion = agent.elegir_accion(obs, explorar=False)
            obs, reward, terminado, truncado, info = env.step(accion)
            recompensa_ep += reward

        resultado = "META" if info["exito"] else ("CHOQUE" if info["colision"] else "TIEMPO")
        print(f"  Episodio {ep}: R={recompensa_ep:+.1f}  Pos={info['pos_agente']:.0f}m  "
              f"Adv={info['adelantamientos']}  [{resultado}]")

    env.close()


if __name__ == "__main__":
    entrenar()