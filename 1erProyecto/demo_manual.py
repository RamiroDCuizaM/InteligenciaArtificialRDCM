"""
demo_manual.py — Control manual del entorno para pruebas visuales.

Teclas:
    ↓  /  S   → Frenar
    →  /  D   → Acelerar
    ↑  /  W   → Mantener velocidad
    ESPACIO   → Cambiar de carril
    R         → Reiniciar episodio
    ESC       → Salir
"""

import sys
import os
import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.overtaking_env import OvertakingEnv


ACCIONES = {
    pygame.K_DOWN:   0,   # Frenar
    pygame.K_s:      0,
    pygame.K_UP:     1,   # Mantener
    pygame.K_w:      1,
    pygame.K_RIGHT:  2,   # Acelerar
    pygame.K_d:      2,
    pygame.K_SPACE:  3,   # Cambiar carril
}


def main():
    pygame.init()
    env = OvertakingEnv(render_mode="human")
    obs, _ = env.reset()

    print("=" * 50)
    print("  DEMO MANUAL — Adelantamiento RL")
    print("=" * 50)
    print("  ↓/S  = Frenar   ↑/W = Mantener")
    print("  →/D  = Acelerar  ESPACIO = Cambiar carril")
    print("  R    = Reiniciar    ESC  = Salir")
    print("=" * 50)

    accion        = 1   # mantener por defecto
    recompensa_ep = 0.0
    episodio      = 1

    running = True
    while running:
        # Leer eventos de teclado
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    recompensa_ep = 0.0
                    episodio += 1
                    print(f"\n[Episodio {episodio}] Reiniciado.")
                elif event.key in ACCIONES:
                    accion = ACCIONES[event.key]

        obs, reward, terminated, truncated, info = env.step(accion)
        recompensa_ep += reward

        # Resetear acción a "mantener" salvo que se pulse otra tecla
        teclas = pygame.key.get_pressed()
        accion = 1
        for k, a in ACCIONES.items():
            if teclas[k]:
                accion = a
                break

        if terminated or truncated:
            resultado = "META" if info["exito"] else ("CHOQUE" if info["colision"] else "TIEMPO")
            print(f"Episodio {episodio} terminó: {resultado}  "
                  f"R={recompensa_ep:+.1f}  Pos={info['pos_agente']:.0f}m  "
                  f"Adv={info['adelantamientos']}")
            obs, _ = env.reset()
            recompensa_ep = 0.0
            episodio += 1
            accion = 1

    env.close()


if __name__ == "__main__":
    main()