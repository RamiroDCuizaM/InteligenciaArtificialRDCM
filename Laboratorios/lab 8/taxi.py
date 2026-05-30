import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

def train(episodes):

    # =========================
    # ENTORNO TAXI
    # =========================
    env = gym.make('Taxi-v4')

    # =========================
    # Q-TABLE
    # 500 estados x 6 acciones
    # =========================
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    # =========================
    # HIPERPARÁMETROS
    # =========================
    learning_rate = 0.1
    discount_factor = 0.95

    epsilon = 1
    epsilon_decay_rate = 0.001

    rng = np.random.default_rng()

    # =========================
    # RECOMPENSAS
    # =========================
    rewards_per_episode = np.zeros(episodes)

    # =========================
    # ENTRENAMIENTO
    # =========================
    for i in range(episodes):

        # ==========================================
        # RENDER CADA 25 EPISODIOS
        # ==========================================
        if (i + 1) % 25 == 0:

            env.close()

            env = gym.make(
                'Taxi-v4',
                render_mode='human'
            )

            print(f'\n=========== EPISODIO {i+1} ===========')

        else:

            env.close()

            env = gym.make('Taxi-v4')

        # Reiniciar entorno
        state = env.reset()[0]

        terminated = False
        truncated = False

        total_reward = 0

        # ==========================================
        # BUCLE DEL EPISODIO
        # ==========================================
        while (not terminated and not truncated):

            # ======================================
            # EXPLORACIÓN / EXPLOTACIÓN
            # ======================================
            if rng.random() < epsilon:

                # EXPLORACIÓN
                action = env.action_space.sample()

            else:

                # EXPLOTACIÓN
                action = np.argmax(q_table[state, :])

            # ======================================
            # EJECUTAR ACCIÓN
            # ======================================
            new_state, reward, terminated, truncated, info = env.step(action)

            # ======================================
            # IMPLEMENTACIÓN INCREMENTAL
            # ======================================
            # Q(s,a) = Q(s,a) + alpha * (
            # reward + gamma * max(Q(s')) - Q(s,a)
            # )

            q_table[state, action] = (
                q_table[state, action]
                +
                learning_rate * (
                    reward
                    +
                    discount_factor *
                    np.max(q_table[new_state, :])
                    -
                    q_table[state, action]
                )
            )

            # Actualizar estado
            state = new_state

            total_reward += reward

            # Mostrar animación cada 25 episodios
            if (i + 1) % 25 == 0:
                env.render()
                time.sleep(0.2)

        # ==========================================
        # DISMINUIR EPSILON
        # ==========================================
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        # Guardar recompensa
        rewards_per_episode[i] = total_reward

        # Mostrar progreso
        print(f'Episodio {i+1} -> Recompensa Total: {total_reward}')

    # ==========================================
    # CERRAR ENTORNO
    # ==========================================
    env.close()

    # ==========================================
    # MOSTRAR Q-TABLE
    # ==========================================
    print('\n================ Q-TABLE FINAL ================\n')

    df_qtable = pd.DataFrame(
        q_table,
        columns=[
            'Sur',
            'Norte',
            'Este',
            'Oeste',
            'Recoger',
            'Dejar'
        ]
    )

    # Mostrar primeras filas
    print(df_qtable.head(30))

    # ==========================================
    # GRAFICAR RECOMPENSAS
    # ==========================================
    sum_rewards = np.zeros(episodes)

    for t in range(episodes):

        sum_rewards[t] = np.sum(
            rewards_per_episode[max(0, t - 100):(t + 1)]
        )

    plt.plot(sum_rewards)

    plt.title('Recompensas Acumuladas')
    plt.xlabel('Episodios')
    plt.ylabel('Suma Recompensas')

    plt.show()

    # ==========================================
    # PRUEBA FINAL DEL AGENTE
    # ==========================================
    print('\n=========== PRUEBA FINAL ===========\n')

    env = gym.make(
        'Taxi-v4',
        render_mode='human'
    )

    state = env.reset()[0]

    terminated = False
    truncated = False

    while not terminated and not truncated:

        # Mejor acción aprendida
        action = np.argmax(q_table[state, :])

        new_state, reward, terminated, truncated, info = env.step(action)

        env.render()

        time.sleep(0.5)

        state = new_state

    env.close()

# ==========================================
# MAIN
# ==========================================
if __name__ == '__main__':

    train(100)