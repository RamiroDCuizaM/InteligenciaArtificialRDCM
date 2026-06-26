"""
Agente Q-Learning con tabla de estados discretizados.
Compatible con el entorno OvertakingEnv.
"""

import numpy as np
import pickle
import os


BINS = [
    np.linspace(0, 1, 8),        
    np.linspace(0, 1, 6),    
    np.linspace(0, 1, 4),    
    np.linspace(0, 1, 8),    
    np.linspace(0, 1, 4),    
    np.array([0.5]),         
]

N_BINS = [len(b) + 1 for b in BINS]   


def discretizar(obs: np.ndarray) -> tuple:
    """Convierte observación continua en índice discreto."""
    indices = []
    for i, val in enumerate(obs):
        idx = int(np.digitize(val, BINS[i]))
        idx = np.clip(idx, 0, N_BINS[i] - 1)
        indices.append(idx)
    return tuple(indices)


class QLearningAgent:
    """Agente tabular Q-Learning con política ε-greedy."""

    def __init__(
        self,
        n_acciones:    int   = 4,
        lr:            float = 0.1,
        gamma:         float = 0.95,
        epsilon:       float = 1.0,
        epsilon_min:   float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.n_acciones    = n_acciones
        self.lr            = lr
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        
        forma = tuple(N_BINS) + (n_acciones,)
        self.Q = np.zeros(forma, dtype=np.float32)

    
    def elegir_accion(self, obs: np.ndarray, explorar: bool = True) -> int:
        estado = discretizar(obs)
        if explorar and np.random.random() < self.epsilon:
            return np.random.randint(self.n_acciones)
        return int(np.argmax(self.Q[estado]))

    
    def actualizar(
        self,
        obs:         np.ndarray,
        accion:      int,
        recompensa:  float,
        obs_nuevo:   np.ndarray,
        terminado:   bool,
    ):
        s  = discretizar(obs)
        s_ = discretizar(obs_nuevo)

        q_actual = self.Q[s][accion]
        q_max    = 0.0 if terminado else float(np.max(self.Q[s_]))
        objetivo = recompensa + self.gamma * q_max
        self.Q[s][accion] += self.lr * (objetivo - q_actual)

    
    def decaer_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    
    def guardar(self, ruta: str = "q_table.pkl"):
        with open(ruta, "wb") as f:
            pickle.dump({"Q": self.Q, "epsilon": self.epsilon}, f)
        print(f"[Q-Learning] Tabla guardada en {ruta}")

    def cargar(self, ruta: str = "q_table.pkl"):
        if not os.path.exists(ruta):
            print(f"[Q-Learning] No se encontró {ruta}. Iniciando desde cero.")
            return
        with open(ruta, "rb") as f:
            data = pickle.load(f)
        self.Q       = data["Q"]
        self.epsilon = data.get("epsilon", self.epsilon_min)
        print(f"[Q-Learning] Tabla cargada desde {ruta}  (ε={self.epsilon:.3f})")