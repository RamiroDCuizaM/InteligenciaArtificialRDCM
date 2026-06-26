"""
Entorno Gymnasium: Simulación de Adelantamiento Inteligente
============================================================
Observaciones (6 valores normalizados):
  [0] dist_vehiculo_delantero   – [0, 1]
  [1] velocidad_agente          – [0, 1]
  [2] velocidad_delantero       – [0, 1]
  [3] dist_vehiculo_contrario   – [0, 1]
  [4] velocidad_contrario       – [0, 1]
  [5] carril_actual             – 0 = derecho, 1 = izquierdo

Acciones discretas:
  0 → Frenar
  1 → Mantener velocidad
  2 → Acelerar
  3 → Cambiar de carril
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

LONGITUD_CARRETERA   = 1000.0   
DT                   = 0.1      

VEL_MIN              = 5.0      
VEL_MAX              = 40.0     
VEL_INICIAL_AGENTE   = 20.0     
DELTA_VEL            = 2.0      

DIST_SEGURA          = 20.0     
DIST_PELIGRO         = 8.0      
DIST_CHOQUE          = 3.0      

VEL_LENTO_MIN        = 8.0
VEL_LENTO_MAX        = 14.0
DIST_LENTO_INICIAL   = 40.0     

VEL_CONTRARIO_MIN    = 10.0
VEL_CONTRARIO_MAX    = 25.0
DIST_REAPARICION     = 300.0    

PASOS_MAX            = 3000     

class OvertakingEnv(gym.Env):
    """Entorno de adelantamiento para un agente Q-Learning / DQN."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )

    
        self.action_space = spaces.Discrete(4)


        self._renderer = None
        
        from env.renderer import GAME_SPEED_FACTOR
        self.game_speed_factor = GAME_SPEED_FACTOR

        self._init_state()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._init_state()
        obs = self._get_obs()
        info = {}
        return obs, info

    def _init_state(self):
        self.pos_agente        = 0.0
        self.vel_agente        = VEL_INICIAL_AGENTE
        self.carril_agente     = 0         
        self.pasos             = 0
        self.recompensa_total  = 0.0
        self.episodios         = 0          
        self.colision          = False
        self.exito             = False

       
        offset = self.np_random.uniform(30.0, 70.0) if hasattr(self, "np_random") and self.np_random is not None else DIST_LENTO_INICIAL
        self.pos_lento = self.pos_agente + offset
        self.vel_lento = self.np_random.uniform(VEL_LENTO_MIN, VEL_LENTO_MAX) if hasattr(self, "np_random") and self.np_random is not None else 10.0

    
        self.pos_contrario    = self.pos_agente + DIST_REAPARICION
        self.vel_contrario    = self._nueva_vel_contrario()

      
        self.adelantamientos  = 0
        self._en_adelant      = False      

    def _nueva_vel_contrario(self):
        if hasattr(self, "np_random") and self.np_random is not None:
            return self.np_random.uniform(VEL_CONTRARIO_MIN, VEL_CONTRARIO_MAX)
        return 18.0

    def step(self, action: int):
        assert self.action_space.contains(action), f"Acción inválida: {action}"

        reward      = 0.0
        terminated  = False
        truncated   = False

        if action == 0:  
            self.vel_agente = max(VEL_MIN, self.vel_agente - DELTA_VEL)
        elif action == 1:  
            pass
        elif action == 2: 
            self.vel_agente = min(VEL_MAX, self.vel_agente + DELTA_VEL)
        elif action == 3:  
            self.carril_agente = 1 - self.carril_agente

        dt_efectivo = DT * (self.game_speed_factor if self.render_mode == "human" else 1.0)
        
        dx_agente           = self.vel_agente  * dt_efectivo
        self.pos_agente    += dx_agente
        self.pos_lento     += self.vel_lento   * dt_efectivo
        self.pos_contrario -= self.vel_contrario * dt_efectivo   


        if self.pos_contrario < self.pos_agente - 50:
            self.pos_contrario = self.pos_agente + DIST_REAPARICION
            self.vel_contrario = self._nueva_vel_contrario()

   
        dist_lento      = self.pos_lento     - self.pos_agente    
        dist_contrario  = self.pos_contrario - self.pos_agente   

        
        reward += dx_agente * 0.1   

        if self.carril_agente == 0 and 0 < dist_lento <= DIST_SEGURA * 2:
            if dist_lento >= DIST_SEGURA:
                reward += 0.5       
            elif dist_lento < DIST_PELIGRO:
                reward -= 20.0      

        if self.carril_agente == 1:
            self._en_adelant = True
        else:
            if self._en_adelant:
                if self.pos_agente > self.pos_lento:
                    reward += 100.0
                    self.adelantamientos += 1
            self._en_adelant = False

        if self.carril_agente == 1 and dist_contrario < DIST_SEGURA * 2:
            reward -= 50.0




        if self.carril_agente == 0 and 0 < dist_lento < DIST_CHOQUE:
            reward    -= 500.0
            terminated = True
            self.colision = True

        if self.carril_agente == 1 and abs(dist_contrario) < DIST_CHOQUE:
            reward    -= 500.0
            terminated = True
            self.colision = True

        if self.pos_agente >= LONGITUD_CARRETERA:
            reward    += 200.0
            terminated = True
            self.exito = True


        self.pasos += 1
        if self.pasos >= PASOS_MAX:
            truncated = True

        self.recompensa_total += reward

        obs  = self._get_obs()
        info = {
            "pos_agente":       self.pos_agente,
            "vel_agente":       self.vel_agente,
            "carril":           self.carril_agente,
            "dist_lento":       dist_lento,
            "dist_contrario":   dist_contrario,
            "adelantamientos":  self.adelantamientos,
            "colision":         self.colision,
            "exito":            self.exito,
        }

        if self.render_mode == "human":
            self.render(interactive=True)

        return obs, reward, terminated, truncated, info

   

    def _get_obs(self) -> np.ndarray:
        dist_lento     = np.clip(self.pos_lento     - self.pos_agente, 0, 200)
        dist_contrario = np.clip(self.pos_contrario - self.pos_agente, 0, 400)

        obs = np.array([
            dist_lento     / 200.0,
            (self.vel_agente   - VEL_MIN) / (VEL_MAX - VEL_MIN),
            (self.vel_lento    - VEL_MIN) / (VEL_MAX - VEL_MIN),
            dist_contrario / 400.0,
            (self.vel_contrario - VEL_MIN) / (VEL_MAX - VEL_MIN),
            float(self.carril_agente),
        ], dtype=np.float32)

        return np.clip(obs, 0.0, 1.0)




    def render(self, interactive: bool = True):
        """Dibuja el frame actual.

        Args:
            interactive: True en demo (limita FPS para suavidad),
                         False en entrenamiento (no bloquea).
        """
        if self._renderer is None:
            from env.renderer import PygameRenderer
            self._renderer = PygameRenderer()

        return self._renderer.render(
            pos_agente       = self.pos_agente,
            vel_agente       = self.vel_agente,
            carril_agente    = self.carril_agente,
            pos_lento        = self.pos_lento,
            vel_lento        = self.vel_lento,
            pos_contrario    = self.pos_contrario,
            vel_contrario    = self.vel_contrario,
            recompensa_total = self.recompensa_total,
            pasos            = self.pasos,
            adelantamientos  = self.adelantamientos,
            colision         = self.colision,
            exito            = self.exito,
            render_mode      = self.render_mode,
            interactive      = interactive,
        )



    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None