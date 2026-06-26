"""
Renderer Pygame para el entorno de adelantamiento.
Muestra la carretera, los vehículos y el HUD en tiempo real.

Diseñado para NO bloquear el entrenamiento:
  - La ventana se crea/destruye bajo demanda.
  - En modo entrenamiento, se renderiza sin espera de FPS.
  - En modo demo, se respeta el FPS para suavidad visual.
"""

import pygame
import numpy as np
import time

# ─────────────────────── configuración de pantalla ───────────────────────────
SCREEN_W   = 900
SCREEN_H   = 540
FPS_DEMO   = 10   # FPS solo para demo interactiva (más lento para apreciar adelantamientos)
FPS_TRAIN  = 0    # 0 = sin límite (no bloquea)
GAME_SPEED_FACTOR = 0.5  # Control de velocidad en modo gráfico (0.1=muy lento, 0.5=mitad, 1.0=normal)

# ─────────────────────── paleta de colores ────────────────────────────────────
C_BG            = (15,  17,  22)    # fondo casi negro
C_ROAD          = (35,  38,  45)    # asfalto oscuro
C_LANE_LINE     = (220, 200,  60)   # línea central amarilla
C_LANE_DASH     = (160, 160, 160)   # guiones blancos laterales
C_SHOULDER      = (25,  90,  40)    # arcén verde

C_AGENT         = (50,  180, 255)   # agente: azul brillante
C_SLOW          = (240, 130,  30)   # vehículo lento: naranja
C_ONCOMING      = (220,  50,  50)   # vehículo contrario: rojo

C_HUD_BG        = (20,  22,  30)
C_HUD_TEXT      = (200, 210, 230)
C_HUD_ACCENT    = (50,  180, 255)
C_HUD_WARN      = (240, 100,  50)
C_HUD_OK        = (60,  210, 120)

C_PROGRESS_BG   = (40,  44,  55)
C_PROGRESS_FG   = (50,  180, 255)

ROAD_LEFT    = 160    # px desde la izquierda donde empieza la carretera
ROAD_RIGHT   = 740
ROAD_WIDTH   = ROAD_RIGHT - ROAD_LEFT

LANE_H       = 90     # altura de cada carril (px)
LANE_TOP_Y   = 170    # y superior del carril superior (contrario)
# Carril 0 (derecho):  LANE_TOP_Y + LANE_H
# Carril 1 (izquierdo): LANE_TOP_Y

CAR_W        = 52
CAR_H        = 30

# Ventana de visualización (metros del mundo que se ven)
VIEW_RANGE   = 150.0   # metros delante y detrás del agente


class PygameRenderer:
    """Renderer no bloqueante para el entorno de adelantamiento.

    Ciclo de vida de la ventana:
      - Se crea la primera vez que se llama ``render()`` con modo "human".
      - Se destruye cuando se llama ``close()`` o cuando el usuario cierra
        la ventana con la X / ESC.
      - Durante el entrenamiento, la ventana se abre y cierra bajo demanda
        sin detener el loop de entrenamiento.
    """

    def __init__(self):
        self._initialized    = False   # ¿pygame.init() fue llamado?
        self.screen          = None
        self.clock           = None
        self._closed_by_user = False   # True si el usuario cerró la ventana

        # Fuentes (se crean una sola vez)
        self.font_big   = None
        self.font_mid   = None
        self.font_small = None

        # Modo de limitación de FPS
        self._interactive = False      # True → demo, False → entrenamiento

    # ──────────────────────────── inicialización lazy ─────────────────────────
    def _ensure_init(self):
        """Inicializa pygame y crea la ventana si no existe."""
        if self._initialized and self.screen is not None:
            return True

        if self._closed_by_user:
            # El usuario cerró explícitamente; no recrear automáticamente.
            return False

        if not self._initialized:
            pygame.init()
            self._initialized = True

        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Simulación de Adelantamiento — RL")
        self.clock = pygame.time.Clock()

        # Fuentes
        self.font_big   = pygame.font.SysFont("consolas", 22, bold=True)
        self.font_mid   = pygame.font.SysFont("consolas", 16)
        self.font_small = pygame.font.SysFont("consolas", 13)

        return True

    # ──────────────────────────── render principal ───────────────────────────
    def render(
        self,
        pos_agente, vel_agente, carril_agente,
        pos_lento, vel_lento,
        pos_contrario, vel_contrario,
        recompensa_total, pasos,
        adelantamientos, colision, exito,
        render_mode="human",
        interactive=False,
    ):
        """Renderiza un frame.

        Args:
            interactive: True durante demo (limita FPS), False durante
                         entrenamiento (sin bloqueo).
        """
        if render_mode not in ("human", "rgb_array"):
            return None

        self._interactive = interactive

        if not self._ensure_init():
            return None   # ventana cerrada por el usuario

        # ── Procesar eventos (imprescindible para que la ventana responda) ──
        should_close = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                should_close = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                should_close = True

        if should_close:
            self._hide_window()
            return None

        surf = self.screen

        # ── Fondo ──────────────────────────────────────────────────
        surf.fill(C_BG)

        # ── Dibujar carretera ──────────────────────────────────────
        self._draw_road(surf, pos_agente)

        # ── Proyectar posiciones al espacio de pantalla ────────────
        def world_to_screen_x(pos_world):
            """Mapea posición mundo → pixel X. El agente está centrado."""
            delta  = pos_world - pos_agente
            frac   = (delta + VIEW_RANGE * 0.4) / (VIEW_RANGE)
            return int(ROAD_LEFT + frac * ROAD_WIDTH)

        def carril_to_y(carril):
            """0=derecho (abajo), 1=izquierdo (arriba)."""
            if carril == 0:
                return LANE_TOP_Y + LANE_H + LANE_H // 2
            else:
                return LANE_TOP_Y + LANE_H // 2

        # ── Dibujar vehículos ──────────────────────────────────────
        # Contrario (carril izquierdo)
        cx = world_to_screen_x(pos_contrario)
        cy = carril_to_y(1)
        if ROAD_LEFT - CAR_W < cx < ROAD_RIGHT + CAR_W:
            self._draw_car(surf, cx, cy, C_ONCOMING, "←", vel_contrario)

        # Lento (carril derecho)
        lx = world_to_screen_x(pos_lento)
        ly = carril_to_y(0)
        if ROAD_LEFT - CAR_W < lx < ROAD_RIGHT + CAR_W:
            self._draw_car(surf, lx, ly, C_SLOW, "→", vel_lento)

        # Agente (siempre centrado)
        ax = world_to_screen_x(pos_agente)
        ay = carril_to_y(carril_agente)
        self._draw_car(surf, ax, ay, C_AGENT, "A", vel_agente, agente=True)

        # ── Etiquetas de carril ────────────────────────────────────
        self._draw_lane_labels(surf)

        # ── Barra de progreso ──────────────────────────────────────
        from env.overtaking_env import LONGITUD_CARRETERA
        progress = min(pos_agente / LONGITUD_CARRETERA, 1.0)
        self._draw_progress(surf, progress, pos_agente, LONGITUD_CARRETERA)

        # ── HUD ────────────────────────────────────────────────────
        self._draw_hud(
            surf, vel_agente, carril_agente,
            recompensa_total, pasos, adelantamientos,
            colision, exito,
            pos_agente, pos_lento, pos_contrario
        )

        # ── Mensaje de fin de episodio ─────────────────────────────
        if colision:
            self._draw_overlay(surf, "¡COLISIÓN!", C_ONCOMING)
        elif exito:
            self._draw_overlay(surf, "¡META ALCANZADA!", C_HUD_OK)

        pygame.display.flip()

        # ── Limitar FPS solo en modo interactivo ───────────────────
        if self._interactive:
            self.clock.tick(FPS_DEMO)
            # Aplicar factor de velocidad: delay adicional para ralentizar
            if GAME_SPEED_FACTOR < 1.0:
                delay = (1.0 / FPS_DEMO) * (1.0 / GAME_SPEED_FACTOR - 1.0)
                time.sleep(delay)
        else:
            # En entrenamiento: solo un tick mínimo para que la ventana
            # procese eventos sin bloquear (0 = sin espera)
            self.clock.tick(FPS_TRAIN)

        if render_mode == "rgb_array":
            return pygame.surfarray.array3d(surf).transpose(1, 0, 2)

        return None

    # ──────────────────────────── ocultar ventana ─────────────────────────────
    def _hide_window(self):
        """Destruye la ventana pero no cierra pygame (puede recrearse)."""
        if self.screen is not None:
            pygame.display.quit()
            self.screen = None
            self.clock  = None
            self._closed_by_user = True

    def reopen(self):
        """Permite reabrir la ventana después de que el usuario la cerró."""
        self._closed_by_user = False

    # ──────────────────────────── carretera ──────────────────────────────────
    def _draw_road(self, surf, pos_agente):
        total_h = LANE_H * 2 + 4
        road_rect = pygame.Rect(ROAD_LEFT, LANE_TOP_Y, ROAD_WIDTH, total_h)

        # Arcenes
        pygame.draw.rect(surf, C_SHOULDER,
                         (0, LANE_TOP_Y, SCREEN_W, total_h))

        # Asfalto
        pygame.draw.rect(surf, C_ROAD, road_rect)

        # Línea central amarilla sólida
        mid_y = LANE_TOP_Y + LANE_H
        pygame.draw.line(surf, C_LANE_LINE,
                         (ROAD_LEFT, mid_y), (ROAD_RIGHT, mid_y), 3)

        # Guiones del borde superior e inferior
        dash_len, gap = 30, 20
        for x in range(ROAD_LEFT, ROAD_RIGHT, dash_len + gap):
            pygame.draw.line(surf, C_LANE_DASH,
                             (x, LANE_TOP_Y + 2),
                             (min(x + dash_len, ROAD_RIGHT), LANE_TOP_Y + 2), 2)
            pygame.draw.line(surf, C_LANE_DASH,
                             (x, LANE_TOP_Y + total_h - 2),
                             (min(x + dash_len, ROAD_RIGHT), LANE_TOP_Y + total_h - 2), 2)

        # Marcadores de distancia (cada 50 m)
        LONGITUD = 1000.0
        for km_mark in range(0, int(LONGITUD) + 1, 50):
            delta = km_mark - pos_agente
            frac  = (delta + VIEW_RANGE * 0.4) / VIEW_RANGE
            px    = int(ROAD_LEFT + frac * ROAD_WIDTH)
            if ROAD_LEFT <= px <= ROAD_RIGHT:
                pygame.draw.line(surf, (80, 85, 100),
                                 (px, LANE_TOP_Y + total_h),
                                 (px, LANE_TOP_Y + total_h + 8), 1)
                label = self.font_small.render(f"{km_mark}m", True, (80, 85, 100))
                surf.blit(label, (px - 14, LANE_TOP_Y + total_h + 9))

    # ──────────────────────────── vehículo ───────────────────────────────────
    def _draw_car(self, surf, cx, cy, color, label, vel, agente=False):
        rect = pygame.Rect(cx - CAR_W // 2, cy - CAR_H // 2, CAR_W, CAR_H)

        # Sombra sutil
        shadow = rect.move(3, 3)
        shadow_surf = pygame.Surface((CAR_W, CAR_H), pygame.SRCALPHA)
        shadow_surf.fill((0, 0, 0, 80))
        surf.blit(shadow_surf, shadow.topleft)

        # Cuerpo
        pygame.draw.rect(surf, color, rect, border_radius=6)

        # Borde más claro
        bright = tuple(min(c + 60, 255) for c in color)
        pygame.draw.rect(surf, bright, rect, width=2, border_radius=6)

        # Ventana (rectángulo interior)
        win_color = (20, 20, 30) if not agente else (10, 40, 80)
        win_rect = rect.inflate(-16, -10)
        pygame.draw.rect(surf, win_color, win_rect, border_radius=3)

        # Etiqueta
        lbl = self.font_small.render(label, True, (240, 240, 240))
        surf.blit(lbl, lbl.get_rect(center=rect.center))

        # Velocidad encima
        vel_kmh = vel * 3.6
        vel_txt = self.font_small.render(f"{vel_kmh:.0f}km/h", True, C_HUD_TEXT)
        surf.blit(vel_txt, vel_txt.get_rect(centerx=cx, bottom=cy - CAR_H // 2 - 3))

    # ──────────────────────────── etiquetas de carril ────────────────────────
    def _draw_lane_labels(self, surf):
        y_izq = LANE_TOP_Y + LANE_H // 2
        y_der = LANE_TOP_Y + LANE_H + LANE_H // 2
        for y, txt, col in [
            (y_izq, "◄ CONTRARIO", C_ONCOMING),
            (y_der, "AGENTE ►",    C_AGENT),
        ]:
            lbl = self.font_small.render(txt, True, col)
            surf.blit(lbl, (ROAD_LEFT - 5 - lbl.get_width(), y - lbl.get_height() // 2))

    # ──────────────────────────── barra de progreso ───────────────────────────
    def _draw_progress(self, surf, progress, pos, total):
        bar_x, bar_y = ROAD_LEFT, SCREEN_H - 50
        bar_w, bar_h = ROAD_WIDTH, 14
        pygame.draw.rect(surf, C_PROGRESS_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=7)
        fill = int(bar_w * progress)
        if fill > 0:
            pygame.draw.rect(surf, C_PROGRESS_FG,
                             (bar_x, bar_y, fill, bar_h), border_radius=7)
        pygame.draw.rect(surf, C_HUD_ACCENT, (bar_x, bar_y, bar_w, bar_h),
                         width=1, border_radius=7)
        pct = self.font_small.render(f"{pos:.0f} / {total:.0f} m  ({progress*100:.1f}%)",
                                     True, C_HUD_TEXT)
        surf.blit(pct, pct.get_rect(centerx=ROAD_LEFT + ROAD_WIDTH // 2,
                                    top=bar_y + bar_h + 4))

    # ──────────────────────────── HUD ─────────────────────────────────────────
    def _draw_hud(
        self, surf, vel_agente, carril_agente,
        recompensa_total, pasos, adelantamientos,
        colision, exito,
        pos_agente, pos_lento, pos_contrario
    ):
        # Panel izquierdo
        px, py = 10, 10
        self._panel(surf, px, py, 145, 260)

        vel_kmh = vel_agente * 3.6
        carril_str = "Izquierdo" if carril_agente == 1 else "Derecho"
        carril_col = C_HUD_WARN if carril_agente == 1 else C_HUD_OK

        lines = [
            ("AGENTE",          C_HUD_ACCENT, True),
            (f"Vel: {vel_kmh:.1f} km/h", C_HUD_TEXT, False),
            (f"Pos: {pos_agente:.1f} m",  C_HUD_TEXT, False),
            ("",                 C_HUD_TEXT, False),
            ("CARRIL",          C_HUD_ACCENT, True),
            (carril_str,        carril_col,  False),
            ("",                 C_HUD_TEXT, False),
            ("SESIÓN",          C_HUD_ACCENT, True),
            (f"Pasos: {pasos}",  C_HUD_TEXT, False),
            (f"Adelant: {adelantamientos}", C_HUD_OK, False),
            ("",                 C_HUD_TEXT, False),
            ("RECOMPENSA",      C_HUD_ACCENT, True),
            (f"{recompensa_total:+.1f}", C_HUD_OK if recompensa_total >= 0 else C_HUD_WARN, False),
        ]
        for i, (txt, col, bold) in enumerate(lines):
            if not txt:
                continue
            fn = self.font_mid if not bold else self.font_big
            rendered = fn.render(txt, True, col)
            surf.blit(rendered, (px + 8, py + 8 + i * 18))

        # Panel derecho — distancias
        rx = SCREEN_W - 165
        self._panel(surf, rx, 10, 155, 130)
        dist_lento     = pos_lento - pos_agente
        dist_contrario = pos_contrario - pos_agente

        r_lines = [
            ("DISTANCIAS",     C_HUD_ACCENT, True),
            (f"→ Lento:  {dist_lento:.1f}m",
             C_HUD_OK if dist_lento > 20 else C_HUD_WARN, False),
            (f"← Contrario: {dist_contrario:.1f}m",
             C_HUD_OK if dist_contrario > 40 else C_HUD_WARN, False),
            ("",              C_HUD_TEXT, False),
            ("TECLADO",       C_HUD_ACCENT, True),
            ("[ESC] Cerrar",  C_HUD_TEXT, False),
        ]
        for i, (txt, col, bold) in enumerate(r_lines):
            if not txt:
                continue
            fn = self.font_mid if not bold else self.font_big
            rendered = fn.render(txt, True, col)
            surf.blit(rendered, (rx + 8, 18 + i * 18))

    def _panel(self, surf, x, y, w, h):
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        s.fill((20, 22, 30, 210))
        surf.blit(s, (x, y))
        pygame.draw.rect(surf, (50, 60, 80), (x, y, w, h), width=1, border_radius=4)

    # ──────────────────────────── overlay ────────────────────────────────────
    def _draw_overlay(self, surf, msg, color):
        overlay = pygame.Surface((SCREEN_W, 60), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        surf.blit(overlay, (0, SCREEN_H // 2 - 30))
        txt = self.font_big.render(msg, True, color)
        surf.blit(txt, txt.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2)))

    # ──────────────────────────── close ──────────────────────────────────────
    def close(self):
        """Cierra pygame completamente."""
        if self._initialized:
            pygame.quit()
            self._initialized    = False
            self.screen          = None
            self.clock           = None
            self._closed_by_user = False