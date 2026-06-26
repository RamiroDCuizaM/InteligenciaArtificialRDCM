# Funcionamiento y Estructura Detallada de la Tabla Q

Este documento explica a fondo el núcleo matemático y conceptual del algoritmo Q-Learning implementado en el simulador: la **Tabla Q (Q-Table)**. Se describe el significado de los números contenidos en ella, la lógica de su almacenamiento en memoria y un ejemplo matemático completo paso a paso de una actualización de valores.

---

## 1. ¿Qué es físicamente la Tabla Q?

La **Tabla Q** es un mapa de memoria (o base de datos tabular) que asocia cada situación posible del entorno con la utilidad estimada de tomar cada una de las decisiones disponibles.

* **Fórmula Conceptual:**
  $$Q(s, a) \approx \text{Retorno total futuro esperado si en el estado } s \text{ tomo la acción } a$$

* **El valor de Q:**
  No es la recompensa inmediata del siguiente paso. Es la suma acumulada de las recompensas futuras que el agente espera recibir desde ese instante hasta el final del episodio si actúa de manera óptima.

---

## 2. La Estructura Multidimensional de la Tabla Q

En Python, la tabla se define como un arreglo de NumPy de 7 dimensiones. Conceptualmente, funciona como un índice jerárquico de búsqueda:

$$\text{Dirección en memoria} \rightarrow \mathbf{Q}[\text{índice}_0][\text{índice}_1][\text{índice}_2][\text{índice}_3][\text{índice}_4][\text{índice}_5][\text{acción}]$$

Cada índice numérico representa la celda discreta resultante de clasificar las variables continuas.

```
       ESTADO DISCRETO S                       ACCIONES (Eje 6)
┌─────────────────────────────┐               ┌────────────────┐
│ [i0,  i1,  i2,  i3,  i4, i5] │  ─────────>   │ Action 0: Q0   │ (Frenar)
└─────────────────────────────┘               │ Action 1: Q1   │ (Mantener)
                                              │ Action 2: Q2   │ (Acelerar)
                                              │ Action 3: Q3   │ (Cambiar Carril)
                                              └────────────────┘
```

---

## 3. Proceso de Discretización (De lo Continuo a la Tabla)

El entorno lee variables físicas en números flotantes continuos (como `30.25 metros` o `18.5 m/s`). La función `discretizar(obs)` convierte estos valores en enteros válidos para indexar la matriz usando límites de partición (`BINS`).

### Ejemplo de límites (`BINS`):
Si definimos `np.linspace(0, 1, 4)` para clasificar una variable normalizada en $[0, 1]$, NumPy crea los siguientes límites divisores:
$$[0.0,\; 0.3333,\; 0.6667,\; 1.0]$$

La función asigna el valor físico a una de las celdas (bins) de acuerdo a dónde cae el número:
* Un valor normalizado de $0.12$ cae entre $0.0$ y $0.3333 \rightarrow$ **Índice 1**
* Un valor normalizado de $0.50$ cae entre $0.3333$ y $0.6667 \rightarrow$ **Índice 2**
* Un valor normalizado de $0.85$ cae entre $0.6667$ y $1.0 \rightarrow$ **Índice 3**

---

## 4. Ejemplo Práctico Paso a Paso (Con Números y Matemáticas)

Imaginemos un instante de la simulación durante el entrenamiento. Vamos a ver cómo se calcula todo en un paso de tiempo ($dt = 0.1\text{ s}$).

### Paso A: Estado Físico del Entorno
* **Distancia al coche lento:** $30.0\text{ m}$ (normalizada: $30 / 200 = 0.15$)
* **Velocidad del agente:** $20.0\text{ m/s}$ (normalizada: $(20 - 5) / 35 \approx 0.4286$)
* **Velocidad del coche lento:** $12.0\text{ m/s}$ (normalizada: $(12 - 5) / 35 = 0.20$)
* **Distancia al contrario:** $300.0\text{ m}$ (normalizada: $300 / 400 = 0.75$)
* **Velocidad del contrario:** $18.0\text{ m/s}$ (normalizada: $(18 - 5) / 35 \approx 0.3714$)
* **Carril actual:** Derecho ($0.0$)

### Paso B: Discretización a Índices de Tabla Q
Al pasar estas variables por los rangos definidos en `BINS`, obtenemos los índices del estado actual $s$:
$$s = (\mathbf{2},\; \mathbf{3},\; \mathbf{1},\; \mathbf{6},\; \mathbf{2},\; \mathbf{0})$$

### Paso C: Consulta de los Valores Q actuales
El agente consulta la fila correspondiente de la tabla Q para este estado:
* $Q(s, \text{Frenar}) = -1.5$
* $Q(s, \text{Mantener}) = \mathbf{2.4}$  $\leftarrow$ *Máximo valor actual*
* $Q(s, \text{Acelerar}) = 1.2$
* $Q(s, \text{Cambiar Carril}) = -10.0$

### Paso D: Selección y Ejecución de la Acción
Suponiendo que el agente ya superó la etapa de exploración aleatoria ($\varepsilon$-greedy decide explotar), el agente elige la mejor acción:
$$\text{Acción seleccionada } a = \text{Mantener } (\text{índice } 1)$$

### Paso E: Transición Física y Recompensa del Entorno
El motor de física avanza un paso de tiempo ($0.1$ segundos):
* El agente avanza $2.0\text{ m}$ ($20.0\text{ m/s} \times 0.1\text{ s}$).
* El coche lento avanza $1.2\text{ m}$ ($12.0\text{ m/s} \times 0.1\text{ s}$).
* La nueva distancia al coche lento es $30.0 + 1.2 - 2.0 = 29.2\text{ m}$.

**Cálculo de Recompensa Recibida ($R$):**
1. Recompensa por avanzar: $+0.2$ (distancia recorrida: $2.0\text{ m} \times 0.1$).
2. Distancia segura del coche lento: $+0.5$ (ya que $29.2\text{ m}$ es una distancia segura en el carril derecho).
$$R = 0.2 + 0.5 = \mathbf{0.7}$$

### Paso F: El Nuevo Estado $s'$
Tras avanzar física y discretizar las nuevas posiciones, el agente determina su nuevo estado discreto $s'$:
$$s' = (\mathbf{2},\; \mathbf{3},\; \mathbf{1},\; \mathbf{6},\; \mathbf{2},\; \mathbf{0})$$
*(Nota: los valores variaron ligeramente pero no lo suficiente como para saltar a otros intervalos de discretización).*

### Paso G: Actualización Matemática de Bellman (Q-Update)
Configuración de parámetros:
* Tasa de aprendizaje ($\alpha$) = $0.1$
* Factor de descuento ($\gamma$) = $0.95$

El agente consulta los valores Q del nuevo estado $s'$ para buscar el mejor futuro posible:
$$\max_{a'} Q(s', a') = 2.4 \quad (\text{correspondiente a la acción de Mantener en } s')$$

Aplicamos la regla de Bellman para actualizar $Q(s, a)$:
$$Q_{\text{nuevo}}(s, a) = Q(s, a) + \alpha \left[ R + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

Reemplazando con nuestros números:
$$Q_{\text{nuevo}}(s, a) = 2.4 + 0.1 \times \left[ 0.7 + 0.95 \times (2.4) - 2.4 \right]$$
$$Q_{\text{nuevo}}(s, a) = 2.4 + 0.1 \times \left[ 0.7 + 2.28 - 2.4 \right]$$
$$Q_{\text{nuevo}}(s, a) = 2.4 + 0.1 \times \left[ 0.58 \right]$$
$$Q_{\text{nuevo}}(s, a) = 2.4 + 0.058 = \mathbf{2.458}$$

### Resultado en la Tabla Q:
El valor de $Q(s, \text{Mantener})$ se actualiza en memoria pasando de $2.4$ a **$2.458$**. El agente ahora tiene una mayor convicción de que mantener la velocidad en ese estado específico es una buena decisión.
