# Guía Sencilla: ¿Cómo aprende una Inteligencia Artificial a adelantar coches?

Este documento explica, de manera muy sencilla y sin tecnicismos matemáticos, cómo funciona el sistema de conducción autónoma inteligente que hemos programado. Pensado para que cualquiera pueda entenderlo de un vistazo.

---

## 1. El Escenario (El Entorno)

Imagina un videojuego clásico de coches de dos carriles en el que hay tres vehículos principales en pantalla:

1. **El Coche Aprendiz (Azul - Nosotros):** Es nuestro piloto automático en prácticas. No sabe conducir al principio y tiene que aprender desde cero a base de experimentar.
2. **El Coche Lento (Naranja):** Va delante de nosotros en el carril derecho. Es un obstáculo al que debemos adelantar para no perder tiempo.
3. **El Coche en Contramano (Rojo):** Viene de frente por el carril izquierdo. Representa el mayor peligro.

---

## 2. El Método de Aprendizaje: Premios y Castigos (Q-Learning)

¿Cómo aprende el coche autónomo si no tiene un profesor que le diga cómo girar el volante? Lo hace igual que una mascota o un niño pequeño: **a base de premios y castigos**.

Cada vez que el coche realiza un movimiento, el simulador le da o le quita "puntos" (llamados *recompensas*):

* **Si avanza por la carretera:** Se le da un pequeño premio constante (ej. $+0.2$ puntos). Esto hace que quiera moverse y no quedarse parado en la salida.
* **Si mantiene una distancia segura:** Recibe un premio extra ($+0.5$ puntos).
* **Si se pega demasiado por detrás al coche de adelante:** Se le resta puntos por conducción temeraria ($-20$ puntos).
* **Si choca (accidente):** Recibe un castigo gigante ($-500$ puntos) y el juego termina inmediatamente.
* **Si logra adelantar con éxito:** Recibe un premio enorme ($+100$ puntos).

Al principio del entrenamiento, el coche conduce de manera totalmente caótica y aleatoria (choca constantemente). Pero, tras cientos de intentos, empieza a darse cuenta de qué movimientos le dan más puntos y cuáles le provocan castigos.

---

## 3. ¿Qué es la Tabla Q? (El "Cuaderno de Experiencias")

Para recordar lo aprendido, el coche tiene una memoria en forma de **cuaderno de notas** gigante llamado **Tabla Q**. 

En este cuaderno, el coche apunta todas las situaciones posibles y les asigna una "nota de calidad" (el valor Q) a las 4 decisiones que puede tomar: **Frenar, Mantener velocidad, Acelerar** o **Cambiar de carril**.

### ¿Cómo simplifica el coche el mundo? (Discretización)
El coche no puede procesar los infinitos decimales de la física (ej. estar a *30.4578 metros*). Su "cerebro" simplifica las cosas agrupando los datos en categorías sencillas:

* **Distancias:** En lugar de metros exactos, piensa en: *¿El coche de adelante está "Muy Cerca", "Cerca" o "Lejos"?*
* **Velocidades:** Piensa en términos de: *¿Voy "Lento", "Medio" o "Rápido"?*
* **Carril:** Solo hay dos opciones: *¿Estoy en la "Derecha" o en la "Izquierda"?*

---

## 4. Un Ejemplo Cotidiano del Cuaderno del Coche

Imagina que el coche está circulando y su cerebro simplifica la situación actual de esta manera:

* **Situación:** Estoy **Muy Cerca** del coche lento, voy a velocidad **Media** y el coche que viene de frente está **Muy Lejos**.

El coche busca esa situación exacta en su cuaderno y ve las siguientes "notas" acumuladas por su experiencia pasada para cada acción:

| Acción | Nota de Calidad (Valor Q) | Lo que piensa el coche |
| :--- | :---: | :--- |
| 🛑 **Frenar** | **`+10`** | *"Es seguro, pero iré muy despacio."* |
| 🛡️ **Mantener velocidad** | **`+5`** | *"Terminaré acercándose demasiado al de adelante."* |
| 🚀 **Acelerar** | **`-500`** | *"¡Peligro! Si acelero ahora chocaré por detrás."* |
| ↩️ **Cambiar de carril** | **`+150`**** (Mejor Nota)** | *"El coche de frente está lejísimos. ¡Es mi oportunidad para adelantar!"* |

### La Decisión:
Como la acción **Cambiar de carril** tiene la nota más alta (**`+150`**), el coche decide pasarse al carril izquierdo.

### Actualizando el Cuaderno:
Si el coche cambia de carril y efectivamente realiza el adelantamiento ganando muchos puntos, al final del movimiento borra el `150` anterior y escribe una nota aún mejor (por ejemplo, `160`). 

Si por el contrario hubiese venido un coche de frente que no vio bien y hubese estado a punto de chocar, la nota de esa acción bajaría drásticamente (por ejemplo, a `-100`), para que en el futuro no vuelva a cometer el mismo error en esa situación.

---

## 5. ¿Cómo se lee el "Cuaderno de Notas" en la Consola?

Cuando ejecutas el script `visualizar_tabla_q.py` en la consola, verás líneas de texto parecidas a esta:

```text
Estado [2][3][1][6][2][0]  |  [Fren: -1.5, *Mant: +2.4, Acel: +1.2, Camb: -10.0]  |  Mantener (+2.4)
```

Aquí te explicamos qué significa cada parte detalladamente:

### A. La parte izquierda: `Estado [2][3][1][6][2][0]` (El estado del Entorno)
Cada número dentro de los corchetes representa la lectura simplificada de uno de los "sensores" de nuestro coche:

1. **`[2]` $\rightarrow$ Distancia al coche naranja (lento):** Nos dice qué tan lejos está el coche que tenemos enfrente. Los valores van de `0` (pegados a él) a `8` (muy lejos). Un `2` significa que estamos bastante cerca.
2. **`[3]` $\rightarrow$ Nuestra velocidad (coche azul):** Mide lo rápido que vamos. Va de `0` (velocidad mínima de $18\text{ km/h}$) a `6` (velocidad máxima de $144\text{ km/h}$). Un `3` significa una velocidad media-alta ideal (unos $80\text{ km/h}$).
3. **`[1]` $\rightarrow$ Velocidad del coche naranja:** Al igual que el anterior, mide la velocidad del coche lento. Un `1` indica que va bastante despacio.
4. **`[6]` $\rightarrow$ Distancia al coche rojo (de frente):** Mide a qué distancia viene el peligro. Va de `0` (choque frontal inminente) a `8` (no viene nadie). Un `6` significa que está muy lejos y el carril contrario está bastante despejado.
5. **`[2]` $\rightarrow$ Velocidad del coche rojo:** Mide lo rápido que se nos acerca el coche que viene de frente. Un `2` indica velocidad media-baja.
6. **`[0]` $\rightarrow$ Nuestro carril actual:** Solo tiene dos opciones:
   * **`0`**: Estamos en el carril derecho (seguro, detrás del coche lento).
   * **`1`**: Estamos en el carril izquierdo (sentido contrario, adelantando).

---

### B. La parte derecha: `[Fren: -1.5, *Mant: +2.4, Acel: +1.2, Camb: -10.0]` (Los puntajes de las Acciones)
Representa la "calificación" que la Inteligencia Artificial le otorga a cada una de sus 4 acciones posibles en esta situación. La IA siempre elegirá la acción con la calificación más alta (que viene marcada con un asterisco `*`):

* **`Fren: -1.5` (Frenar):** Calificación negativa baja. Frenar en este momento nos haría perder el ritmo de forma innecessaria.
* **`*Mant: +2.4` (Mantener velocidad) $\leftarrow$ ¡La Ganadora!:** Es la mejor acción evaluada por la IA. Considera que mantener el curso actual le dará el mejor resultado acumulado.
* **`Acel: +1.2` (Acelerar):** Calificación regular. Acelerar ahora mismo nos pondría demasiado cerca del coche naranja antes de cambiar de carril.
* **`Camb: -10.0` (Cambiar Carril):** Calificación muy mala. Considera que aún no es el momento oportuno para cruzar de carril.

---

## 6. La Fórmula de Aprendizaje y su Conexión Total

Para entender cómo se relaciona la física de la carretera con el cerebro de la Inteligencia Artificial, debemos observar la fórmula matemática que la IA usa cada vez que toma una decisión:

$$Q(\text{antiguo}) \leftarrow Q(\text{antiguo}) + \alpha \left[ \text{Premio} + \gamma \max Q(\text{futuro}) - Q(\text{antiguo}) \right]$$

Aunque parezca complicada, en lenguaje cotidiano funciona exactamente como un **mecanismo de ajuste de expectativas**:

$$\text{Nueva Nota} = \text{Nota Antigua} + \text{Tasa de Aprendizaje} \times \left( \text{Premio Real} + \text{Visión de Futuro} \times \text{Mejor Nota de la Siguiente Situación} - \text{Nota Antigua} \right)$$

---

### ¿Cómo se conecta todo en la práctica? (El Ciclo del Conductor)

Cada vez que el juego avanza un paso de tiempo ($0.1$ segundos), ocurre este ciclo de conexión:

```
  1. LEER EL ESTADO ────────> 2. ELEGIR ACCIÓN ────────> 3. MEDIR RESULTADO (Premio)
  (Corchetes izquierdos)      (Ej. Mantener velocidad)   (Ej. +0.7 puntos de carretera)
           ▲                                                           │
           │                                                           ▼
  5. ESCRIBIR NUEVA NOTA <─────── 4. MIRAR EL FUTURO <─────── NUEVA SITUACIÓN
  (Fórmula de Aprendizaje)       (Mejor Nota del Siguiente)   (A dónde se movieron los coches)
```

1. **La Situación Actual (Estado):** El coche lee sus sensores `[2][3][1][6][2][0]` (corchetes izquierdos).
2. **La Acción:** Mira las 4 opciones a la derecha y elige la de mejor nota: **Mantener velocidad** (con nota de `+2.4`).
3. **El Entorno y el Premio:** El coche ejecuta la acción. En la simulación física real, el coche avanza, consume tiempo y el coche de adelante también se mueve. Como todo ha salido de forma segura, el simulador le otorga un **Premio Real** de `+0.7` puntos.
4. **La Nueva Situación (Mirar al Futuro):** Tras moverse, el coche ahora está en una nueva posición física. Vuelve a mirar sus sensores y ve que está en una nueva situación. Busca en su cuaderno esta nueva situación y se pregunta: *"¿Cuál es la mejor nota que tengo apuntada aquí para el siguiente paso?"*. Su cuaderno le responde que la mejor nota a futuro es de `+2.4`.
5. **La Fórmula conecta el Pasado, Presente y Futuro:**
   * **Lo que esperaba ganar:** La "Nota Antigua" era `+2.4`.
   * **Lo que de verdad ganó en total:** El premio inmediato (`+0.7`) más lo que espera ganar en el futuro descontando su impaciencia (`0.95 × +2.4 = +2.28`). El total real fue `0.7 + 2.28 = 2.98`.
   * **El ajuste (La Sorpresa):** La IA se da cuenta de que la realidad (`2.98`) fue mejor que su expectativa anterior (`2.4`). Hay una "sorpresa positiva" de `+0.58` puntos.
   * **El Aprendizaje:** Multiplica esa sorpresa por la Tasa de Aprendizaje (`0.1`) para no sobreactuar: $0.1 \times 0.58 = 0.058$.
   * **El Resultado final en la libreta:** Suma ese pequeño ajuste a la nota anterior: $2.4 + 0.058 = \mathbf{2.458}$.

¡Así es como los corchetes de estado, las opciones de acción, las físicas del videojuego y los premios se integran segundo a segundo para enseñarle a conducir de forma perfecta!
