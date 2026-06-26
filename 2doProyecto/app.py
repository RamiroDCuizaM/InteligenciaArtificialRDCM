# app.py
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import io

app = Flask(__name__)

# ── Definición del modelo (igual que en el notebook) ──────────────────────────
model = torch.nn.Sequential(
    torch.nn.Linear(30000, 128),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(128, 15)
)
model.load_state_dict(torch.load("modelo/modelo_formaciones.pth", map_location="cpu"))
model.eval()

# ── Clases de formaciones (índice 0–14 → mostrar 1–15) ────────────────────────
FORMACIONES = {
    0:  {"nombre":"4-4-2",         "descripcion":"La formación más utilizada y tradicional del fútbol mundial.",              "fortaleza":"Su estructura equilibrada con mediocampistas y laterales que asisten eficazmente a los dos delanteros.", "enfoque":"Equilibrado; funciona bastante bien tanto para atacar como para defender."},
    1:  {"nombre":"4-3-3",         "descripcion":"El esquema base que dio origen al Fútbol Total y al juego de posesión.",     "fortaleza":"La presión alta, el marcaje zonal y la facilidad para crear patrones de pases triangulares.",             "enfoque":"Dedicada a atacar a través del control del balón."},
    2:  {"nombre":"4-5-1",         "descripcion":"Un esquema táctico compacto reconocido históricamente por el Chelsea de Mourinho.", "fortaleza":"Un mediocampista defensivo que destruye el juego rival y distribuye rápido el balón.",         "enfoque":"Dedicada a defender y contraatacar."},
    3:  {"nombre":"4-3-2-1",       "descripcion":"Formación conocida como árbol de Navidad.",                                  "fortaleza":"Superioridad numérica en el mediocampo.",                                                             "enfoque":"Atacar mediante la posesión."},
    4:  {"nombre":"4-1-3-2",       "descripcion":"Versión más ofensiva del 4-4-2.",                                            "fortaleza":"Amenaza constante con dos delanteros.",                                                               "enfoque":"Ataque equilibrado."},
    5:  {"nombre":"5-4-1",         "descripcion":"Evolución táctica del catenaccio italiano.",                                  "fortaleza":"Gran solidez defensiva.",                                                                            "enfoque":"Defensivo."},
    6:  {"nombre":"4-1-2-1-2 (Diamante)", "descripcion":"Versión moderna del 4-4-2.",                                         "fortaleza":"Excelente uso de las bandas.",                                                                        "enfoque":"Ofensivo equilibrado."},
    7:  {"nombre":"3-5-2",         "descripcion":"Esquema revolucionario popularizado por Argentina 1986.",                    "fortaleza":"Dominio del mediocampo.",                                                                            "enfoque":"Equilibrado."},
    8:  {"nombre":"5-3-2",         "descripcion":"Variante defensiva del 3-5-2.",                                              "fortaleza":"Cierre de espacios por bandas.",                                                                     "enfoque":"Principalmente defensivo."},
    9:  {"nombre":"4-2-3-1",       "descripcion":"Configuración moderna muy flexible.",                                        "fortaleza":"Seguridad defensiva y creatividad ofensiva.",                                                        "enfoque":"Ataque con equilibrio."},
    10: {"nombre":"3-4-3",         "descripcion":"Sistema moderno de alta exigencia táctica.",                                 "fortaleza":"Ataque de cinco jugadores.",                                                                         "enfoque":"Muy ofensivo."},
    11: {"nombre":"3-2-4-1",       "descripcion":"Formación moderna usada en la Premier League.",                              "fortaleza":"Transiciones extremadamente rápidas.",                                                               "enfoque":"Ataque veloz."},
    12: {"nombre":"WM (3-2-5)",    "descripcion":"Primera gran revolución táctica del fútbol.",                                "fortaleza":"Separación clara entre defensa y ataque.",                                                           "enfoque":"Equilibrado."},
    13: {"nombre":"2-3-2-3",       "descripcion":"El famoso Metodo de Vittorio Pozzo.",                                        "fortaleza":"Control táctico del juego.",                                                                         "enfoque":"Equilibrado."},
    14: {"nombre":"4-2-4",         "descripcion":"Sistema inmortalizado por Brasil 1958 y 1970.",                              "fortaleza":"Laterales muy ofensivos.",                                                                           "enfoque":"Ataque y defensa fuertes."},
}

def preprocesar_imagen(image_bytes):
    """Convierte la imagen a un vector de 30 000 valores (100×100×3 aplanado)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((100, 100))
    arr = np.array(img, dtype=np.float32) / 255.0   # normaliza 0-1
    tensor = torch.tensor(arr.flatten()).unsqueeze(0) # shape (1, 30000)
    return tensor

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predecir", methods=["POST"])
def predecir():
    if "imagen" not in request.files:
        return jsonify({"error": "No se recibió imagen"}), 400

    imagen_bytes = request.files["imagen"].read()
    tensor = preprocesar_imagen(imagen_bytes)

    with torch.no_grad():
        salida = model(tensor)
        clase_idx = int(torch.argmax(salida, dim=1).item())  # 0–14

    info = FORMACIONES[clase_idx]
    return jsonify({
        "clase":       clase_idx + 1,          # mostrar 1–15
        "nombre":      info["nombre"],
        "descripcion": info["descripcion"],
        "fortaleza":   info["fortaleza"],
        "enfoque":     info["enfoque"],
        "confianza":   round(float(torch.softmax(salida, dim=1).max().item()) * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)