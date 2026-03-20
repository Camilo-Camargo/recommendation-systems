"""
Servidor Web — Presentación + Demo Interactiva de Sistemas Recomendadores
Ejecutar: python3 server.py
Abrir: http://localhost:8080
"""

import json
import time
import numpy as np
import os
from collections import defaultdict
from flask import Flask, render_template_string, request, jsonify

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════

OPENROUTER_API_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    "YOUR_OPENROUTER_API_KEY",
)
LLM_MODEL = "openrouter/free"
TOP_K = 5

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

app = Flask(__name__)

# ═══════════════════════════════════════════════════════
# CATÁLOGO
# ═══════════════════════════════════════════════════════

CATALOGO = [
    {"id": 1, "nombre": "Nike Air Max 90", "categoria": "zapatos", "subcategoria": "deportivos",
     "precio": 150, "tags": "running correr deporte casual urbano amortiguación nike",
     "descripcion": "Zapatilla deportiva con cámara de aire visible, ideal para running y uso diario.",
     "popularidad": 95, "rating_promedio": 4.5, "num_reviews": 320},
    {"id": 2, "nombre": "Adidas Ultraboost 23", "categoria": "zapatos", "subcategoria": "deportivos",
     "precio": 180, "tags": "running correr deporte boost energía adidas maratón",
     "descripcion": "Zapatilla de alto rendimiento con tecnología Boost para máxima devolución de energía.",
     "popularidad": 90, "rating_promedio": 4.7, "num_reviews": 280},
    {"id": 3, "nombre": "Converse Chuck Taylor", "categoria": "zapatos", "subcategoria": "casual",
     "precio": 65, "tags": "casual urbano clásico lona skate moda street converse",
     "descripcion": "El clásico zapatilla de lona, ícono del estilo urbano desde 1917.",
     "popularidad": 88, "rating_promedio": 4.3, "num_reviews": 510},
    {"id": 4, "nombre": "Dr. Martens 1460", "categoria": "zapatos", "subcategoria": "botas",
     "precio": 170, "tags": "botas cuero resistente punk rock alternativo formal casual",
     "descripcion": "Bota de cuero con suela air-cushion, símbolo de rebeldía y durabilidad.",
     "popularidad": 85, "rating_promedio": 4.6, "num_reviews": 190},
    {"id": 5, "nombre": "Vans Old Skool", "categoria": "zapatos", "subcategoria": "casual",
     "precio": 70, "tags": "skate casual urbano lona gamuza street vans",
     "descripcion": "Zapatilla de skate con la icónica franja lateral, perfecta para el día a día.",
     "popularidad": 87, "rating_promedio": 4.4, "num_reviews": 420},
    {"id": 6, "nombre": "New Balance 574", "categoria": "zapatos", "subcategoria": "deportivos",
     "precio": 100, "tags": "retro classic deporte casual cómodo new balance",
     "descripcion": "Silueta retro-running que combina estilo clásico con comodidad moderna.",
     "popularidad": 82, "rating_promedio": 4.3, "num_reviews": 250},
    {"id": 7, "nombre": "Zapato Oxford de Cuero", "categoria": "zapatos", "subcategoria": "formal",
     "precio": 120, "tags": "formal elegante oficina traje cuero vestir negocios",
     "descripcion": "Zapato formal de cuero genuino con cordones, ideal para oficina y eventos.",
     "popularidad": 70, "rating_promedio": 4.2, "num_reviews": 85},
    {"id": 8, "nombre": "Birkenstock Arizona", "categoria": "zapatos", "subcategoria": "sandalias",
     "precio": 110, "tags": "sandalia cómodo verano descanso corcho ortopédico",
     "descripcion": "Sandalia con plantilla de corcho anatómica para máxima comodidad.",
     "popularidad": 78, "rating_promedio": 4.5, "num_reviews": 310},
    {"id": 9, "nombre": "Salomon Speedcross 5", "categoria": "zapatos", "subcategoria": "trail",
     "precio": 140, "tags": "trail montaña senderismo agarre barro outdoor aventura",
     "descripcion": "Zapatilla de trail running con agarre agresivo para terrenos extremos.",
     "popularidad": 80, "rating_promedio": 4.7, "num_reviews": 150},
    {"id": 10, "nombre": "Crocs Classic Clog", "categoria": "zapatos", "subcategoria": "casual",
     "precio": 45, "tags": "cómodo ligero casual playa jardinería barato crocs",
     "descripcion": "Zueco ultraligero y cómodo, ideal para uso casual y doméstico.",
     "popularidad": 75, "rating_promedio": 4.1, "num_reviews": 600},
    {"id": 11, "nombre": "iPhone 15 Pro", "categoria": "electrónica", "subcategoria": "smartphones",
     "precio": 1199, "tags": "teléfono smartphone apple ios cámara titanio profesional",
     "descripcion": "Smartphone premium con chip A17 Pro, cámara de 48MP y cuerpo de titanio.",
     "popularidad": 97, "rating_promedio": 4.6, "num_reviews": 890},
    {"id": 12, "nombre": "Samsung Galaxy S24 Ultra", "categoria": "electrónica", "subcategoria": "smartphones",
     "precio": 1299, "tags": "teléfono smartphone samsung android cámara AI galaxy",
     "descripcion": "Flagship Android con S Pen, IA generativa integrada y cámara de 200MP.",
     "popularidad": 95, "rating_promedio": 4.5, "num_reviews": 720},
    {"id": 13, "nombre": "MacBook Air M3", "categoria": "electrónica", "subcategoria": "laptops",
     "precio": 1099, "tags": "laptop computadora apple mac portátil programación diseño",
     "descripcion": "Laptop ultradelgada con chip M3, 18h de batería y pantalla Liquid Retina.",
     "popularidad": 94, "rating_promedio": 4.8, "num_reviews": 560},
    {"id": 14, "nombre": "Sony WH-1000XM5", "categoria": "electrónica", "subcategoria": "audio",
     "precio": 350, "tags": "audífonos auriculares bluetooth noise cancelling sony música",
     "descripcion": "Auriculares premium con cancelación de ruido líder en la industria.",
     "popularidad": 92, "rating_promedio": 4.7, "num_reviews": 430},
    {"id": 15, "nombre": "iPad Air M2", "categoria": "electrónica", "subcategoria": "tablets",
     "precio": 599, "tags": "tablet apple ipad dibujo notas estudio portátil",
     "descripcion": "Tablet versátil con chip M2, compatible con Apple Pencil Pro.",
     "popularidad": 89, "rating_promedio": 4.5, "num_reviews": 340},
    {"id": 16, "nombre": "Chaqueta North Face Nuptse", "categoria": "ropa", "subcategoria": "abrigos",
     "precio": 280, "tags": "chaqueta abrigo invierno plumas frío montaña north face",
     "descripcion": "Chaqueta de plumón 700-fill para condiciones de frío extremo.",
     "popularidad": 88, "rating_promedio": 4.6, "num_reviews": 210},
    {"id": 17, "nombre": "Levi's 501 Original", "categoria": "ropa", "subcategoria": "jeans",
     "precio": 70, "tags": "jeans pantalón denim clásico casual moda levi",
     "descripcion": "El jean original straight-fit que definió la cultura denim.",
     "popularidad": 90, "rating_promedio": 4.4, "num_reviews": 780},
    {"id": 18, "nombre": "Camiseta Uniqlo Supima", "categoria": "ropa", "subcategoria": "camisetas",
     "precio": 15, "tags": "camiseta básica algodón casual minimalista uniqlo barato",
     "descripcion": "Camiseta de algodón Supima premium, suave y duradera.",
     "popularidad": 80, "rating_promedio": 4.3, "num_reviews": 1200},
    {"id": 19, "nombre": "Hoodie Nike Tech Fleece", "categoria": "ropa", "subcategoria": "sudaderas",
     "precio": 120, "tags": "sudadera hoodie deportivo casual nike tech fleece urbano",
     "descripcion": "Sudadera con capucha de tejido tech fleece, ligera y térmica.",
     "popularidad": 86, "rating_promedio": 4.5, "num_reviews": 390},
    {"id": 20, "nombre": "Zapatillas Puma RS-X", "categoria": "zapatos", "subcategoria": "deportivos",
     "precio": 110, "tags": "retro running chunky deporte casual puma colorido",
     "descripcion": "Zapatilla retro-futurista con suela chunky y diseño llamativo.",
     "popularidad": 76, "rating_promedio": 4.2, "num_reviews": 180},
    {"id": 21, "nombre": "Sapiens — Yuval N. Harari", "categoria": "libros", "subcategoria": "no-ficción",
     "precio": 20, "tags": "libro historia humanidad ciencia sociedad pensamiento",
     "descripcion": "Recorrido por la historia de la humanidad desde los cazadores-recolectores.",
     "popularidad": 96, "rating_promedio": 4.7, "num_reviews": 2100},
    {"id": 22, "nombre": "El Principito — Saint-Exupéry", "categoria": "libros", "subcategoria": "ficción",
     "precio": 12, "tags": "libro ficción clásico filosofía infantil poesía imaginación",
     "descripcion": "Fábula poética sobre la amistad, el amor y lo esencial invisible.",
     "popularidad": 98, "rating_promedio": 4.9, "num_reviews": 5000},
    {"id": 23, "nombre": "Deep Learning — Goodfellow", "categoria": "libros", "subcategoria": "técnico",
     "precio": 65, "tags": "libro técnico IA inteligencia artificial machine learning redes neuronales",
     "descripcion": "Referencia académica sobre fundamentos de deep learning y redes neuronales.",
     "popularidad": 85, "rating_promedio": 4.4, "num_reviews": 320},
    {"id": 24, "nombre": "Cien Años de Soledad — García Márquez", "categoria": "libros", "subcategoria": "ficción",
     "precio": 16, "tags": "libro ficción realismo mágico colombia latinoamérica novela",
     "descripcion": "Obra maestra del realismo mágico que narra la saga de los Buendía.",
     "popularidad": 97, "rating_promedio": 4.8, "num_reviews": 3800},
    {"id": 25, "nombre": "Cafetera Nespresso Vertuo", "categoria": "hogar", "subcategoria": "cocina",
     "precio": 200, "tags": "café cafetera cápsulas espresso nespresso cocina mañana",
     "descripcion": "Sistema de café con centrífuga para espresso y café largo perfecto.",
     "popularidad": 88, "rating_promedio": 4.5, "num_reviews": 450},
    {"id": 26, "nombre": "Dyson V15 Detect", "categoria": "hogar", "subcategoria": "limpieza",
     "precio": 750, "tags": "aspiradora inalámbrica limpieza hogar tecnología dyson láser",
     "descripcion": "Aspiradora inalámbrica con láser para revelar polvo invisible.",
     "popularidad": 85, "rating_promedio": 4.6, "num_reviews": 280},
    {"id": 27, "nombre": "Echo Dot 5ta Gen", "categoria": "hogar", "subcategoria": "smart home",
     "precio": 50, "tags": "alexa altavoz inteligente smart home domótica música amazon",
     "descripcion": "Altavoz inteligente con Alexa para controlar tu hogar con la voz.",
     "popularidad": 90, "rating_promedio": 4.3, "num_reviews": 1500},
    {"id": 28, "nombre": "Silla Ergonómica Herman Miller", "categoria": "hogar", "subcategoria": "oficina",
     "precio": 1200, "tags": "silla ergonómica oficina trabajo postura espalda premium",
     "descripcion": "Silla de oficina ergonómica premium con soporte lumbar ajustable.",
     "popularidad": 82, "rating_promedio": 4.8, "num_reviews": 190},
    {"id": 29, "nombre": "Kindle Paperwhite", "categoria": "electrónica", "subcategoria": "e-readers",
     "precio": 140, "tags": "kindle lector ebook lectura amazon pantalla tinta electrónica",
     "descripcion": "E-reader con pantalla anti-reflejos de 6.8\" y semanas de batería.",
     "popularidad": 91, "rating_promedio": 4.6, "num_reviews": 680},
    {"id": 30, "nombre": "Mochila Herschel Retreat", "categoria": "accesorios", "subcategoria": "mochilas",
     "precio": 90, "tags": "mochila urbana viaje laptop casual moda herschel",
     "descripcion": "Mochila de estilo retro con compartimento para laptop de 15\".",
     "popularidad": 79, "rating_promedio": 4.3, "num_reviews": 340},
]

HISTORIALES = [
    {"user": "deportista_1", "compras": [1, 2, 9, 19, 14], "ratings": [5, 5, 4, 4, 5]},
    {"user": "deportista_2", "compras": [2, 9, 6, 16, 1], "ratings": [5, 5, 4, 5, 4]},
    {"user": "deportista_3", "compras": [1, 6, 20, 9, 19], "ratings": [4, 5, 4, 5, 3]},
    {"user": "tech_1", "compras": [11, 13, 14, 15, 23], "ratings": [5, 5, 5, 4, 4]},
    {"user": "tech_2", "compras": [12, 13, 14, 27, 29], "ratings": [5, 4, 5, 4, 5]},
    {"user": "tech_3", "compras": [11, 15, 23, 29, 28], "ratings": [4, 5, 5, 5, 4]},
    {"user": "casual_1", "compras": [3, 5, 17, 18, 30], "ratings": [5, 5, 4, 3, 4]},
    {"user": "casual_2", "compras": [5, 3, 10, 17, 22], "ratings": [4, 5, 4, 5, 5]},
    {"user": "casual_3", "compras": [3, 4, 17, 19, 30], "ratings": [5, 4, 5, 4, 4]},
    {"user": "lector_1", "compras": [21, 22, 24, 29, 25], "ratings": [5, 5, 5, 4, 3]},
    {"user": "lector_2", "compras": [22, 24, 21, 23, 18], "ratings": [5, 5, 4, 4, 3]},
    {"user": "hogar_1", "compras": [25, 26, 27, 28, 8], "ratings": [5, 4, 5, 5, 3]},
    {"user": "hogar_2", "compras": [27, 25, 26, 10, 18], "ratings": [5, 5, 4, 3, 4]},
]

# ═══════════════════════════════════════════════════════
# SISTEMAS RECOMENDADORES
# ═══════════════════════════════════════════════════════

class RecomendadorContenido:
    def __init__(self):
        textos = [f"{p['nombre']} {p['categoria']} {p['subcategoria']} {p['tags']} {p['descripcion']}" for p in CATALOGO]
        self.tfidf = TfidfVectorizer(stop_words=None)
        self.matriz = self.tfidf.fit_transform(textos)

    def recomendar(self, query):
        t0 = time.time()
        query_vec = self.tfidf.transform([query])
        sims = cosine_similarity(query_vec, self.matriz).flatten()
        indices = sims.argsort()[::-1][:TOP_K]
        lat = time.time() - t0
        return [{"nombre": CATALOGO[i]["nombre"], "categoria": CATALOGO[i]["categoria"],
                 "precio": CATALOGO[i]["precio"], "score": round(float(sims[i]), 3),
                 "razon": f"Similitud TF-IDF: {sims[i]:.3f}"} for i in indices], round(lat * 1000, 1)


class RecomendadorColaborativo:
    def __init__(self):
        self.n = len(CATALOGO)
        self.mat = np.zeros((len(HISTORIALES), self.n))
        for i, h in enumerate(HISTORIALES):
            for item_id, rating in zip(h["compras"], h["ratings"]):
                idx = next((j for j, p in enumerate(CATALOGO) if p["id"] == item_id), None)
                if idx is not None:
                    self.mat[i, idx] = rating

    def recomendar(self, query):
        t0 = time.time()
        textos = [f"{p['nombre']} {p['tags']} {p['descripcion']}" for p in CATALOGO]
        tfidf = TfidfVectorizer()
        matriz = tfidf.fit_transform(textos)
        qv = tfidf.transform([query])
        sims = cosine_similarity(qv, matriz).flatten()
        top_items = sims.argsort()[::-1][:3]
        profile = np.zeros(self.n)
        for idx in top_items:
            profile[idx] = 5.0
        user_sims = []
        for i in range(len(HISTORIALES)):
            mask = (profile > 0) | (self.mat[i] > 0)
            if mask.sum() == 0:
                user_sims.append(0)
                continue
            a, b = profile[mask], self.mat[i][mask]
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            user_sims.append(np.dot(a, b) / norm if norm > 0 else 0)
        user_sims = np.array(user_sims)
        scores = np.zeros(self.n)
        for u_idx in user_sims.argsort()[::-1][:5]:
            if user_sims[u_idx] > 0:
                scores += user_sims[u_idx] * self.mat[u_idx]
        for idx in top_items:
            scores[idx] = 0
        ranked = scores.argsort()[::-1][:TOP_K]
        lat = time.time() - t0
        mx = scores.max() if scores.max() > 0 else 1
        return [{"nombre": CATALOGO[i]["nombre"], "categoria": CATALOGO[i]["categoria"],
                 "precio": CATALOGO[i]["precio"], "score": round(float(scores[i] / mx), 3),
                 "razon": f"Usuarios similares (score: {scores[i]:.2f})"} for i in ranked if scores[i] > 0], round(lat * 1000, 1)


class RecomendadorConocimiento:
    REGLAS = {
        "zapato": {"categoria": "zapatos"}, "zapatilla": {"categoria": "zapatos"},
        "tenis": {"categoria": "zapatos", "subcategoria": "deportivos"},
        "bota": {"categoria": "zapatos", "subcategoria": "botas"},
        "sandalia": {"categoria": "zapatos", "subcategoria": "sandalias"},
        "formal": {"subcategoria": "formal"}, "deporte": {"subcategoria": "deportivos"},
        "correr": {"subcategoria": "deportivos"}, "running": {"subcategoria": "deportivos"},
        "montaña": {"subcategoria": "trail"}, "senderismo": {"subcategoria": "trail"},
        "teléfono": {"subcategoria": "smartphones"}, "celular": {"subcategoria": "smartphones"},
        "laptop": {"subcategoria": "laptops"}, "computadora": {"subcategoria": "laptops"},
        "audífono": {"subcategoria": "audio"}, "auricular": {"subcategoria": "audio"},
        "tablet": {"subcategoria": "tablets"}, "libro": {"categoria": "libros"},
        "leer": {"categoria": "libros"}, "lectura": {"categoria": "libros"},
        "café": {"tags_match": "café"}, "ropa": {"categoria": "ropa"},
        "chaqueta": {"subcategoria": "abrigos"}, "jean": {"subcategoria": "jeans"},
        "camiseta": {"subcategoria": "camisetas"}, "sudadera": {"subcategoria": "sudaderas"},
        "mochila": {"subcategoria": "mochilas"},
        "barato": {"max_precio": 50}, "económico": {"max_precio": 70},
        "premium": {"min_precio": 200}, "caro": {"min_precio": 300},
    }

    def recomendar(self, query):
        t0 = time.time()
        q = query.lower()
        cands = list(CATALOGO)
        reglas_usadas = []
        for kw, filtro in self.REGLAS.items():
            if kw in q:
                reglas_usadas.append(kw)
                nuevos = []
                for p in cands:
                    ok = True
                    if "categoria" in filtro and p["categoria"] != filtro["categoria"]: ok = False
                    if "subcategoria" in filtro and p["subcategoria"] != filtro["subcategoria"]: ok = False
                    if "max_precio" in filtro and p["precio"] > filtro["max_precio"]: ok = False
                    if "min_precio" in filtro and p["precio"] < filtro["min_precio"]: ok = False
                    if "tags_match" in filtro and filtro["tags_match"] not in p["tags"]: ok = False
                    if ok: nuevos.append(p)
                if nuevos: cands = nuevos
        cands.sort(key=lambda p: p["popularidad"] * p["rating_promedio"], reverse=True)
        lat = time.time() - t0
        return [{"nombre": p["nombre"], "categoria": p["categoria"], "precio": p["precio"],
                 "score": round((p["popularidad"] * p["rating_promedio"]) / 500, 3),
                 "razon": f"Reglas: {', '.join(reglas_usadas) if reglas_usadas else 'popularidad general'}"
                 } for p in cands[:TOP_K]], round(lat * 1000, 1)


class RecomendadorLLM:
    def __init__(self, modo):
        self.modo = modo
        self.catalogo_str = "\n".join(
            f"ID:{p['id']}|{p['nombre']}|{p['categoria']}/{p['subcategoria']}|${p['precio']}|R:{p['rating_promedio']}|{p['descripcion']}"
            for p in CATALOGO)

    def recomendar(self, query):
        if self.modo == "zero-shot":
            prompt = f"""Eres un sistema recomendador de productos experto.
CATÁLOGO:
{self.catalogo_str}

CONSULTA: "{query}"
Recomienda exactamente {TOP_K} productos del catálogo.
Responde SOLO JSON: {{"recommendations": [{{"id": <num>, "nombre": "...", "score": <0-1>, "razon": "..."}}]}}"""
        else:
            prompt = f"""Eres un sistema recomendador de productos experto. Ejemplos:
Consulta: "necesito algo para correr"
Respuesta: {{"recommendations": [{{"id": 2, "nombre": "Adidas Ultraboost 23", "score": 0.95, "razon": "Alto rendimiento para running"}}]}}

CATÁLOGO:
{self.catalogo_str}

CONSULTA: "{query}"
Recomienda exactamente {TOP_K} productos. SOLO JSON:
{{"recommendations": [{{"id": <num>, "nombre": "...", "score": <0-1>, "razon": "..."}}]}}"""

        for attempt in range(3):
            try:
                if attempt > 0: time.sleep(5)
                t0 = time.time()
                resp = client.chat.completions.create(
                    model=LLM_MODEL, messages=[{"role": "user", "content": prompt}],
                    temperature=0.3, max_tokens=2000)
                lat = time.time() - t0
                content = resp.choices[0].message.content
                if not content: continue
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    content = content.rstrip(", \n\r\t")
                    if not content.endswith("]}"): content += "]}" if content.endswith("}") else '"}]}'
                    result = json.loads(content)
                recs = []
                for r in result.get("recommendations", [])[:TOP_K]:
                    prod = next((p for p in CATALOGO if p["id"] == r.get("id")), None)
                    recs.append({"nombre": r.get("nombre", prod["nombre"] if prod else "?"),
                                 "categoria": prod["categoria"] if prod else "?",
                                 "precio": prod["precio"] if prod else 0,
                                 "score": round(float(r.get("score", 0.5)), 3),
                                 "razon": r.get("razon", "Recomendado por LLM")})
                if recs: return recs, round(lat * 1000, 1)
            except Exception as e:
                last_err = str(e)
        return [{"nombre": "Error", "categoria": "-", "precio": 0, "score": 0, "razon": last_err[:80]}], 0


# Inicializar motores
rec_contenido = RecomendadorContenido()
rec_colaborativo = RecomendadorColaborativo()
rec_conocimiento = RecomendadorConocimiento()
rec_llm_zero = RecomendadorLLM("zero-shot")
rec_llm_few = RecomendadorLLM("few-shot")


# ═══════════════════════════════════════════════════════
# RUTAS
# ═══════════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template_string(PAGE_HOME)

@app.route("/presentacion")
def presentacion():
    with open("presentacion/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.route("/demo")
def demo():
    return render_template_string(PAGE_DEMO)

@app.route("/api/recomendar", methods=["POST"])
def api_recomendar():
    query = request.json.get("query", "")
    if not query:
        return jsonify({"error": "Query vacía"}), 400

    resultados = {}

    # ML (rápidos)
    r, l = rec_contenido.recomendar(query)
    resultados["Basado en Contenido (TF-IDF)"] = {"items": r, "latencia": l, "tipo": "ml"}
    r, l = rec_colaborativo.recomendar(query)
    resultados["Filtrado Colaborativo"] = {"items": r, "latencia": l, "tipo": "ml"}
    r, l = rec_conocimiento.recomendar(query)
    resultados["Basado en Conocimiento"] = {"items": r, "latencia": l, "tipo": "ml"}

    # LLM
    r, l = rec_llm_zero.recomendar(query)
    resultados["LLM Zero-shot"] = {"items": r, "latencia": l, "tipo": "llm"}
    r, l = rec_llm_few.recomendar(query)
    resultados["LLM Few-shot"] = {"items": r, "latencia": l, "tipo": "llm"}

    return jsonify(resultados)


# ═══════════════════════════════════════════════════════
# PÁGINAS HTML
# ═══════════════════════════════════════════════════════

PAGE_HOME = """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sistemas Recomendadores — ML vs LLM</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',sans-serif;background:#fafafa;color:#111;min-height:100vh;display:flex;align-items:center;justify-content:center}
.container{text-align:center;max-width:600px;padding:2rem}
h1{font-size:2.8rem;font-weight:700;letter-spacing:-.03em;line-height:1.1;margin-bottom:.3rem}
.sub{color:#666;font-size:1.1rem;font-weight:300;margin-bottom:3rem}
.links{display:flex;flex-direction:column;gap:1rem}
a.btn{display:block;padding:1.2rem 2rem;border:2px solid #111;border-radius:8px;text-decoration:none;color:#111;font-weight:500;font-size:1rem;transition:all .2s}
a.btn:hover{background:#111;color:#fff}
a.btn .desc{display:block;font-size:.8rem;color:#888;font-weight:300;margin-top:.3rem}
a.btn:hover .desc{color:#ccc}
.sep{border:none;border-top:1px solid #ddd;margin:1rem 0}
</style>
</head>
<body>
<div class="container">
    <h1>Sistemas<br>Recomendadores</h1>
    <p class="sub">Machine Learning vs Large Language Models</p>
    <div class="links">
        <a class="btn" href="/presentacion">
            Presentacion
            <span class="desc">Slides interactivos en Reveal.js — tipos, comparativas y ejemplos reales</span>
        </a>
        <hr class="sep">
        <a class="btn" href="/demo">
            Demo Interactiva
            <span class="desc">Escribe una consulta y 5 sistemas la evaluan en tiempo real</span>
        </a>
    </div>
</div>
</body>
</html>
"""

PAGE_DEMO = r"""
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Demo — Sistemas Recomendadores</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',sans-serif;background:#fafafa;color:#111;min-height:100vh}
.header{background:#111;color:#fff;padding:1.5rem 2rem;display:flex;align-items:center;justify-content:space-between}
.header h1{font-size:1.3rem;font-weight:600;letter-spacing:-.02em}
.header a{color:#888;text-decoration:none;font-size:.85rem}
.header a:hover{color:#fff}

/* Metodología */
.method-section{max-width:900px;margin:2rem auto 0;padding:0 1.5rem}
.method-toggle{background:none;border:2px solid #ddd;border-radius:8px;padding:.7rem 1.2rem;font-size:.85rem;color:#444;cursor:pointer;font-family:inherit;width:100%;text-align:left;display:flex;justify-content:space-between;align-items:center;transition:all .2s}
.method-toggle:hover{border-color:#111;color:#111}
.method-toggle .arrow{transition:transform .2s;font-size:1.1rem}
.method-toggle.open .arrow{transform:rotate(180deg)}
.method-cards{display:none;margin-top:1rem;display:grid;grid-template-columns:1fr;gap:.8rem}
.method-cards.visible{display:grid}
.method-card{background:#fff;border:1px solid #e8e8e8;border-radius:8px;padding:1.2rem 1.4rem;transition:border-color .2s}
.method-card:hover{border-color:#bbb}
.method-card-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.6rem}
.method-card h3{font-size:.95rem;font-weight:600;margin:0}
.method-card .mtag{padding:.15rem .6rem;border-radius:12px;font-size:.65rem;font-weight:600;letter-spacing:.03em}
.mtag-ml{background:#111;color:#fff}
.mtag-llm{background:#e0e0e0;color:#444}
.method-card p{font-size:.8rem;color:#555;line-height:1.6;margin:0}
.method-card .label{font-size:.65rem;text-transform:uppercase;letter-spacing:.06em;color:#999;font-weight:600;margin-top:.6rem;margin-bottom:.2rem}
.method-card code{font-family:'JetBrains Mono',monospace;font-size:.72rem;background:#f5f5f5;padding:.15rem .4rem;border-radius:3px;color:#333}
.method-card .steps{list-style:none;padding:0;margin:.3rem 0 0}
.method-card .steps li{font-size:.78rem;color:#555;padding:.2rem 0;padding-left:1.2rem;position:relative}
.method-card .steps li::before{content:"→";position:absolute;left:0;color:#bbb}

.search-area{max-width:900px;margin:1.5rem auto;padding:0 1.5rem}
.search-box{display:flex;gap:.8rem}
.search-box input{flex:1;padding:.9rem 1.2rem;border:2px solid #ddd;border-radius:8px;font-size:1rem;font-family:inherit;outline:none;transition:border .2s}
.search-box input:focus{border-color:#111}
.search-box button{padding:.9rem 2rem;background:#111;color:#fff;border:none;border-radius:8px;font-size:1rem;font-weight:500;cursor:pointer;font-family:inherit;transition:background .2s}
.search-box button:hover{background:#333}
.search-box button:disabled{background:#999;cursor:wait}
.examples{margin-top:.8rem;display:flex;gap:.5rem;flex-wrap:wrap}
.examples span{font-size:.75rem;color:#888}
.examples button{background:none;border:1px solid #ddd;border-radius:20px;padding:.3rem .8rem;font-size:.75rem;color:#666;cursor:pointer;font-family:inherit;transition:all .15s}
.examples button:hover{border-color:#111;color:#111}
.loading{text-align:center;padding:3rem;color:#888;font-size:.9rem}
.loading .spinner{display:inline-block;width:24px;height:24px;border:3px solid #ddd;border-top-color:#111;border-radius:50%;animation:spin .6s linear infinite;margin-bottom:.5rem}
@keyframes spin{to{transform:rotate(360deg)}}
.results{max-width:900px;margin:0 auto;padding:0 1.5rem 3rem}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:1rem;margin-bottom:2rem}
.stat-card{background:#fff;border:1px solid #e8e8e8;border-radius:8px;padding:1.2rem}
.stat-card h4{font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;color:#888;margin-bottom:.8rem}
.bar-row{display:flex;align-items:center;margin:.4rem 0;font-size:.8rem}
.bar-label{width:160px;color:#444;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bar-track{flex:1;height:22px;background:#f0f0f0;border-radius:4px;overflow:hidden;margin:0 .5rem}
.bar-fill{height:100%;border-radius:4px;transition:width .6s ease}
.bar-fill.ml{background:#111}
.bar-fill.llm{background:#888}
.bar-value{width:50px;text-align:right;font-family:'JetBrains Mono',monospace;font-size:.75rem;color:#666}
.system-results{margin-top:1.5rem}
.system-card{background:#fff;border:1px solid #e8e8e8;border-radius:8px;margin-bottom:1rem;overflow:hidden}
.system-header{display:flex;justify-content:space-between;align-items:center;padding:1rem 1.2rem;border-bottom:1px solid #f0f0f0}
.system-name{font-weight:600;font-size:.95rem}
.system-meta{display:flex;gap:1rem;font-size:.75rem;color:#888}
.system-meta .tag{padding:.15rem .5rem;border-radius:12px;font-weight:500}
.tag-ml{background:#111;color:#fff}
.tag-llm{background:#e8e8e8;color:#444}
.system-body{padding:0 1.2rem 1rem}
.item-row{display:flex;align-items:center;padding:.6rem 0;border-bottom:1px solid #f8f8f8}
.item-row:last-child{border:none}
.item-rank{width:28px;height:28px;background:#f5f5f5;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.75rem;font-weight:600;color:#888;margin-right:.8rem;flex-shrink:0}
.item-info{flex:1;min-width:0}
.item-name{font-weight:500;font-size:.9rem}
.item-reason{font-size:.75rem;color:#888;margin-top:.1rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.item-score{text-align:right;margin-left:.8rem}
.item-score .num{font-family:'JetBrains Mono',monospace;font-size:.85rem;font-weight:600}
.item-score .price{font-size:.7rem;color:#888}
.no-results{padding:1.5rem;text-align:center;color:#888;font-size:.85rem}
</style>
</head>
<body>
<div class="header">
    <h1>Demo Comparativa — Sistemas Recomendadores</h1>
    <a href="/">Volver al inicio</a>
</div>

<!-- ══════ METODOLOGÍA ══════ -->
<div class="method-section">
    <button class="method-toggle" id="methodToggle" onclick="toggleMethod()">
        <span>Como se construyo cada modelo</span>
        <span class="arrow">▼</span>
    </button>
    <div class="method-cards" id="methodCards">

        <div class="method-card">
            <div class="method-card-header">
                <h3>1. Basado en Contenido (TF-IDF)</h3>
                <span class="mtag mtag-ml">ML</span>
            </div>
            <p>Analiza el <strong>texto descriptivo</strong> de cada producto (nombre, categoria, tags, descripcion) y lo compara con la consulta del usuario.</p>
            <div class="label">Algoritmo</div>
            <p><code>TF-IDF</code> (Term Frequency - Inverse Document Frequency) + <code>Similitud Coseno</code></p>
            <div class="label">Pasos de construccion</div>
            <ul class="steps">
                <li>Se concatena nombre + categoria + tags + descripcion de cada producto</li>
                <li>Se construye una <strong>matriz TF-IDF</strong> con <code>sklearn.TfidfVectorizer</code></li>
                <li>La consulta del usuario se transforma al mismo espacio vectorial</li>
                <li>Se calcula <strong>similitud coseno</strong> entre la query y cada producto</li>
                <li>Se devuelven los Top-K productos mas similares</li>
            </ul>
            <div class="label">Libreria</div>
            <p><code>scikit-learn</code> — TfidfVectorizer, cosine_similarity</p>
        </div>

        <div class="method-card">
            <div class="method-card-header">
                <h3>2. Filtrado Colaborativo</h3>
                <span class="mtag mtag-ml">ML</span>
            </div>
            <p>Encuentra <strong>usuarios con gustos similares</strong> y recomienda lo que ellos compraron. Se basa en patrones de comportamiento, no en el contenido.</p>
            <div class="label">Algoritmo</div>
            <p><code>User-based CF</code> con similitud coseno sobre una matriz usuario-item</p>
            <div class="label">Pasos de construccion</div>
            <ul class="steps">
                <li>Se construye una <strong>matriz de ratings</strong> (13 usuarios x 30 productos) con historiales simulados</li>
                <li>Se identifica que productos le interesan al usuario actual (via TF-IDF rapido)</li>
                <li>Se crea un <strong>perfil temporal</strong> del usuario con esos productos</li>
                <li>Se calcula <strong>similitud coseno</strong> entre el perfil y cada usuario historico</li>
                <li>Se agregan los ratings de los 5 usuarios mas similares, ponderados por similitud</li>
                <li>Se devuelven los Top-K productos con mayor score agregado</li>
            </ul>
            <div class="label">Libreria</div>
            <p><code>numpy</code> — operaciones matriciales, <code>scikit-learn</code> — TF-IDF auxiliar</p>
        </div>

        <div class="method-card">
            <div class="method-card-header">
                <h3>3. Basado en Conocimiento (Reglas)</h3>
                <span class="mtag mtag-ml">ML</span>
            </div>
            <p>Usa <strong>reglas explicitas del dominio</strong> definidas manualmente. Filtra productos segun restricciones inferidas de la consulta.</p>
            <div class="label">Algoritmo</div>
            <p><code>Constraint-based filtering</code> — mapeo de palabras clave a filtros categoricos y numericos</p>
            <div class="label">Pasos de construccion</div>
            <ul class="steps">
                <li>Se define un <strong>diccionario de 30+ reglas</strong>: "zapato" → categoria:zapatos, "correr" → subcategoria:deportivos, etc.</li>
                <li>Se analizan las palabras de la consulta y se activan las reglas que coincidan</li>
                <li>Se aplican filtros en cascada: categoria, subcategoria, rango de precio, tags</li>
                <li>Los candidatos se ordenan por <code>popularidad x rating_promedio</code></li>
                <li>Se devuelven los Top-K productos que pasan todos los filtros</li>
            </ul>
            <div class="label">Libreria</div>
            <p>Ninguna externa — logica pura de Python con diccionarios y filtros</p>
        </div>

        <div class="method-card">
            <div class="method-card-header">
                <h3>4. LLM Zero-shot</h3>
                <span class="mtag mtag-llm">LLM</span>
            </div>
            <p>Envia el catalogo completo y la consulta a un <strong>Large Language Model</strong> sin ningun ejemplo previo. El modelo usa su conocimiento general para razonar.</p>
            <div class="label">Modelo</div>
            <p><code>openrouter/free</code> — enruta automaticamente al mejor modelo gratuito disponible (Nemotron, Llama, etc.) via <strong>OpenRouter API</strong></p>
            <div class="label">Pasos de construccion</div>
            <ul class="steps">
                <li>Se serializa el catalogo de 30 productos como texto: ID, nombre, categoria, precio, rating, descripcion</li>
                <li>Se construye un <strong>prompt estructurado</strong> con: catalogo + consulta + instruccion de formato JSON</li>
                <li>Se envia al LLM via API REST (<code>openai.ChatCompletion</code> apuntando a OpenRouter)</li>
                <li>El LLM <strong>razona</strong> sobre la consulta y selecciona los productos mas relevantes</li>
                <li>Se parsea la respuesta JSON con scores y razones en lenguaje natural</li>
            </ul>
            <div class="label">Prompt</div>
            <p>Solo instruccion + catalogo + consulta. <strong>Sin ejemplos</strong> de como responder.</p>
            <div class="label">Libreria</div>
            <p><code>openai</code> SDK — client.chat.completions.create()</p>
        </div>

        <div class="method-card">
            <div class="method-card-header">
                <h3>5. LLM Few-shot</h3>
                <span class="mtag mtag-llm">LLM</span>
            </div>
            <p>Igual que zero-shot pero se incluyen <strong>2 ejemplos de consulta-respuesta</strong> en el prompt para guiar al modelo sobre el formato y calidad esperados.</p>
            <div class="label">Modelo</div>
            <p><code>openrouter/free</code> — mismo modelo que zero-shot para comparacion justa</p>
            <div class="label">Pasos de construccion</div>
            <ul class="steps">
                <li>Se construye el mismo prompt base con catalogo + consulta</li>
                <li>Se agregan <strong>2 ejemplos demostrativos</strong>: "necesito algo para correr" → Adidas Ultraboost (0.95), y "quiero un buen libro" → Cien Anos de Soledad (0.96)</li>
                <li>Estos ejemplos ensenan al modelo: formato JSON esperado, rango de scores, estilo de razones</li>
                <li>Se envia al LLM y se parsea identicamente al zero-shot</li>
            </ul>
            <div class="label">Diferencia clave vs zero-shot</div>
            <p>El <strong>few-shot learning</strong> mejora consistencia del formato y calidad de las razones. Tipicamente produce scores mas altos y recomendaciones mas precisas.</p>
            <div class="label">Libreria</div>
            <p><code>openai</code> SDK — client.chat.completions.create()</p>
        </div>

    </div>
</div>

<!-- ══════ BÚSQUEDA ══════ -->
<div class="search-area">
    <div class="search-box">
        <input type="text" id="query" placeholder="Escribe tu consulta... ej: recomiendame un zapato deportivo" autocomplete="off">
        <button id="btn" onclick="buscar()">Buscar</button>
    </div>
    <div class="examples">
        <span>Prueba:</span>
        <button onclick="ejemplo(this)">zapato para correr</button>
        <button onclick="ejemplo(this)">quiero un buen libro</button>
        <button onclick="ejemplo(this)">necesito audífonos</button>
        <button onclick="ejemplo(this)">laptop para programar</button>
        <button onclick="ejemplo(this)">ropa barata</button>
        <button onclick="ejemplo(this)">regalo premium</button>
    </div>
</div>

<div id="loading" class="loading" style="display:none">
    <div class="spinner"></div>
    <div>Evaluando con 5 sistemas simultaneamente...</div>
    <div style="font-size:.75rem;color:#aaa;margin-top:.3rem">Los LLMs pueden tardar unos segundos</div>
</div>

<div id="results" class="results"></div>

<script>
const input = document.getElementById('query');
const btn = document.getElementById('btn');
input.addEventListener('keydown', e => { if (e.key === 'Enter') buscar(); });

function toggleMethod() {
    const cards = document.getElementById('methodCards');
    const toggle = document.getElementById('methodToggle');
    const visible = cards.classList.contains('visible');
    if (visible) {
        cards.classList.remove('visible');
        cards.style.display = 'none';
        toggle.classList.remove('open');
    } else {
        cards.classList.add('visible');
        cards.style.display = 'grid';
        toggle.classList.add('open');
    }
}
// Start collapsed
document.getElementById('methodCards').style.display = 'none';

function ejemplo(el) {
    input.value = el.textContent;
    buscar();
}

async function buscar() {
    const q = input.value.trim();
    if (!q) return;
    btn.disabled = true;
    btn.textContent = 'Evaluando...';
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').innerHTML = '';

    try {
        const res = await fetch('/api/recomendar', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query: q})
        });
        const data = await res.json();
        renderResults(data, q);
    } catch(e) {
        document.getElementById('results').innerHTML = '<p class="no-results">Error de conexion</p>';
    }
    btn.disabled = false;
    btn.textContent = 'Buscar';
    document.getElementById('loading').style.display = 'none';
}

const MODEL_INFO = {
    "Basado en Contenido (TF-IDF)": "TF-IDF + Similitud Coseno sobre texto descriptivo",
    "Filtrado Colaborativo": "User-based CF — similitud coseno sobre matriz usuario-item",
    "Basado en Conocimiento": "Constraint-based — 30+ reglas de dominio + ranking por popularidad",
    "LLM Zero-shot": "Prompt con catalogo completo, sin ejemplos — openrouter/free",
    "LLM Few-shot": "Prompt con catalogo + 2 ejemplos demostrativos — openrouter/free"
};

function renderResults(data, query) {
    const c = document.getElementById('results');
    const systems = Object.entries(data);
    const maxScore = Math.max(...systems.map(([,v]) => {
        const scores = v.items.filter(i => i.score > 0).map(i => i.score);
        return scores.length ? scores.reduce((a,b)=>a+b,0)/scores.length : 0;
    }));
    const maxLat = Math.max(...systems.map(([,v]) => v.latencia || 1));

    let html = '<div class="stats">';
    html += '<div class="stat-card"><h4>Score Promedio (Relevancia)</h4>';
    systems.forEach(([name, v]) => {
        const avg = v.items.filter(i=>i.score>0).length ? v.items.filter(i=>i.score>0).reduce((a,b)=>a+b.score,0)/v.items.filter(i=>i.score>0).length : 0;
        const pct = maxScore > 0 ? (avg/maxScore*100) : 0;
        const cls = v.tipo === 'ml' ? 'ml' : 'llm';
        html += '<div class="bar-row">';
        html += '<div class="bar-label">' + name.replace('Basado en ','') + '</div>';
        html += '<div class="bar-track"><div class="bar-fill ' + cls + '" style="width:' + pct + '%"></div></div>';
        html += '<div class="bar-value">' + avg.toFixed(3) + '</div></div>';
    });
    html += '</div>';

    html += '<div class="stat-card"><h4>Latencia</h4>';
    systems.forEach(([name, v]) => {
        const pct = maxLat > 0 ? (v.latencia/maxLat*100) : 0;
        const cls = v.tipo === 'ml' ? 'ml' : 'llm';
        const latStr = v.latencia > 1000 ? (v.latencia/1000).toFixed(1)+'s' : v.latencia.toFixed(0)+'ms';
        html += '<div class="bar-row">';
        html += '<div class="bar-label">' + name.replace('Basado en ','') + '</div>';
        html += '<div class="bar-track"><div class="bar-fill ' + cls + '" style="width:' + pct + '%"></div></div>';
        html += '<div class="bar-value">' + latStr + '</div></div>';
    });
    html += '</div></div>';

    html += '<div class="system-results">';
    systems.forEach(([name, v]) => {
        const tagCls = v.tipo === 'ml' ? 'tag-ml' : 'tag-llm';
        const tagLabel = v.tipo === 'ml' ? 'ML' : 'LLM';
        const latStr = v.latencia > 1000 ? (v.latencia/1000).toFixed(1)+'s' : v.latencia.toFixed(0)+'ms';
        const info = MODEL_INFO[name] || '';
        html += '<div class="system-card">';
        html += '<div class="system-header">';
        html += '<div><span class="system-name">' + name + '</span>';
        html += '<div style="font-size:.72rem;color:#999;margin-top:.15rem">' + info + '</div></div>';
        html += '<div class="system-meta"><span class="tag ' + tagCls + '">' + tagLabel + '</span><span>' + latStr + '</span><span>' + v.items.filter(i=>i.score>0).length + ' items</span></div>';
        html += '</div><div class="system-body">';
        if (!v.items.length || (v.items.length === 1 && v.items[0].nombre === 'Error')) {
            html += '<div class="no-results">' + (v.items[0]?.razon || 'Sin resultados') + '</div>';
        } else {
            v.items.forEach((item, i) => {
                if (item.nombre === 'Error') { html += '<div class="no-results">' + item.razon + '</div>'; return; }
                html += '<div class="item-row">';
                html += '<div class="item-rank">' + (i+1) + '</div>';
                html += '<div class="item-info"><div class="item-name">' + item.nombre + '</div>';
                html += '<div class="item-reason">' + item.razon + '</div></div>';
                html += '<div class="item-score"><div class="num">' + item.score.toFixed(2) + '</div>';
                html += '<div class="price">$' + item.precio + '</div></div></div>';
            });
        }
        html += '</div></div>';
    });
    html += '</div>';
    c.innerHTML = html;
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import socket
    port = int(os.environ.get("PORT", 8080))
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "0.0.0.0"
    print()
    print("  ╔═══════════════════════════════════════════════════════╗")
    print("  ║  Servidor — Sistemas Recomendadores                  ║")
    print("  ╠═══════════════════════════════════════════════════════╣")
    print(f"  ║  Local:   http://localhost:{port}                      ║")
    print(f"  ║  Red:     http://{local_ip}:{port}                  ║")
    print("  ║                                                       ║")
    print("  ║  /              → Página principal                    ║")
    print("  ║  /presentacion  → Slides Reveal.js                   ║")
    print("  ║  /demo          → Demo interactiva                   ║")
    print("  ╚═══════════════════════════════════════════════════════╝")
    print()
    app.run(host="0.0.0.0", port=port, debug=False)
