"""
Demo Interactiva — Comparación de Sistemas Recomendadores
Escribe una consulta y todos los enfoques generan recomendaciones simultáneamente.
"""

import json
import time
import numpy as np
import os
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# ═══════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════

OPENROUTER_API_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    "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",
)
LLM_MODELS = {
    "LLM (auto-free)": "openrouter/free",
}
TOP_K = 5

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


# ═══════════════════════════════════════════════════════
# CATÁLOGO DE PRODUCTOS
# ═══════════════════════════════════════════════════════

CATALOGO = [
    # Zapatos
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

    # Electrónica
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

    # Ropa
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

    # Libros
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

    # Hogar
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

# ═══════════════════════════════════════════════════════
# HISTORIAL SIMULADO DE USUARIOS (para filtrado colaborativo)
# ═══════════════════════════════════════════════════════

# Patrones de compra simulados: usuarios con gustos similares
HISTORIALES = [
    # Perfil deportista
    {"user": "deportista_1", "compras": [1, 2, 9, 19, 14], "ratings": [5, 5, 4, 4, 5]},
    {"user": "deportista_2", "compras": [2, 9, 6, 16, 1], "ratings": [5, 5, 4, 5, 4]},
    {"user": "deportista_3", "compras": [1, 6, 20, 9, 19], "ratings": [4, 5, 4, 5, 3]},
    # Perfil tech
    {"user": "tech_1", "compras": [11, 13, 14, 15, 23], "ratings": [5, 5, 5, 4, 4]},
    {"user": "tech_2", "compras": [12, 13, 14, 27, 29], "ratings": [5, 4, 5, 4, 5]},
    {"user": "tech_3", "compras": [11, 15, 23, 29, 28], "ratings": [4, 5, 5, 5, 4]},
    # Perfil casual/urbano
    {"user": "casual_1", "compras": [3, 5, 17, 18, 30], "ratings": [5, 5, 4, 3, 4]},
    {"user": "casual_2", "compras": [5, 3, 10, 17, 22], "ratings": [4, 5, 4, 5, 5]},
    {"user": "casual_3", "compras": [3, 4, 17, 19, 30], "ratings": [5, 4, 5, 4, 4]},
    # Perfil lector
    {"user": "lector_1", "compras": [21, 22, 24, 29, 25], "ratings": [5, 5, 5, 4, 3]},
    {"user": "lector_2", "compras": [22, 24, 21, 23, 18], "ratings": [5, 5, 4, 4, 3]},
    # Perfil hogar
    {"user": "hogar_1", "compras": [25, 26, 27, 28, 8], "ratings": [5, 4, 5, 5, 3]},
    {"user": "hogar_2", "compras": [27, 25, 26, 10, 18], "ratings": [5, 5, 4, 3, 4]},
]

# ═══════════════════════════════════════════════════════
# SISTEMA 1: BASADO EN CONTENIDO (TF-IDF)
# ═══════════════════════════════════════════════════════

class RecomendadorContenido:
    def __init__(self, catalogo):
        self.catalogo = catalogo
        self.nombres = [p["nombre"] for p in catalogo]
        # Construir texto descriptivo por producto
        textos = []
        for p in catalogo:
            texto = f"{p['nombre']} {p['categoria']} {p['subcategoria']} {p['tags']} {p['descripcion']}"
            textos.append(texto)
        self.tfidf = TfidfVectorizer(stop_words=None)
        self.matriz = self.tfidf.fit_transform(textos)

    def recomendar(self, query, k=TOP_K):
        t0 = time.time()
        query_vec = self.tfidf.transform([query])
        similitudes = cosine_similarity(query_vec, self.matriz).flatten()
        indices = similitudes.argsort()[::-1][:k]
        latencia = time.time() - t0

        resultados = []
        for idx in indices:
            p = self.catalogo[idx]
            resultados.append({
                "nombre": p["nombre"],
                "categoria": p["categoria"],
                "precio": p["precio"],
                "score": float(similitudes[idx]),
                "razon": f"Similitud TF-IDF: {similitudes[idx]:.3f} — coincidencia de contenido con '{query}'",
            })
        return resultados, latencia


# ═══════════════════════════════════════════════════════
# SISTEMA 2: FILTRADO COLABORATIVO (Similitud de usuarios)
# ═══════════════════════════════════════════════════════

class RecomendadorColaborativo:
    def __init__(self, catalogo, historiales):
        self.catalogo = catalogo
        self.historiales = historiales
        self.n_items = len(catalogo)

        # Construir matriz usuario-ítem
        self.usuarios = [h["user"] for h in historiales]
        self.matriz_ratings = np.zeros((len(historiales), self.n_items))
        for i, h in enumerate(historiales):
            for item_id, rating in zip(h["compras"], h["ratings"]):
                idx = next((j for j, p in enumerate(catalogo) if p["id"] == item_id), None)
                if idx is not None:
                    self.matriz_ratings[i, idx] = rating

    def recomendar(self, query, k=TOP_K):
        """Simula un usuario nuevo basándose en la query para encontrar usuarios similares."""
        t0 = time.time()

        # Identificar productos relevantes a la query usando TF-IDF rápido
        textos = [f"{p['nombre']} {p['tags']} {p['descripcion']}" for p in self.catalogo]
        tfidf = TfidfVectorizer()
        matriz = tfidf.fit_transform(textos)
        query_vec = tfidf.transform([query])
        sims = cosine_similarity(query_vec, matriz).flatten()
        top_items = sims.argsort()[::-1][:3]

        # Crear perfil simulado del usuario (interesado en esos productos)
        user_profile = np.zeros(self.n_items)
        for idx in top_items:
            user_profile[idx] = 5.0

        # Encontrar usuarios más similares
        user_sims = []
        for i in range(len(self.historiales)):
            mask = (user_profile > 0) | (self.matriz_ratings[i] > 0)
            if mask.sum() == 0:
                user_sims.append(0)
                continue
            # Similitud coseno entre perfiles
            a = user_profile[mask]
            b = self.matriz_ratings[i][mask]
            norm = (np.linalg.norm(a) * np.linalg.norm(b))
            sim = np.dot(a, b) / norm if norm > 0 else 0
            user_sims.append(sim)

        user_sims = np.array(user_sims)
        top_users = user_sims.argsort()[::-1][:5]

        # Agregar scores ponderados por similitud
        scores = np.zeros(self.n_items)
        for u_idx in top_users:
            if user_sims[u_idx] > 0:
                scores += user_sims[u_idx] * self.matriz_ratings[u_idx]

        # Excluir ítems ya "conocidos"
        for idx in top_items:
            scores[idx] = 0

        ranked = scores.argsort()[::-1][:k]
        latencia = time.time() - t0

        resultados = []
        for idx in ranked:
            if scores[idx] <= 0:
                continue
            p = self.catalogo[idx]
            resultados.append({
                "nombre": p["nombre"],
                "categoria": p["categoria"],
                "precio": p["precio"],
                "score": float(scores[idx] / scores.max()) if scores.max() > 0 else 0,
                "razon": f"Usuarios similares compraron este producto (score: {scores[idx]:.2f})",
            })
        return resultados, latencia


# ═══════════════════════════════════════════════════════
# SISTEMA 3: BASADO EN CONOCIMIENTO (Reglas)
# ═══════════════════════════════════════════════════════

class RecomendadorConocimiento:
    # Reglas del dominio: mapeo de intención → filtros
    REGLAS = {
        "zapato": {"categoria": "zapatos"},
        "zapatilla": {"categoria": "zapatos"},
        "tenis": {"categoria": "zapatos", "subcategoria": "deportivos"},
        "bota": {"categoria": "zapatos", "subcategoria": "botas"},
        "sandalia": {"categoria": "zapatos", "subcategoria": "sandalias"},
        "formal": {"subcategoria": "formal"},
        "deporte": {"subcategoria": "deportivos"},
        "correr": {"subcategoria": "deportivos"},
        "running": {"subcategoria": "deportivos"},
        "montaña": {"subcategoria": "trail"},
        "senderismo": {"subcategoria": "trail"},
        "teléfono": {"subcategoria": "smartphones"},
        "celular": {"subcategoria": "smartphones"},
        "smartphone": {"subcategoria": "smartphones"},
        "laptop": {"subcategoria": "laptops"},
        "computadora": {"subcategoria": "laptops"},
        "audífono": {"subcategoria": "audio"},
        "auricular": {"subcategoria": "audio"},
        "tablet": {"subcategoria": "tablets"},
        "libro": {"categoria": "libros"},
        "leer": {"categoria": "libros"},
        "lectura": {"categoria": "libros"},
        "café": {"tags_match": "café"},
        "cocina": {"subcategoria": "cocina"},
        "limpieza": {"subcategoria": "limpieza"},
        "ropa": {"categoria": "ropa"},
        "chaqueta": {"subcategoria": "abrigos"},
        "jean": {"subcategoria": "jeans"},
        "camiseta": {"subcategoria": "camisetas"},
        "sudadera": {"subcategoria": "sudaderas"},
        "mochila": {"subcategoria": "mochilas"},
        "barato": {"max_precio": 50},
        "económico": {"max_precio": 70},
        "premium": {"min_precio": 200},
        "caro": {"min_precio": 300},
    }

    def __init__(self, catalogo):
        self.catalogo = catalogo

    def recomendar(self, query, k=TOP_K):
        t0 = time.time()
        query_lower = query.lower()
        filtros_aplicados = []
        candidatos = list(self.catalogo)

        for keyword, filtro in self.REGLAS.items():
            if keyword in query_lower:
                filtros_aplicados.append(f"{keyword} → {filtro}")
                nuevos = []
                for p in candidatos:
                    match = True
                    if "categoria" in filtro and p["categoria"] != filtro["categoria"]:
                        match = False
                    if "subcategoria" in filtro and p["subcategoria"] != filtro["subcategoria"]:
                        match = False
                    if "max_precio" in filtro and p["precio"] > filtro["max_precio"]:
                        match = False
                    if "min_precio" in filtro and p["precio"] < filtro["min_precio"]:
                        match = False
                    if "tags_match" in filtro and filtro["tags_match"] not in p["tags"]:
                        match = False
                    if match:
                        nuevos.append(p)
                if nuevos:
                    candidatos = nuevos

        # Ordenar por popularidad × rating
        candidatos.sort(key=lambda p: p["popularidad"] * p["rating_promedio"], reverse=True)
        latencia = time.time() - t0

        resultados = []
        for p in candidatos[:k]:
            score = (p["popularidad"] * p["rating_promedio"]) / (100 * 5)
            resultados.append({
                "nombre": p["nombre"],
                "categoria": p["categoria"],
                "precio": p["precio"],
                "score": score,
                "razon": f"Reglas: {', '.join(filtros_aplicados) if filtros_aplicados else 'sin filtro específico'} | Pop: {p['popularidad']} | Rating: {p['rating_promedio']}",
            })
        return resultados, latencia


# ═══════════════════════════════════════════════════════
# SISTEMA 4: LLM (OpenRouter)
# ═══════════════════════════════════════════════════════

class RecomendadorLLM:
    def __init__(self, catalogo, modo="few-shot", modelo="stepfun/step-3.5-flash:free", nombre_modelo="LLM"):
        self.catalogo = catalogo
        self.modo = modo
        self.modelo = modelo
        self.nombre_modelo = nombre_modelo
        catalogo_str = "\n".join(
            f"ID:{p['id']} | {p['nombre']} | {p['categoria']}/{p['subcategoria']} | ${p['precio']} | Rating:{p['rating_promedio']} | {p['descripcion']}"
            for p in catalogo
        )
        self.catalogo_str = catalogo_str

    def recomendar(self, query, k=TOP_K):
        if self.modo == "zero-shot":
            prompt = f"""Eres un sistema recomendador de productos experto.

CATÁLOGO DISPONIBLE:
{self.catalogo_str}

CONSULTA DEL USUARIO: "{query}"

Recomienda exactamente {k} productos del catálogo anterior que mejor satisfagan la consulta.
Responde SOLO con un JSON válido sin texto adicional:
{{"recommendations": [{{"id": <number>, "nombre": "...", "score": <0.0-1.0>, "razon": "..."}}]}}"""

        else:  # few-shot
            prompt = f"""Eres un sistema recomendador de productos experto. Aquí hay ejemplos de cómo debes responder:

EJEMPLO 1:
Consulta: "necesito algo para correr"
Respuesta: {{"recommendations": [{{"id": 2, "nombre": "Adidas Ultraboost 23", "score": 0.95, "razon": "Zapatilla de alto rendimiento diseñada específicamente para running"}}, {{"id": 1, "nombre": "Nike Air Max 90", "score": 0.88, "razon": "Zapatilla deportiva versátil con excelente amortiguación para correr"}}]}}

EJEMPLO 2:
Consulta: "quiero un buen libro"
Respuesta: {{"recommendations": [{{"id": 24, "nombre": "Cien Años de Soledad", "score": 0.96, "razon": "Obra maestra de la literatura universal con rating 4.8"}}, {{"id": 21, "nombre": "Sapiens", "score": 0.93, "razon": "Best-seller que explora la historia de la humanidad"}}]}}

CATÁLOGO DISPONIBLE:
{self.catalogo_str}

CONSULTA DEL USUARIO: "{query}"

Recomienda exactamente {k} productos del catálogo que mejor satisfagan la consulta.
Responde SOLO con JSON válido sin texto adicional:
{{"recommendations": [{{"id": <number>, "nombre": "...", "score": <0.0-1.0>, "razon": "..."}}]}}"""

        max_retries = 3
        last_error = "Sin respuesta del modelo"
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(5)  # Espera entre reintentos

                t0 = time.time()
                response = client.chat.completions.create(
                    model=self.modelo,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2000,
                )
                latencia = time.time() - t0
                content = response.choices[0].message.content
                if not content:
                    continue
                content = content.strip()

                # Limpiar markdown
                if content.startswith("```"):
                    content = content.split("\n", 1)[1]
                    content = content.rsplit("```", 1)[0].strip()

                # Intentar parsear JSON, con reparación si está truncado
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    # Intentar reparar JSON truncado
                    content = content.rstrip(", \n\r\t")
                    if not content.endswith("]}"):
                        # Cerrar el array y objeto
                        if content.endswith("}"):
                            content += "]}"
                        elif content.endswith('"'):
                            content += "}]}"
                        else:
                            content += '"}]}'
                    result = json.loads(content)

                resultados = []
                for r in result.get("recommendations", [])[:k]:
                    rid = r.get("id")
                    prod = next((p for p in self.catalogo if p["id"] == rid), None)
                    resultados.append({
                        "nombre": r.get("nombre", prod["nombre"] if prod else "?"),
                        "categoria": prod["categoria"] if prod else "?",
                        "precio": prod["precio"] if prod else 0,
                        "score": float(r.get("score", 0.5)),
                        "razon": r.get("razon", "Recomendado por LLM"),
                    })
                if resultados:
                    return resultados, latencia

            except Exception as e:
                last_error = e
                continue

        return [{"nombre": f"Error: {last_error}", "categoria": "-", "precio": 0, "score": 0, "razon": str(last_error)}], 0.0


# ═══════════════════════════════════════════════════════
# VISUALIZACIÓN DE RESULTADOS
# ═══════════════════════════════════════════════════════

def barra(valor, max_ancho=30, char="█"):
    """Genera una barra visual ASCII."""
    lleno = int(valor * max_ancho)
    vacio = max_ancho - lleno
    return char * lleno + "░" * vacio


def mostrar_resultados(nombre_sistema, resultados, latencia, color_code=""):
    """Muestra resultados de un sistema de forma visual."""
    print(f"\n  ┌─────────────────────────────────────────────────────────────────┐")
    print(f"  │  {nombre_sistema:<40} Latencia: {latencia*1000:>6.1f}ms  │")
    print(f"  ├─────────────────────────────────────────────────────────────────┤")

    if not resultados:
        print(f"  │  Sin resultados                                                │")
    else:
        for i, r in enumerate(resultados, 1):
            nombre = r['nombre'][:35].ljust(35)
            print(f"  │  {i}. {nombre} ${r['precio']:>6}       │")
            score_bar = barra(r['score'], 20)
            print(f"  │     Score: {score_bar} {r['score']:.2f}              │")
            razon = r['razon'][:60]
            print(f"  │     {razon:<60}│")
            if i < len(resultados):
                print(f"  │                                                                 │")

    print(f"  └─────────────────────────────────────────────────────────────────┘")


def mostrar_estadisticas(todos_resultados):
    """Muestra tabla comparativa de estadísticas."""
    print("\n")
    print("  ╔═══════════════════════════════════════════════════════════════════════╗")
    print("  ║              ESTADÍSTICAS COMPARATIVAS                               ║")
    print("  ╠════════════════════════╦══════════╦══════════╦═══════════╦════════════╣")
    print("  ║ Sistema                ║ Latencia ║ Score Ø  ║ Precio Ø  ║ Resultados ║")
    print("  ╠════════════════════════╬══════════╬══════════╬═══════════╬════════════╣")

    stats = {}
    for nombre, (resultados, latencia) in todos_resultados.items():
        n = len(resultados)
        avg_score = np.mean([r["score"] for r in resultados]) if resultados else 0
        avg_precio = np.mean([r["precio"] for r in resultados]) if resultados else 0
        stats[nombre] = {
            "latencia": latencia,
            "score": avg_score,
            "precio": avg_precio,
            "n": n,
        }
        lat_str = f"{latencia*1000:.0f}ms" if latencia < 1 else f"{latencia:.1f}s"
        print(f"  ║ {nombre:<22} ║ {lat_str:>8} ║ {avg_score:>8.3f} ║ ${avg_precio:>8.0f} ║ {n:>10} ║")

    print("  ╚════════════════════════╩══════════╩══════════╩═══════════╩════════════╝")

    # Análisis
    print("\n  ── ANÁLISIS ──────────────────────────────────────────────────────")
    if stats:
        mejor_score = max(stats.items(), key=lambda x: x[1]["score"])
        mejor_latencia = min(stats.items(), key=lambda x: x[1]["latencia"] if x[1]["latencia"] > 0 else float("inf"))
        print(f"  Mayor relevancia:  {mejor_score[0]} (score promedio: {mejor_score[1]['score']:.3f})")
        print(f"  Más rápido:        {mejor_latencia[0]} ({mejor_latencia[1]['latencia']*1000:.1f}ms)")

        # Gráfico de barras comparativo
        print("\n  ── SCORE PROMEDIO ────────────────────────────────────────────────")
        max_score = max(s["score"] for s in stats.values()) if stats else 1
        for nombre, s in stats.items():
            bar = barra(s["score"] / max_score if max_score > 0 else 0, 30)
            print(f"  {nombre:<24} {bar} {s['score']:.3f}")

        print("\n  ── LATENCIA ─────────────────────────────────────────────────────")
        max_lat = max(s["latencia"] for s in stats.values()) if stats else 1
        for nombre, s in stats.items():
            bar = barra(s["latencia"] / max_lat if max_lat > 0 else 0, 30)
            lat_str = f"{s['latencia']*1000:.0f}ms" if s['latencia'] < 1 else f"{s['latencia']:.1f}s"
            print(f"  {nombre:<24} {bar} {lat_str}")


# ═══════════════════════════════════════════════════════
# MAIN — LOOP INTERACTIVO
# ═══════════════════════════════════════════════════════

def main():
    print()
    print("  ╔═══════════════════════════════════════════════════════════════╗")
    print("  ║    DEMO COMPARATIVA — SISTEMAS RECOMENDADORES               ║")
    print("  ║    ML Tradicional vs LLM                                    ║")
    print("  ╠═══════════════════════════════════════════════════════════════╣")
    print("  ║  Escribe una consulta y todos los sistemas la evalúan.      ║")
    print("  ║  Escribe 'salir' para terminar.                             ║")
    print("  ╚═══════════════════════════════════════════════════════════════╝")
    print()

    # Inicializar sistemas
    print("  Inicializando sistemas...")
    rec_contenido = RecomendadorContenido(CATALOGO)
    rec_colaborativo = RecomendadorColaborativo(CATALOGO, HISTORIALES)
    rec_conocimiento = RecomendadorConocimiento(CATALOGO)
    # LLMs gratuitos: Step 3.5 Flash y Llama 3.3 70B
    rec_llms = {}
    for nombre, modelo_id in LLM_MODELS.items():
        for modo in ["zero-shot", "few-shot"]:
            key = f"{nombre} ({modo})"
            rec_llms[key] = RecomendadorLLM(CATALOGO, modo=modo, modelo=modelo_id, nombre_modelo=nombre)
    print(f"  LLMs configurados: {', '.join(LLM_MODELS.keys())}")
    print("  Todos los sistemas listos.\n")

    while True:
        query = input("  🔍 Tu consulta: ").strip()
        if not query or query.lower() in ("salir", "exit", "quit"):
            print("\n  ¡Hasta luego!")
            break

        print(f"\n  Evaluando: \"{query}\"")
        print("  " + "═" * 67)

        todos_resultados = {}

        # 1. Basado en Contenido
        print("\n  ⏳ Basado en Contenido (TF-IDF)...")
        res, lat = rec_contenido.recomendar(query)
        todos_resultados["Contenido (TF-IDF)"] = (res, lat)
        mostrar_resultados("BASADO EN CONTENIDO (TF-IDF)", res, lat)

        # 2. Filtrado Colaborativo
        print("\n  ⏳ Filtrado Colaborativo...")
        res, lat = rec_colaborativo.recomendar(query)
        todos_resultados["Colaborativo"] = (res, lat)
        mostrar_resultados("FILTRADO COLABORATIVO", res, lat)

        # 3. Basado en Conocimiento
        print("\n  ⏳ Basado en Conocimiento (Reglas)...")
        res, lat = rec_conocimiento.recomendar(query)
        todos_resultados["Conocimiento (Reglas)"] = (res, lat)
        mostrar_resultados("BASADO EN CONOCIMIENTO (REGLAS)", res, lat)

        # 4+. LLMs (todos los modelos × modos)
        for i, (key, rec_llm) in enumerate(rec_llms.items()):
            if i > 0:
                time.sleep(4)  # Evitar rate limiting entre llamadas
            print(f"\n  ⏳ {key} (OpenRouter)...")
            res, lat = rec_llm.recomendar(query)
            todos_resultados[key] = (res, lat)
            mostrar_resultados(key.upper(), res, lat)

        # Estadísticas comparativas
        mostrar_estadisticas(todos_resultados)

        print("\n" + "  " + "═" * 67 + "\n")


if __name__ == "__main__":
    main()
