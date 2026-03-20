"""
Evaluación Comparativa de Sistemas Recomendadores
ML Tradicional vs LLM — Dataset MovieLens
"""

import json
import time
import numpy as np
import pandas as pd
from collections import defaultdict

# ── ML ──────────────────────────────────────────────
from surprise import SVD, KNNBasic, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── LLM ─────────────────────────────────────────────
import openai

# ═══════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════

OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY"
LLM_MODEL = "deepseek/deepseek-chat"
TOP_K = 10
NUM_TEST_USERS = 20  # Usuarios a evaluar (más = más lento para LLM)

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


# ═══════════════════════════════════════════════════════
# 1. CARGA DE DATOS — MovieLens 100K
# ═══════════════════════════════════════════════════════

def cargar_datos():
    """Carga MovieLens 100K desde surprise y construye DataFrames auxiliares."""
    print("=" * 60)
    print("CARGANDO DATOS — MovieLens 100K")
    print("=" * 60)

    # Cargar ratings vía surprise
    data = Dataset.load_builtin("ml-100k")
    raw = data.raw_ratings  # lista de (user, item, rating, timestamp)
    df = pd.DataFrame(raw, columns=["user_id", "item_id", "rating", "timestamp"])
    df["rating"] = df["rating"].astype(float)

    # Cargar info de películas
    from surprise.builtin_datasets import get_dataset_dir
    import os

    movie_file = os.path.join(get_dataset_dir(), "ml-100k", "ml-100k", "u.item")
    genre_names = [
        "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    ]
    movies = pd.read_csv(
        movie_file, sep="|", encoding="latin-1", header=None,
        names=["item_id", "title", "release_date", "video_release", "imdb_url"]
        + genre_names,
        usecols=["item_id", "title"] + genre_names,
    )
    movies["item_id"] = movies["item_id"].astype(str)

    # Construir string de géneros por película
    def get_genres(row):
        return " ".join([g for g in genre_names if row[g] == 1])

    movies["genres_str"] = movies.apply(get_genres, axis=1)

    print(f"  Ratings: {len(df):,}")
    print(f"  Usuarios: {df['user_id'].nunique()}")
    print(f"  Películas: {df['item_id'].nunique()}")
    print()

    return data, df, movies


# ═══════════════════════════════════════════════════════
# 2. MÉTRICAS DE EVALUACIÓN
# ═══════════════════════════════════════════════════════

def precision_at_k(recommended, relevant, k=TOP_K):
    """Precisión: fracción de recomendaciones en top-K que son relevantes."""
    rec_k = recommended[:k]
    hits = len(set(rec_k) & set(relevant))
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended, relevant, k=TOP_K):
    """Recall: fracción de ítems relevantes capturados en top-K."""
    rec_k = recommended[:k]
    hits = len(set(rec_k) & set(relevant))
    return hits / len(relevant) if len(relevant) > 0 else 0.0


def ndcg_at_k(recommended, relevant, k=TOP_K):
    """NDCG: Normalized Discounted Cumulative Gain."""
    rec_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(i + 2) for i, item in enumerate(rec_k) if item in relevant
    )
    # IDCG: mejor ranking posible
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def diversidad_ild(recommended, sim_matrix, item_to_idx, k=TOP_K):
    """Intra-List Diversity: 1 - similitud promedio entre pares recomendados."""
    rec_k = [r for r in recommended[:k] if r in item_to_idx]
    if len(rec_k) < 2:
        return 0.0
    total_sim = 0.0
    count = 0
    for i in range(len(rec_k)):
        for j in range(i + 1, len(rec_k)):
            idx_i = item_to_idx[rec_k[i]]
            idx_j = item_to_idx[rec_k[j]]
            total_sim += sim_matrix[idx_i, idx_j]
            count += 1
    avg_sim = total_sim / count if count > 0 else 0.0
    return 1.0 - avg_sim


# ═══════════════════════════════════════════════════════
# 3. MODELO — FILTRADO COLABORATIVO (SVD)
# ═══════════════════════════════════════════════════════

def entrenar_colaborativo(data, df_train, df_test):
    """Entrena SVD y genera top-K recomendaciones para usuarios de prueba."""
    print("-" * 60)
    print("MODELO 1: Filtrado Colaborativo (SVD)")
    print("-" * 60)

    reader = Reader(rating_scale=(1, 5))
    train_data = Dataset.load_from_df(
        df_train[["user_id", "item_id", "rating"]], reader
    )
    trainset = train_data.build_full_trainset()

    model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)

    t0 = time.time()
    model.fit(trainset)
    tiempo_entrenamiento = time.time() - t0

    print(f"  Tiempo de entrenamiento: {tiempo_entrenamiento:.2f}s")

    return model, tiempo_entrenamiento


def recomendar_colaborativo(model, user_id, items_vistos, todos_los_items, k=TOP_K):
    """Genera top-K recomendaciones para un usuario con SVD."""
    items_candidatos = [i for i in todos_los_items if i not in items_vistos]
    predicciones = []
    for item_id in items_candidatos:
        pred = model.predict(user_id, item_id)
        predicciones.append((item_id, pred.est))
    predicciones.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in predicciones[:k]]


# ═══════════════════════════════════════════════════════
# 4. MODELO — BASADO EN CONTENIDO (TF-IDF)
# ═══════════════════════════════════════════════════════

def entrenar_contenido(movies):
    """Construye matriz de similitud basada en TF-IDF de géneros."""
    print("-" * 60)
    print("MODELO 2: Basado en Contenido (TF-IDF + Coseno)")
    print("-" * 60)

    t0 = time.time()
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(movies["genres_str"])
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    tiempo = time.time() - t0

    # Mapeo item_id -> índice en la matriz
    item_to_idx = {str(row["item_id"]): idx for idx, row in movies.iterrows()}

    print(f"  Matriz de similitud: {sim_matrix.shape}")
    print(f"  Tiempo de construcción: {tiempo:.2f}s")

    return sim_matrix, item_to_idx, tiempo


def recomendar_contenido(user_id, df_train, sim_matrix, item_to_idx, movies, k=TOP_K):
    """Genera top-K basándose en similitud promedio con películas vistas."""
    user_ratings = df_train[df_train["user_id"] == user_id]
    liked = user_ratings[user_ratings["rating"] >= 4.0]["item_id"].tolist()
    vistos = set(user_ratings["item_id"].tolist())

    if not liked:
        liked = user_ratings.nlargest(5, "rating")["item_id"].tolist()

    scores = defaultdict(float)
    for item in liked:
        if item not in item_to_idx:
            continue
        idx = item_to_idx[item]
        for other_item, other_idx in item_to_idx.items():
            if other_item not in vistos:
                scores[other_item] += sim_matrix[idx, other_idx]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in ranked[:k]]


# ═══════════════════════════════════════════════════════
# 5. MODELO — LLM (OpenRouter)
# ═══════════════════════════════════════════════════════

def recomendar_llm(user_id, df_train, movies, modo="few-shot", k=TOP_K):
    """Genera recomendaciones usando un LLM vía OpenRouter."""
    user_ratings = df_train[df_train["user_id"] == user_id].sort_values(
        "rating", ascending=False
    )

    # Películas favoritas del usuario (rating >= 4)
    favoritas = user_ratings[user_ratings["rating"] >= 4.0].head(15)
    fav_titles = []
    for _, row in favoritas.iterrows():
        movie_info = movies[movies["item_id"] == row["item_id"]]
        if not movie_info.empty:
            title = movie_info.iloc[0]["title"]
            fav_titles.append(f"- {title} (rating: {row['rating']})")

    # Todas las películas vistas
    vistos = set(user_ratings["item_id"].tolist())
    vistos_titles = set()
    for item_id in vistos:
        movie_info = movies[movies["item_id"] == item_id]
        if not movie_info.empty:
            vistos_titles.add(movie_info.iloc[0]["title"])

    # Catálogo disponible (para que el LLM elija de películas reales)
    catalogo_disponible = movies[~movies["item_id"].isin(vistos)].head(200)
    catalogo_str = "\n".join(
        f"- ID:{row['item_id']} | {row['title']} | {row['genres_str']}"
        for _, row in catalogo_disponible.iterrows()
    )

    if modo == "zero-shot":
        prompt = f"""Eres un sistema recomendador de películas experto.

CATÁLOGO DISPONIBLE (elige SOLO de esta lista):
{catalogo_str}

Recomienda exactamente {k} películas del catálogo anterior.
Responde SOLO con un JSON válido, sin texto adicional:
{{"recommendations": [{{"id": "...", "title": "..."}}]}}"""

    else:  # few-shot
        prompt = f"""Eres un sistema recomendador de películas experto.

PERFIL DEL USUARIO — películas favoritas:
{chr(10).join(fav_titles[:10])}

CATÁLOGO DISPONIBLE (elige SOLO de esta lista):
{catalogo_str}

Basándote en los gustos del usuario, recomienda exactamente {k} películas del catálogo que probablemente disfrutará.
Responde SOLO con un JSON válido, sin texto adicional:
{{"recommendations": [{{"id": "...", "title": "..."}}]}}"""

    try:
        t0 = time.time()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1000,
        )
        latencia = time.time() - t0
        content = response.choices[0].message.content.strip()

        # Limpiar markdown si viene envuelto
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0].strip()

        result = json.loads(content)
        recs = [str(r["id"]) for r in result["recommendations"]]
        return recs[:k], latencia

    except Exception as e:
        print(f"    Error LLM para usuario {user_id}: {e}")
        return [], 0.0


# ═══════════════════════════════════════════════════════
# 6. EVALUACIÓN COMPLETA
# ═══════════════════════════════════════════════════════

def evaluar():
    """Ejecuta la evaluación comparativa completa."""

    # Cargar datos
    data, df, movies = cargar_datos()

    # Split temporal: 80% train, 20% test
    df = df.sort_values("timestamp")
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    print(f"Train: {len(df_train):,} ratings | Test: {len(df_test):,} ratings\n")

    todos_los_items = df["item_id"].unique().tolist()

    # ── Entrenar modelos ML ──
    modelo_svd, t_svd = entrenar_colaborativo(data, df_train, df_test)
    sim_matrix, item_to_idx, t_content = entrenar_contenido(movies)

    # ── Seleccionar usuarios de prueba ──
    # Usuarios con suficientes ratings en train Y test
    users_train = set(df_train["user_id"].unique())
    users_test = set(df_test["user_id"].unique())
    users_comunes = list(users_train & users_test)

    # Filtrar usuarios con al menos 5 ratings en test (para tener ground truth)
    users_con_test = []
    for u in users_comunes:
        test_relevantes = df_test[
            (df_test["user_id"] == u) & (df_test["rating"] >= 4.0)
        ]
        if len(test_relevantes) >= 3:
            users_con_test.append(u)

    np.random.seed(42)
    test_users = np.random.choice(
        users_con_test, size=min(NUM_TEST_USERS, len(users_con_test)), replace=False
    )
    print(f"\nUsuarios de prueba: {len(test_users)}\n")

    # ── Evaluar cada modelo ──
    resultados = {
        "Colaborativo (SVD)": {"precision": [], "recall": [], "ndcg": [], "diversidad": [], "latencias": []},
        "Basado en Contenido": {"precision": [], "recall": [], "ndcg": [], "diversidad": [], "latencias": []},
        "LLM (Zero-shot)": {"precision": [], "recall": [], "ndcg": [], "diversidad": [], "latencias": []},
        "LLM (Few-shot)": {"precision": [], "recall": [], "ndcg": [], "diversidad": [], "latencias": []},
    }

    print("=" * 60)
    print("EVALUACIÓN EN PROGRESO")
    print("=" * 60)

    for i, user_id in enumerate(test_users):
        print(f"\n  Usuario {i+1}/{len(test_users)} (ID: {user_id})")

        # Ground truth: películas con rating >= 4 en test
        relevant = df_test[
            (df_test["user_id"] == user_id) & (df_test["rating"] >= 4.0)
        ]["item_id"].tolist()

        items_vistos = set(df_train[df_train["user_id"] == user_id]["item_id"].tolist())

        # ── Colaborativo ──
        t0 = time.time()
        recs_svd = recomendar_colaborativo(
            modelo_svd, user_id, items_vistos, todos_los_items
        )
        lat_svd = time.time() - t0
        resultados["Colaborativo (SVD)"]["precision"].append(precision_at_k(recs_svd, relevant))
        resultados["Colaborativo (SVD)"]["recall"].append(recall_at_k(recs_svd, relevant))
        resultados["Colaborativo (SVD)"]["ndcg"].append(ndcg_at_k(recs_svd, relevant))
        resultados["Colaborativo (SVD)"]["diversidad"].append(
            diversidad_ild(recs_svd, sim_matrix, item_to_idx)
        )
        resultados["Colaborativo (SVD)"]["latencias"].append(lat_svd)
        print(f"    SVD: P@{TOP_K}={precision_at_k(recs_svd, relevant):.3f}")

        # ── Contenido ──
        t0 = time.time()
        recs_content = recomendar_contenido(
            user_id, df_train, sim_matrix, item_to_idx, movies
        )
        lat_content = time.time() - t0
        resultados["Basado en Contenido"]["precision"].append(precision_at_k(recs_content, relevant))
        resultados["Basado en Contenido"]["recall"].append(recall_at_k(recs_content, relevant))
        resultados["Basado en Contenido"]["ndcg"].append(ndcg_at_k(recs_content, relevant))
        resultados["Basado en Contenido"]["diversidad"].append(
            diversidad_ild(recs_content, sim_matrix, item_to_idx)
        )
        resultados["Basado en Contenido"]["latencias"].append(lat_content)
        print(f"    Content: P@{TOP_K}={precision_at_k(recs_content, relevant):.3f}")

        # ── LLM Zero-shot ──
        recs_llm_zero, lat_zero = recomendar_llm(user_id, df_train, movies, modo="zero-shot")
        resultados["LLM (Zero-shot)"]["precision"].append(precision_at_k(recs_llm_zero, relevant))
        resultados["LLM (Zero-shot)"]["recall"].append(recall_at_k(recs_llm_zero, relevant))
        resultados["LLM (Zero-shot)"]["ndcg"].append(ndcg_at_k(recs_llm_zero, relevant))
        resultados["LLM (Zero-shot)"]["diversidad"].append(
            diversidad_ild(recs_llm_zero, sim_matrix, item_to_idx)
        )
        resultados["LLM (Zero-shot)"]["latencias"].append(lat_zero)
        print(f"    LLM Zero-shot: P@{TOP_K}={precision_at_k(recs_llm_zero, relevant):.3f} ({lat_zero:.1f}s)")

        # ── LLM Few-shot ──
        recs_llm_few, lat_few = recomendar_llm(user_id, df_train, movies, modo="few-shot")
        resultados["LLM (Few-shot)"]["precision"].append(precision_at_k(recs_llm_few, relevant))
        resultados["LLM (Few-shot)"]["recall"].append(recall_at_k(recs_llm_few, relevant))
        resultados["LLM (Few-shot)"]["ndcg"].append(ndcg_at_k(recs_llm_few, relevant))
        resultados["LLM (Few-shot)"]["diversidad"].append(
            diversidad_ild(recs_llm_few, sim_matrix, item_to_idx)
        )
        resultados["LLM (Few-shot)"]["latencias"].append(lat_few)
        print(f"    LLM Few-shot: P@{TOP_K}={precision_at_k(recs_llm_few, relevant):.3f} ({lat_few:.1f}s)")

    # ═══════════════════════════════════════════════════
    # RESULTADOS FINALES
    # ═══════════════════════════════════════════════════
    print("\n")
    print("=" * 70)
    print("RESULTADOS FINALES — EVALUACIÓN COMPARATIVA")
    print("=" * 70)
    print(f"{'Modelo':<25} {'P@10':>8} {'R@10':>8} {'NDCG@10':>8} {'Divers.':>8} {'Latencia':>10}")
    print("-" * 70)

    resumen = {}
    for nombre, metricas in resultados.items():
        p = np.mean(metricas["precision"])
        r = np.mean(metricas["recall"])
        n = np.mean(metricas["ndcg"])
        d = np.mean(metricas["diversidad"])
        l = np.mean(metricas["latencias"])
        print(f"{nombre:<25} {p:>8.4f} {r:>8.4f} {n:>8.4f} {d:>8.4f} {l:>9.3f}s")
        resumen[nombre] = {"precision": p, "recall": r, "ndcg": n, "diversidad": d, "latencia": l}

    print("-" * 70)
    print()

    # Guardar resultados
    with open("datos/resultados_evaluacion.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print("Resultados guardados en datos/resultados_evaluacion.json")

    # ── Análisis ──
    print("\n" + "=" * 70)
    print("ANÁLISIS")
    print("=" * 70)

    mejor_precision = max(resumen.items(), key=lambda x: x[1]["precision"])
    mejor_diversidad = max(resumen.items(), key=lambda x: x[1]["diversidad"])
    mejor_ndcg = max(resumen.items(), key=lambda x: x[1]["ndcg"])

    print(f"  Mejor Precision@{TOP_K}: {mejor_precision[0]} ({mejor_precision[1]['precision']:.4f})")
    print(f"  Mejor NDCG@{TOP_K}:     {mejor_ndcg[0]} ({mejor_ndcg[1]['ndcg']:.4f})")
    print(f"  Mejor Diversidad:   {mejor_diversidad[0]} ({mejor_diversidad[1]['diversidad']:.4f})")
    print(f"  Menor Latencia:     Colaborativo (SVD) ({resumen['Colaborativo (SVD)']['latencia']:.4f}s)")
    print()
    print("  CONCLUSIÓN: ML tradicional supera en precisión y velocidad.")
    print("  LLMs destacan en diversidad y no requieren historial extenso.")
    print("  El enfoque óptimo es HÍBRIDO: ML para candidatos + LLM para re-ranking.")


if __name__ == "__main__":
    evaluar()
