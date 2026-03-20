# Sistemas Recomendadores — ML vs LLM

Proyecto educativo que compara cinco enfoques de sistemas de recomendacion: tres basados en Machine Learning tradicional y dos basados en Large Language Models. Incluye una presentacion interactiva en Reveal.js y una demo web donde todos los sistemas evaluan la misma consulta en tiempo real.

## Contenido

- Presentacion con slides interactivos (tipos, casos reales, metricas, comparativas)
- Demo web interactiva: escribe una consulta y 5 sistemas la evaluan simultaneamente
- Codigo fuente de cada sistema recomendador
- Estadisticas comparativas de relevancia y latencia

## Sistemas implementados

| # | Sistema | Tipo | Tecnica |
|---|---------|------|---------|
| 1 | Basado en Contenido | ML | TF-IDF + Similitud Coseno sobre texto descriptivo de productos |
| 2 | Filtrado Colaborativo | ML | User-based CF con similitud coseno sobre matriz usuario-item |
| 3 | Basado en Conocimiento | ML | 30+ reglas de dominio con filtros en cascada |
| 4 | LLM Zero-shot | LLM | Prompt con catalogo completo, sin ejemplos (OpenRouter API) |
| 5 | LLM Few-shot | LLM | Prompt con catalogo + 2 ejemplos demostrativos (OpenRouter API) |

## Requisitos

- Python 3.9 o superior
- Cuenta gratuita en OpenRouter (https://openrouter.ai) para los modelos LLM

## Instalacion

```bash
git clone https://github.com/Camilo-Camargo/recommendation-systems.git
cd recommendation-systems
pip install -r requirements.txt
```

## Uso

### Servidor web (presentacion + demo)

```bash
export OPENROUTER_API_KEY=tu_api_key_aqui
python3 server.py
```

Abrir en el navegador:

- http://localhost:8080 — Pagina principal
- http://localhost:8080/presentacion — Slides Reveal.js
- http://localhost:8080/demo — Demo interactiva

### Demo por terminal

```bash
export OPENROUTER_API_KEY=tu_api_key_aqui
python3 codigo/demo_comparativa.py
```

Escribe cualquier consulta (por ejemplo: "zapato para correr", "quiero un buen libro", "laptop para programar") y los 5 sistemas generan recomendaciones con scores, razones y estadisticas comparativas.

## Estructura del proyecto

```
recommendation-systems/
  server.py                  -- Servidor Flask con presentacion y demo web
  requirements.txt           -- Dependencias de Python
  Dockerfile                 -- Para despliegue en contenedor
  presentacion/
    index.html               -- Slides Reveal.js (27 slides)
  codigo/
    demo_comparativa.py      -- Demo interactiva por terminal
    evaluacion_completa.py   -- Evaluacion con dataset MovieLens
```

## Despliegue gratuito

El proyecto puede desplegarse en Render.com o Railway.app:

1. Subir el repositorio a GitHub
2. Crear un Web Service conectado al repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn server:app --bind 0.0.0.0:$PORT`
5. Agregar variable de entorno: `OPENROUTER_API_KEY`

## Tecnologias

- Python, Flask, Gunicorn
- scikit-learn (TF-IDF, similitud coseno)
- NumPy (operaciones matriciales para filtrado colaborativo)
- OpenAI SDK (cliente para OpenRouter API)
- Reveal.js (presentacion)

## Referencias

- Linden, G., Smith, B., York, J. — "Amazon.com Recommendations: Item-to-Item Collaborative Filtering" — IEEE Internet Computing, 2003
- Netflix Prize — en.wikipedia.org/wiki/Netflix_Prize
- Covington, P., Adams, J., Sargin, E. — "Deep Neural Networks for YouTube Recommendations" — ACM RecSys, 2016
- Music Genome Project — en.wikipedia.org/wiki/Music_Genome_Project
- Burke, R. — "Knowledge-based recommender systems" — Encyclopedia of Library and Information Systems, 2000
