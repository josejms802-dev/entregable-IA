import re
import random
from collections import Counter

import numpy as np
import pandas as pd
import requests
import streamlit as st
import tiktoken
from groq import Groq

st.set_page_config(page_title="Tokenización & Embeddings con Groq", page_icon="🧠", layout="wide")

st.title("🧠 Tokenización, Bag of Words & Embeddings")
st.caption(
    "Demo educativa: cómo un modelo tipo GPT tokeniza texto, arma su bag of words "
    "y cómo se compara la similitud semántica entre dos frases mediante embeddings."
)

# =========================================================
# Sidebar: configuración
# =========================================================
with st.sidebar:
    st.header("⚙️ Configuración")
    api_key = st.text_input("Groq API Key", type="password")
    st.markdown("[Obtén tu API Key aquí](https://console.groq.com/keys)")

    encoding_name = st.selectbox(
        "Tokenizador (familia GPT)",
        ["o200k_base", "cl100k_base"],
        index=0,
        help="o200k_base: usado por GPT-4o / gpt-oss. cl100k_base: usado por GPT-3.5/GPT-4 clásico.",
    )

    st.divider()
    st.caption("Solo se listan modelos GPT (gpt-oss) del catálogo de Groq. No se usa Ollama.")


@st.cache_data(show_spinner=False, ttl=300)
def obtener_catalogo_modelos(key: str):
    """Consulta el catálogo real de modelos de Groq."""
    resp = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {key}"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json().get("data", [])
    # Solo modelos GPT (gpt-oss), nunca Ollama ni otras familias
    return [m for m in data if "gpt" in m.get("id", "").lower()]


@st.cache_resource(show_spinner=False)
def obtener_encoder(nombre_encoding: str):
    return tiktoken.get_encoding(nombre_encoding)


PALETA = [
    "#FDE68A", "#A7F3D0", "#BFDBFE", "#FBCFE8", "#FCA5A5",
    "#DDD6FE", "#FCD34D", "#99F6E4", "#C4B5FD", "#FDBA74",
]

tab1, tab2 = st.tabs(["🔤 Tokenización + Bag of Words", "📐 Embeddings & Similitud coseno"])

# =========================================================
# TAB 1: Tokenización
# =========================================================
with tab1:
    st.subheader("Catálogo de modelos GPT disponibles en Groq")

    modelos_gpt = []
    if api_key:
        try:
            modelos_gpt = obtener_catalogo_modelos(api_key)
            if modelos_gpt:
                df_modelos = pd.DataFrame(
                    [
                        {
                            "id": m.get("id"),
                            "propietario": m.get("owned_by"),
                            "context_window": m.get("context_window"),
                            "activo": m.get("active"),
                        }
                        for m in modelos_gpt
                    ]
                )
                st.dataframe(df_modelos, use_container_width=True, hide_index=True)
            else:
                st.info("No se encontraron modelos GPT (gpt-oss) en tu cuenta de Groq.")
        except Exception as e:
            st.error(f"No se pudo obtener el catálogo de modelos: {e}")
    else:
        st.info("Ingresa tu API Key en la barra lateral para cargar el catálogo real de modelos.")

    ids_modelos = [m["id"] for m in modelos_gpt] if modelos_gpt else ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]

    col_a, col_b = st.columns(2)
    with col_a:
        modelo_sel = st.selectbox("Modelo GPT a usar", ids_modelos)
    with col_b:
        temperatura = st.slider("Temperatura", 0.0, 2.0, 0.7, 0.05)

    # Learning rate: NO es un parámetro real de inferencia, se muestra solo con fines ilustrativos.
    seed_lr = sum(ord(c) for c in modelo_sel)
    learning_rate_ilustrativo = round(random.Random(seed_lr).uniform(1e-5, 5e-4), 6)
    st.caption(
        f"🎓 **Learning rate (ilustrativo):** `{learning_rate_ilustrativo}` — "
        "esto **no** es un dato real de la API: el learning rate es un hiperparámetro de "
        "*entrenamiento*, no de inferencia, y Groq (como ninguna API de completions) no lo expone. "
        "Se muestra solo con fines educativos, fijo por modelo para esta demo."
    )

    st.divider()
    st.subheader("Ingresa un prompt para tokenizar")
    prompt = st.text_area("Prompt", placeholder="Escribe aquí el texto a tokenizar...", height=100)

    if prompt:
        enc = obtener_encoder(encoding_name)
        token_ids = enc.encode(prompt)

        st.markdown(f"**Total de tokens:** {len(token_ids)}")

        # --- Visualización coloreada ---
        st.markdown("#### Tokenización coloreada")
        html_spans = []
        filas_tabla = []
        for i, tid in enumerate(token_ids):
            texto_tok = enc.decode([tid])
            color = PALETA[i % len(PALETA)]
            texto_html = texto_tok.replace(" ", "·").replace("\n", "⏎")
            html_spans.append(
                f'<span style="background-color:{color}; padding:2px 4px; '
                f'margin:2px; border-radius:4px; display:inline-block; '
                f'font-family:monospace; font-size:0.9em;" '
                f'title="token_id: {tid}">{texto_html}</span>'
            )
            filas_tabla.append({"posición": i, "token": texto_tok, "token_id": tid})

        st.markdown(" ".join(html_spans), unsafe_allow_html=True)

        st.markdown("#### Detalle token por token")
        st.dataframe(pd.DataFrame(filas_tabla), use_container_width=True, hide_index=True)

        # --- Bag of Words ---
        st.markdown("#### Bag of Words (a nivel de palabra)")
        palabras = re.findall(r"\b\w+\b", prompt.lower())
        bow = Counter(palabras)
        df_bow = pd.DataFrame(bow.items(), columns=["palabra", "frecuencia"]).sort_values(
            "frecuencia", ascending=False
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(df_bow, use_container_width=True, hide_index=True)
        with c2:
            st.bar_chart(df_bow.set_index("palabra"))

        # --- Generación real opcional ---
        st.divider()
        if st.button("🚀 Generar respuesta real con este modelo y temperatura"):
            if not api_key:
                st.error("Ingresa tu API Key para poder generar una respuesta.")
            else:
                try:
                    client = Groq(api_key=api_key)
                    resp = client.chat.completions.create(
                        model=modelo_sel,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperatura,
                    )
                    st.markdown("**Respuesta del modelo:**")
                    st.write(resp.choices[0].message.content)
                except Exception as e:
                    st.error(f"Error al llamar a la API de Groq: {e}")
    else:
        st.info("Escribe un prompt arriba para ver su tokenización.")

# =========================================================
# TAB 2: Embeddings y similitud coseno
# =========================================================
with tab2:
    st.subheader("Comparar dos frases por similitud coseno")
    st.caption(
        "Los embeddings se generan localmente con un modelo de sentence-transformers "
        "(no requieren la API de Groq, que no ofrece endpoint de embeddings)."
    )

    @st.cache_resource(show_spinner="Cargando modelo de embeddings...")
    def cargar_modelo_embeddings():
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")

    col1, col2 = st.columns(2)
    with col1:
        frase1 = st.text_area("Frase 1", placeholder="Ej: El gato duerme en el sofá", height=100)
    with col2:
        frase2 = st.text_area("Frase 2", placeholder="Ej: El felino descansa en el mueble", height=100)

    if st.button("📐 Calcular similitud"):
        if not frase1 or not frase2:
            st.warning("Ingresa ambas frases para comparar.")
        else:
            try:
                modelo_emb = cargar_modelo_embeddings()
                emb1 = modelo_emb.encode(frase1)
                emb2 = modelo_emb.encode(frase2)

                similitud = float(
                    np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                )

                st.metric("Similitud coseno", f"{similitud:.4f}")
                st.progress(min(max((similitud + 1) / 2, 0.0), 1.0))

                if similitud > 0.8:
                    interpretacion = "🟢 Muy similares semánticamente."
                elif similitud > 0.5:
                    interpretacion = "🟡 Moderadamente relacionadas."
                else:
                    interpretacion = "🔴 Poco o nada relacionadas."
                st.write(interpretacion)

                with st.expander("Ver vectores de embedding (primeras 20 dimensiones)"):
                    df_emb = pd.DataFrame(
                        {
                            "dim": list(range(20)),
                            "frase_1": emb1[:20],
                            "frase_2": emb2[:20],
                        }
                    )
                    st.dataframe(df_emb, use_container_width=True, hide_index=True)
                    st.caption(f"Dimensión total del embedding: {len(emb1)}")

            except ModuleNotFoundError:
                st.error(
                    "Falta instalar `sentence-transformers`. Ejecuta: "
                    "`pip install -r requirements.txt`"
                )
            except Exception as e:
                st.error(f"Error calculando embeddings: {e}")