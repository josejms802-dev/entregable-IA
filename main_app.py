import streamlit as st
from groq import Groq

st.set_page_config(page_title="Chat con Groq", page_icon="💬")

st.title("💬 Chat con Groq")
st.caption("Ingresa tu API Key de Groq y hazle preguntas al modelo.")

# --- Sidebar: configuración ---
with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("Groq API Key", type="password")
    model = st.selectbox(
        "Modelo",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
        ],
        index=0,
    )
    st.markdown("[Obtén tu API Key aquí](https://console.groq.com/keys)")

# --- Estado de la conversación ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Mostrar historial ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Input del usuario ---
pregunta = st.chat_input("Escribe tu pregunta...")

if pregunta:
    if not api_key:
        st.error("Por favor ingresa tu API Key de Groq en la barra lateral.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    try:
        client = Groq(api_key=api_key)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            respuesta_completa = ""

            stream = client.chat.completions.create(
                model=model,
                messages=st.session_state.messages,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                respuesta_completa += delta
                placeholder.markdown(respuesta_completa + "▌")

            placeholder.markdown(respuesta_completa)

        st.session_state.messages.append(
            {"role": "assistant", "content": respuesta_completa}
        )

    except Exception as e:
        st.error(f"Ocurrió un error al conectar con Groq: {e}")

# --- Botón para limpiar chat ---
if st.session_state.messages:
    if st.sidebar.button("🗑️ Limpiar conversación"):
        st.session_state.messages = []
        st.rerun()
