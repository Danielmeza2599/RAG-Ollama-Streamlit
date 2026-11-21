import streamlit as st
from rag_core import (EMBEDDING_MODEL, DEFAULT_LLM, extract_pdf, split_pages, build_index, retrieve, generate_answer, check_ollama, DEFAULT_OLLAMA_URL)
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="RAG | MTW410", layout="wide")
st.title("RAG + Hugging Face + Ollama")
st.caption("MTW410 · PDF → fragmentos → embeddings → FAISS → Ollama local")
@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

with st.sidebar:
    st.header("Configuración")
    ollama_url = st.text_input("Servidor Ollama", value=DEFAULT_OLLAMA_URL)
    model = st.text_input("Modelo local", value=DEFAULT_LLM)
    if st.button("Comprobar Ollama"):
        try:
            available = check_ollama(ollama_url)
            st.success("Conectado. Modelos: " + (", ".join(available) if available else "ninguno"))
            if model not in available:
                st.warning(f"Descarga el modelo con: ollama pull {model}")
        except Exception as exc:
            st.error(f"No se pudo conectar con Ollama: {exc}")
    k = st.slider("Fragmentos recuperados", 1, 6, 3)
    st.info("Los PDF escaneados requieren OCR. Los PDF se procesan localmente. La primera ejecución descarga el modelo de embeddings.")

files = st.file_uploader("Carga uno o varios archivos PDF", type=["pdf"], accept_multiple_files=True)
if files:
    signatures = [(f.name, f.size, f.file_id) for f in files]
    if st.session_state.get("signatures") != signatures:
        with st.spinner("Extrayendo texto, calculando embeddings e indexando…"):
            try:
                pages = []
                for f in files:
                    pages.extend(extract_pdf(f.getvalue(), f.name))
                chunks = split_pages(pages)
                embedder = get_embedder()
                index = build_index(chunks, embedder)
                st.session_state.update(signatures=signatures, chunks=chunks, index=index, history=[])
            except Exception as exc:
                st.error(f"No fue posible procesar el PDF: {exc}")
                st.stop()
    st.success(f"Índice listo: {len(st.session_state['chunks'])} fragmentos de {len(files)} archivo(s).")
    for msg in st.session_state.get("history", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("hits"):
                with st.expander("Ver fragmentos recuperados"):
                    for h in msg["hits"]:
                        st.markdown(f"**{h['source']} · página {h['page']} · similitud {h['score']:.3f}**")
                        st.write(h["text"])
    question = st.chat_input("Pregunta algo sobre tus documentos")
    if question:
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        hits = retrieve(question, st.session_state.index, st.session_state.chunks, get_embedder(), k=k)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Consultando el modelo generativo…"):
                    answer = generate_answer(question, hits, model=model, base_url=ollama_url)
            except Exception as exc:
                answer = f"No fue posible generar la respuesta: {exc}\n\nPuedes revisar los fragmentos recuperados a continuación y comprobar que Ollama esté abierto y que el modelo esté descargado."
            st.markdown(answer)
            with st.expander("Ver fragmentos recuperados"):
                for h in hits:
                    st.markdown(f"**{h['source']} · página {h['page']} · similitud {h['score']:.3f}**")
                    st.write(h["text"])
        st.session_state.history.append({"role": "assistant", "content": answer, "hits": hits})
else:
    st.info("Comienza cargando el PDF de ejemplo o uno de tus documentos.")
