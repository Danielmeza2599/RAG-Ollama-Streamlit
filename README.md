# RAG Ollama Streamlit

Aplicación didáctica de Retrieval-Augmented Generation (RAG) que permite subir uno o varios PDF, recuperarlos por similitud semántica y hacerles preguntas con un modelo local de Ollama.

El sistema extrae texto seleccionado de las páginas con PyMuPDF, crea fragmentos de hasta 850 caracteres con solapamiento de 150 caracteres, genera embeddings con Sentence Transformers y los busca en un índice FAISS temporal. Las respuestas se producen localmente con Ollama y se acompañan de la fuente, página y fragmentos recuperados.

## Características

- Carga de uno o varios archivos PDF.
- Extracción de texto por página.
- Fragmentación configurable en el código.
- Embeddings multilingües locales.
- Recuperación vectorial con FAISS.
- Generación con Ollama mediante su API HTTP local.
- Chat e índice mantenidos en memoria durante la sesión de Streamlit.
- Citas con nombre de archivo y número de página.

## Arquitectura rápida

- `app.py`: interfaz web, carga de PDFs y estado de Streamlit.
- `rag_core.py`: extracción, chunking, embeddings, FAISS y llamadas a Ollama.
- `requirements.txt`: dependencias Python.
- `.env.example`: plantilla de configuración local.
- `Context/ContextProject-Agents.md`: contexto técnico completo del proyecto.

## Requisitos

- Python 3. No está fijada una versión mínima en el proyecto.
- Ollama instalado y accesible.
- Dependencias Python indicadas en `requirements.txt`.
- Conexión a Internet en la primera ejecución si el modelo de embeddings no está descargado.

## Instalación

Desde la raíz del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Instala y inicia Ollama:

```bash
ollama pull qwen2.5:3b
ollama serve
```

El comando `ollama serve` puede omitirse si Ollama ya está ejecutándose como servicio.

## Configuración

El archivo `.env` acepta estas variables:

| Variable | Valor predeterminado | Uso |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | URL del servidor Ollama |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Modelo generado por Ollama |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Modelo de embeddings |

La configuración puede modificarse desde la barra lateral de la aplicación. Las variables permiten establecer valores iniciales al iniciar la sesión.

## Ejecutar

```bash
streamlit run app.py
```

Luego:

1. Comprueba la conexión con Ollama desde la barra lateral.
2. Confirma que el modelo aparece en Ollama o ejecútalo con `ollama pull`.
3. Carga uno o varios PDF con texto seleccionable.
4. Ajusta la cantidad de fragmentos recuperados.
5. Formula preguntas en el chat.

Comprobación manual de Ollama:

```bash
curl http://localhost:11434/api/tags
```

## Limitaciones

- No procesa OCR; los PDF escaneados necesitan texto extraíble.
- El índice y el historial se mantienen solo en memoria.
- No tiene autenticación, API propia, base de datos ni persistencia.
- No incluye tests automatizados ni pipeline CI/CD.
- Se recomienda no subir PDFs al repositorio si contienen información sensible.

## Validación

No existe una suite de pruebas configurada. Para una comprobación básica de sintaxis:

```bash
python -m compileall app.py rag_core.py
```

Para validar funcionalmente la aplicación, carga un PDF con texto seleccionable y una pregunta cuya respuesta esté contenida en el documento.

## Datos y privacidad

La extracción y generación están diseñadas para ejecutarse localmente. Los archivos se procesan en memoria y no se escribe una base de datos persistente. Ollama recibe únicamente el fragmento recuperado, el prompt y la pregunta.
