# Resumen del proyecto

Proyecto didáctico de recuperación aumentada por generación (RAG) con una interfaz web creada en Streamlit. Permite cargar uno o varios PDF, extraer su texto seleccionable, dividirlo en fragmentos, generar embeddings locales y recuperar pasajes relevantes mediante FAISS. La respuesta se genera con un modelo local de Ollama y se presenta junto con las fuentes recuperadas.

El proyecto está pensado para ejecutarse en local y no utiliza bases de datos persistentes ni servicios de IA externos. Su código fuente principal se limita a dos módulos Python: la interfaz en `app.py` y las operaciones RAG en `rag_core.py`.

# Propósito y alcance

- **Propósito:** consultar en lenguaje natural el contenido textual de documentos PDF y obtener respuestas limitadas a ese contexto.
- **Audiencia:** estudiantes o desarrolladores que quieran estudiar o demostrar un flujo RAG básico con Hugging Face Sentence Transformers, FAISS y Ollama.
- **Objetivo principal:** convertir PDF locales en un índice vectorial temporal y responder preguntas citando archivo y página.
- **Entrada principal:** uno o varios archivos PDF cargados desde la interfaz.
- **Salida principal:** una respuesta en español basada únicamente en los fragmentos recuperados, acompañada por el texto, archivo, página y similitud de cada fragmento.

Incluido:
- Extracción de texto por páginas con PyMuPDF.
- División de texto en fragmentos de 850 caracteres con solapamiento de 150 caracteres.
- Embeddings normalizados de texto con Sentence Transformers.
- Búsqueda de similitud mediante FAISS (`IndexFlatIP`).
- Generación mediante la API HTTP local de Ollama.
- Conversación e índice mantenidos en `st.session_state` durante la sesión de Streamlit.

Fuera del alcance encontrado en el código:
- OCR para PDF escaneados.
- Persistencia de índices o documentos.
- Autenticación, autorización o usuarios.
- API HTTP propia para consumidores externos.
- Base de datos.
- Streaming de respuestas.
- Evaluación automática de calidad RAG.

# Arquitectura y estructura de carpetas

```text
RagOllamaStd/
├── app.py                         # Interfaz Streamlit y orquestación de UI/estado
├── rag_core.py                    # Extracción, chunking, embeddings, FAISS y Ollama
├── requirements.txt               # Dependencias directas con rangos
├── .venv/                         # Entorno virtual local observado (no forma parte del código fuente)
├── __pycache__/                   # Caché local de Python
└── Context/
    └── ContextProject-Agents.md   # Contexto técnico para agentes
```

## Responsabilidades

| Ruta | Responsabilidad |
|---|---|
| `app.py` | Configura Streamlit, carga PDF, mantiene sesiones, muestra controles y chat, y coordina las funciones de `rag_core.py`. |
| `rag_core.py` | Implementa el motor RAG: lectura de PDF, normalización/chunking, creación y consulta del índice, y llamadas HTTP a Ollama. |
| `requirements.txt` | Declara las dependencias directas y sus rangos permitidos. |
| `Context/ContextProject-Agents.md` | Documenta el proyecto para futuros agentes de IA. |

El diseño es un ejemplo monolítico y modular en dos capas: [app.py](../app.py) controla presentación y flujo, mientras [rag_core.py](../rag_core.py) contiene la lógica independiente de interfaz. No hay patrón de dominio, servicios separados, contenedores ni configuración mediante framework.

# Flujo principal de funcionamiento

1. Streamlit crea la página y obtiene los valores predeterminados de Ollama y del modelo.
2. El usuario carga uno o varios PDF desde el componente `file_uploader`.
3. `app.py` calcula una firma con nombre, tamaño y `file_id` para detectar cambios en los archivos.
4. Si los archivos cambiaron, `extract_pdf()` abre cada PDF con PyMuPDF y toma el texto obtenido por página.
5. `split_pages()` elimina espacios repetidos y divide cada página en fragmentos de hasta 850 caracteres, con solapamiento de 150 caracteres.
6. `get_embedder()` carga una vez el modelo `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` gracias a `@st.cache_resource`.
7. `build_index()` codifica los fragmentos, normaliza los vectores y crea un `faiss.IndexFlatIP`. Al usar vectores normalizados, el producto interno representa coseno.
8. Los fragmentos, el índice y el historial quedan en `st.session_state`. No se escriben en disco.
9. El usuario envía una pregunta. `retrieve()` convierte la pregunta en embedding y busca hasta `k` fragmentos; `k` se configura con el deslizador de 1 a 6 y vale 3 por defecto.
10. `make_messages()` prepara un prompt de sistema y usuario con el contexto recuperado.
11. `generate_answer()` envía el prompt a Ollama por `POST /api/chat`.
12. Streamlit muestra la respuesta y un desplegable con archivo, página, puntuación y texto recuperado.
13. Si Ollama falla, Streamlit muestra un mensaje de error y conserva los fragmentos recuperados disponibles para inspección.

# Stack tecnológico

| Tecnología | Uso | Versión declarada o identificada |
|---|---|---|
| Python | Lenguaje principal | `No verificado` como requisito mínimo; `.venv/pyvenv.cfg` indica Python 3.14.6 en el entorno local observado |
| Streamlit | UI web y estado de sesión | `>=1.35,<2` en `requirements.txt` |
| PyMuPDF (`fitz`) | Extracción de texto por página | `>=1.24,<2` |
| Sentence Transformers | Modelo y generación de embeddings | `>=3,<6` |
| FAISS CPU | Índice vectorial y búsqueda por coseno | `>=1.8,<2` |
| NumPy | Vectores y conversión de tipos | `>=1.26,<3` |
| Requests | Comunicación HTTP con Ollama | `>=2.31,<3` |
| python-dotenv | Carga de configuración desde `.env` | `>=1.0,<2` |
| Ollama | Servidor local de generación | Versión `No verificada` |
| Modelo de embeddings | Embeddings multilingües | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Modelo generativo | Respuestas locales | `qwen2.5:3b`, configurable desde la UI |

No se encontró un archivo de lock o fijación exacta de dependencias. Las versiones de `requirements.txt` son rangos, no versiones completas bloqueadas.

# Configuración y ejecución local

## Requisitos operativos

- Python compatible con las versiones de las dependencias. El entorno observado usa Python 3.14.6, pero no existe una declaración formal del mínimo de Python.
- Ollama ejecutándose y accesible por HTTP.
- Descarga del modelo generativo seleccionado, por defecto `qwen2.5:3b`.
- Conexión a Internet potencialmente necesaria en la primera ejecución para descargar el modelo de embeddings.
- RAM suficiente para cargar Sentence Transformers y el modelo de Ollama; los requisitos exactos dependen de los modelos usados.

## Variables de entorno

El proyecto carga variables locales con `python-dotenv` desde `.env`. `.env` está ignorado por Git y `.env.example` contiene la plantilla versionable:

| Configuración | Ubicación | Valor predeterminado |
|---|---|---|
| URL de Ollama | `OLLAMA_URL` en `.env`; `rag_core.py` y campo lateral | `http://localhost:11434` |
| Modelo generativo | `OLLAMA_MODEL` en `.env`; `rag_core.py` y campo lateral | `qwen2.5:3b` |
| Modelo de embeddings | `EMBEDDING_MODEL` en `.env`; `rag_core.py` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Tamaño de fragmento | `rag_core.py` | 850 caracteres |
| Solapamiento | `rag_core.py` | 150 caracteres |
| Fragmentos recuperados | `app.py` | 3, configurable de 1 a 6 |
| Temperatura de Ollama | `rag_core.py` | 0.1 |
| Máximo de tokens generados | `rag_core.py` | 500 |

## Instalación y ejecución

Desde la raíz del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama serve
ollama pull qwen2.5:3b
streamlit run app.py
```

Notas:
- `ollama serve` puede no ser necesario si Ollama ya está iniciado como servicio local.
- El comando exacto para Windows `No verificado`; la activación mostrada es para macOS/Linux.
- No existe un script de arranque o Makefile.
- Streamlit suele abrir la aplicación en `http://localhost:8501`, pero no está fijado en código ni documentado.

## Comprobación rápida de Ollama

Desde la barra lateral se puede pulsar “Comprobar Ollama”. Esto llama a `GET /api/tags`. También se puede comprobar manualmente:

```bash
curl http://localhost:11434/api/tags
```

# Modelos de datos y APIs

## Datos de documentos y fragmentos

No existe esquema persistente ni ORM. Las estructuras principales son diccionarios Python.

**Página extraída**

```python
{"source": str, "page": int, "text": str}
```

**Fragmento**

```python
{"source": str, "page": int, "text": str}
```

**Resultado recuperado**

```python
{"source": str, "page": int, "text": str, "score": float}
```

## Estado de Streamlit

`st.session_state` conserva:

| Clave | Tipo lógico | Uso |
|---|---|---|
| `signatures` | lista de tuplas `(nombre, tamaño, file_id)` | Detecta si deben reconstruirse el índice y los fragmentos |
| `chunks` | lista de fragmentos | Contexto disponible para recuperación |
| `index` | `faiss.IndexFlatIP` | Índice vectorial en memoria |
| `history` | lista de mensajes | Historial mostrado en el chat |

El índice, los vectores, los fragmentos y el historial se pierden al terminar o reiniciar la sesión aplicable.

## API de Ollama

El proyecto consume directamente los endpoints locales:

| Método | Endpoint | Uso |
|---|---|---|
| `GET` | `/api/tags` | Comprueba disponibilidad y obtiene nombres de modelos |
| `POST` | `/api/chat` | Solicita una respuesta al modelo generativo |

Contrato aproximado de `POST /api/chat` usado por el proyecto:

```json
{
  "model": "modelo seleccionado",
  "messages": [
    {"role": "system", "content": "instrucciones"},
    {"role": "user", "content": "contexto y pregunta"}
  ],
  "stream": false,
  "options": {"temperature": 0.1, "num_predict": 500}
}
```

La aplicación lee `response.json()["message"]["content"]`. No implementa autenticación de Ollama ni maneja respuestas con esquema distinto al esperado.

No existe una API propia de Streamlit; la interfaz se sirve directamente en su servidor local.

# Pruebas y despliegue

## Pruebas

- No se encontraron archivos `test_*.py`, `*_test.py`, directorio `tests/` ni configuración de pytest.
- No existe cobertura configurada ni comando de pruebas definido.
- Validación mínima existente: `app.py` y `rag_core.py` deben compilarse; se recomienda ejecutar:

```bash
python -m compileall app.py rag_core.py
```

- Validación funcional manual: iniciar Ollama, descargar el modelo configurado, iniciar Streamlit, cargar al menos un PDF con texto seleccionable y formular una pregunta.
- Casos manuales recomendados: PDF vacío o escaneado, varios PDF, cambio de archivo, modelo inexistente, Ollama apagado, pregunta fuera del contexto y límite de fragmentos.

## Despliegue

No se encontraron Dockerfile, Docker Compose, pipeline CI/CD, manifiestos de despliegue ni documentación de release. El despliegue previsto por el código es local.

Entornos definidos: ninguno formal. La URL y el modelo son editables en la barra lateral, por lo que pueden apuntar a otra instancia de Ollama accesible.

Proceso de release `No verificado`.

# Convenciones y guías para agentes

## Qué hacer

- Mantener la separación entre UI (`app.py`) y lógica RAG (`rag_core.py`).
- Añadir funciones nuevas de recuperación o generación a `rag_core.py`, no mezclar lógica HTTP o FAISS con widgets.
- Usar rutas existentes y conservar compatibilidad con las estructuras de diccionario actuales.
- Validar cambios con `python -m compileall app.py rag_core.py`.
- Validar manualmente un PDF con texto seleccionable y Ollama activo antes de considerar completa una modificación funcional.
- Añadir pruebas unitarias si se modifica chunking, recuperación, construcción del índice o construcción del prompt.
- Conservar la salida en español y las citas de archivo/página en los cambios del prompt.
- Declarar y documentar explícitamente cualquier nueva variable de entorno o dependencia.

## Qué evitar

- No asumir que los PDF escaneados contienen texto: el código no ejecuta OCR.
- No persistir la sesión en disco a menos que el diseño se modifique deliberadamente.
- No cambiar silenciosamente los valores de chunking o el modelo de embeddings sin evaluar la calidad y compatibilidad de los índices.
- No añadir llamadas a servicios de pago o APIs externas al reemplazar Ollama; el diseño actual es local.
- No modificar `make_messages()` para permitir que instrucciones dentro de los documentos tengan prioridad sobre las reglas del sistema.
- No usar identificadores de índice fuera del rango devuelto por FAISS.
- No editar `.venv/`, `__pycache__/` ni artefactos generados como parte del proyecto fuente.
- No eliminar las comprobaciones de errores HTTP y disponibilidad de Ollama.

## Convenciones observadas

- Código Python directo, sin clases ni patrones de diseño complejos.
- Nombres de funciones en inglés y variables de uso compartido como `chunks`, `index`, `hits`, `question` y `embedder`.
- Mensajes visibles al usuario en español.
- Constantes de modelos y URL en mayúsculas en `rag_core.py`.
- Funciones pequeñas y con una responsabilidad principal.
- Error de negocio presentado mediante excepciones y mostrado por Streamlit, no mediante un framework de logging.
- Sin formateador, linter o configuración de estilo formal encontrados.

# Puntos de atención y deuda técnica

- **Sin OCR:** los PDF escaneados producen cero texto y `build_index()` falla con una indicación de OCR requerido.
- **Reconstrucción costosa:** cambiar los PDFs reconstruye embeddings e índice completos; no hay caché persistente ni indexación incremental.
- **Modelo descargado en la primera ejecución:** el inicio puede ser lento o fallar sin conexión o permisos de caché.
- **Estado acoplado a Streamlit:** la sesión y el índice viven en `st.session_state`; no son reutilizables desde una API o proceso backend.
- **Sin persistencia:** reiniciar Streamlit pierde documentos, fragmentos e índice.
- **Dependencia fuerte de Ollama:** `app.py` no puede generar respuestas si el servicio está apagado o el modelo no está descargado.
- **Cobertura RAG básica:** no hay reranking, búsqueda híbrida, metadatos avanzados, chunking semántico ni evaluación.
- **Prompt limitado por contexto:** el contexto se forma solo con los fragmentos seleccionados y puede truncar información relevante fuera de los mejores resultados.
- **Búsqueda no protegida ante índice vacío:** `build_index()` detecta ausencia de fragmentos, pero debe mantenerse la validación antes de recuperar.
- **Firma de archivos incompleta para detectar cambios:** usa nombre, tamaño y `file_id`; dos PDF del mismo nombre/tamaño podrían compartir firma según el comportamiento de `file_id`.
- **Historial inconsistente:** al reconstruir el índice se reinicia `history=[]`, por lo que el historial no persiste entre cambios de documentos.
- **Sin límite de archivos o bytes:** cargar muchos PDF grandes puede aumentar mucho memoria, tiempo de extracción y tamaño del índice.
- **Chunking de caracteres simple:** puede separar palabras o estructuras semánticas sin respetar párrafos, tablas o títulos.
- **Sin pruebas ni CI:** los cambios no cuentan con una validación automatizada del repositorio.
- **Documentación actualizada:** ahora existe `README.md` con instalación y ejecución; conservarlo sincronizado al cambiar dependencias o configuración.
- **Rango amplio de dependencias:** futuras instalaciones podrían recibir versiones distintas dentro de los rangos declarados.
- **Python mínimo no fijado:** solo se observó Python 3.14.6 dentro de `.venv/`; no está garantizado que sea el mínimo compatible.
- **Despliegue no verificado:** no existe evidencia de cómo se ejecuta fuera de un entorno local.

# Glosario de términos específicos del proyecto

| Término | Significado en este proyecto |
|---|---|
| RAG | Retrieval-Augmented Generation: recuperar fragmentos relevantes de documentos y enviarlos a un LLM como contexto. |
| Chunk / fragmento | Bloque de texto de hasta 850 caracteres tomado de una página y usado como unidad de búsqueda. |
| Embedding | Vector numérico que representa semánticamente un fragmento o una pregunta. |
| FAISS | Biblioteca local usada para almacenar y consultar vectores con similitud de producto interno. |
| IndexFlatIP | Índice plano de FAISS que calcula producto interno; equivale a coseno cuando los vectores están normalizados. |
| Ollama | Servidor local que expone modelos generativos mediante una API HTTP. |
| Hit / resultado recuperado | Fragmento encontrado por FAISS junto con su fuente, página y puntuación de similitud. |
| Firma de archivos | Identificador de cambios formado por nombre, tamaño y `file_id` de los PDF cargados. |
| Contexto recuperado | Texto formado por los mejores fragmentos enviados al modelo generativo. |
| PDF seleccionable | PDF cuyo texto puede extraerse directamente; excluye PDF que requieren OCR. |
| Streamlit session | Estado en memoria asociado a la sesión activa de la interfaz web. |