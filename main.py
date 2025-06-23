# -*- coding: utf-8 -*-

import os
import json
import tempfile
import gradio as gr
import base64
import io
import atexit
from unstructured.partition.pdf import partition_pdf
from langchain_openai import AzureChatOpenAI
from langchain_core.caches import BaseCache
from langchain_core.callbacks.manager import Callbacks
AzureChatOpenAI.model_rebuild()
import warnings
warnings.filterwarnings("ignore", module="langchain")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from openai import AzureOpenAI
import uuid
from chromadb import PersistentClient
from pdf2image import convert_from_path
import PIL.Image

# --- Crucial Import Change ---
"""## Extract the data

Extract the elements of the PDF that we will be able to use in the retrieval process. These elements can be: Text, Images, Tables, etc.

### Partition PDF tables, text, and images
"""


# Función para dividir el PDF en elementos como texto, imágenes o tablas
def get_chunks(file_path):
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,            # Activamos la detección de tablas
        strategy="hi_res",                     # Estrategia que permite extraer mejor la estructura
        extract_image_block_types=["Image", "Table"],   # Incluimos tablas como imágenes
        extract_image_block_to_payload=True,   # Extraemos imágenes en base64
        chunking_strategy="by_title",          # Agrupamos según títulos del documento
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )
    return chunks


# Obtenemos las tablas en base64 de los elementos compuestos
def get_tables_base64(chunks):
    tables = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Table" in str(type(el)):
                    tables.append(el.metadata.image_base64)
    return tables


# Obtenemos las tablas como objetos directamente
def get_chunk_tables(chunks):
    chunk_tables = []
    for c in chunks:
        elements = c.metadata.orig_elements
        chunk_tables += [el for el in elements if 'Table' in str(type(el))]

    return chunk_tables


# Obtenemos las imágenes codificadas en base64
def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64


# Obtenemos los objetos de tipo imagen directamente
def get_chunk_images(chunks):
    chunk_images = []
    for c in chunks:
        elements = c.metadata.orig_elements
        chunk_images += [el for el in elements if 'Image' in str(type(el))]

    return chunk_images


# Función para resumir fragmentos de texto usando LLM
def summarize_texts(texts, subscription_key, endpoint, api_version):
    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additional comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}
    """

    prompt = ChatPromptTemplate.from_template(prompt_text)

    model = AzureChatOpenAI(
        deployment_name="gpt-4o-mini",
        openai_api_key=subscription_key,
        azure_endpoint=endpoint,
        openai_api_version=api_version,
        temperature=0.5,
    )

    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # Ejecutamos el resumen en paralelo para múltiples fragmentos(máx. 3 al mismo tiempo)
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})

    return text_summaries


# Resumimos tablas con imágenes codificadas
def summarize_tables_with_images(tables, subscription_key, endpoint, api_version):
    """
    tables: lista de strings con imágenes codificadas en base64 para las tablas
    """
    prompt_template = """Describe the image in detail. It is a table taken from a document, such as a scientific or legal paper.
    Be specific and precise in describing the structure and contents of the table."""

    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}},
            ],
        )
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    model = AzureChatOpenAI(
        deployment_name="gpt-4o-mini",
        openai_api_key=subscription_key,
        azure_endpoint=endpoint,
        openai_api_version=api_version,
        temperature=0.5,
    )

    chain = prompt | model | StrOutputParser()

    # tables es una lista de strings con la parte base64 de la imagen
    table_summaries = chain.batch(tables)

    return table_summaries


# Resumimos imágenes incrustadas en el documento
def summarize_images(images, subscription_key, endpoint, api_version):
    """
    images: lista de strings con imágenes codificadas en base64
    """
    prompt_template = """Describe the image in detail. For context,
    the image is taken from a document, such as a scientific or legal paper.
    Be specific about graphs, such as bar plots."""

    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}},
            ],
        )
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    model = AzureChatOpenAI(
        deployment_name="gpt-4o-mini",
        openai_api_key=subscription_key,
        azure_endpoint=endpoint,
        openai_api_version=api_version,
        temperature=0.5,
    )

    chain = prompt | model | StrOutputParser()

    image_summaries = chain.batch(images)

    return image_summaries




# Obtenemos el embedding de un texto dado
def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(
        input=text,
        model=model,
    )
    return response.data[0].embedding



# Función para insertar documentos en Chroma

def add_document_to_chroma(summary, full_doc): 
    if isinstance(full_doc, list):
        full_text = "\n".join(str(el) for el in full_doc)
    else:
        full_text = str(full_doc)

    doc_id = str(uuid.uuid4())
    emb = get_embedding(summary)

    pagina_inicio = full_doc.metadata.orig_elements[0].to_dict()['metadata']['page_number']
    pagina_fin = full_doc.metadata.orig_elements[len(full_doc.metadata.orig_elements)-1].to_dict()['metadata']['page_number']

    collection.add(
        ids=[doc_id],
        embeddings=[emb],
        metadatas=[{"original": full_text, "pagina_inicio":pagina_inicio, "pagina_fin":pagina_fin}],
        documents=[summary]
    )


# Añadimos una imagen o tabla como documento en Chroma
def add_image_to_chroma(summary, full_doc):

    doc_id = str(uuid.uuid4())
    emb = get_embedding(summary)

    pagina_inicio = full_doc.to_dict()['metadata']['page_number']
    pagina_fin = full_doc.to_dict()['metadata']['page_number']

    if full_doc.to_dict()['type'] == 'Table':
        full_doc.metadata.image_base64 = "Table " + full_doc.metadata.image_base64

    collection.add(
        ids=[doc_id],
        embeddings=[emb],
        metadatas=[{"original": full_doc.to_dict()['metadata']['image_base64'], "pagina_inicio":pagina_inicio, "pagina_fin":pagina_fin}],
        documents=[summary]
    )


# Recuperamos los documentos más relevantes a partir de una consulta
def retrieve(query, top_k=3):
    q_emb = get_embedding(query)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["metadatas", "documents"] 
    )

    # Devolvemos los documentos originales
    return [[m.get("original"), m.get("pagina_inicio", None), m.get("pagina_fin", None)] for m in results["metadatas"][0]]



# Clasificamos los documentos recuperados por tipo: imágenes, tablas o texto.
# Aqui los docs es una Lista de Lista con [[documento, pagina_inicio, pagina_fin], ..., [documento, pagina_inicio, pagina_fin]]
def parse_docs(docs):
    images = []
    texts = []
    tables=[]
    for doc in docs:
        if doc[0][:5] == 'Table':
            tables.append(doc)

        elif doc[0][0] == '/':
            images.append(doc)

        else:
            texts.append(doc)

    for t in tables:
        t[0] = t[0][6:]

    return {"images": images, "texts": texts, "tables": tables}


# Construimos el prompt que se enviará al modelo
def build_prompt(docs_by_type, user_question):
    context_text = "\n".join(
    text.text if hasattr(text, "text") else str(text)
    for text in docs_by_type["texts"]
) if docs_by_type["texts"] else ""
   
    prompt_text = f"""You are a helpful assistant. Please answer the user's question using the context below as your main source of information.
    Context: {context_text}

    Question: {user_question}
    When you answer, please mention the page numbers where the information was found and the name of the section, if available.
    """

    prompt_content = [{"type": "text", "text": prompt_text}]

    for table in docs_by_type["tables"]:
        prompt_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{table}"},
        })

    for image in docs_by_type["images"]:
        prompt_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        })

    return prompt_content


# Ejecutamos la consulta contra el modelo con los documentos recuperados
def invoke_chain_with_retrieval(query_text, top_k=3):
    # Primero hacemos el retrieve automáticamente
    retrieved_docs = retrieve(query_text, top_k=top_k)

    # Parseamos los docs recuperados
    docs_by_type = parse_docs(retrieved_docs)

    docs_by_type_procesado = docs_by_type.copy()

    # Nos quedamos solo con la parte de textos y de imagenes (ya que tenemos [[texto/imagen, pagina_inicio, pagina_fin], ..., [...]])
    docs_by_type_procesado["texts"] = [sublist[0] for sublist in docs_by_type["texts"]]
    docs_by_type_procesado["images"] = [sublist[0] for sublist in docs_by_type["images"]]
    docs_by_type_procesado["tables"] = [sublist[0] for sublist in docs_by_type["tables"]]

    # Construimos el prompt
    prompt_content = build_prompt(docs_by_type_procesado, query_text)

    # Ejecutamos la llamada a la API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_content},
        ],
    )

    return response.choices[0].message.content


def invoke_chain_with_sources_and_retrieval(query_text, top_k=3):
    global chat_history  # Para usar el historial en la construcción del prompt

    # Hacemos el retrieve automáticamente
    retrieved_docs = retrieve(query_text, top_k=top_k)

    # Parseamos los docs recuperados
    docs_by_type = parse_docs(retrieved_docs)
    docs_by_type_procesado = docs_by_type.copy()

    docs_by_type_procesado["texts"] = [sublist[0] for sublist in docs_by_type["texts"]]
    docs_by_type_procesado["images"] = [sublist[0] for sublist in docs_by_type["images"]]
    docs_by_type_procesado["tables"] = [sublist[0] for sublist in docs_by_type["tables"]]

    # Construimos el prompt con contexto relevante
    prompt_content = build_prompt(docs_by_type_procesado, query_text)

    # Construimos la lista de mensajes incluyendo el historial anterior
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    max_turns = 5
    for role, msg in chat_history[-max_turns*2:]:  # Cada turno son 2 mensajes (usuario, asistente)
        if role.lower() == "usuario":
            messages.append({"role": "user", "content": msg})
        elif role.lower() == "asistente":
            messages.append({"role": "assistant", "content": msg})

    # Agregamos el nuevo turno con el prompt construido a partir del contexto
    messages.append({"role": "user", "content": prompt_content})

    # Llamada al modelo
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    return {
        "response": response.choices[0].message.content,
        "context": docs_by_type
    }



# Configuración inicial
REGISTRY_PATH = "pdf_registry.json"
chat_history = []

client = None
chroma = None
collection = None

# Funciones para manejar el registro de PDFs y colecciones
def load_registry():
    if not os.path.exists(REGISTRY_PATH):
        return []
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry):
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def add_to_registry(pdf_name, collection_name):
    registry = load_registry()
    exists = any(r['pdf_name'] == pdf_name for r in registry)
    if not exists:
        registry.append({
            "pdf_name": pdf_name,
            "collection_name": collection_name,
        })
        save_registry(registry)


def list_registered_pdfs():
    registry = load_registry()
    return [r['pdf_name'] for r in registry]


def get_collection_name_by_pdf(pdf_name):
    registry = load_registry()
    for r in registry:
        if r['pdf_name'] == pdf_name:
            return r['collection_name']
    return None



# Inicializamos el cliente de Azure OpenAI con las variables de entorno
def initialize_azure_client():
    global client
    api_version = os.environ.get("OPENAI_API_VERSION")
    endpoint = os.environ.get("OPENAI_API_BASE")
    subscription_key = os.environ.get("API_KEY")

    if not (api_version and endpoint and subscription_key):
        raise RuntimeError("Faltan variables de entorno necesarias.")

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

# Función para procesar los PDF y guardar la colección
def process_pdf_with_registry(pdf_file, pdf_name):
    global chroma, collection

    if not pdf_file:
        return "Error: Debes subir un archivo PDF."

    # Obtenemos la ruta al PDF tal como lo da Gradio:
    if hasattr(pdf_file, "name"):
        tmp_path = pdf_file.name           # Cuando pdf_file es un objeto con .name
    elif isinstance(pdf_file, dict) and "name" in pdf_file:
        tmp_path = pdf_file["name"]        # Si Gradio devuelve un dict {'name': ..., 'data': ...}
    else:
        return "Error inesperado: no pude determinar la ruta del PDF."

    try:
        initialize_azure_client()
    except Exception as e:
        return f"Error al inicializar cliente Azure/OpenAI: {e}"

    # Dividimos el PDF en fragmentos de texto, tablas e imágenes
    chunks = get_chunks(tmp_path)
    tables, texts = [], []
    for chunk in chunks:
        if "Table" in str(type(chunk)): tables.append(chunk)
        if "CompositeElement" in str(type(chunk)): texts.append(chunk)

    # Convertimos las tablas e imágenes a formato base64 y las asociamos a los fragmentos
    tables_base64 = get_tables_base64(chunks)
    chunk_tables = get_chunk_tables(chunks)
    images_base64 = get_images_base64(chunks)
    chunk_images = get_chunk_images(chunks)

    api_version = os.environ["OPENAI_API_VERSION"]
    endpoint = os.environ["OPENAI_API_BASE"]
    subscription_key = os.environ["API_KEY"]

    # Generamos resúmenes para cada tipo de contenido
    text_summaries = summarize_texts(texts, subscription_key, endpoint, api_version)
    print("\nResumen de textos realizado con éxito\n")
    table_summaries = summarize_tables_with_images(tables, subscription_key, endpoint, api_version)
    print("\nResumen de tablas realizado con éxito\n")
    image_summaries = summarize_images(images_base64, subscription_key, endpoint, api_version)
    print("\nResumen de imagenes realizado con éxito\n")

    # Creamos una colección nueva o reutilizamos una existente con el nombre del PDF
    collection_name = f"collection_{pdf_name.strip().replace(' ', '_')}"
    chroma = PersistentClient(path="./chroma_data")
    collection = chroma.get_or_create_collection(name=collection_name)
   
    print("\nColeccion de Chroma creada con exito\n")

    # Guardamos todos los resúmenes junto con el contenido original en la colección
    for summary, full in zip(text_summaries, texts):
        add_document_to_chroma(summary, full)
    for summary, full in zip(table_summaries, chunk_tables):
        add_image_to_chroma(summary, full)
    for summary, full in zip(image_summaries, chunk_images):
        add_image_to_chroma(summary, full)

    add_to_registry(pdf_name.strip(), collection_name)
    return f"PDF procesado y guardado en colección: {collection_name}"

# Para cargar una colección existente
def load_collection_by_name(pdf_name):
    global chroma, collection
    try:
        initialize_azure_client()
    except Exception as e:
        return f"Error al inicializar cliente Azure/OpenAI: {e}"

    collection_name = get_collection_name_by_pdf(pdf_name)
    if not collection_name:
        return f"No se encontró colección para PDF '{pdf_name}'."

    chroma = PersistentClient(path="./chroma_data")
    collection = chroma.get_or_create_collection(name=collection_name)
    return f"Colección '{collection_name}' cargada para consultas."


TEMP_IMAGE_PATHS = []  # Lista global de imágenes temporales

# Guardamos una imagen temporal a partir de base64
def save_temp_image(base64_img_data_uri_or_raw_b64):
    base64_data = base64_img_data_uri_or_raw_b64
    img_format = "png"  

    # Si es un data URI, extraemos el formato y los datos base64
    if isinstance(base64_img_data_uri_or_raw_b64, str) and base64_img_data_uri_or_raw_b64.startswith("data:image"):
        try:
            header, base64_data = base64_img_data_uri_or_raw_b64.split(",", 1)
            img_format = header.split('/')[1].split(';')[0]
            if not img_format: img_format = "png"
        except ValueError:
            print(f"Warning: Malformed data URI, attempting to decode as raw base64.")
            pass 

    try:
        img_bytes = base64.b64decode(base64_data)
        pil_image_object = PIL.Image.open(io.BytesIO(img_bytes))
        
        temp_dir = tempfile.gettempdir()
        os.makedirs(temp_dir, exist_ok=True)

        # Convertimos imágenes con paleta a RGB si vamos a guardarlas como JPEG
        if img_format.lower() == "jpg":
            img_format = "jpeg"
        
        temp_filename = os.path.join(temp_dir, f"{uuid.uuid4().hex}.{img_format}")
        
        if pil_image_object.mode == 'P' and img_format.lower() == 'jpeg':
            pil_image_object = pil_image_object.convert('RGB')
            
        pil_image_object.save(temp_filename, format=img_format.upper())
        TEMP_IMAGE_PATHS.append(temp_filename)
        return temp_filename
    
    except base64.binascii.Error as b64_err:
        print(f"Error decodificando base64: {b64_err}")
        print(f"Problematic base64 preview: {str(base64_data)[:100]}...")
        return None
    
    except PIL.UnidentifiedImageError:
        print(f"Error: PIL no pudo identificar el formato de la imagen desde los bytes decodificados.")
        print(f"Problematic base64 preview: {str(base64_data)[:100]}...")
        return None
    
    except Exception as e:
        print(f"Error general al guardar imagen temporal: {e} (type: {type(e)})")
        print(f"Problematic base64 preview: {str(base64_img_data_uri_or_raw_b64)[:100]}...")
        return None


# Función para convertir páginas PDF a imágenes base64 (data URI strings)
def get_pdf_pages_as_base64(file_path, first_page, last_page):
    images_b64_data_uris = []
    if not os.path.exists(file_path):
        print(f"Error: PDF file not found at {file_path}")
        return []
    try:
        pages = convert_from_path(file_path, dpi=200, first_page=first_page, last_page=last_page, fmt="png", poppler_path=None) # Set poppler_path if needed
        for page_pil_image in pages:
            buffered = io.BytesIO()
            page_pil_image.save(buffered, format="PNG")
            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
            images_b64_data_uris.append("data:image/png;base64," + img_b64_str)
    except Exception as e:
        print(f"Error convirtiendo paginas PDF a base64 ({file_path}, paginas {first_page}-{last_page}): {e}")
    return images_b64_data_uris

# Manejamos una consulta tipo chat usando RAG multimodal
def chat_with_rag(message: str, selected_pdf_name: str):
    global chat_history, collection 

    output_chat_history = list(chat_history) 
    error_message_str = ""
    output_gallery_images = []

    if not selected_pdf_name:
        error_message_str = "Selecciona un PDF para consultar."
        return output_chat_history, error_message_str, output_gallery_images

    # Construimos la ruta al archivo PDF local
    pdf_file_for_page_extraction = os.path.join("data", selected_pdf_name + ".pdf")
    
    if not os.path.isfile(pdf_file_for_page_extraction):
        print(f"Advertencia: Archivo PDF '{pdf_file_for_page_extraction}' no encontrado para extracción de páginas.")

    # Nos aseguramos de que la colección correcta esté cargada
    current_collection_name_for_pdf = get_collection_name_by_pdf(selected_pdf_name)
    if not collection or not current_collection_name_for_pdf or collection.name != current_collection_name_for_pdf:
        status = load_collection_by_name(selected_pdf_name) # This sets global `collection`
        if "Error" in status or not collection:
            error_message_str = f"Fallo al cargar la colección para '{selected_pdf_name}': {status}"
            return output_chat_history, error_message_str, output_gallery_images
        print(f"Colección '{collection.name}' cargada para '{selected_pdf_name}'.")


    if not message or not message.strip():
        error_message_str = "Por favor, escribe una pregunta válida."
        return output_chat_history, error_message_str, output_gallery_images

    # Ejecutamos la consulta con recuperación de contexto
    try:
        response_sources = invoke_chain_with_sources_and_retrieval(message)
    except Exception as e:
        error_message_str = f"Error al consultar: {e}"
        output_chat_history.append(("System", error_message_str)) # Add error to chat
        return output_chat_history, error_message_str, output_gallery_images

    response_text = f"Respuesta:\n{response_sources['response']}" # \n\nContexto:\n

    # Extraemos las páginas del PDF correspondientes al contexto de texto
    if response_sources['context'].get('texts'):
        for text_content, start_page, end_page in response_sources['context']['texts']:
            if os.path.isfile(pdf_file_for_page_extraction):
                page_data_uris = get_pdf_pages_as_base64(pdf_file_for_page_extraction, start_page, end_page)
                for data_uri in page_data_uris:
                    temp_path = save_temp_image(data_uri)
                    if temp_path:
                        output_gallery_images.append(temp_path)
            else:
                print(f"Skipping PDF page image extraction for text: {pdf_file_for_page_extraction} not found.")


    # Extraemos imágenes asociadas al contexto
    if response_sources['context'].get('images'):
        for img_b64_content, start_page, end_page in response_sources['context']['images']:
            temp_img_path = save_temp_image(img_b64_content)
            if temp_img_path:
                output_gallery_images.append(temp_img_path)
            else:
                print(f"Failed to save image from context for gallery.")

    # Procesamos las tablas si vienen en formato de imagen
    if response_sources['context'].get('tables'):
        for table_content, start_page, end_page in response_sources['context']['tables']:
            temp_img_path = save_temp_image(table_content)
            if temp_img_path:
                output_gallery_images.append(temp_img_path)
            else:
                print(f"Failed to save table from context for gallery.")

    output_chat_history.append(("Usuario", message))
    output_chat_history.append(("Asistente", response_text))
    chat_history.clear()
    chat_history.extend(output_chat_history)

    return output_chat_history, error_message_str, output_gallery_images


# Limpiamos el historial del chat
def reset_chat():
    global chat_history
    chat_history.clear()
    return [], "", []


# Para actualizar la lista de PDFs registrados 
def update_pdf_list_dropdown():
    return gr.update(choices=list_registered_pdfs())


# Gradio UI
initial_pdfs = list_registered_pdfs()


theme = gr.themes.Default(
    primary_hue=gr.themes.Color(c100="#dbeafe", c200="#bfdbfe", c300="#93c5fd", c400="#60a5fa", c50="#eff6ff", c500="#3b82f6", c600="rgba(0, 56.74232954545456, 181.20937500000002, 1)", c700="#1d4ed8", c800="#1e40af", c900="#1e3a8a", c950="#1d3660"),
    neutral_hue="slate",
).set(
    body_background_fill='*primary_50',
    body_text_color='*secondary_900',
    body_text_color_dark='*primary_50',
    link_text_color='*neutral_600',
    link_text_color_active='*neutral_500',
    block_label_background_fill='*primary_100',
    block_label_shadow='*shadow_drop',
    block_label_text_color='*neutral_600',
    chatbot_text_size='*text_md',
    checkbox_background_color_selected='*neutral_700',
    checkbox_label_text_color='*checkbox_background_color',
    input_background_fill='*neutral_50',
    input_background_fill_hover_dark='*body_background_fill',
    input_border_color_dark='*border_color_accent_subdued',
    input_border_color_focus='*secondary_400',
    button_primary_background_fill='*primary_600',
    button_secondary_background_fill='*secondary_200'
)

with gr.Blocks(theme=theme) as demo:

    gr.Markdown("# 💡 Gestor RAG Multimodal de PDFs")

    with gr.Tab("Subir y procesar PDF"):
        pdf_input_file = gr.File(file_types=[".pdf"], label="Sube un archivo PDF")
        pdf_name_input = gr.Textbox(label="Nombre para el PDF (sin extensión .pdf)", placeholder="Ej: Reporte_2025")
        process_button = gr.Button("Procesar PDF")
        process_output_text = gr.Textbox(label="Estado Procesamiento", interactive=False, lines=3)

    with gr.Tab("Consultar PDF procesado"):
        # Sección Superior: Selección de PDF y Estado
        with gr.Row():
            pdf_list_dropdown = gr.Dropdown(
                choices=initial_pdfs, 
                label="Selecciona un PDF cargado", 
                allow_custom_value=False, 
                scale=3
            )
            load_collection_button = gr.Button("Cargar colección", scale=1)
        
        load_collection_output_text = gr.Textbox(
            label="Estado carga colección", 
            interactive=False,
            lines=1
        )

        gr.Markdown("---") # Separador visual

        # Sección Principal: Dos Columnas (Chat y Galería)
        with gr.Row(equal_height=False): # Las columnas pueden tener alturas ligeramente diferentes si es necesario

            # Columna Izquierda: Interfaz de Chat
            with gr.Column(scale=1): # Ocupa la mitad del espacio disponible en la fila
                chatbot_display = gr.Chatbot(
                    label="Conversación", 
                    height=450 
                )
                question_input_text = gr.Textbox(
                    label="Tu pregunta", 
                    placeholder="Escribe aquí tu pregunta", 
                    lines=2
                )
                with gr.Row():
                    ask_button = gr.Button("Enviar", variant="primary", scale=3)
                    clear_button = gr.Button("Limpiar chat", scale=1)
                
                error_output_box = gr.Textbox(
                    label="ERROR",
                    interactive=False, 
                    visible=False,
                    lines=2
                )

            # Columna Derecha: Galería de Imágenes
            with gr.Column(scale=1): # Ocupa la otra mitad del espacio
                pdf_pages_gallery_display = gr.Gallery(
                    label="Páginas del contexto / Imágenes Relevantes", 
                    elem_id="image-gallery", 
                    height=630,
                    columns=4,
                    object_fit="contain", 
                    show_label=True 
                )

        def process_and_update_dropdown(pdf_file_obj, pdf_name_val):
            status = process_pdf_with_registry(pdf_file_obj, pdf_name_val) 
            return status, gr.update(choices=list_registered_pdfs())

        process_button.click(
            fn=process_and_update_dropdown,
            inputs=[pdf_input_file, pdf_name_input],
            outputs=[process_output_text, pdf_list_dropdown]
        )

        load_collection_button.click(
            fn=load_collection_by_name,
            inputs=[pdf_list_dropdown], 
            outputs=[load_collection_output_text]
        )
        
        ask_button.click(
            fn=chat_with_rag,
            inputs=[question_input_text, pdf_list_dropdown],
            outputs=[chatbot_display, error_output_box, pdf_pages_gallery_display]
        ).then(lambda: gr.update(value=""), outputs=[question_input_text])

        clear_button.click(
            fn=reset_chat,
            outputs=[chatbot_display, error_output_box, pdf_pages_gallery_display]
        )

    demo.load(update_pdf_list_dropdown, outputs=pdf_list_dropdown)

@atexit.register
def cleanup_temp_images():
    print(f"Limpiando {len(TEMP_IMAGE_PATHS)} imágenes temporales...")
    cleaned_count = 0
    for path in TEMP_IMAGE_PATHS:
        try:
            if os.path.exists(path):
                os.remove(path)
                cleaned_count += 1
        except Exception as e:
            print(f"No se pudo borrar imagen temporal {path}: {e}")
    if cleaned_count > 0:
        print(f"Se limpiaron {cleaned_count} imágenes temporales.")
    TEMP_IMAGE_PATHS.clear()


if __name__ == "__main__":
    if not os.path.exists(REGISTRY_PATH):
        save_registry([])

    if not os.path.exists("./chroma_data") and PersistentClient.__name__ != "MockPersistentClient":
        os.makedirs("./chroma_data", exist_ok=True)

    demo.launch(debug=True)







    
