# TFM: Implementación de un Sistema RAG para la Recuperación y Generación de Respuestas Basadas en Documentos Propios

Este repositorio contiene el código y los recursos utilizados en mi Trabajo de Fin de Máster (TFM) titulado **Implementación de un Sistema RAG para la Recuperación y Generación de Respuestas Basadas en Documentos Propios**, una técnica que combina modelos generativos con recuperación de información para mejorar la generación de texto basada en contexto, en este caso proporcionado archivos en formato PDF.

---

## 🛠 Requisitos previos

- Python **3.11.4**
- Cuenta activa en **Azure OpenAI** con acceso a modelos GPT
- Clave de API válida de Azure OpenAI (ver sección de configuración)

---

## 🚀 Instalación

Antes de ejecutar el proyecto, se recomienda crear un entorno virtual con Python 3.11.4:

```bash
python3.11.4 -m venv env
source env/bin/activate  # En Windows: env\Scripts\activate
```
Pasos para la instalación:
### 1. Clonar el repositorio

### 2. Instalar Poppler y Tesseract
- Poppler: se utiliza para extraer texto de archivos PDF.
  - En Windows:
     - Descarga los binarios desde: https://github.com/oschwartz10612/poppler-windows/releases/
      
     - Extrae el contenido en una carpeta (por ejemplo, C:\poppler)
      
     - Añade la ruta del directorio C:\poppler\Library\bin a tu variable de entorno PATH.
   
- Tesseract: se usa para realizar OCR (reconocimiento óptico de caracteres) sobre imágenes o PDFs escaneados.
  - En Windows:
     - Descarga el instalador desde: https://github.com/UB-Mannheim/tesseract/wiki
      
     - Instálalo normalmente (por ejemplo, en C:\Program Files\Tesseract-OCR)
      
     - Añade C:\Program Files\Tesseract-OCR a la variable de entorno PATH.

### 3. Instalar dependencias
Instala los paquetes necesarios definidos en el archivo requirements.txt:
```bash
pip install -r requirements.txt
```

### 4. Configurar credenciales de Azure OpenAI
Debes exportar las siguientes variables de entorno o definirlas en un archivo .env:
```bash
OPENAI_API_BASE= "Tu endpoint"
OPENAI_API_VERSION="2025-01-01-preview"
OPENAI_DEPLOYMENT_NAME="gpt-4o-mini"
API_KEY = "Tu API_KEY"
```
O crea un archivo .env en la raíz del proyecto con el contenido:

---

## 📄 Archivos principales
a) ```main.py```
En este script esta todo el trabajo, al ejercutarlo nos da una ruta para cargar la interfaz y poder utilizar todo el sistema. Haciendo control + click en ```http://localhost:7860``` se nos cargaría la página.
```bash
  chatbot_display = gr.Chatbot(
* Running on local URL:  http://localhost:7860
* To create a public link, set `share=True` in `launch()`.
```

b) ```gt.ipynb```
Este archivo es un notebook que contiene las preguntas y respuestas generadas para la evalaución del modelo.


---

## 📁 Estructura del Proyecto
```bash
tfm-rag-multimodal/
│
├── data/                      # Documentos fuente (entrada del sistema)
├── gt.ipynb                   # Evaluación del RAG en base a dos documentos
├── main.py                    # Ejecución del pipeline RAG
├── pdf_registry.json          # Contiene información sobre los PDF procesados
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Archivo actual
```
---

## 🧠 Descripción General
Este proyecto es una aplicación web interactiva basada en Gradio que permite a los usuarios subir, procesar y consultar archivos PDF mediante técnicas de RAG (Retrieval-Augmented Generation) multimodal. La solución combina texto, tablas e imágenes extraídas del PDF para generar respuestas enriquecidas a preguntas del usuario.

Funcionamiento:
### 1. Subida y procesamiento de PDFs
- El usuario sube un PDF con el que quiera interactuar.

- El sistema divide el contenido del PDF en fragmentos (chunks) de texto, tablas e imágenes.

- Resume cada tipo de contenido mediante llamadas a la API de Azure OpenAI.

- Guarda todos estos resúmenes y los datos originales en una colección persistente usando ChromaDB.

- Registra el nombre del PDF en un archivo local para futuras consultas.

### 2. Consulta interactiva estilo chatbot
- El usuario elige un PDF previamente procesado y formula una pregunta.

- El sistema recupera los fragmentos más relevantes desde la colección usando RAG.

- Genera una respuesta en lenguaje natural combinando texto, imágenes y tablas.

- Muestra en una galería visual las páginas o elementos del PDF que sirvieron como contexto.

- Mantiene el historial de la conversación en una interfaz tipo chat.

---

## Contacto
Para dudas o sugerencias, puedes contactarme: joseluiscn9@gmail.com
