import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pyttsx3
import speech_recognition as sr

# Configurar modelo de texto
modelo = SentenceTransformer('all-MiniLM-L6-v2')

# Base de conocimientos
base_preguntas = [
    # Física
    {"tema": "física", "pregunta": "¿Qué es la fuerza?", 
     "respuesta": "La fuerza es una magnitud que puede cambiar el estado de movimiento o deformar un cuerpo."},
    {"tema": "física", "pregunta": "¿Qué es la velocidad?", 
     "respuesta": "La velocidad es la relación entre el espacio recorrido y el tiempo que se tarda en recorrerlo."},
    {"tema": "física", "pregunta": "¿Qué dice la primera ley de Newton?", 
     "respuesta": "La primera ley dice que un cuerpo permanece en reposo o movimiento rectilíneo uniforme si no hay una fuerza externa."},
    {"tema": "física", "pregunta": "¿Qué es la energía cinética?", 
     "respuesta": "Es la energía que tiene un cuerpo debido a su movimiento."},
    {"tema": "física", "pregunta": "¿Qué diferencia hay entre masa y peso?", 
     "respuesta": "La masa es la cantidad de materia, y el peso es la fuerza con la que la gravedad actúa sobre esa masa."},

    # Inglés
    {"tema": "inglés", "pregunta": "¿Qué es un verbo en inglés?", 
     "respuesta": "Un verbo en inglés es una palabra que indica una acción, como 'run', 'eat' o 'study'."},
    {"tema": "inglés", "pregunta": "¿Qué es el presente simple?", 
     "respuesta": "Es un tiempo verbal que se usa para hablar de acciones habituales, como: 'She walks to school every day'."},
    {"tema": "inglés", "pregunta": "¿Cómo se forma una pregunta en inglés?", 
     "respuesta": "Usualmente se invierte el sujeto con el auxiliar. Ejemplo: 'Do you like apples?''"},
    {"tema": "inglés", "pregunta": "¿Cuál es la diferencia entre 'much' y 'many'?", 
     "respuesta": "'Much' se usa con incontables y 'many' con contables. Ej: 'much water', 'many books'."},
    {"tema": "inglés", "pregunta": "¿Qué son los adjetivos posesivos?", 
     "respuesta": "Son palabras como 'my', 'your', 'his', que indican posesión."},

    # Matemáticas
    {"tema": "matemáticas", "pregunta": "¿Qué es una fracción?", 
     "respuesta": "Una fracción representa una parte de un todo, se expresa como a/b."},
    {"tema": "matemáticas", "pregunta": "¿Qué es el mínimo común múltiplo?", 
     "respuesta": "Es el menor número que es múltiplo común de dos o más números."},
    {"tema": "matemáticas", "pregunta": "¿Cómo se resuelve una ecuación de primer grado?", 
     "respuesta": "Se despeja la incógnita aislándola en un lado de la ecuación."},
    {"tema": "matemáticas", "pregunta": "¿Qué es una raíz cuadrada?", 
     "respuesta": "Es el número que, multiplicado por sí mismo, da como resultado el número original."},
    {"tema": "matemáticas", "pregunta": "¿Qué es un número primo?", 
     "respuesta": "Es un número que solo es divisible entre 1 y él mismo."},

    # Lenguaje y Literatura
    {"tema": "lenguaje", "pregunta": "¿Qué es un sustantivo?", 
     "respuesta": "Un sustantivo es una palabra que nombra personas, animales, cosas o ideas."},
    {"tema": "lenguaje", "pregunta": "¿Qué es un adjetivo?", 
     "respuesta": "Es una palabra que describe o califica a un sustantivo."},
    {"tema": "lenguaje", "pregunta": "¿Qué es una metáfora?", 
     "respuesta": "Una metáfora es una figura literaria que compara dos elementos sin usar 'como'."},
    {"tema": "lenguaje", "pregunta": "¿Qué es un sinónimo?", 
     "respuesta": "Es una palabra que tiene un significado similar a otra."},
    {"tema": "literatura", "pregunta": "¿Qué es una novela?", 
     "respuesta": "Es una obra literaria extensa de ficción, con personajes y trama desarrollados."},
    {"tema": "literatura", "pregunta": "¿Qué es el narrador omnisciente?", 
     "respuesta": "Es un narrador que conoce todo lo que sienten y piensan los personajes."}
]

# Codificación de preguntas
preguntas_texto = [item["pregunta"] for item in base_preguntas]
preguntas_codificadas = modelo.encode(preguntas_texto)

# Función para hablar
def hablar_texto(texto):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    engine.say(texto)
    engine.runAndWait()

# Función para escuchar
def escuchar_microfono():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Escuchando... habla ahora.")
        try:
            audio = r.listen(source, timeout=5)
            texto = r.recognize_google(audio, language="es-ES")
            st.success(f"🗣️ Dijiste: {texto}")
            return texto
        except sr.UnknownValueError:
            st.error("No entendí lo que dijiste.")
        except sr.RequestError:
            st.error("Error de conexión con el servicio de voz.")
        except sr.WaitTimeoutError:
            st.warning("⏱️ No detecté ninguna voz.")
    return ""

# Interfaz Streamlit
st.title("🤖 Asistente Virtual para el Aula")
st.write("Haz una pregunta sobre física, inglés, matemáticas, lenguaje o literatura.")

modo = st.radio("Selecciona modo de entrada:", ["✍️ Escribir", "🎤 Hablar"])

pregunta_usuario = ""

if modo == "✍️ Escribir":
    pregunta_usuario = st.text_input("Escribe tu pregunta aquí:")
elif modo == "🎤 Hablar":
    if st.button("🎙️ Hablar ahora"):
        pregunta_usuario = escuchar_microfono()

if pregunta_usuario:
    pregunta_codificada = modelo.encode(pregunta_usuario)
    similitudes = util.cos_sim(pregunta_codificada, preguntas_codificadas)
    indice_mas_cercano = similitudes.argmax()
    mejor_respuesta = base_preguntas[indice_mas_cercano]["respuesta"]
    tema_detectado = base_preguntas[indice_mas_cercano]["tema"]

    st.markdown(f"### 📚 Tema detectado: `{tema_detectado.capitalize()}`")
    st.markdown("### ✅ Respuesta del asistente:")
    st.write(mejor_respuesta)

    if st.checkbox("🔊 Leer respuesta en voz alta"):
        hablar_texto(mejor_respuesta)
