 MRI + SIGNAL-AND-SYSTEMS
 Simulador de Resonancia Magnética (MRI) - Libro Electrónico Interactivo
Descripción

Esta aplicación educativa está diseñada para estudiantes y profesionales de Ingeniería Biomédica que desean comprender los fundamentos de la Resonancia Magnética (MRI). Combina un libro electrónico interactivo con explicaciones visuales, un simulador biomédico en tiempo real para experimentar con parámetros de MRI, actividades y cuestionarios generados por IA (Gemini) para reforzar el aprendizaje, y un sistema de subida de tareas para entornos educativos.

La aplicación incluye una sección especializada de Control de Señales que permite diseñar y analizar controladores PID, visualizar funciones de transferencia, y evaluar la estabilidad de sistemas mediante diagramas de Bode, Root Locus y Nyquist, integrando conceptos de teoría de control con aplicaciones biomédicas.

 Características principales

- Libro Electrónico: Páginas educativas sobre Física de MRI, Relajación T1/T2, Contraste, Espacio-k y Secuencias de Pulso.
- Simulador Biomédico: Ajusta parámetros como TR, TE, Flip Angle, filtros y observa el efecto en señales y reconstrucción de imágenes.
- Actividades Interactivas: Genera preguntas de opción múltiple con IA (Gemini) y guarda los resultados.
- Sistema de Tareas: Permite a los estudiantes subir archivos de tareas (PDF, Word, imágenes) que se guardan localmente.
- Análisis de Control: Herramienta tipo MATLAB para diseño de controladores PID, análisis de funciones de transferencia, polos/ceros, diagramas de Bode, Root Locus y Nyquist.

Requisitos previos
- Python 3.9 o superior
- Pip (gestor de paquetes de Python)
- Conexión a Internet (para la generación de preguntas con Gemini)



 Instalación
 1. Crear un entorno virtual (recomendado)
En Windows:

python -m venv venv
venv\Scripts\activate

pip install dash plotly numpy scipy scikit-image google-generativeai sympy

GEMINI_API_KEY = "AQUI_PONER_API_KEY"
Nota: Si no tienes API key, la aplicación usará un banco de preguntas predefinido.

SimuMRI/
├── appSimuMRI.py                  # Código principal de la aplicación
├── assets/                        # Archivos estáticos (imágenes, videos)
│   ├── mri.png                    # Imagen del escáner
│   ├── espines.png                # Imagen de espines
│   ├── dLarmor.png                # Diagrama de Larmor
│   ├── contraste.png              # Comparativa de contrastes
│   ├── kspace.png                 # Trayectorias de espacio-k
│   ├── kspaceMRI.png              # Parámetros físicos desde espacio-k
│   ├── secuencia.png              # Diagrama de secuencia PRESS
│   └── MRI.mp4                    # Video explicativo de bobinas
├── tareas_subidas/                # Carpeta donde se guardan las tareas subidas
├── archive/                       # Dataset de MRI (descargar por separado)
├── resultados_estudiantes.csv     # Registro de resultados de actividades
└── tareas_estudiantes.txt         # Registro de tareas enviadas


Descarga del Dataset
El dataset de imágenes de MRI necesario para el simulador está disponible en:

https://drive.google.com/file/d/1zZU-pbUkpqfbOPCdbCfb_sC6Dm6GBebs/view?usp=sharing

Descargar y extraer el contenido en la carpeta archive/ dentro del proyecto.

Ejecución

python appSimuMRI.py
Luego abre tu navegador y ve a: http://127.0.0.1:8050

Compartir en línea (Cloudflare Tunnel)

cloudflared tunnel --url http://localhost:8050

Cómo usar la aplicación
Navegación
Inicio: Presentación general y acceso a todas las secciones.
Física de la MRI: Espín, campo B0, frecuencia de Larmor y precesión.
Señal y Relajación: Procesos T1 y T2, FID y curvas de relajación.
Contraste en MRI: Ponderaciones T1, T2 y Densidad Protónica (PD).
Espacio-k: Centro vs. Periferia, trayectorias de adquisición.
Secuencias de Pulso: Spin-Echo, PRESS y línea de tiempo.
Simulación: Ajusta parámetros en tiempo real y observa los cambios en las señales.
Actividades: Genera preguntas, responde y obtén tu calificación.
Control: Diseño de controladores PID y análisis de sistemas.

Simulación
En la pestaña "Simulación" puedes:
Seleccionar una patología y un paciente del dataset.
Ajustar parámetros como:
Ponderación (T1, T2, PD)
Filtrado del espacio-k (Centro, Periferia, Completo)
Factor de aceleración R (1, 2, 4)
Ruido térmico y filtros espaciales
Observar los resultados en tiempo real en los gráficos de pipeline.

Actividades
Ingresa tu nombre y correo.
Haz clic en "Generar Nuevas Preguntas".
Selecciona una opción para cada pregunta.
Haz clic en "Verificar Respuestas" para ver tu calificación.
Usa "Enviar Resultados" para guardar tu puntaje.

Control de Señales
En la pestaña "Control" puedes:
Escribir una función de transferencia en el campo de texto (ej: 1/(s^2 + 0.6*s + 1)).
Hacer clic en "Mostrar" para visualizar polos/ceros en el plano S.
Ajustar los parámetros PID (Kp, Ki, Kd) con los sliders.
Hacer clic en "Aplicar PID" para simular la respuesta del sistema controlado.
Usar los presets rápidos: P, PI, PID, Agresivo.
Explorar herramientas avanzadas: Bode, Root Locus, Nyquist.
Exportar datos de simulación a CSV.
Formato de Funciones de Transferencia
El campo de texto acepta expresiones matemáticas estándar. 
Ejemplos:
Sistema de 1er orden: 1/(s + 1)

Sistema de 2do orden: 1/(s^2 + 0.6*s + 1)

Sistema con cero: (s + 1)/(s^2 + 0.5*s + 1)

Sistema con retardo: exp(-0.5*s)/(s + 1)

Sistema factorizado: 1/((s + 1)*(s + 2))

Reglas básicas:
Usa s como variable de Laplace
Usa ** o ^ para potencias

Usa * para multiplicación (ej: 2*s en lugar de 2s)

Para retardos: exp(-T*s) o e**(-T*s)

Registros y almacenamiento
resultados_estudiantes.csv: Guarda nombre, correo, puntaje y fecha de cada actividad.

tareas_estudiantes.txt: Registro de tareas enviadas con descripción y metadatos.

tareas_subidas/: Carpeta con los archivos de tareas subidos.

Notas adicionales
La aplicación usa debug=False por defecto para evitar recargas innecesarias.

Los gráficos interactivos permiten zoom, pan y descarga de imágenes.

El dataset debe estar en la carpeta archive/ para que el simulador funcione correctamente.

La sección de Control requiere sympy para el parseo de funciones de transferencia (instalado automáticamente con las dependencias).

Solución de problemas
Error: No se puede importar sympy

pip install sympy

Error: API Key no configurada
La aplicación funcionará con preguntas predefinidas, pero se recomienda configurar la API Key.

El dataset no carga
Verifica que la carpeta archive/ contenga las subcarpetas con las imágenes de MRI.

La pestaña Control no muestra polos/ceros
Asegúrate de escribir una función de transferencia válida y hacer clic en "Mostrar".

Error al parsear función de transferencia
Revisa que la sintaxis sea correcta. Usa s como variable y * para multiplicación.
