# RESONANCIA MAGNÉTICA MRI
# Libro electrónico interactivo + Simulador biomédico integrado

import os
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal, ndimage  #Filtros y procesamientos de señales
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize

import dash
from dash import dcc, html, Input, Output, State, ALL, callback_context, no_update
from dash.exceptions import PreventUpdate

#Agente
import google.generativeai as genai
import datetime
import random
import base64

import re
import io
import sympy as sp

_SYMPY_OK = True
try:
    import sympy as sp
except ImportError:
    _SYMPY_OK = False

GEMINI_API_KEY = "Aqui ponen el APY KEY de su cuenta"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# ---- 1. ESCANEO Y GESTIÓN DEL DATASET 

def index_kaggle_dataset(base_dir=".", max_samples_per_class=150):
    valid_exts = ('.jpg', '.jpeg', '.png', '.tif')
    categories = {
        'Glioma': [],
        'Meningioma': [],
        'Pituitary (Hipófisis)': [],
        'No Tumor (Sano)': [],
        'Otros': []
    }

    for root, _, files in os.walk(base_dir):
        if 'venv' in root or '.git' in root:
            continue
        for file in files:
            if file.lower().endswith(valid_exts):
                full_path = os.path.join(root, file)
                path_lower = full_path.lower()

                if 'glioma' in path_lower and len(categories['Glioma']) < max_samples_per_class:
                    categories['Glioma'].append(full_path)
                elif 'meningioma' in path_lower and len(categories['Meningioma']) < max_samples_per_class:
                    categories['Meningioma'].append(full_path)
                elif 'pituitary' in path_lower and len(categories['Pituitary (Hipófisis)']) < max_samples_per_class:
                    categories['Pituitary (Hipófisis)'].append(full_path)
                elif ('no_tumor' in path_lower or 'notumor' in path_lower or 'healthy' in path_lower) and len(categories['No Tumor (Sano)']) < max_samples_per_class:
                    categories['No Tumor (Sano)'].append(full_path)
                elif len(categories['Otros']) < max_samples_per_class:
                    categories['Otros'].append(full_path)

    for cat in categories:
        categories[cat].sort()

    available_categories = {k: v for k, v in categories.items() if len(v) > 0}
    if not available_categories:
        available_categories['Sintético Demo'] = ['synthetic']

    return available_categories

DATASET_INDEX = index_kaggle_dataset(base_dir=".", max_samples_per_class=150)

def load_mri_image(filepath, target_size=(160, 160)):
    if filepath == 'synthetic' or not os.path.exists(filepath):
        y, x = np.ogrid[:target_size[0], :target_size[1]]
        cy, cx = target_size[0] / 2, target_size[1] / 2
        brain_mask = ((x - cx)**2 / (60**2) + (y - cy)**2 / (70**2)) <= 1.0
        tumor_mask = ((x - (cx + 18))**2 / (16**2) + (y - (cy - 14))**2 / (14**2)) <= 1.0
        img = np.zeros(target_size, dtype=float)
        img[brain_mask] = 0.58
        img[tumor_mask] = 0.95
        return ndimage.gaussian_filter(img, sigma=0.8)

    try:
        raw_img = imread(filepath)
        if raw_img.ndim == 3:
            gray_img = rgb2gray(raw_img)
        else:
            gray_img = raw_img.astype(float)

        img_resized = resize(gray_img, target_size, anti_aliasing=True)
        if np.max(img_resized) > 0:
            img_resized = img_resized / np.max(img_resized)
        return img_resized
    except Exception:
        return np.zeros(target_size)


# ---- 2. MODELADO FÍSICO MRI ----

def simulate_mri_tissue_contrast(base_image, weight_type, tr, te):
    img = base_image.copy()
    mask_bg = (img < 0.04)

    pd_map = img.copy()
    t1_map = 350.0 + img * 1900.0
    t2_map = 35.0 + img * 220.0

    #Ec señal Spin-Echo
    signal_map = pd_map * (1.0 - np.exp(-tr / t1_map)) * np.exp(-te / t2_map)
    signal_map[mask_bg] = 0.0

    if np.max(signal_map) > 0:
        signal_map = signal_map / np.max(signal_map)
    return signal_map

def simulate_kspace_pipeline(image, r_factor=1, noise_level=0.0, k_filter='all'):
    kspace = np.fft.fftshift(np.fft.fft2(image)) #transf Fourier
    h, w = kspace.shape
    cy, cx = h // 2, w // 2

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - cx)**2 / (cy**2) + (y - cy)**2 / (cx**2))
    cutoff_rad = 0.18

    #filtro de frecuencias
    if k_filter == 'center_only':
        #frecuencias altas
        mask_freq = (dist_from_center <= cutoff_rad).astype(float)
        kspace = kspace * mask_freq
    elif k_filter == 'periphery_only':
        #frecuencias altas
        mask_freq = (dist_from_center > cutoff_rad).astype(float)
        kspace = kspace * mask_freq

    #submuestreo
    if r_factor > 1:
        mask_alias = np.zeros_like(kspace)
        mask_alias[::r_factor, :] = 1.0
        center_band = slice(cy - 6, cy + 6)
        mask_alias[center_band, :] = 1.0
        kspace = kspace * mask_alias

    #ruido termino
    if noise_level > 0:
        sigma = noise_level * np.max(np.abs(kspace)) * 0.035
        noise = np.random.normal(0, sigma, kspace.shape) + 1j * np.random.normal(0, sigma, kspace.shape)
        kspace = kspace + noise

    #reconstrucción
    recon = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace)))
    if np.max(recon) > 0:
        recon = recon / np.max(recon)
    return kspace, recon

#ajuste de contraste
def apply_window_level(img, window, level):
    img_min = level - window / 2.0
    img_max = level + window / 2.0
    clipped = np.clip(img, img_min, img_max)
    if (img_max - img_min) > 0:
        return (clipped - img_min) / (img_max - img_min)
    return clipped

def calculate_snr(image):
    h, w = image.shape
    signal_roi = image[h//3:2*h//3, w//3:2*w//3]
    noise_roi = image[:h//8, :w//8]
    mean_sig = np.mean(signal_roi)
    std_noise = np.std(noise_roi)
    if std_noise == 0:
        return 999.0
    return float(mean_sig / std_noise)

def generate_rf_pulse(pulse_type, duration_ms, flip_angle_deg, points=500):
    t = np.linspace(-duration_ms / 2, duration_ms / 2, points)
    if pulse_type == 'sinc':
        rf = np.sinc(t / (duration_ms / 4))
    elif pulse_type == 'gauss':
        sigma = duration_ms / 6
        rf = np.exp(-t**2 / (2 * sigma**2))
    elif pulse_type == 'rect':
        rf = np.ones_like(t)
    else:
        rf = np.zeros_like(t)

    integral = np.sum(rf) * (duration_ms / points)
    if integral != 0:
        #area bajo la curva - flip angle
        rf = rf * (np.radians(flip_angle_deg) / integral)
    return t, rf

def simulate_fid(t_max_ms, t1_ms, t2_ms, f0_hz=50.0, points=1000):
    #decaimiento exponencial y oscilación
    t = np.linspace(0, t_max_ms, points)
    s_FID = np.exp(-t / t2_ms) * np.exp(1j * 2 * np.pi * f0_hz * (t / 1000.0))
    return t, s_FID

#filtros de procesamiento de señal
def apply_filter(sig, fs, filter_type, cutoff_hz):
    nyquist = 0.5 * fs
    norm_cutoff = min(cutoff_hz / nyquist, 0.99)
    if norm_cutoff <= 0:
        norm_cutoff = 0.01
    b, a = signal.butter(4, norm_cutoff, btype=filter_type)
    return signal.filtfilt(b, a, sig)

def generate_circle_phantom_kspace(size=64, radius=16):
    y, x = np.ogrid[:size, :size]
    center = size // 2
    mask = (x - center)**2 + (y - center)**2 <= radius**2
    img = np.zeros((size, size))
    img[mask] = 1.0
    kspace = np.fft.fftshift(np.fft.fft2(img))
    recon = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace)))
    return img, kspace, recon

# 3. ESTILOS Y CSS OSCURO PERSONALIZADO

DARK_CSS = """
.dash-dropdown,
.dash-dropdown .Select-control,
.Select-control,
div[class*="control"] {
    background-color: #1e293b !important;
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}

.dash-dropdown .Select-value-label,
.Select-value-label,
.Select-placeholder,
div[class*="singleValue"],
div[class*="placeholder"],
div[class*="ValueContainer"] * {
    color: #ffffff !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

.Select-menu-outer,
.Select-menu,
.Select-menu-outer > div,
.Select-menu > div,
div[class*="menu"],
div[class*="MenuList"] {
    background-color: #1e293b !important;
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.7) !important;
    z-index: 99999 !important;
}

.Select-input,
.Select-input > input,
div[class*="Input"],
div[class*="Input"] input,
div[class*="search"],
div[class*="search"] input {
    background-color: #0f172a !important;
    background: #0f172a !important;
    color: #ffffff !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
}

.VirtualizedSelectOption,
.VirtualizedSelectOption *,
.Select-option,
.Select-option *,
div[class*="option"],
div[class*="option"] *,
.Select-menu-outer div:not(.is-selected):not(.is-focused):not([class*="-is-selected"]):not([class*="-is-focused"]),
.Select-menu-outer span,
.Select-menu div,
.Select-menu span {
    background-color: #1e293b !important;
    background: #1e293b !important;
    color: #ffffff !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

.VirtualizedSelectFocusedOption,
.VirtualizedSelectFocusedOption *,
.Select-option.is-selected,
.Select-option.is-selected *,
.Select-option.is-focused,
.Select-option.is-focused *,
.VirtualizedSelectOption.is-selected,
.VirtualizedSelectOption.is-selected *,
div[class*="-is-selected"],
div[class*="-is-selected"] *,
div[class*="-is-focused"],
div[class*="-is-focused"] * {
    background-color: #1e293b !important;
    background: #1e293b !important;
    color: #c084fc !important;
    font-weight: 700 !important;
}

.Select-arrow {
    border-color: #94a3b8 transparent transparent !important;
}

.rc-slider-mark-text {
    color: #94a3b8 !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    background: transparent !important;
}

.rc-slider-mark-text-active {
    color: #38bdf8 !important;
    font-weight: 700 !important;
}

.rc-slider-rail {
    background-color: #334155 !important;
    height: 6px !important;
}

.rc-slider-track {
    background-color: #a855f7 !important;
    height: 6px !important;
}

.rc-slider-handle {
    border: 2px solid #c084fc !important;
    background-color: #9333ea !important;
    width: 16px !important;
    height: 16px !important;
    margin-top: -5px !important;
}

input[type="number"],
.dash-input,
div[class*="number-input"],
.rc-slider-tooltip-inner {
    background-color: #1e293b !important;
    color: #38bdf8 !important;
    border: 1px solid #475569 !important;
    border-radius: 6px !important;
    font-weight: 800 !important;
    font-size: 13.5px !important;
    text-align: center !important;
    padding: 4px 8px !important;
}

/* Estilo para imágenes dentro de tarjetas educativas */
.edu-image {
    width: 100%;
    border-radius: 10px;
    margin: 8px 0 12px 0;
    border: 1px solid #1f2937;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}
.edu-image-container {
    display: flex;
    justify-content: center;
    margin: 10px 0 6px 0;
}
.img-caption {
    color: #94a3b8;
    font-size: 12px;
    text-align: center;
    margin-top: 4px;
    font-style: italic;
}
.sidebar-link i {
    color: #38bdf8 !important; 
    transition: color 0.15s ease;
}
.sidebar-link:hover i {
    color: #7dd3fc !important; 
}
.sidebar-link .fa-magnet {
    color: #38bdf8 !important;
}
.sidebar-link .fa-flask {
    color: #38bdf8 !important;
}
.sidebar-link .fa-house {
    color: #38bdf8 !important;
}
.sidebar-link .fa-atom {
    color: #38bdf8 !important;
}
.sidebar-link .fa-signal {
    color: #38bdf8 !important;
}
.sidebar-link .fa-palette {
    color: #38bdf8 !important;
}
.sidebar-link .fa-clock {
    color: #38bdf8 !important;
}

"""

THEME = {
    'bg': '#0a0f1d',
    'card': '#111827',
    'card_border': '#1f2937',
    'text': '#f3f4f6',
    'text_muted': '#9ca3af',
    'accent_cyan': '#38bdf8',
    'accent_emerald': '#10b981',
    'accent_amber': '#f59e0b'
}

STYLE_CONTAINER = {
    'fontFamily': '"Inter", "Segoe UI", -apple-system, sans-serif',
    'backgroundColor': THEME['bg'],
    'minHeight': '100vh',
    'padding': '20px 35px',
    'color': THEME['text']
}

STYLE_HEADER = {
    'background': 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%)',
    'padding': '26px 32px',
    'borderRadius': '16px',
    'border': '1px solid #334155',
    'boxShadow': '0 10px 25px -5px rgba(0, 0, 0, 0.5)',
    'marginBottom': '24px'
}

STYLE_CARD = {
    'backgroundColor': THEME['card'],
    'borderRadius': '14px',
    'padding': '22px',
    'border': '1px solid #ffffff',
    'boxShadow': '0 4px 12px rgba(0, 0, 0, 0.3)',
    'marginBottom': '20px'
}

STYLE_CALLOUT = {
    'background': 'linear-gradient(90deg, rgba(2,132,199,0.15) 0%, rgba(17,24,39,0.6) 100%)',
    'borderLeft': '4px solid #38bdf8',
    'padding': '14px 18px',
    'borderRadius': '8px',
    'fontSize': '15px',
    'color': '#e0f2fe',
    'lineHeight': '1.5',
    'marginBottom': '16px'
}

STYLE_SECTION_TITLE = {
    'fontSize': '20px',
    'fontWeight': '700',
    'color': '#f8fafc',
    'borderBottom': '1px solid #374151',
    'paddingBottom': '8px',
    'marginTop': '0px',
    'marginBottom': '14px'
}

STYLE_LABEL = {
    'fontWeight': '600',
    'fontSize': '15px',
    'color': '#cbd5e1',
    'marginTop': '14px',
    'marginBottom': '6px',
    'display': 'block'
}

STYLE_GRAPH_DESC = {
    'color': '#94a3b8',
    'fontSize': '14px',
    'lineHeight': '1.5',
    'marginTop': '10px',
    'paddingTop': '10px',
    'borderTop': '1px solid #1f2937'
}

def make_slider_marks(marks_dict):
    return {k: {'label': v, 'style': {'color': '#94a3b8', 'fontSize': '11px'}} for k, v in marks_dict.items()}

categories_list = list(DATASET_INDEX.keys())
initial_category = categories_list[0]
initial_max_samples = max(len(DATASET_INDEX[initial_category]) - 1, 0)

# ---- 4. LAYOUT DE LA SIMULACIÓN 
#Estructura de la simulacion/Interfaz

simulacion_layout = html.Div(style=STYLE_CONTAINER, children=[

    html.Div(style=STYLE_HEADER, children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'flexWrap': 'wrap'}, children=[
            html.Div([
                html.Div([
                    html.Span("INGENIERÍA BIOMÉDICA", style={
                        'backgroundColor': 'rgba(56, 189, 248, 0.2)', 'color': '#38bdf8', 'padding': '4px 12px',
                        'borderRadius': '20px', 'fontSize': '11px', 'fontWeight': '700', 'border': '1px solid rgba(56,189,248,0.4)'
                    })
                ]),
                html.H1("Simulación de Resonancia Magnética",
                        style={'margin': '10px 0 6px 0', 'fontSize': '44px', 'color': '#ffffff', 'fontWeight': '800'}),

            ]),
            html.Div(style={'textAlign': 'right', 'marginTop': '10px'}, children=[
                html.Div("Campo B₀ de Referencia", style={'fontSize': '12px', 'color': '#94a3b8'}),
                html.Div("1.5 Tesla (63.87 MHz)", style={'fontSize': '17px', 'fontWeight': '800', 'color': '#38bdf8'})
            ])
        ])
    ]),

    dcc.Tabs(id="tabs-mri", value='tab-signals', style={'marginBottom': '22px'}, children=[

        # ----------------- PESTAÑA 1: SEÑALES -----------------
        dcc.Tab(
            label=' 1. Física Cuántica de Señales RF y Relajación',
            value='tab-signals',
            style={'backgroundColor': '#111827', 'color': '#9ca3af', 'border': '1px solid #1f2937', 'padding': '14px', 'fontWeight': '600'},
            selected_style={'backgroundColor': '#1e293b', 'color': '#38bdf8', 'borderTop': '3px solid #38bdf8', 'fontWeight': '700', 'padding': '14px'},
            children=[
                html.Div(style={'display': 'flex', 'gap': '22px', 'marginTop': '18px'}, children=[

                    html.Div(style={'flex': '1', 'maxWidth': '380px'}, children=[
                        html.Div(style=STYLE_CARD, children=[
                            html.H3(" Pulso Excitatriz B₁(t)", style=STYLE_SECTION_TITLE),

                            html.Div(style=STYLE_CALLOUT, children=[
                                html.Strong("Flip Angle (α): "),
                                "El pulso RF inclina el vector Mz hacia Mxy: ",
                                html.I("α = γ ∫ B₁(t) dt. "), "90° produce la máxima señal FID."
                            ]),

                            html.Label("Forma de Onda del Pulso RF:", style=STYLE_LABEL),
                            dcc.Dropdown(
                                id='rf-type',
                                options=[
                                    {'label': 'Sinc (Selectivo de corte)', 'value': 'sinc'},
                                    {'label': 'Gaussiano (Suave)', 'value': 'gauss'},
                                    {'label': 'Rectangular (Hard pulse)', 'value': 'rect'}
                                ],
                                value='sinc',
                                clearable=False
                            ),

                            html.Label("Duración del Pulso (ms):", style=STYLE_LABEL),
                            dcc.Slider(id='rf-duration', min=1, max=20, step=1, value=10,
                                       marks=make_slider_marks({1: '1ms', 10: '10ms', 20: '20ms'})),

                            html.Label("Flip Angle α (°):", style=STYLE_LABEL),
                            dcc.Slider(id='rf-angle', min=10, max=180, step=10, value=90,
                                       marks=make_slider_marks({30: '30°', 90: '90°', 180: '180°'})),

                            html.Hr(style={'margin': '20px 0', 'borderColor': '#374151'}),
                            html.H3("⏱ Tiempos de Relajación Tisular", style=STYLE_SECTION_TITLE),

                            html.Label("T1 (Recuperación Longitudinal - ms):", style=STYLE_LABEL),
                            dcc.Slider(id='t1-val', min=100, max=2000, step=100, value=800,
                                       marks=make_slider_marks({200: 'Grasa (250)', 800: 'GM (900)', 1600: 'CSF (2500)'})),

                            html.Label("T2 (Decaimiento Transversal - ms):", style=STYLE_LABEL),
                            dcc.Slider(id='t2-val', min=10, max=500, step=10, value=100,
                                       marks=make_slider_marks({30: 'Hueso (30)', 100: 'Tejido (100)', 300: 'Agua (300)'})),

                            html.Hr(style={'margin': '20px 0', 'borderColor': '#374151'}),
                            html.H3("Filtrado del Receptor", style=STYLE_SECTION_TITLE),

                            html.Label("Tipo de Filtro:", style=STYLE_LABEL),
                            dcc.Dropdown(
                                id='filter-type',
                                options=[
                                    {'label': 'Pasa Bajos (Anti-Aliasing)', 'value': 'lowpass'},
                                    {'label': 'Pasa Altos (Elimina DC)', 'value': 'highpass'}
                                ],
                                value='lowpass',
                                clearable=False
                            ),

                            html.Label("Frecuencia de Corte (Hz):", style=STYLE_LABEL),
                            dcc.Slider(id='filter-cutoff', min=5, max=200, step=5, value=60,
                                       marks=make_slider_marks({10: '10Hz', 60: '60Hz', 150: '150Hz'}))
                        ])
                    ]),

                    html.Div(style={'flex': '2'}, children=[
                        # Gráfico 1: Pulso RF
                        html.Div(style=STYLE_CARD, children=[
                            dcc.Graph(id='graph-rf-pulse'),
                            html.Div([
                                html.Strong("Dominio Temporal: "), "Envolvente de RF modulada en amplitud B₁(t). ",
                                html.Br(),
                                html.Strong("Clave Señales: "), "El pulso Sinc en tiempo genera una banda de excitación rectangular en frecuencia por propiedades de Fourier, permitiendo seleccionar un corte anatómico limpio.",
                                html.Br(),
                                html.Strong("Efecto Físico: "), "El área bajo la curva define el ángulo de inclinación transversal (Flip Angle α)."
                            ], style=STYLE_GRAPH_DESC)
                        ]),
                        # Gráfico 2: FID
                        html.Div(style=STYLE_CARD, children=[
                            dcc.Graph(id='graph-fid-filtered'),
                            html.Div([
                                html.Strong("Fenómeno Bioeléctrico: "), "Voltaje oscilatorio inducido en la bobina por la precesión de espines (Ley de Faraday).",
                                html.Br(),
                                html.Strong("Procesamiento: "), "Compara la señal pura vs. filtrada: el filtro pasa-bajos suprime el ruido de alta frecuencia a expensas de suavizar transiciones rápidas.",
                                html.Br(),
                                html.Strong("Envolvente: "), "El decaimiento temporal de la amplitud está dictado directamente por la constante de tiempo tisular T2."
                            ], style=STYLE_GRAPH_DESC)
                        ]),
                        # Gráfico 3: FFT
                        html.Div(style=STYLE_CARD, children=[
                            dcc.Graph(id='graph-fft'),
                            html.Div([
                                html.Strong("Análisis espectral: "), "Transformada de Fourier de la señal FID. La magnitud muestra las frecuencias presentes y la fase la coherencia de los espines."
                            ], style=STYLE_GRAPH_DESC)
                        ]),
                        # Gráfico 4: Relajación Tisular

                        html.Div(style=STYLE_CARD, children=[
                            dcc.Graph(id='graph-relaxation-curves'),
                            html.Div(style={
                                'background': 'linear-gradient(90deg, rgba(56,189,248,0.10) 0%, rgba(17,24,39,0.65) 100%)',
                                'borderLeft': '4px solid #38bdf8',
                                'padding': '12px 16px',
                                'borderRadius': '8px',
                                'fontSize': '13px',
                                'lineHeight': '1.5',
                                'color': '#e2e8f0',
                                'marginTop': '10px'
                            }, children=[
                                html.Div("• Análisis de Respuesta Temporal (T₁ vs T₂):", style={'fontWeight': '700', 'color': '#f8fafc', 'marginBottom': '5px'}),
                                html.Div([
                                    html.Strong("1. Recuperación Longitudinal T₁ (Dominio de Reconstrucción Mz): ", style={'color': '#38bdf8'}),
                                    "Sigue la ecuación diferencial Mz(t) = M₀(1 - e^(-t/T₁)). Un T₁ corto implica constante de tiempo rápida. A TR = 500 ms, la sustancia blanca alcanza mayor amplitud que el LCR, generando contraste anatómico."
                                ]),
                                html.Div([
                                    html.Strong("2. Decaimiento Transversal T₂ (Dominio de Pérdida de Coherencia Mxy): ", style={'color': '#a855f7'}),
                                    "Sigue la ley de decaimiento exponencial Mxy(t) = M₀ · e^(-t/T₂). Tejidos con T₂ largo conservan desfase de espines por más tiempo. A TE = 15 ms, el LCR mantiene mayor nivel de señal transversal."
                                ]),
                                html.Div([
                                    html.Strong("Criterio de Diseño: ", style={'color': '#f59e0b'}),
                                    "El contraste de imagen es un muestreo en instantes TR y TE específicos de la respuesta temporal del tejido biológico."
                                ], style={'marginTop': '5px'})
                            ]),
                            html.Div("Curvas de respuesta temporal tisular. Los cursores verticales de muestreo (TR y TE) actúan como discriminadores de señal para maximizar la relación contraste-ruido (CNR) entre tejidos biológicos.", style=STYLE_GRAPH_DESC)
                        ]),
                        # Gráfico 5: Fantoma Espacio-k
                        html.Div(style=STYLE_CARD, children=[
                            dcc.Graph(id='graph-kspace-circle'),
                            html.Div("Representación de un objeto canónico (fantoma circular): dominio de la frecuencia espacial 2D (kx, ky) a la izquierda y reconstrucción espacial 2D f(x,y) mediante Transformada Inversa 2D de Fourier (iFFT2) a la derecha.", style=STYLE_GRAPH_DESC)
                        ]),
                    ])
                ])
            ]
        ),
        # ----------------- PESTAÑA 2: ESPACIO K Y DATASET
        dcc.Tab(
            label=' 2. Espacio-k 2D, Ponderaciones Clínicas y Reconstrucción',
            value='tab-images',
            style={'backgroundColor': '#111827', 'color': '#9ca3af', 'border': '1px solid #1f2937', 'padding': '14px', 'fontWeight': '600'},
            selected_style={'backgroundColor': '#1e293b', 'color': '#38bdf8', 'borderTop': '3px solid #38bdf8', 'fontWeight': '700', 'padding': '14px'},
            children=[
                html.Div(style={'display': 'flex', 'gap': '22px', 'marginTop': '18px'}, children=[

                    html.Div(style={'flex': '1', 'maxWidth': '390px'}, children=[
                        html.Div(style=STYLE_CARD, children=[
                            html.H3("Dataset Clínico Real", style=STYLE_SECTION_TITLE),

                            html.Label("Patología Cerebral:", style=STYLE_LABEL),
                            dcc.Dropdown(
                                id='category-select',
                                options=[{'label': f" {k} ({len(v)} casos)", 'value': k} for k, v in DATASET_INDEX.items()],
                                value=initial_category,
                                clearable=False
                            ),

                            html.Label("Muestra de Paciente:", style=STYLE_LABEL),
                            dcc.Slider(
                                id='sample-slider',
                                min=0,
                                max=initial_max_samples,
                                step=1,
                                value=0,
                                marks=make_slider_marks({0: 'Caso 0', initial_max_samples: f'Caso {initial_max_samples}'})
                            ),

                            html.Hr(style={'margin': '18px 0', 'borderColor': '#374151'}),
                            html.H3("Contraste Spin-Echo", style=STYLE_SECTION_TITLE),

                            html.Label("Ponderación de Tejido:", style=STYLE_LABEL),
                            dcc.Dropdown(
                                id='weight-type',
                                options=[
                                    {'label': 'Ponderación T1 (TR=500ms, TE=15ms) - Anatomía', 'value': 'T1'},
                                    {'label': 'Ponderación T2 (TR=3000ms, TE=90ms) - Edema/LCR', 'value': 'T2'},
                                    {'label': 'Densidad Protónica PD (TR=3000ms, TE=15ms) - Contenido H⁺', 'value': 'PD'}
                                ],
                                value='T1',
                                clearable=False
                            ),

                            html.Hr(style={'margin': '18px 0', 'borderColor': '#374151'}),
                            html.H3("🌐 Didáctica del Espacio-k", style=STYLE_SECTION_TITLE),

                            html.Label("Filtrado de Frecuencias Espaciales:", style=STYLE_LABEL),
                            dcc.Dropdown(
                                id='kspace-filter-select',
                                options=[
                                    {'label': 'Espacio-k Completo (Nyquist 100%)', 'value': 'all'},
                                    {'label': 'Solo Centro del Espacio-k (Contraste Tisular)', 'value': 'center_only'},
                                    {'label': 'Solo Periferia del Espacio-k (Bordes y Resolución)', 'value': 'periphery_only'}
                                ],
                                value='all',
                                clearable=False
                            ),

                            html.Label("Aceleración R (Submuestreo de Fase):", style=STYLE_LABEL),
                            dcc.Dropdown(
                                id='undersample-factor',
                                options=[
                                    {'label': 'R=1 (Sin Aliasing)', 'value': 1},
                                    {'label': 'R=2 (Líneas Alternadas - Aliasing)', 'value': 2},
                                    {'label': 'R=4 (Aliasing Severo)', 'value': 4}
                                ],
                                value=1,
                                clearable=False
                            ),

                            html.Label("Ruido Térmico (Inyección Rician):", style=STYLE_LABEL),
                            dcc.Slider(id='noise-slider', min=0.0, max=0.5, step=0.05, value=0.0,
                                       marks=make_slider_marks({0.0: '0%', 0.25: '25%', 0.5: '50%'})),

                            html.Hr(style={'margin': '18px 0', 'borderColor': '#374151'}),
                            html.H3("🖥️ Visor DICOM", style=STYLE_SECTION_TITLE),

                            html.Label("Window (Contraste):", style=STYLE_LABEL),
                            dcc.Slider(id='slider-window', min=0.1, max=2.0, step=0.05, value=1.0,
                                       marks=make_slider_marks({0.1: '0.1', 1.0: '1.0', 2.0: '2.0'})),

                            html.Label("Level (Brillo):", style=STYLE_LABEL),
                            dcc.Slider(id='slider-level', min=0.0, max=1.0, step=0.05, value=0.5,
                                       marks=make_slider_marks({0.0: '0.0', 0.5: '0.5', 1.0: '1.0'})),

                            html.Label("Filtro Espacial:", style=STYLE_LABEL),
                            dcc.Dropdown(
                                id='spatial-filter',
                                options=[
                                    {'label': 'Ninguno', 'value': 'none'},
                                    {'label': 'Gaussiano (Suavizado)', 'value': 'gauss'},
                                    {'label': 'Sobel (Detección de Bordes)', 'value': 'sobel'}
                                ],
                                value='none',
                                clearable=False
                            ),

                            html.Div(id='metrics-panel', style={'marginTop': '18px'})
                        ])
                    ]),

                    html.Div(style={'flex': '2'}, children=[
                        html.Div(id='biomedical-insight-box', style=STYLE_CARD),

                        # Gráfico Pipeline Imagen
                        html.Div(style=STYLE_CARD, children=[
                            dcc.Graph(id='graph-image-pipeline'),
                            html.Div("Pipeline del sistema de procesamiento de imágenes biomédicas: (1) Dominio K bidimensional F{f(x,y)}, (2) Reconstrucción mediante iFFT2 evidenciando solapamiento por submuestreo de Nyquist, (3) Post-procesamiento espacial y ajuste dinámico de contraste (Window/Level).", style=STYLE_GRAPH_DESC)
                        ]),
                        # Gráfico Secuencia Spin-Echo
                        html.Div(style=STYLE_CARD, children=[
                            dcc.Graph(id='graph-sequence-diagram'),
                            html.Div("Diagrama de tiempos de la secuencia Spin-Echo. Ilustra la aplicación del pulso de 90° (t=0), el pulso de re-fasamiento de 180° (t=TE/2) para cancelar inhomogeneidades del campo B₀, y la formación del eco en el tiempo TE.", style=STYLE_GRAPH_DESC)
                        ]),
                        # Gráfico Perfil 1D
                        html.Div(style=STYLE_CARD, children=[
                            dcc.Graph(id='graph-line-profile'),
                            html.Div("Muestreo espacial 1D (corte transversal de intensidad en la fila Y). Permite evaluar cuantitativamente la modulación del contraste, la pendiente de bordes (respuesta en frecuencia espacial) y la estimación estadística de la relación señal-ruido (SNR).", style=STYLE_GRAPH_DESC)
                        ])
                    ])
                ])
            ]
        )
    ])
])


# ---- 5. CALLBACKS DE LA SIMULACIÓN

def register_simulation_callbacks(app):

    @app.callback(
        [Output('sample-slider', 'max'),
         Output('sample-slider', 'marks'),
         Output('sample-slider', 'value')],
        [Input('category-select', 'value')]
    )
    def update_sample_slider(category):
        files = DATASET_INDEX.get(category, [])
        max_idx = max(len(files) - 1, 0)
        mid_idx = max_idx // 2
        marks = make_slider_marks({0: '0', mid_idx: f'{mid_idx}', max_idx: f'{max_idx}'})
        return max_idx, marks, 0

    #call de señales
    @app.callback(
        [Output('graph-rf-pulse', 'figure'),
         Output('graph-fid-filtered', 'figure'),
         Output('graph-fft', 'figure'),
         Output('graph-relaxation-curves', 'figure'),
         Output('graph-kspace-circle', 'figure')],
        [Input('rf-type', 'value'),
         Input('rf-duration', 'value'),
         Input('rf-angle', 'value'),
         Input('t1-val', 'value'),
         Input('t2-val', 'value'),
         Input('filter-type', 'value'),
         Input('filter-cutoff', 'value')]
    )
    def update_signals(rf_type, rf_dur, rf_ang, t1, t2, f_type, f_cutoff):
        template = "plotly_dark"
        plot_bg = THEME['card']
        paper_bg = THEME['card']

        t_rf, rf_sig = generate_rf_pulse(rf_type, rf_dur, rf_ang)
        fig_rf = go.Figure()
        fig_rf.add_trace(go.Scatter(x=t_rf, y=rf_sig, mode='lines', name='B₁(t)', line=dict(color=THEME['accent_cyan'], width=2.5)))
        fig_rf.update_layout(
            title=f"<b>Envolvente del Pulso RF B₁(t)</b> - {rf_type.upper()} | {rf_ang}°",
            xaxis_title="Tiempo (ms)", yaxis_title="Amplitud B₁", height=240, template=template,
            plot_bgcolor=plot_bg, paper_bgcolor=paper_bg, margin=dict(l=40, r=20, t=40, b=30)
        )

        t_fid, fid_complex = simulate_fid(t_max_ms=200, t1_ms=t1, t2_ms=t2, f0_hz=50.0)
        fid_real = np.real(fid_complex)
        fs = 1000.0 / (t_fid[1] - t_fid[0])
        fid_filt = apply_filter(fid_real, fs, f_type, f_cutoff)

        fig_fid = go.Figure()
        fig_fid.add_trace(go.Scatter(x=t_fid, y=fid_real, mode='lines', name='FID Real', opacity=0.35, line=dict(color='#94a3b8')))
        fig_fid.add_trace(go.Scatter(x=t_fid, y=fid_filt, mode='lines', name=f'FID Filtrada ({f_type.upper()})', line=dict(color=THEME['accent_emerald'], width=2)))
        fig_fid.update_layout(
            title="<b>Señal Free Induction Decay (FID)</b>",
            xaxis_title="Tiempo post-pulso (ms)", yaxis_title="Voltaje (mV)", height=240, template=template,
            plot_bgcolor=plot_bg, paper_bgcolor=paper_bg, margin=dict(l=40, r=20, t=40, b=30)
        )

        n = len(fid_real)
        fft_vals = np.fft.fftshift(np.fft.fft(fid_real))
        freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0/fs))
        mag = np.abs(fft_vals)
        phase = np.angle(fft_vals)

        fig_fft = make_subplots(rows=1, cols=2, subplot_titles=("<b>Magnitud |S(f)|</b>", "<b>Fase ∠S(f)</b>"))
        fig_fft.add_trace(go.Scatter(x=freqs, y=mag, mode='lines', name='Magnitud', line=dict(color=THEME['accent_cyan'], width=2)), row=1, col=1)
        fig_fft.add_trace(go.Scatter(x=freqs, y=phase, mode='lines', name='Fase', line=dict(color=THEME['accent_amber'], width=1.5)), row=1, col=2)
        fig_fft.update_layout(height=240, template=template, plot_bgcolor=plot_bg, paper_bgcolor=paper_bg, margin=dict(l=40, r=20, t=40, b=30))
        fig_fft.update_xaxes(range=[-120, 120])

        # GRÁFICO PEDAGÓGICO T1 vs T2 (con ID ÚNICO: graph-relaxation-curves)

        t_t1 = np.linspace(0, 3000, 500)
        t_t2 = np.linspace(0, 300, 500)

        tissue_t1 = {
            'Sust. Blanca (WM)': 600.0,
            'Sust. Gris (GM)': 900.0,
            'LCR/Agua (CSF)': 2500.0
        }
        tissue_t2 = {
            'Sust. Blanca (WM)': 70.0,
            'Sust. Gris (GM)': 100.0,
            'LCR/Agua (CSF)': 300.0
        }

        fig_relax = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "<b>Recuperación T₁ (Mz) - Cursor TR = 500 ms</b>",
                "<b>Decaimiento T₂ (Mxy) - Cursor TE = 15 ms</b>"
            ),
            horizontal_spacing=0.12
        )

        tissue_colors = {
            'Sust. Blanca (WM)': THEME['accent_cyan'],
            'Sust. Gris (GM)': '#a855f7',
            'LCR/Agua (CSF)': THEME['accent_emerald']
        }

        for tissue, t1_ref in tissue_t1.items():
            fig_relax.add_trace(
                go.Scatter(
                    x=t_t1, y=1 - np.exp(-t_t1 / t1_ref),
                    mode='lines', name=tissue,
                    line=dict(color=tissue_colors[tissue], width=2.2)
                ),
                row=1, col=1
            )

        fig_relax.add_trace(
            go.Scatter(
                x=t_t1, y=1 - np.exp(-t_t1 / t1),
                mode='lines', name=f'Personalizado (T₁={t1} ms)',
                line=dict(color=THEME['accent_amber'], width=2, dash='dash')
            ),
            row=1, col=1
        )

        for tissue, t2_ref in tissue_t2.items():
            fig_relax.add_trace(
                go.Scatter(
                    x=t_t2, y=np.exp(-t_t2 / t2_ref),
                    mode='lines', name=tissue,
                    line=dict(color=tissue_colors[tissue], width=2.2),
                    showlegend=False
                ),
                row=1, col=2
            )

        fig_relax.add_trace(
            go.Scatter(
                x=t_t2, y=np.exp(-t_t2 / t2),
                mode='lines', name=f'Personalizado (T₂={t2} ms)',
                line=dict(color=THEME['accent_amber'], width=2, dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )

        fig_relax.add_vline(
            x=500, line_dash='dot', line_color='#cbd5e1',
            annotation_text='TR=500 ms', annotation_position='top right',
            row=1, col=1
        )
        fig_relax.add_vline(
            x=15, line_dash='dot', line_color='#cbd5e1',
            annotation_text='TE=15 ms', annotation_position='top right',
            row=1, col=2
        )

        t1_cursor_values = {name: 1 - np.exp(-500 / val) for name, val in tissue_t1.items()}
        t2_cursor_values = {name: np.exp(-15 / val) for name, val in tissue_t2.items()}

        fig_relax.add_annotation(
            x=500, y=t1_cursor_values['Sust. Blanca (WM)'],
            xref='x', yref='y',
            text='↑ Rec. rápida',
            showarrow=True, arrowhead=2, ax=45, ay=-25,
            font=dict(size=10, color=THEME['accent_cyan'])
        )

        fig_relax.add_annotation(
            x=15, y=t2_cursor_values['LCR/Agua (CSF)'],
            xref='x2', yref='y2',
            text='↓ Dec. lento',
            showarrow=True, arrowhead=2, ax=50, ay=25,
            font=dict(size=10, color=THEME['accent_emerald'])
        )

        fig_relax.update_layout(
            height=320,
            template=template,
            plot_bgcolor=plot_bg,
            paper_bgcolor=paper_bg,
            margin=dict(l=45, r=25, t=50, b=65),
            legend=dict(
                orientation='h',
                yanchor='top', y=-0.22,
                xanchor='center', x=0.5,
                font=dict(size=10)
            )
        )

        fig_relax.update_xaxes(title_text='Tiempo tras pulso 90° (ms)', range=[0, 3000], row=1, col=1)
        fig_relax.update_xaxes(title_text='Tiempo tras pulso 90° (ms)', range=[0, 300], row=1, col=2)
        fig_relax.update_yaxes(title_text='Mz / M₀ (Recuperación)', range=[0, 1.05], row=1, col=1)
        fig_relax.update_yaxes(title_text='Mxy / M₀ (Señal Transversal)', range=[0, 1.05], row=1, col=2)

        _, kspace, recon = generate_circle_phantom_kspace(size=64, radius=16)
        kspace_log = np.log10(np.abs(kspace) + 1)

        fig_circ = make_subplots(rows=1, cols=2, subplot_titles=("<b>Espacio-k 2D</b>", "<b>Reconstrucción iFFT</b>"))
        fig_circ.add_trace(go.Heatmap(z=kspace_log, colorscale='Viridis', showscale=False), row=1, col=1)
        fig_circ.add_trace(go.Heatmap(z=recon, colorscale='Gray', showscale=False), row=1, col=2)
        fig_circ.update_layout(height=270, template=template, plot_bgcolor=plot_bg, paper_bgcolor=paper_bg, margin=dict(l=20, r=20, t=40, b=20))
        fig_circ.update_xaxes(showticklabels=False)
        fig_circ.update_yaxes(showticklabels=False)

        return fig_rf, fig_fid, fig_fft, fig_relax, fig_circ

    #callback de imagenes
    @app.callback(
        [Output('graph-image-pipeline', 'figure'),
         Output('graph-sequence-diagram', 'figure'),
         Output('graph-line-profile', 'figure'),
         Output('metrics-panel', 'children'),
         Output('biomedical-insight-box', 'children')],
        [Input('category-select', 'value'),
         Input('sample-slider', 'value'),
         Input('weight-type', 'value'),
         Input('kspace-filter-select', 'value'),
         Input('undersample-factor', 'value'),
         Input('noise-slider', 'value'),
         Input('slider-window', 'value'),
         Input('slider-level', 'value'),
         Input('spatial-filter', 'value')]
    )
    def update_image_pipeline(category, sample_idx, weight_type, k_filter, r_factor, noise_lvl, window, level, spat_filt):
        template = "plotly_dark"
        plot_bg = THEME['card']
        paper_bg = THEME['card']

        if weight_type == 'T1':
            tr, te = 500.0, 15.0
            contrast_desc = "TR Corto (500 ms) y TE Corto (15 ms): Maximiza la ponderación por T1. Grasa y sustancia blanca presentan alta intensidad; el LCR es hipointenso."
        elif weight_type == 'T2':
            tr, te = 3000.0, 90.0
            contrast_desc = "TR Largo (3000 ms) y TE Largo (90 ms): Maximiza la ponderación por T2. Fluidos libres como el LCR y edemas tisulares presentan hiperintensidad."
        else:
            tr, te = 3000.0, 15.0
            contrast_desc = "TR Largo (3000 ms) y TE Corto (15 ms): Minimiza los efectos de T1 y T2; la amplitud de la señal depende directamente de la concentración de protones (H⁺)."

        files = DATASET_INDEX.get(category, ['synthetic'])
        idx = int(np.clip(sample_idx, 0, len(files) - 1)) if len(files) > 0 else 0
        filepath = files[idx] if len(files) > 0 else 'synthetic'
        base_img = load_mri_image(filepath)

        img_contrast = simulate_mri_tissue_contrast(base_img, weight_type, tr, te)
        kspace, recon = simulate_kspace_pipeline(img_contrast, r_factor=r_factor, noise_level=noise_lvl, k_filter=k_filter)
        kspace_mag = np.log10(np.abs(kspace) + 1)
        wl_img = apply_window_level(recon, window, level)

        #filtro del dominio espacial
        if spat_filt == 'gauss':
            final_img = ndimage.gaussian_filter(wl_img, sigma=1.2)
        elif spat_filt == 'sobel':
            final_img = np.abs(ndimage.sobel(wl_img))
        else:
            final_img = wl_img

        snr_val = calculate_snr(final_img)
        filename_display = os.path.basename(filepath) if filepath != 'synthetic' else 'Muestra_Sintetica'

        metrics_html = html.Div(style={
            'backgroundColor': '#1e293b', 'padding': '16px', 'borderRadius': '10px',
            'border': '1px solid #334155'
        }, children=[
            html.Div([
                html.Span("Caso: ", style={'color': '#94a3b8', 'fontSize': '12px'}),
                html.Span(filename_display, style={'fontWeight': 'bold', 'color': '#38bdf8', 'fontSize': '12.5px', 'wordBreak': 'break-all'})
            ]),
            html.Div([
                html.Span("SNR Calculado: ", style={'color': '#94a3b8', 'fontSize': '12px'}),
                html.Strong(f"{snr_val:.2f}", style={'color': THEME['accent_emerald'], 'fontSize': '15px'})
            ], style={'marginTop': '6px'}),
            html.Div(f"⏱️ TR: {tr:.0f} ms | TE: {te:.0f} ms", style={'fontSize': '12px', 'color': '#cbd5e1', 'marginTop': '4px'}),
            html.Div(f"⚡ Factor Aceleración: R = {r_factor}x", style={'fontSize': '12px', 'color': THEME['accent_cyan'], 'fontWeight': 'bold', 'marginTop': '4px'})
        ])

        k_filter_msg = ""

        #filtro en espacio k - frecuencias espaciales
        if k_filter == 'center_only':
            k_filter_msg = " | [Filtro K: Pasa-Bajos] Retiene frecuencias centrales: preserva el contraste tisular global pero degrada la resolución espacial de bordes."
        elif k_filter == 'periphery_only':
            k_filter_msg = " | [Filtro K: Pasa-Altos] Retiene altas frecuencias: resalta contornos e interfaces anatómicas, eliminando la información de contraste."

        insight_html = html.Div(children=[
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'marginBottom': '8px'}, children=[
                html.Span("BIOFÍSICA EN TIEMPO REAL", style={
                    'backgroundColor': 'rgba(56, 189, 248, 0.2)', 'color': '#38bdf8', 'padding': '4px 10px',
                    'borderRadius': '6px', 'fontSize': '15px', 'fontWeight': '700', 'border': '1px solid rgba(56,189,248,0.3)'
                }),
                html.Span(f"Secuencia: {weight_type}", style={'fontWeight': 'bold', 'color': '#f8fafc', 'fontSize': '15px'})
            ]),
            html.P(contrast_desc + k_filter_msg, style={'fontSize': '13px', 'margin': '0 0 8px 0', 'lineHeight': '1.5', 'color': '#e2e8f0'}),
            html.Div(style={'fontSize': '14px', 'color': '#94a3b8'}, children=[
                html.Strong("Efecto de Aliasing Espacial: ") if r_factor > 1 else html.Span(),
                f"El submuestreo periódico en el Espacio-k (R={r_factor}) reduce la frecuencia de muestreo por debajo del límite de Nyquist, produciendo el solapamiento o repliegue de la imagen reconstruida." if r_factor > 1 else "Muestreo completo que cumple el Teorema de Nyquist-Shannon."
            ])
        ])

        fig_pipeline = make_subplots(rows=1, cols=3, subplot_titles=(
            "<b>1. Espacio-k (Frecuencias)</b>",
            f"<b>2. Reconstrucción iFFT (R={r_factor})</b>",
            f"<b>3. Imagen Diagnóstica ({spat_filt.upper()})</b>"
        ))

        fig_pipeline.add_trace(go.Heatmap(z=kspace_mag, colorscale='Viridis', showscale=False), row=1, col=1)
        fig_pipeline.add_trace(go.Heatmap(z=recon, colorscale='Gray', showscale=False), row=1, col=2)
        fig_pipeline.add_trace(go.Heatmap(z=final_img, colorscale='Gray', showscale=False), row=1, col=3)

        fig_pipeline.update_layout(
            height=350,
            template=template,
            plot_bgcolor=plot_bg,
            paper_bgcolor=paper_bg,
            margin=dict(l=15, r=15, t=50, b=15)
        )
        fig_pipeline.update_xaxes(showticklabels=False)
        fig_pipeline.update_yaxes(showticklabels=False)

        t_seq = np.linspace(0, 120, 600)
        rf_pulse_train = np.exp(-((t_seq - 10)/2.5)**2) + 1.2 * np.exp(-((t_seq - 55)/2.5)**2)
        echo_signal = 0.9 * np.exp(-((t_seq - 100)/5.0)**2) * np.cos(2*np.pi*0.25*(t_seq - 100))

        fig_seq = go.Figure()
        fig_seq.add_trace(go.Scatter(x=t_seq, y=rf_pulse_train, mode='lines', name='Pulsos RF (90° / 180°)', line=dict(color=THEME['accent_cyan'], width=2)))
        fig_seq.add_trace(go.Scatter(x=t_seq, y=echo_signal, mode='lines', name='Eco de Espín', line=dict(color=THEME['accent_emerald'], width=2)))

        fig_seq.add_vline(x=10, line_dash="dot", line_color="#64748b", annotation_text="90° RF", annotation_position="top left")
        fig_seq.add_vline(x=55, line_dash="dot", line_color="#64748b", annotation_text="180° RF (TE/2)", annotation_position="top left")
        fig_seq.add_vline(x=100, line_dash="dot", line_color=THEME['accent_amber'], annotation_text=f"Eco (TE={te:.0f}ms)", annotation_position="top right")

        fig_seq.update_layout(
            title="<b>Cronograma de Tiempos Spin-Echo</b>",
            xaxis_title="Tiempo Relativo (ms)",
            yaxis_title="Amplitud",
            height=190,
            template=template,
            plot_bgcolor=plot_bg,
            paper_bgcolor=paper_bg,
            margin=dict(l=40, r=20, t=35, b=25)
        )

        mid_row = final_img.shape[0] // 2
        profile_data = final_img[mid_row, :]

        fig_profile = go.Figure()
        fig_profile.add_trace(go.Scatter(
            y=profile_data, mode='lines+markers',
            line=dict(color=THEME['accent_cyan'], width=2),
            marker=dict(size=4),
            name=f'Fila Y = {mid_row}'
        ))
        fig_profile.update_layout(
            title=f"<b>Perfil de Intensidad 1D (Fila Y = {mid_row})</b>",
            xaxis_title="Píxel (X)",
            yaxis_title="Intensidad",
            height=190,
            template=template,
            plot_bgcolor=plot_bg,
            paper_bgcolor=paper_bg,
            margin=dict(l=40, r=20, t=35, b=25)
        )

        return fig_pipeline, fig_seq, fig_profile, metrics_html, insight_html


# SECCIÓN B - LIBRO ELECTRÓNICO EDUCATIVO

STYLE_SIDEBAR = {
    'width': '270px',
    'minWidth': '270px',
    'backgroundColor': '#0d1424',
    'borderRight': f"1px solid {THEME['card_border']}",
    'padding': '28px 18px',
    'position': 'sticky',
    'top': '0',
    'height': '100vh',
    'overflowY': 'auto',
    'boxSizing': 'border-box',
    'fontFamily': '"Inter", "Segoe UI", -apple-system, sans-serif',
}

STYLE_CONTENT_AREA = {
    'flex': '1',
    'minWidth': '0',
    'backgroundColor': THEME['bg'],
    'fontFamily': '"Inter", "Segoe UI", -apple-system, sans-serif',
}

STYLE_EDU_CARD = {
    'backgroundColor': THEME['card'],
    'borderRadius': '16px',
    'padding': '26px 28px',
    'border': '1px solid #ffffff',
    'boxShadow': '0 4px 14px rgba(0, 0, 0, 0.35)',
    'marginBottom': '22px',
}

STYLE_EDU_PAGE_HEADER = {
    'background': 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%)',
    'padding': '30px 34px',
    'borderRadius': '16px',
    'border': '1px solid #ffffff',
    'boxShadow': '0 10px 25px -5px rgba(0, 0, 0, 0.5)',
    'marginBottom': '26px',
}

STYLE_PILL = {
    'backgroundColor': 'rgba(56, 189, 248, 0.2)', 'color': THEME['accent_cyan'], 'padding': '4px 12px',
    'borderRadius': '20px', 'fontSize': '11px', 'fontWeight': '700', 'border': '1px solid rgba(56,189,248,0.4)',
    'letterSpacing': '0.5px'
}

STYLE_MODAL_OVERLAY_HIDDEN = {'display': 'none'}
STYLE_MODAL_OVERLAY_VISIBLE = {
    'display': 'flex',
    'position': 'fixed',
    'top': '0', 'left': '0', 'width': '100%', 'height': '100%',
    'backgroundColor': 'rgba(2,6,15,0.78)',
    'zIndex': '9998',
    'alignItems': 'center',
    'justifyContent': 'center',
}

STYLE_MODAL_BOX = {
    'backgroundColor': THEME['card'],
    'border': '1px solid #ffffff',
    'borderTop': f"3px solid {THEME['accent_cyan']}",
    'borderRadius': '16px',
    'padding': '26px 30px',
    'maxWidth': '580px',
    'width': '90%',
    'maxHeight': '80vh',
    'overflowY': 'auto',
    'boxShadow': '0 20px 50px rgba(0,0,0,0.6)',
    'fontFamily': '"Inter", "Segoe UI", -apple-system, sans-serif',
}

EXTRA_CSS = """
.sidebar-link {
    display: flex;
    align-items: center;
    padding: 11px 14px;
    border-radius: 10px;
    color: #9ca3af;
    text-decoration: none;
    transition: all 0.15s ease;
    margin-bottom: 3px;
}
.sidebar-link:hover {
    background-color: rgba(56, 189, 248, 0.12);
    color: #38bdf8;
}
.term-link {
    color: #38bdf8;
    background: none;
    border: none;
    border-bottom: 1px dashed #38bdf8;
    padding: 0 1px;
    font: inherit;
    font-weight: 700;
    cursor: pointer;
    transition: color 0.15s ease;
}
.term-link:hover {
    color: #f59e0b;
    border-bottom-color: #f59e0b;
}
.edu-card:hover {
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.5);
    transition: box-shadow 0.2s ease;
}
.modal-close-btn {
    background: rgba(148,163,184,0.15);
    border: none;
    color: #cbd5e1;
    width: 30px;
    height: 30px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 15px;
}
.modal-close-btn:hover {
    background: rgba(239,68,68,0.25);
    color: #f87171;
}
.nav-cta-btn {
    background: linear-gradient(135deg, #0369a1, #0284c7);
    color: #ffffff;
    border: none;
    padding: 12px 22px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 14px;
    cursor: pointer;
    text-decoration: none;
    display: inline-block;
}
.nav-cta-btn:hover {
    background: linear-gradient(135deg, #0284c7, #38bdf8);
}
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: #0a0f1d; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #475569; }
"""

FULL_CSS = DARK_CSS + EXTRA_CSS


# ---- Glosario-

def _modal_body(paragraphs):
    return html.Div([html.P(p, style={'fontSize': '14.5px', 'lineHeight': '1.6', 'color': '#e2e8f0', 'marginBottom': '10px'}) for p in paragraphs])


#glosarios
GLOSSARY = {
    'espin': {
        'title': 'Espín del Protón',
        'body': _modal_body([
            "El espín es una propiedad cuántica intrínseca del protón (núcleo de hidrógeno), análoga a un pequeño imán en rotación permanente.",
            "En ausencia de campo magnético externo, los espines de los protones del tejido apuntan en direcciones aleatorias, por lo que su suma vectorial es prácticamente nula.",
            "Al aplicar un campo magnético externo B₀, los espines tienden a alinearse en paralelo o antiparalelo al campo, generando una magnetización neta M₀ a favor del eje del campo (ligero exceso de espines en paralelo, de menor energía).",
        ])
    },
    'b0': {
        'title': 'Campo Magnético B₀',
        'body': _modal_body([
            "B₀ es el campo magnético principal y estático del equipo de resonancia, generado típicamente por un imán superconductor. Se mide en Tesla (T); los equipos clínicos habituales usan 1.5 T o 3 T.",
            "Su función es alinear los espines de los protones del cuerpo y establecer el eje longitudinal (eje Z) sobre el que se mide la magnetización.",
            "Cuanto mayor es B₀, mayor es la magnetización neta disponible y, en general, mayor la relación señal-ruido (SNR), pero también aumentan ciertos artefactos.",
        ])
    },
    'larmor': {
        'title': 'Frecuencia de Larmor',
        'body': _modal_body([
            "Es la frecuencia de precesión de los espines alrededor del eje de B₀, dada por la ecuación de Larmor: ω₀ = γ · B₀ (en rad/s), o f₀ = γ/2π · B₀ (en Hz).",
            "γ es la razón giromagnética, una constante propia de cada núcleo (para el hidrógeno, γ/2π ≈ 42.58 MHz/T).",
            "A 1.5 T, la frecuencia de Larmor del hidrógeno es de aproximadamente 63.87 MHz; a 3 T, se duplica a ~127.74 MHz. Esta es exactamente la frecuencia que debe tener el pulso de radiofrecuencia (RF) para excitar los espines (condición de resonancia).",
        ])
    },
    't1': {
        'title': 'Relajación T1 (Longitudinal)',
        'body': _modal_body([
            "T1 es la constante de tiempo con la que la magnetización longitudinal Mz se recupera hacia su valor de equilibrio M₀ después de un pulso de excitación, siguiendo Mz(t) = M₀·(1 - e^(−t/T1)).",
            "Físicamente refleja cuán eficientemente los protones excitados ceden energía a la red molecular circundante (relajación espín-red).",
            "Tejidos como la grasa tienen T1 corto (recuperan rápido, se ven brillantes en secuencias T1); el líquido cefalorraquídeo (LCR) tiene T1 largo (se ve oscuro).",
        ])
    },
    't2': {
        'title': 'Relajación T2 (Transversal)',
        'body': _modal_body([
            "T2 es la constante de tiempo con la que la magnetización transversal Mxy decae debido a la pérdida de coherencia de fase entre espines vecinos, siguiendo Mxy(t) = M₀·e^(-t/T2).",
            "Refleja interacciones espín-espín: pequeñas variaciones locales de campo magnético hacen que los espines precesen a frecuencias ligeramente distintas y se desfasen entre sí.",
            "Los fluidos libres (como el LCR) tienen T2 largo y se ven brillantes en secuencias T2; tejidos más organizados (hueso, ligamentos) tienen T2 corto y se ven oscuros.",
        ])
    },
    'pd': {
        'title': 'Densidad Protónica (PD)',
        'body': _modal_body([
            "La ponderación por Densidad Protónica busca minimizar los efectos de T1 y T2, de modo que el contraste de la imagen dependa principalmente de la concentración de protones de hidrógeno (agua/grasa) en cada tejido.",
            "Se logra usando un TR largo (para minimizar el efecto T1) junto con un TE corto (para minimizar el efecto T2).",
            "Es útil para evaluar estructuras con contenido de agua similar pero distinta densidad de protones, como el cartílago articular.",
        ])
    },
    'kspace': {
        'title': 'Espacio-k',
        'body': _modal_body([
            "El espacio-k es el dominio de frecuencias espaciales donde se almacenan crudos los datos adquiridos por el escáner de MRI, antes de reconstruir la imagen.",
            "Cada punto del espacio-k no corresponde a un píxel de la imagen final: contiene información sobre una frecuencia espacial particular presente en toda la imagen.",
            "El centro del espacio-k (bajas frecuencias) determina principalmente el contraste global de la imagen; la periferia (altas frecuencias) determina los bordes y detalles finos.",
            "La imagen final se obtiene aplicando una Transformada Inversa de Fourier 2D (iFFT2) sobre los datos del espacio-k.",
        ])
    },
    'aliasing': {
        'title': 'Aliasing (Solapamiento Espacial)',
        'body': _modal_body([
            "El aliasing ocurre cuando el espacio-k se muestrea a una tasa inferior a la exigida por el Teorema de Nyquist-Shannon, típicamente al usar técnicas de aceleración (submuestreo, factor R) para acortar el tiempo de adquisición.",
            "El resultado visual es un 'repliegue' o superposición de partes de la imagen sobre zonas opuestas del campo de visión (FOV), generando un artefacto de solapamiento característico.",
            "Técnicas de reconstrucción paralela (como SENSE o GRAPPA) usan información adicional de múltiples bobinas receptoras para corregir este solapamiento y permitir factores de aceleración R > 1 sin perder demasiada calidad de imagen.",
        ])
    },
    'tr': {
        'title': 'Tiempo de Repetición (TR)',
        'body': _modal_body([
            "TR es el tiempo transcurrido entre la aplicación de un pulso de excitación de 90° y el siguiente pulso de 90° del ciclo posterior, medido en milisegundos.",
            "Controla cuánta recuperación longitudinal (T1) ha ocurrido antes de la siguiente excitación: un TR corto acentúa las diferencias de T1 entre tejidos (ponderación T1); un TR largo permite recuperación casi completa, reduciendo el efecto T1.",
        ])
    },
    'te': {
        'title': 'Tiempo de Eco (TE)',
        'body': _modal_body([
            "TE es el tiempo transcurrido desde el pulso de excitación de 90° hasta el momento en que se mide el eco de la señal, medido en milisegundos.",
            "Controla cuánto decaimiento transversal (T2) ha ocurrido antes de medir la señal: un TE corto minimiza el efecto T2; un TE largo acentúa las diferencias de T2 entre tejidos (ponderación T2).",
        ])
    },
    'flipangle': {
        'title': 'Ángulo de Inclinación (Flip Angle)',
        'body': _modal_body([
            "El flip angle (α) es el ángulo en el que el pulso de radiofrecuencia inclina el vector de magnetización neta M respecto al eje longitudinal (Z), dado por α = γ ∫ B₁(t) dt.",
            "Un pulso de 90° lleva toda la magnetización al plano transversal, produciendo la máxima señal FID inicial disponible.",
            "Ángulos menores a 90° se usan en secuencias rápidas de gradiente para preservar magnetización longitudinal entre repeticiones sucesivas.",
        ])
    },
    'spinecho': {
        'title': 'Secuencia Spin-Echo',
        'body': _modal_body([
            "La secuencia Spin-Echo es una de las secuencias de pulso fundamentales en MRI. Combina un pulso de excitación de 90° con un pulso de re-fasamiento de 180° aplicado en TE/2.",
            "El pulso de 180° invierte el desfase acumulado por inhomogeneidades del campo B₀, generando un 'eco' de la señal en el instante TE, con amplitud que depende principalmente de T2 (no de T2*, que incluye efectos de inhomogeneidad del campo).",
            "Es la base de las ponderaciones clínicas clásicas T1, T2 y PD, según los valores elegidos de TR y TE.",
        ])
    },
    'fid': {
        'title': 'FID (Free Induction Decay)',
        'body': _modal_body([
            "El FID es la señal oscilatoria que se induce en la bobina receptora inmediatamente después del pulso de excitación, por efecto de la precesión de los espines (Ley de inducción de Faraday).",
            "Su amplitud decae exponencialmente con una constante de tiempo T2* (que incluye tanto la relajación T2 real como inhomogeneidades del campo magnético local).",
            "Contiene, en el dominio del tiempo, toda la información de frecuencias que luego se extrae mediante la Transformada de Fourier.",
        ])
    },
    'rmn': {
        'title': 'Resonancia Magnética Nuclear (RMN)',
        'body': _modal_body([
            "La Resonancia Magnética Nuclear es el fenómeno físico en el que los núcleos atómicos con espín (como el hidrógeno) absorben y luego emiten energía electromagnética al ser excitados por un pulso de radiofrecuencia sintonizado exactamente en su frecuencia de Larmor, mientras están inmersos en un campo magnético estático B₀.",
            "Este fenómeno fue descrito originalmente en física y química para analizar la composición de sustancias; aplicado al cuerpo humano y combinado con gradientes de campo para codificar posición espacial, da lugar a la Resonancia Magnética por Imágenes (MRI).",
            "La señal de RMN que se detecta al final (el FID y su transformada) contiene información sobre cuántos protones hay en cada punto y con qué rapidez se relajan, lo cual se traduce en contraste de imagen.",
        ])
    },
    'imagenologia': {
        'title': 'Formación de Imágenes Anatómicas y Funcionales',
        'body': _modal_body([
            "A partir de la señal de RMN, y usando gradientes de campo magnético para codificar espacialmente cada punto del cuerpo, es posible reconstruir imágenes anatómicas detalladas de tejidos blandos (cerebro, músculo, órganos) sin usar radiación ionizante.",
            "Además de la anatomía estática, existen variantes de MRI 'funcionales' (como la fMRI) que detectan cambios de oxigenación sanguínea asociados a la actividad neuronal, permitiendo mapear qué zonas del cerebro se activan durante una tarea.",
            "Otras variantes miden difusión de agua (para detectar isquemias cerebrales tempranas) o flujo sanguíneo, ampliando el uso de la MRI más allá de la simple anatomía.",
        ])
    },
    'diagnostico': {
        'title': 'Importancia Clínica y Diagnóstica',
        'body': _modal_body([
            "La MRI es una de las herramientas más importantes en el diagnóstico médico moderno gracias a su excelente contraste entre tejidos blandos, algo que otras modalidades como la radiografía o la tomografía computarizada logran con menor detalle.",
            "Se usa ampliamente para detectar tumores, lesiones cerebrales, daño en ligamentos y articulaciones, patologías de columna, enfermedades cardiovasculares y muchas otras condiciones.",
            "Al no utilizar radiación ionizante (a diferencia de los rayos X o la TC), es una técnica considerada segura para estudios repetidos y para poblaciones sensibles, aunque tiene contraindicaciones propias (por ejemplo, ciertos implantes metálicos).",
        ])
    },
    'espinred': {
        'title': 'Relajación Espín-Red (mecanismo de T1)',
        'body': _modal_body([
            "La relajación espín-red es el mecanismo físico detrás de T1: los protones excitados ceden el exceso de energía que recibieron del pulso de RF a la 'red' molecular circundante (las moléculas vecinas y su movimiento térmico), regresando gradualmente a su alineación de equilibrio con B₀.",
            "Cuanto más eficiente sea este intercambio de energía entre el protón y su entorno molecular, más rápido se recupera Mz - es decir, más corto es T1.",
        ])
    },
    'espinespin': {
        'title': 'Relajación Espín-Espín (mecanismo de T2)',
        'body': _modal_body([
            "La relajación espín-espín es el mecanismo físico detrás de T2: los espines vecinos interactúan entre sí y experimentan pequeñas variaciones de campo magnético local, lo que hace que precesen a frecuencias ligeramente distintas y pierdan coherencia de fase entre ellos.",
            "A diferencia de T1, este proceso no involucra pérdida de energía hacia el entorno, sino simplemente pérdida de sincronía entre los espines - por eso T2 es siempre igual o más corto que T1 en un mismo tejido.",
        ])
    },
}


def term(text, key):
    """Botón de término clickeable que abre el modal de 'aprender más'."""
    return html.Button(text, id={'type': 'term-link', 'term': key}, n_clicks=0, className='term-link')


def edu_card(title, icon, children):
    return html.Div(className='edu-card', style=STYLE_EDU_CARD, children=[
        html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'marginBottom': '14px'}, children=[
            html.Span(icon, style={'fontSize': '22px'}),
            html.H3(title, style={'margin': '0', 'fontSize': '19px', 'fontWeight': '700', 'color': '#f8fafc'})
        ]),
        html.Div(children)
    ])


def page_header(pill_text, title, subtitle):
    return html.Div(style=STYLE_EDU_PAGE_HEADER, children=[
        html.Span(pill_text, style=STYLE_PILL),
        html.H1(title, style={'margin': '12px 0 6px 0', 'fontSize': '34px', 'color': '#ffffff', 'fontWeight': '800'}),
        html.P(subtitle, style={'margin': '0', 'fontSize': '15px', 'color': '#cbd5e1', 'maxWidth': '780px', 'lineHeight': '1.5'}),
    ])


# ---- Gráficos estáticos ilustrativos (complementan el texto de cada página) ----

STATIC_FIG_LAYOUT_KWARGS = dict(
    template='plotly_dark',
    plot_bgcolor=THEME['card'],
    paper_bgcolor=THEME['card'],
    font=dict(family='"Inter", "Segoe UI", -apple-system, sans-serif', color=THEME['text']),
)

# Barras: campo B₀ vs frecuencia
def fig_field_comparison():
    """Gráfico de barras: intensidad de campo B0 vs frecuencia de Larmor."""
    campos = ['0.5 T', '1.5 T', '3.0 T']
    frecuencias = [21.29, 63.87, 127.74]
    colores = [THEME['accent_emerald'], THEME['accent_cyan'], THEME['accent_amber']]

    fig = go.Figure(go.Bar(
        x=campos, y=frecuencias,
        marker_color=colores,
        text=[f"{f} MHz" for f in frecuencias],
        textposition='outside'
    ))
    fig.update_layout(
        title="<b>Frecuencia de Larmor según Intensidad de Campo B₀</b>",
        xaxis_title="Campo Magnético Principal (B₀)",
        yaxis_title="Frecuencia de Larmor (MHz)",
        height=300,
        margin=dict(l=50, r=20, t=50, b=40),
        **STATIC_FIG_LAYOUT_KWARGS
    )
    fig.update_yaxes(range=[0, 145])
    return fig

#curvas T1/T2
def fig_static_relaxation_curves():
    """Curvas ilustrativas (estáticas) de recuperación T1 y decaimiento T2 por tejido."""
    t_t1 = np.linspace(0, 3000, 300)
    t_t2 = np.linspace(0, 300, 300)

    tissue_t1 = {'Sust. Blanca (WM)': 600.0, 'Sust. Gris (GM)': 900.0, 'LCR/Agua (CSF)': 2500.0}
    tissue_t2 = {'Sust. Blanca (WM)': 70.0, 'Sust. Gris (GM)': 100.0, 'LCR/Agua (CSF)': 300.0}
    colors = {'Sust. Blanca (WM)': THEME['accent_cyan'], 'Sust. Gris (GM)': '#a855f7', 'LCR/Agua (CSF)': THEME['accent_emerald']}

    fig = make_subplots(rows=1, cols=2, subplot_titles=("<b>Recuperación T₁ (Mz)</b>", "<b>Decaimiento T₂ (Mxy)</b>"), horizontal_spacing=0.12)

    for tissue, t1_ref in tissue_t1.items():
        fig.add_trace(go.Scatter(x=t_t1, y=1 - np.exp(-t_t1 / t1_ref), mode='lines', name=tissue, line=dict(color=colors[tissue], width=2.2)), row=1, col=1)
    for tissue, t2_ref in tissue_t2.items():
        fig.add_trace(go.Scatter(x=t_t2, y=np.exp(-t_t2 / t2_ref), mode='lines', name=tissue, line=dict(color=colors[tissue], width=2.2), showlegend=False), row=1, col=2)

    fig.update_layout(
        height=300,
        margin=dict(l=45, r=25, t=45, b=60),
        legend=dict(orientation='h', yanchor='top', y=-0.22, xanchor='center', x=0.5, font=dict(size=10)),
        **STATIC_FIG_LAYOUT_KWARGS
    )
    fig.update_xaxes(title_text='Tiempo (ms)', row=1, col=1)
    fig.update_xaxes(title_text='Tiempo (ms)', row=1, col=2)
    fig.update_yaxes(title_text='Mz / M₀', range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text='Mxy / M₀', range=[0, 1.05], row=1, col=2)
    return fig

#intensidad para los tejidos
def fig_contrast_bars():
    """Comparación ilustrativa de intensidad relativa de señal por tejido y ponderación."""
    tejidos = ['Grasa', 'Sust. Gris', 'LCR']
    t1_vals = [0.90, 0.55, 0.15]
    t2_vals = [0.35, 0.50, 0.95]
    pd_vals = [0.70, 0.65, 0.55]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='T1', x=tejidos, y=t1_vals, marker_color=THEME['accent_cyan']))
    fig.add_trace(go.Bar(name='T2', x=tejidos, y=t2_vals, marker_color=THEME['accent_emerald']))
    fig.add_trace(go.Bar(name='PD', x=tejidos, y=pd_vals, marker_color=THEME['accent_amber']))
    fig.update_layout(
        title="<b>Intensidad Relativa de Señal por Tejido y Ponderación</b>",
        xaxis_title="Tejido",
        yaxis_title="Intensidad Relativa (0–1, ilustrativo)",
        barmode='group',
        height=300,
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        **STATIC_FIG_LAYOUT_KWARGS
    )
    return fig

#visualizacion de espacio-k
def fig_kspace_matrix():
    """Representación visual de una matriz de espacio-k con centro y periferia resaltados."""
    size = 64
    y, x = np.ogrid[:size, :size]
    cy, cx = size // 2, size // 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (size / 2)

    matrix = np.zeros((size, size))
    matrix[dist <= 0.18] = 2.0     # centro
    matrix[(dist > 0.18)] = 1.0    # periferia

    fig = go.Figure(go.Heatmap(
        z=matrix,
        colorscale=[[0.0, '#1e293b'], [0.5, THEME['accent_amber']], [1.0, THEME['accent_cyan']]],
        showscale=False
    ))
    fig.add_annotation(x=cx, y=cy, text="CENTRO<br>(contraste)", showarrow=False, font=dict(size=11, color='#0a0f1d', family='"Inter", sans-serif'))
    fig.add_annotation(x=size - 10, y=8, text="PERIFERIA (bordes)", showarrow=False, font=dict(size=10, color='#0a0f1d', family='"Inter", sans-serif'))
    fig.update_layout(
        title="<b>Matriz de Espacio-k: Centro vs. Periferia</b>",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        **STATIC_FIG_LAYOUT_KWARGS
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig

def next_section_link(href, label):
    return html.Div(style={'marginTop': '10px', 'marginBottom': '30px'}, children=[
        dcc.Link(f"Siguiente: {label} →", href=href, className='nav-cta-btn')
    ])
# ---- Página: Inicio ----

def page_inicio():
    return html.Div(style=STYLE_CONTAINER, children=[
        page_header(
            "LIBRO ELECTRÓNICO INTERACTIVO",
            "Fundamentos de Resonancia Magnética (MRI)",
            "Un recorrido visual por la física, el procesamiento de señales y la reconstrucción de imágenes en MRI, con una simulación biomédica completamente funcional al final."
        ),

        edu_card("¿Qué es la Resonancia Magnética?", "", [
            html.P([
                "La Resonancia Magnética (MRI, por sus siglas en inglés) es una técnica de diagnóstico por imágenes que aprovecha el fenómeno físico de la ",
                term("Resonancia Magnética Nuclear (RMN)", "rmn"),
                " para observar el interior del cuerpo humano ",
                html.Strong("sin usar radiación ionizante"), "."
            ], style={'color': '#e2e8f0', 'fontSize': '15px', 'lineHeight': '1.7', 'marginBottom': '14px'}),

            html.Ul(style={'color': '#e2e8f0', 'fontSize': '14.5px', 'lineHeight': '1.9', 'paddingLeft': '22px', 'margin': '0 0 14px 0'}, children=[
                html.Li([
                    html.Strong("Fenómeno físico (RMN): ", style={'color': THEME['accent_cyan']}),
                    "los núcleos de hidrógeno del cuerpo (abundantes en agua y grasa) se alinean con un campo magnético intenso y absorben energía de pulsos de radiofrecuencia sintonizados a su frecuencia de resonancia; al relajarse, emiten una señal medible."
                ]),
                html.Li([
                    html.Strong("Obtención de imágenes: ", style={'color': THEME['accent_emerald']}),
                    "usando gradientes de campo magnético para codificar la posición espacial de cada señal, es posible reconstruir ",
                    term("imágenes anatómicas y funcionales", "imagenologia"),
                    " del cuerpo con un nivel de detalle excepcional en tejidos blandos."
                ]),
                html.Li([
                    html.Strong("Importancia clínica: ", style={'color': THEME['accent_amber']}),
                    "es una herramienta clave para el ",
                    term("diagnóstico médico", "diagnostico"),
                    " de tumores, lesiones neurológicas, patologías articulares y muchas otras condiciones, gracias a su excelente contraste entre tejidos."
                ]),
            ]),
            
    html.Div(className='edu-image-container', children=[
        html.Img(
            src='/assets/mri.png',
            className='edu-image',
            style={'maxHeight': '400px', 'objectFit': 'contain', 'margin': '6px 0 6px 0'}
        )
    ]),
    html.Div("Equipo de Resonancia Magnética", className='img-caption'),
    html.Div(style={'margin': '16px 0 12px 0'}, children=[
        html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px', 'marginBottom': '10px'}, children=[
            html.Span("", style={'fontSize': '18px'}),
            html.Strong("Funcionamiento de bobinas y captura de señales MRI", 
                   style={'color': '#f8fafc', 'fontSize': '14px'})
    ]),
        html.Div(style={'display': 'flex', 'justifyContent': 'center'}, children=[
            html.Video(
                src='/assets/MRI.mp4',
                controls=True,
                autoPlay=False,
                style={
                    'width': '100%',
                    'maxWidth': '700px',
                    'borderRadius': '12px',
                    'border': '1px solid #1f2937',
                    'boxShadow': '0 8px 24px rgba(0, 0, 0, 0.4)',
                    'backgroundColor': '#000'
                }
            )
        ]),

        html.Div(
            "▶ Explicación visual de cómo las bobinas transmisoras generan el pulso RF y las bobinas receptoras captan la señal de los espines.",
            style={'color': '#94a3b8', 'fontSize': '12px', 'textAlign': 'center', 'marginTop': '6px', 'fontStyle': 'italic'}
        )
    ]),
            html.Div(style=STYLE_CALLOUT, children=[
                html.Strong("En resumen: "),
                "la MRI convierte una propiedad cuántica invisible del núcleo atómico el espín en imágenes clínicas detalladas, combinando física, procesamiento de señales y reconstrucción matemática de imágenes."
            ]),
        ]),

        html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(260px, 1fr))', 'gap': '18px'}, children=[
            edu_card("Física de la MRI", "⚛️", [
                html.P("Espín del protón, campo B₀ y frecuencia de Larmor: el punto de partida de toda la resonancia magnética.", style={'color': '#cbd5e1', 'fontSize': '14px'}),
                dcc.Link("Explorar →", href='/fisica', style={'color': THEME['accent_cyan'], 'fontWeight': '700', 'fontSize': '13px', 'textDecoration': 'none'})
            ]),
            edu_card("Señal y Relajación", "📡", [
                html.P("Cómo se genera la señal de MRI y por qué los procesos T1 y T2 son la clave del contraste.", style={'color': '#cbd5e1', 'fontSize': '14px'}),
                dcc.Link("Explorar →", href='/senal-relajacion', style={'color': THEME['accent_cyan'], 'fontWeight': '700', 'fontSize': '13px', 'textDecoration': 'none'})
            ]),
            edu_card("Contraste en MRI", " ", [
                html.P("Cómo las combinaciones de TR y TE generan imágenes ponderadas en T1, T2 y Densidad Protónica.", style={'color': '#cbd5e1', 'fontSize': '14px'}),
                dcc.Link("Explorar →", href='/contraste', style={'color': THEME['accent_cyan'], 'fontWeight': '700', 'fontSize': '13px', 'textDecoration': 'none'})
            ]),
            edu_card("Espacio-k", "🌐", [
                html.P("El dominio de frecuencias donde se adquieren realmente los datos crudos de la imagen.", style={'color': '#cbd5e1', 'fontSize': '14px'}),
                dcc.Link("Explorar →", href='/espacio-k', style={'color': THEME['accent_cyan'], 'fontWeight': '700', 'fontSize': '13px', 'textDecoration': 'none'})
            ]),
            edu_card("Secuencias de Pulso", "⏱️", [
                html.P("Introducción a la secuencia Spin-Echo: pulso 90°, pulso 180°, TE y TR.", style={'color': '#cbd5e1', 'fontSize': '14px'}),
                dcc.Link("Explorar →", href='/secuencias', style={'color': THEME['accent_cyan'], 'fontWeight': '700', 'fontSize': '13px', 'textDecoration': 'none'})
            ]),
            edu_card("Simulación Interactiva", "🧪", [
                html.P("El simulador biomédico completo: ajusta parámetros en tiempo real y observa el efecto físico.", style={'color': '#cbd5e1', 'fontSize': '14px'}),
                dcc.Link("Abrir simulador →", href='/simulacion', style={'color': THEME['accent_emerald'], 'fontWeight': '700', 'fontSize': '13px', 'textDecoration': 'none'})
            ]),
        ]),
        html.Div(style={**STYLE_CALLOUT, 'marginTop': '22px'}, children=[
            html.Strong("Consejo: "),
            "En el texto de cada sección, los términos resaltados en ",
            html.Span("cian", style={'color': THEME['accent_cyan'], 'fontWeight': '700'}),
            " son clickeables - pulsa sobre ellos para abrir una explicación ampliada."
        ])
    ])


# ---- Página: Física de la MRI ----

def page_fisica():
    return html.Div(style=STYLE_CONTAINER, children=[
        page_header("1 · FÍSICA DE LA MRI", "Espín, Campo B₀ y Frecuencia de Larmor",
                    "Los tres pilares físicos sobre los que se construye toda la resonancia magnética."),

        edu_card("Espín del Protón", "", [
            html.P([
                "Cada protón de hidrógeno (abundante en el agua y la grasa del cuerpo) posee una propiedad cuántica llamada ",
                term("espín", "espin"),
                ", que se comporta como un pequeño imán girando. Sin un campo externo, estos diminutos imanes apuntan en direcciones aleatorias y su efecto neto se cancela."
            ], style={'color': '#e2e8f0', 'fontSize': '15px', 'lineHeight': '1.6'}),

            html.Div(className='edu-image-container', children=[
                html.Img(
                    src='/assets/espines.png',
                    className='edu-image',
                    style={'maxHeight': '200px', 'objectFit': 'contain', 'margin': '6px 0 6px 0'}
                )
            ]),
            html.Div("Representación de espines alineados bajo el campo B₀", className='img-caption'),
        ]),

        edu_card("Campo Magnético B₀", "", [
            html.P([
                "Al introducir al paciente en el escáner, se aplica un campo magnético intenso y estático llamado ",
                term("B₀", "b0"),
                ". Los espines se alinean (mayoritariamente) a favor de este campo, generando una magnetización neta M₀ a lo largo del eje longitudinal."
            ], style={'color': '#e2e8f0', 'fontSize': '15px', 'lineHeight': '1.6'}),
            html.Div(style=STYLE_CALLOUT, children=[
                "Equipos clínicos habituales: 1.5 T y 3 T. A mayor B₀, mayor señal disponible."
            ]),
        ]),

        edu_card("Frecuencia de Larmor", "📐", [
            html.P(["Bajo la influencia de B₀, los espines no permanecen estáticos: precesan (giran) alrededor del eje del campo a una frecuencia característica llamada ",
                term("frecuencia de Larmor", "larmor"),
                ", dada por ω₀ = γ · B₀."
            ], style={'color': '#e2e8f0', 'fontSize': '15px', 'lineHeight': '1.6'}),
            html.P("Esta es exactamente la frecuencia que debe emitir el pulso de radiofrecuencia para excitar los espines - es la condición de resonancia que da nombre a la técnica.",
                   style={'color': '#94a3b8', 'fontSize': '14px', 'lineHeight': '1.6'}),
                edu_card("Precesión y Frecuencia de Larmor", "🔄", [
            html.P(["La ", term("frecuencia de Larmor", "larmor"), 
                " determina la velocidad a la que los espines precesan alrededor del campo B₀. Esta frecuencia es única para cada núcleo y depende directamente de la intensidad del campo magnético."
            ], style={'color': '#e2e8f0', 'fontSize': '15px', 'lineHeight': '1.6'}),
            
            html.P(["El diagrama muestra cómo los espines de hidrógeno (protones) giran alrededor del eje del campo magnético principal. Cuanto mayor es B₀, mayor es la frecuencia de precesión."
            ], style={'color': '#94a3b8', 'fontSize': '14px', 'lineHeight': '1.6'}),

            html.Div(className='edu-image-container', children=[
                html.Img(
                    src='/assets/dLarmor.png',
                    className='edu-image',
                    style={'maxHeight': '300px', 'objectFit': 'contain', 'margin': '6px 0 6px 0'}
                )
            ]),
            html.Div("Diagrama de precesión de espines alrededor del campo B₀", className='img-caption'),]),
        ]),

        edu_card("Comparación de Campos Magnéticos Clínicos", "", [
            dcc.Graph(figure=fig_field_comparison(), config={'displayModeBar': False}),
            html.Div("A mayor intensidad de B₀, mayor es la frecuencia de Larmor necesaria para excitar los espines, y en general mayor la señal disponible (mejor SNR).", style=STYLE_GRAPH_DESC)
        ]),

        next_section_link('/senal-relajacion', 'Señal y Relajación'),
    ])


# ---- Página: Señal y Relajación ----

def page_senal_relajacion():
    return html.Div(style=STYLE_CONTAINER, children=[
        page_header("2 · SEÑAL Y RELAJACIÓN", "Cómo se genera la señal: procesos T1 y T2",
                    "Tras el pulso de excitación, los espines regresan al equilibrio siguiendo dos procesos simultáneos e independientes."),

        edu_card("Generación de la Señal (FID)", "📶", [
            html.P([
                "Un pulso de radiofrecuencia a la frecuencia de Larmor inclina la magnetización hacia el plano transversal. Al apagarse el pulso, esta magnetización en precesión induce una corriente oscilante en la bobina receptora: la señal ",
                term("FID", "fid"),
                " (Free Induction Decay)."
            ], style={'color': '#e2e8f0', 'fontSize': '15px', 'lineHeight': '1.6'}),
        ]),

        html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(320px, 1fr))', 'gap': '18px'}, children=[
            edu_card("Relajación T1 (Longitudinal)", "⬆️", [
                html.P([
                    "La relajación ", term("T1", "t1"),
                    " describe cómo Mz se recupera hacia M₀ tras el pulso, cediendo energía al entorno molecular. Tejidos con T1 corto (grasa) recuperan rápido y se ven brillantes en imágenes T1."
                ], style={'color': '#e2e8f0', 'fontSize': '14.5px', 'lineHeight': '1.6'}),
                html.P([
                    "El mecanismo físico detrás de este proceso se llama ",
                    term("relajación espín-red", "espinred"), "."
                ], style={'color': '#94a3b8', 'fontSize': '13.5px', 'lineHeight': '1.6'}),
            ]),
            edu_card("Relajación T2 (Transversal)", "↘️", [
                html.P([
                    "La relajación ", term("T2", "t2"),
                    " describe cómo Mxy decae por pérdida de coherencia entre espines vecinos. Fluidos libres (LCR) tienen T2 largo y se ven brillantes en imágenes T2."
                ], style={'color': '#e2e8f0', 'fontSize': '14.5px', 'lineHeight': '1.6'}),
                html.P([
                    "El mecanismo físico detrás de este proceso se llama ",
                    term("relajación espín-espín", "espinespin"), "."
                ], style={'color': '#94a3b8', 'fontSize': '13.5px', 'lineHeight': '1.6'}),
            ]),
        ]),

        edu_card("La Diferencia Fundamental entre T1 y T2", "💡", [
            html.P([
                "Aunque ambos procesos empiezan en el mismo instante (justo después del pulso de 90°), son ",
                html.Strong("físicamente independientes"),
                " y describen dos preguntas distintas sobre lo que le pasa a la magnetización:"
            ], style={'color': '#e2e8f0', 'fontSize': '14.5px', 'lineHeight': '1.7', 'marginBottom': '10px'}),

            html.Ul(style={'color': '#e2e8f0', 'fontSize': '14px', 'lineHeight': '1.8', 'paddingLeft': '20px', 'margin': '0 0 12px 0'}, children=[
                html.Li([html.Strong("T1 pregunta: ", style={'color': THEME['accent_cyan']}), "¿cuánta energía ha cedido el sistema de espines de vuelta al entorno molecular (la 'red')? Es un intercambio de ", html.Em("energía"), "."]),
                html.Li([html.Strong("T2 pregunta: ", style={'color': THEME['accent_emerald']}), "¿cuánto han perdido los espines la sincronía de fase entre ellos? Es una pérdida de ", html.Em("coherencia"), ", sin intercambio neto de energía con el entorno."]),
            ]),

            html.P([
                html.Strong("Por qué ocurren a escalas de tiempo distintas: "),
                "T2 siempre es igual o más corto que T1 en un mismo tejido porque desincronizarse (T2) es un proceso mucho más sensible y rápido que ceder energía térmicamente al entorno (T1). Basta con pequeñísimas diferencias de campo local entre espines vecinos para que pierdan la fase entre sí, mientras que transferir energía a la red molecular requiere un mecanismo de acoplamiento más lento."
            ], style={'color': '#e2e8f0', 'fontSize': '14px', 'lineHeight': '1.7', 'marginBottom': '10px'}),

            html.Div(style=STYLE_CALLOUT, children=[
                html.Strong("Analogía: "),
                "imagina un grupo de corredores que parten juntos de la línea de salida (justo tras el pulso de 90°, todos en fase). ",
                html.Strong("T2"),
                " es cuánto tardan en desordenarse entre sí y perder el paso sincronizado - basta con que unos corran ligeramente más rápido que otros por el terreno irregular (variaciones locales de campo). ",
                html.Strong("T1"),
                ", en cambio, es cuánto tardan en detenerse por completo y 'entregar' toda su energía cinética al terreno - un proceso de agotamiento mucho más lento y de otra naturaleza. Por eso los corredores se desordenan (T2) mucho antes de detenerse del todo (T1)."
            ]),

            html.P("Esta diferencia de mecanismos y de escalas de tiempo (T1: cientos a miles de ms; T2: decenas a cientos de ms) es precisamente lo que permite, eligiendo TR y TE, generar distintos tipos de contraste entre tejidos en la imagen final.",
                   style={'color': '#94a3b8', 'fontSize': '14px', 'lineHeight': '1.6', 'marginTop': '4px'}),
        ]),

        edu_card("Curvas Típicas de Relajación por Tejido", "📈", [
            dcc.Graph(figure=fig_static_relaxation_curves(), config={'displayModeBar': False}),
            html.Div("Nótese que el eje de tiempo de T1 (hasta 3000 ms) es diez veces más largo que el de T2 (hasta 300 ms): son procesos que ocurren a escalas completamente distintas.", style=STYLE_GRAPH_DESC)
        ]),

        next_section_link('/contraste', 'Contraste en MRI'),
    ])


# ---- Página: Contraste en MRI ----

def page_contraste():
    return html.Div(style=STYLE_CONTAINER, children=[
        page_header("3 · CONTRASTE EN MRI", "Imágenes T1, T2 y Densidad Protónica",
                    "El operador del escáner elige el contraste de la imagen ajustando dos parámetros de la secuencia de pulso."),

        edu_card("Los dos parámetros clave", "🎛️", [
            html.P([
                "El ", term("TR", "tr"), " (Tiempo de Repetición) controla cuánta recuperación T1 ocurre entre pulsos. El ",
                term("TE", "te"), " (Tiempo de Eco) controla cuánto decaimiento T2 ocurre antes de medir la señal."
            ], style={'color': '#e2e8f0', 'fontSize': '15px', 'lineHeight': '1.6'}),
        ]),

        # ===== TARJETAS DE PONDERACIONES (lado a lado) =====
        html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(240px, 1fr))', 'gap': '16px'}, children=[
            edu_card("Ponderación T1", "🟦", [
                html.P("TR corto + TE corto.", style={'color': THEME['accent_cyan'], 'fontWeight': '700', 'fontSize': '13px'}),
                html.P("Buena para anatomía. Grasa brillante, LCR oscuro.", style={'color': '#94a3b8', 'fontSize': '13.5px'}),
            ]),
            edu_card("Ponderación T2", "🟩", [
                html.P("TR largo + TE largo.", style={'color': THEME['accent_emerald'], 'fontWeight': '700', 'fontSize': '13px'}),
                html.P("Buena para edema/patología. LCR y líquidos brillantes.", style={'color': '#94a3b8', 'fontSize': '13.5px'}),
            ]),
            edu_card("Densidad Protónica (PD)", "🟨", [
                html.P([
                    "TR largo + TE corto - minimiza efectos de ", term("T1", "t1"), " y ", term("T2", "t2"), "."
                ], style={'color': THEME['accent_amber'], 'fontWeight': '700', 'fontSize': '13px'}),
                html.P([
                    "El contraste depende de la ", term("Densidad Protónica", "pd"), " del tejido."
                ], style={'color': '#94a3b8', 'fontSize': '13.5px'}),
            ]),
        ]),

        # ===== NUEVA TARJETA: Imagen de Contraste (propia) =====
        edu_card("Comparativa Visual de Ponderaciones T1, T2 y PD", "🖼️", [
            html.P([
                "La elección de ", term("TR", "tr"), " y ", term("TE", "te"), 
                " determina cómo se ven los diferentes tejidos en la imagen final."
            ], style={'color': '#e2e8f0', 'fontSize': '15px', 'lineHeight': '1.6'}),
            
            html.Ul(style={'color': '#e2e8f0', 'fontSize': '14px', 'lineHeight': '1.8', 'paddingLeft': '20px', 'margin': '0 0 14px 0'}, children=[
                html.Li([
                    html.Strong("T1 (TR corto, TE corto): ", style={'color': THEME['accent_cyan']}),
                    "La grasa y la sustancia blanca aparecen brillantes; el LCR y los fluidos aparecen oscuros. Ideal para evaluar anatomía."
                ]),
                html.Li([
                    html.Strong("T2 (TR largo, TE largo): ", style={'color': THEME['accent_emerald']}),
                    "Los fluidos y el LCR aparecen brillantes; la grasa aparece oscura. Ideal para detectar patologías y edemas."
                ]),
                html.Li([
                    html.Strong("Densidad Protónica (TR largo, TE corto): ", style={'color': THEME['accent_amber']}),
                    "Minimiza los efectos de T1 y T2; el contraste depende de la concentración de protones. Útil para evaluar cartílago."
                ]),
            ]),

            html.Div(className='edu-image-container', children=[
                html.Img(
                    src='/assets/contraste.png',
                    className='edu-image',
                    style={'maxHeight': '350px', 'objectFit': 'contain', 'margin': '6px 0 6px 0'}
                )
            ]),
            html.Div("Comparación de la misma región cerebral en T1, T2 y Densidad Protónica (PD)", className='img-caption'),
        ]),

        # ===== GRÁFICO DE INTENSIDAD (se mantiene igual) =====
        edu_card("Intensidad de Señal por Tejido y Ponderación", "📊", [
            dcc.Graph(figure=fig_contrast_bars(), config={'displayModeBar': False}),
            html.Div("Valores ilustrativos (no clínicos exactos) para mostrar el patrón general: la grasa domina en T1, el LCR domina en T2, y la Densidad Protónica ofrece un contraste intermedio y más uniforme entre tejidos.", style=STYLE_GRAPH_DESC)
        ]),

        html.Div(style={**STYLE_CALLOUT, 'marginTop': '4px'}, children=[
            "Puedes comparar estos tres contrastes en tiempo real, sobre casos clínicos reales, en la sección de ",
            dcc.Link("Simulación", href='/simulacion', style={'color': THEME['accent_cyan'], 'fontWeight': '700'}),
            "."
        ]),

        next_section_link('/espacio-k', 'Espacio-k'),
    ])


# ---- Página: Espacio-k ----

def page_kspace():
    return html.Div(style=STYLE_CONTAINER, children=[
        page_header("4 · ESPACIO-K", "Cómo se adquieren y procesan los datos",
                    "La imagen de MRI no se adquiere directamente: primero se llena una matriz de frecuencias espaciales."),

        edu_card("¿Qué es el Espacio-k?", "🌐", [
            html.P([
                "El ", term("espacio-k", "kspace"),
                " es el dominio de frecuencias espaciales donde el escáner almacena los datos crudos. Cada punto no representa un píxel: contiene información de una frecuencia espacial presente en toda la imagen."
            ], style={'color': '#e2e8f0', 'fontSize': '15px', 'lineHeight': '1.6'}),
        ]),

        # ===== DISPOSICIÓN EN DOS COLUMNAS: Centro + Periferia (izquierda) | Matriz (derecha) =====
        html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px', 'marginBottom': '22px', 'alignItems': 'stretch'}, children=[
            
            # ===== COLUMNA IZQUIERDA: dos tarjetas apiladas verticalmente
            html.Div(style={'flex': '1', 'minWidth': '280px', 'display': 'flex', 'flexDirection': 'column', 'gap': '18px'}, children=[
                edu_card("Centro del Espacio-k", "🎯", [
                    html.P(
                        "Bajas frecuencias espaciales → determinan el contraste global de la imagen.",
                        style={'color': '#e2e8f0', 'fontSize': '14px', 'lineHeight': '1.6'}
                    ),
                    html.Div(style=STYLE_CALLOUT, children=[
                        "El centro del espacio-k contiene la información de brillo y contraste general de la imagen."
                    ]),
                ]),
                edu_card("Periferia del Espacio-k", "🔲", [
                    html.P(
                        "Altas frecuencias espaciales → determinan bordes y detalles finos.",
                        style={'color': '#e2e8f0', 'fontSize': '14px', 'lineHeight': '1.6'}
                    ),
                    html.Div(style=STYLE_CALLOUT, children=[
                        "La periferia del espacio-k contiene la información de bordes y detalles anatómicos."
                    ]),
                ]),
            ]),
            
            # ===== COLUMNA DERECHA
            html.Div(style={'flex': '1.5', 'minWidth': '320px', 'display': 'flex'}, children=[
                edu_card("Visualización de la Matriz de Espacio-k", "🗺️", [
                    html.P(
                        "La matriz completa del espacio-k combina la información de todas las frecuencias espaciales. "
                        "La región central (ámbar) define el contraste global, mientras que la periferia (cian) define los bordes y detalles.",
                        style={'color': '#e2e8f0', 'fontSize': '14px', 'lineHeight': '1.6'}
                    ),
                    dcc.Graph(figure=fig_kspace_matrix(), config={'displayModeBar': False}),
                ]),
            ]),
        ]),

        # ===== TARJETA: Trayectorias de Adquisición (kspace.png)
        edu_card("Trayectorias de Adquisición en el Espacio-k", "🗺️", [
            html.P([
                "El ", term("espacio-k", "kspace"), 
                " se puede llenar siguiendo diferentes patrones o trayectorias. La imagen muestra las cuatro formas más comunes en las que el escáner adquiere los datos."
            ], style={'color': '#e2e8f0', 'fontSize': '15px', 'lineHeight': '1.6'}),
            
            html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(140px, 1fr))', 'gap': '12px', 'margin': '12px 0'}, children=[
                html.Div([
                    html.Strong("Cartesiano", style={'color': THEME['accent_cyan']}),
                    html.P("Fila por fila. El método tradicional, aún el más usado.", style={'color': '#94a3b8', 'fontSize': '12px'})
                ]),
                html.Div([
                    html.Strong("Espiral", style={'color': THEME['accent_emerald']}),
                    html.P("Desde el centro hacia afuera. Rápido y eficiente.", style={'color': '#94a3b8', 'fontSize': '12px'})
                ]),
                html.Div([
                    html.Strong("Radial", style={'color': THEME['accent_amber']}),
                    html.P("Líneas que atraviesan el centro. Robusto ante movimiento.", style={'color': '#94a3b8', 'fontSize': '12px'})
                ]),
                html.Div([
                    html.Strong("Zig-Zag", style={'color': '#a855f7'}),
                    html.P("Patrón ondulado. Usado en secuencias rápidas.", style={'color': '#94a3b8', 'fontSize': '12px'})
                ]),
            ]),

            html.Div(className='edu-image-container', children=[
                html.Img(
                    src='/assets/kspace.png',
                    className='edu-image',
                    style={'maxHeight': '320px', 'objectFit': 'contain', 'margin': '6px 0 6px 0'}
                )
            ]),
            html.Div("Trayectorias de adquisición en el espacio-k: Cartesiano, Espiral, Radial y Zig-Zag", className='img-caption'),
        ]),

        # ===== TARJETA: Parámetros Físicos (kspaceMRI.png) =====
        edu_card("Visualización de Parámetros Físicos desde el Espacio-k", "📊", [
            html.P([
                "La imagen muestra cómo diferentes parámetros físicos pueden ser extraídos y visualizados a partir de la información contenida en el espacio-k."
            ], style={'color': '#e2e8f0', 'fontSize': '15px', 'lineHeight': '1.6'}),
            
            html.P([
                html.Strong("Los parámetros representados son:"),
            ], style={'color': '#94a3b8', 'fontSize': '14px', 'lineHeight': '1.8'}),
            
            html.Ul(style={'color': '#e2e8f0', 'fontSize': '14px', 'lineHeight': '1.8', 'paddingLeft': '20px', 'margin': '0 0 14px 0'}, children=[
                html.Li([
                    html.Strong("Desplazamiento normal: ", style={'color': THEME['accent_cyan']}),
                    "Medición del movimiento tisular (elástico) en respuesta a la excitación."
                ]),
                html.Li([
                    html.Strong("Intensidad estructural: ", style={'color': THEME['accent_emerald']}),
                    "Refleja la densidad y organización de los tejidos."
                ]),
                html.Li([
                    html.Strong("Potencia inyectada: ", style={'color': THEME['accent_amber']}),
                    "Energía depositada por los pulsos de RF en el tejido."
                ]),
                html.Li([
                    html.Strong("Intensidad acústica: ", style={'color': '#a855f7'}),
                    "Medición de las ondas sonoras generadas por la excitación del tejido."
                ]),
            ]),

            html.Div(className='edu-image-container', children=[
                html.Img(
                    src='/assets/kspaceMRI.png',
                    className='edu-image',
                    style={'maxHeight': '300px', 'objectFit': 'contain', 'margin': '6px 0 6px 0'}
                )
            ]),
            html.Div("Distribución de parámetros físicos (desplazamiento, intensidad, potencia y acústica) desde datos de MRI", className='img-caption'),
        ]),

        edu_card("De vuelta a la imagen: la iFFT2", "↔️", [
            html.P("La imagen final se reconstruye aplicando una Transformada Inversa de Fourier 2D (iFFT2) sobre los datos del espacio-k completo.",
                   style={'color': '#e2e8f0', 'fontSize': '14.5px', 'lineHeight': '1.6'}),
        ]),

        edu_card("Submuestreo y Aliasing", "⚠️", [
            html.P([
                "Para acelerar la adquisición, a veces se omiten líneas del espacio-k (factor de aceleración R). Si el muestreo cae por debajo del límite de Nyquist, aparece el fenómeno de ",
                term("aliasing", "aliasing"), ": un solapamiento visible de la imagen reconstruida."
            ], style={'color': '#e2e8f0', 'fontSize': '14.5px', 'lineHeight': '1.6'}),
        ]),

        next_section_link('/secuencias', 'Secuencias de Pulso'),
    ])

# ---- Página: Secuencias de Pulso 

def _timeline_marker(label, sublabel, color, left_pct):
    return html.Div(style={
        'position': 'absolute', 'left': f'{left_pct}%', 'top': '0', 'transform': 'translateX(-50%)',
        'textAlign': 'center'
    }, children=[
        html.Div(style={'width': '12px', 'height': '12px', 'borderRadius': '50%', 'backgroundColor': color,
                         'margin': '0 auto 6px auto', 'boxShadow': f'0 0 8px {color}'}),
        html.Div(label, style={'fontSize': '12.5px', 'fontWeight': '700', 'color': color}),
        html.Div(sublabel, style={'fontSize': '10.5px', 'color': '#94a3b8'}),
    ])


def page_secuencias():
    return html.Div(style=STYLE_CONTAINER, children=[
        page_header("5 · SECUENCIAS DE PULSO", "Introducción a la Secuencia Spin-Echo",
                    "Una secuencia de pulso define el orden y el momento exacto en que se aplican pulsos de RF y se mide la señal."),

        edu_card("La secuencia Spin-Echo", "🧲", [
            html.P([
                "La ", term("secuencia Spin-Echo", "spinecho"),
                " combina un pulso de excitación de 90° con un pulso de re-fasamiento de 180°, aplicado en TE/2, que corrige el desfase producido por inhomogeneidades del campo B₀."
            ], style={'color': '#e2e8f0', 'fontSize': '15px', 'lineHeight': '1.6'}),
        ]),

                edu_card("Secuencia PRESS: Espectroscopía por Resonancia Magnética", "🧪", [
            html.P([
                "Además de las secuencias de imagen, existen secuencias diseñadas para medir la composición química de los tejidos. La secuencia ", 
                html.Strong("PRESS", style={'color': THEME['accent_cyan']}),
                " (Point RESolved Spectroscopy) es una de las más utilizadas en espectroscopía por resonancia magnética."
            ], style={'color': '#e2e8f0', 'fontSize': '15px', 'lineHeight': '1.6'}),

            html.P([
                "El diagrama muestra la línea de tiempo de los pulsos de RF y los gradientes necesarios para seleccionar un volumen específico de tejido y generar un eco de señal."
            ], style={'color': '#94a3b8', 'fontSize': '14px', 'lineHeight': '1.6'}),

            html.Div(className='edu-image-container', children=[
                html.Img(
                    src='/assets/secuencia.png',
                    className='edu-image',
                    style={'maxHeight': '320px', 'objectFit': 'contain', 'margin': '6px 0 6px 0'}
                )
            ]),
            html.Div("Diagrama de la secuencia PRESS para espectroscopía por resonancia magnética", className='img-caption'),
            
            html.Div(style=STYLE_CALLOUT, children=[
                html.Strong("Aplicación clínica: "),
                "La espectroscopía PRESS se usa para detectar metabolitos en el cerebro y ayudar en el diagnóstico de tumores, enfermedades metabólicas y epilepsia."
            ]),
        ]),

        edu_card("Línea de tiempo de la secuencia", "⏱️", [
            html.Div(style={'position': 'relative', 'height': '90px', 'margin': '10px 30px 0 30px'}, children=[
                html.Div(style={'position': 'absolute', 'top': '5px', 'left': '0', 'right': '0', 'height': '2px', 'backgroundColor': '#334155'}),
                _timeline_marker("Pulso 90°", "excitación (t=0)", THEME['accent_cyan'], 5),
                _timeline_marker("Pulso 180°", "re-fasamiento (t=TE/2)", THEME['accent_amber'], 50),
                _timeline_marker("Eco", "medición (t=TE)", THEME['accent_emerald'], 95),
            ]),
            html.P([
                "Componentes clave: el ", term("Flip Angle", "flipangle"),
                " del pulso inicial (90°), el ", term("TE", "te"),
                " (instante de medición del eco) y el ", term("TR", "tr"),
                " (tiempo entre repeticiones sucesivas de la secuencia)."
            ], style={'color': '#94a3b8', 'fontSize': '13.5px', 'lineHeight': '1.6', 'marginTop': '18px'}),
        ]),

        html.Div(style=STYLE_CALLOUT, children=[
            "Puedes ver este mismo diagrama actualizarse en tiempo real, junto al resto del pipeline de adquisición, en la sección de ",
            dcc.Link("Simulación", href='/simulacion', style={'color': THEME['accent_cyan'], 'fontWeight': '700'}),
            "."
        ]),

        html.Div(style={'marginTop': '10px', 'marginBottom': '30px'}, children=[
            dcc.Link("Ir al Simulador →", href='/simulacion', className='nav-cta-btn')
        ]),
    ])


# ===== ESTILOS PARA ACTIVIDADES 
STYLE_QUIZ_CARD = {
    'backgroundColor': THEME['card'],
    'borderRadius': '16px',
    'padding': '24px 28px',
    'border': '1px solid #ffffff',
    'boxShadow': '0 4px 14px rgba(0, 0, 0, 0.35)',
    'marginBottom': '18px',
}

STYLE_OPTION_BUTTON = {
    'backgroundColor': '#1e293b',
    'border': '1px solid #334155',
    'borderRadius': '10px',
    'padding': '12px 18px',
    'color': '#e2e8f0',
    'cursor': 'pointer',
    'width': '100%',
    'textAlign': 'left',
    'fontSize': '14px',
    'transition': 'all 0.2s ease',
}

STYLE_OPTION_SELECTED = {
    'backgroundColor': '#1e293b',
    'border': '2px solid #38bdf8',
    'borderRadius': '10px',
    'padding': '12px 18px',
    'color': '#ffffff',
    'cursor': 'pointer',
    'width': '100%',
    'textAlign': 'left',
    'fontSize': '14px',
    'boxShadow': '0 0 15px rgba(56, 189, 248, 0.15)',
}

# ===== FUNCIONES PARA GENERAR PREGUNTAS CON GEMINI =====
def generar_preguntas_con_gemini(tema="resonancia magnética MRI", cantidad=5):
    """Genera preguntas de opción múltiple usando Gemini API, basadas en contenido educativo de MRI."""
    try:
        prompt = f"""
        Eres un experto en física de resonancia magnética (MRI) y educación en ingeniería biomédica.
        Tu tarea es generar {cantidad} preguntas de opción múltiple para estudiantes universitarios.
        Las preguntas deben basarse en los siguientes conceptos clave que se enseñan en un curso de MRI:
        
        **CONCEPTOS FUNDAMENTALES (del libro electrónico de MRI):**
        1. Espín del protón y campo magnético B₀
        2. Frecuencia de Larmor y precesión
        3. Relajación T1 (recuperación longitudinal) y T2 (decaimiento transversal)
        4. Ponderaciones T1, T2 y Densidad Protónica (PD)
        5. Espacio-k: centro (bajas frecuencias = contraste) y periferia (altas frecuencias = bordes)
        6. Secuencia Spin-Echo: pulso 90°, pulso 180°, TE y TR
        7. FID (Free Induction Decay) y procesamiento de señales
        8. Adquisición y reconstrucción de imágenes (iFFT)
        9. Aliasing y submuestreo (factor R)
        10. Aplicaciones clínicas de la MRI
        
        **REFERENCIA ESPECÍFICA SOBRE ESPACIO-K:**
        Según MRI Questions (https://mriquestions.com/big-spot-in-middle.html):
        - La región central del espacio-k es brillante porque contiene las frecuencias espaciales bajas
        - Estas frecuencias representan: nivel general de señal, estructuras anatómicas grandes y contraste
        - La periferia del espacio-k contiene altas frecuencias: detalles finos, bordes nítidos y estructuras pequeñas
        - La fila central (ky=0) se adquiere sin gradiente de fase neto → señales se suman coherentemente
        - La columna central (kx=0) se muestrea en el pico del eco → señal máxima
        - El centro del espacio-k determina: brillo general, contraste y relación señal-ruido (SNR)
        - La periferia es esencial para: resolución espacial y definición de bordes
        
        **INSTRUCCIONES PARA LAS PREGUNTAS:**
        1. Cada pregunta debe ser clara, educativa y relevante para la ingeniería biomédica
        2. Incluir 4 opciones (A, B, C, D) con una sola correcta
        3. Las opciones incorrectas deben ser plausibles pero claramente erróneas
        4. Incluir preguntas que conecten la teoría con aplicaciones clínicas
        5. Variar la dificultad: algunas fáciles, otras más desafiantes
        
        **FORMATO DE SALIDA (JSON):**
        [
            {{
                "pregunta": "¿Cuál es la principal función del centro del espacio-k en la formación de una imagen MRI?",
                "opciones": {{
                    "A": "Define el contraste y brillo general de la imagen",
                    "B": "Define los bordes y detalles finos",
                    "C": "Elimina el ruido de la imagen",
                    "D": "Acelera el tiempo de adquisición"
                }},
                "correcta": "A"
            }},
            {{
                "pregunta": "¿Qué frecuencia de Larmor corresponde a un campo B₀ de 1.5 Tesla?",
                "opciones": {{
                    "A": "21.29 MHz",
                    "B": "42.58 MHz",
                    "C": "63.87 MHz",
                    "D": "127.74 MHz"
                }},
                "correcta": "C"
            }}
        ]
        
        **IMPORTANTE:**
        - Devuelve SOLO el JSON, sin texto adicional
        - No uses markdown, solo el JSON puro
        - Asegúrate de que las preguntas sean precisas y estén bien redactadas
        - Incluye al menos 2 preguntas sobre el espacio-k y su centro/periferia
        """
        
        response = model.generate_content(prompt)
        texto = response.text.strip()
        
        # Limpiar la respuesta 
        if texto.startswith('```json'):
            texto = texto[7:]
        if texto.startswith('```'):
            texto = texto[3:]
        if texto.endswith('```'):
            texto = texto[:-3]
        
        preguntas = json.loads(texto)
        return preguntas
        
    except Exception as e:
        print(f"⚠️ Error: {e}. Usando preguntas de respaldo.")
        return generar_preguntas_fallback(cantidad)


def generar_preguntas_fallback(cantidad=5):
    """Preguntas predefinidas como respaldo."""
    banco_preguntas = [
        {
            "pregunta": "¿Qué es la frecuencia de Larmor?",
            "opciones": {
                "A": "La frecuencia de precesión de los espines alrededor de B₀",
                "B": "La frecuencia del pulso de radiofrecuencia",
                "C": "La frecuencia de relajación T1",
                "D": "La frecuencia de la imagen final"
            },
            "correcta": "A"
        },
        {
            "pregunta": "¿Qué parámetros definen la ponderación T1 en MRI?",
            "opciones": {
                "A": "TR corto, TE corto",
                "B": "TR largo, TE largo",
                "C": "TR largo, TE corto",
                "D": "TR corto, TE largo"
            },
            "correcta": "A"
        },
        {
            "pregunta": "¿Qué es el espacio-k en MRI?",
            "opciones": {
                "A": "El dominio de frecuencias espaciales",
                "B": "La imagen final reconstruida",
                "C": "El campo magnético principal",
                "D": "La señal FID"
            },
            "correcta": "A"
        },
        {
            "pregunta": "¿Cuál es la función del pulso de 180° en una secuencia Spin-Echo?",
            "opciones": {
                "A": "Refasar los espines para formar un eco",
                "B": "Excitar los espines por primera vez",
                "C": "Eliminar la señal FID",
                "D": "Aumentar el campo B₀"
            },
            "correcta": "A"
        },
        {
            "pregunta": "¿Qué tejido tiene T1 más corto (recupera más rápido)?",
            "opciones": {
                "A": "Grasa",
                "B": "Líquido cefalorraquídeo (LCR)",
                "C": "Sustancia gris",
                "D": "Músculo"
            },
            "correcta": "A"
        },
        {
            "pregunta": "¿Qué es la FID (Free Induction Decay)?",
            "opciones": {
                "A": "La señal que decae después del pulso de excitación",
                "B": "La imagen final reconstruida",
                "C": "El campo magnético principal",
                "D": "El pulso de 180°"
            },
            "correcta": "A"
        },
        {
            "pregunta": "¿Cuál es la ventaja de la MRI sobre la radiografía?",
            "opciones": {
                "A": "No usa radiación ionizante",
                "B": "Es más rápida",
                "C": "Es más económica",
                "D": "Tiene mejor resolución en hueso"
            },
            "correcta": "A"
        },
        {
            "pregunta": "¿Qué técnica se usa para acelerar la adquisición en MRI?",
            "opciones": {
                "A": "Submuestreo del espacio-k (R>1)",
                "B": "Aumentar TE",
                "C": "Reducir TR",
                "D": "Usar un campo B₀ más bajo"
            },
            "correcta": "A"
        },
    ]
    
    seleccionadas = random.sample(banco_preguntas, min(cantidad, len(banco_preguntas)))
    return seleccionadas

# ===== PÁGINA DE ACTIVIDADES =====

def page_actividades():
    return html.Div(style=STYLE_CONTAINER, children=[
        page_header(
            "ACTIVIDADES INTERACTIVAS",
            "Prueba de conocimientos de MRI",
            "Responde las preguntas y recibe tu retroalimentación."
        ),
        
        html.Div(style={'display': 'flex', 'gap': '15px', 'marginBottom': '20px', 'flexWrap': 'wrap'}, children=[
            html.Button(
                "🔄 Generar Nuevas Preguntas",
                id='btn-generar-preguntas',
                n_clicks=0,
                className='nav-cta-btn',
                style={'background': 'linear-gradient(135deg, #0369a1, #0284c7)'}
            ),
            html.Button(
                "📤 Enviar Resultados",
                id='btn-enviar-resultados',
                n_clicks=0,
                className='nav-cta-btn',
                style={'background': 'linear-gradient(135deg, #065f46, #059669)'}
            ),
        ]),
        
        # ===== INFORMACIÓN DEL ESTUDIANTE =====
        html.Div(style=STYLE_QUIZ_CARD, children=[
            html.H4("Datos del Estudiante", style={'color': '#f8fafc', 'marginBottom': '12px', 'fontSize': '16px'}),
            html.Div(style={'display': 'flex', 'gap': '15px', 'flexWrap': 'wrap'}, children=[
                html.Div(style={'flex': '1', 'minWidth': '200px'}, children=[
                    html.Label("Nombre completo:", style={'color': '#94a3b8', 'fontSize': '13px', 'display': 'block', 'marginBottom': '4px'}),
                    dcc.Input(
                        id='input-nombre-estudiante',
                        type='text',
                        placeholder='Ingresa tu nombre',
                        style={
                            'width': '100%',
                            'padding': '10px 14px',
                            'backgroundColor': '#1e293b',
                            'border': '1px solid #334155',
                            'borderRadius': '8px',
                            'color': '#ffffff',
                            'fontSize': '14px'
                        }
                    )
                ]),
                html.Div(style={'flex': '1', 'minWidth': '200px'}, children=[
                    html.Label("Correo electrónico:", style={'color': '#94a3b8', 'fontSize': '13px', 'display': 'block', 'marginBottom': '4px'}),
                    dcc.Input(
                        id='input-correo-estudiante',
                        type='email',
                        placeholder='ejemplo@universidad.edu',
                        style={
                            'width': '100%',
                            'padding': '10px 14px',
                            'backgroundColor': '#1e293b',
                            'border': '1px solid #334155',
                            'borderRadius': '8px',
                            'color': '#ffffff',
                            'fontSize': '14px'
                        }
                    )
                ]),
            ]),
        ]),
        
        # ===== CONTENEDOR DE PREGUNTAS =====
        html.Div(id='container-preguntas', children=[
            html.Div(style=STYLE_QUIZ_CARD, children=[
                html.P("Presiona 'Generar Nuevas Preguntas' para comenzar.", 
                       style={'color': '#94a3b8', 'textAlign': 'center', 'fontSize': '16px'})
            ])
        ]),
        
        # ===== ÁREA DE TAREAS CON SUBIDA DE ARCHIVOS =====
        html.Div(style=STYLE_QUIZ_CARD, children=[
            html.H4("📝 Subir Tarea", style={'color': '#f8fafc', 'marginBottom': '12px', 'fontSize': '16px'}),
            html.P("Sube tu tarea en formato PDF, Word, o imagen. El archivo se guardará en la carpeta de tareas.", 
                   style={'color': '#94a3b8', 'fontSize': '13px', 'marginBottom': '12px'}),
            
            html.Div(style={'display': 'flex', 'gap': '15px', 'flexWrap': 'wrap'}, children=[
                html.Div(style={'flex': '1', 'minWidth': '200px'}, children=[
                    html.Label("Nombre completo:", style={'color': '#94a3b8', 'fontSize': '13px', 'display': 'block', 'marginBottom': '4px'}),
                    dcc.Input(
                        id='input-nombre-tarea',
                        type='text',
                        placeholder='Tu nombre completo',
                        style={
                            'width': '100%',
                            'padding': '10px 14px',
                            'backgroundColor': '#1e293b',
                            'border': '1px solid #334155',
                            'borderRadius': '8px',
                            'color': '#ffffff',
                            'fontSize': '14px'
                        }
                    )
                ]),
                html.Div(style={'flex': '1', 'minWidth': '200px'}, children=[
                    html.Label("Asignatura / Curso:", style={'color': '#94a3b8', 'fontSize': '13px', 'display': 'block', 'marginBottom': '4px'}),
                    dcc.Input(
                        id='input-curso-tarea',
                        type='text',
                        placeholder='Ej: Imagenología Médica',
                        style={
                            'width': '100%',
                            'padding': '10px 14px',
                            'backgroundColor': '#1e293b',
                            'border': '1px solid #334155',
                            'borderRadius': '8px',
                            'color': '#ffffff',
                            'fontSize': '14px'
                        }
                    )
                ]),
            ]),
            
            html.Div(style={'display': 'flex', 'gap': '15px', 'flexWrap': 'wrap', 'marginTop': '12px'}, children=[
                html.Div(style={'flex': '2', 'minWidth': '250px'}, children=[
                    html.Label("Descripción de la tarea:", style={'color': '#94a3b8', 'fontSize': '13px', 'display': 'block', 'marginBottom': '4px'}),
                    dcc.Textarea(
                        id='textarea-tarea-descripcion',
                        placeholder='Describe brevemente tu tarea...',
                        style={
                            'width': '100%',
                            'minHeight': '80px',
                            'padding': '12px 14px',
                            'backgroundColor': '#1e293b',
                            'border': '1px solid #334155',
                            'borderRadius': '8px',
                            'color': '#e2e8f0',
                            'fontSize': '14px',
                            'resize': 'vertical'
                        }
                    )
                ]),
                
                html.Div(style={'flex': '1', 'minWidth': '200px'}, children=[
                    html.Label("Archivo de la tarea:", style={'color': '#94a3b8', 'fontSize': '13px', 'display': 'block', 'marginBottom': '4px'}),
                    dcc.Upload(
                        id='upload-tarea-archivo',
                        children=html.Div([
                            html.I(className="fa-solid fa-cloud-arrow-up", style={'fontSize': '24px', 'color': '#38bdf8', 'display': 'block', 'marginBottom': '6px'}),
                            html.Div([
                                "📎 Arrastra o ",
                                html.Span("haz clic para subir", style={'color': '#38bdf8', 'textDecoration': 'underline'}),
                                " tu archivo"
                            ])
                        ], style={
                            'border': '2px dashed #334155',
                            'borderRadius': '8px',
                            'padding': '24px 20px',
                            'textAlign': 'center',
                            'color': '#94a3b8',
                            'fontSize': '13px',
                            'cursor': 'pointer',
                            'height': '100%',
                            'minHeight': '80px',
                            'display': 'flex',
                            'flexDirection': 'column',
                            'alignItems': 'center',
                            'justifyContent': 'center'
                        }),
                        multiple=False
                    ),
                    html.Div(id='upload-tarea-info', style={'color': '#38bdf8', 'fontSize': '12px', 'marginTop': '6px'})
                ]),
            ]),
            
            html.Div(style={'marginTop': '16px', 'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'alignItems': 'center'}, children=[
                html.Button(
                    "📤 Enviar Tarea",
                    id='btn-enviar-tarea',
                    n_clicks=0,
                    className='nav-cta-btn',
                    style={'background': 'linear-gradient(135deg, #7c3aed, #8b5cf6)'}
                ),
                html.Div(id='mensaje-tarea-estado', style={'color': '#10b981', 'fontSize': '14px', 'fontWeight': '600'})
            ]),
            
            html.Hr(style={'borderColor': '#374151', 'margin': '18px 0 12px 0'}),
            html.Div([
                html.Span("📂 Tareas enviadas recientemente: ", style={'color': '#94a3b8', 'fontSize': '13px'}),
                html.Span(id='contador-tareas', style={'color': '#38bdf8', 'fontSize': '13px', 'fontWeight': '700'})
            ]),
        ]),
        
        # ===== MODAL DE RESULTADOS =====
        html.Div(id='modal-resultados', style=STYLE_MODAL_OVERLAY_HIDDEN, children=[
            html.Div(style=STYLE_MODAL_BOX, children=[
                html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'flex-start', 'marginBottom': '14px'}, children=[
                    html.H3("📊 Resultados del Quiz", style={'margin': '0', 'color': '#f8fafc', 'fontSize': '20px', 'fontWeight': '800'}),
                    html.Button("✕", id='btn-cerrar-resultados', n_clicks=0, className='modal-close-btn')
                ]),
                html.Div(id='modal-resultados-body')
            ])
        ]),
        
        # ===== STORE PARA GUARDAR ESTADO =====
        dcc.Store(id='store-preguntas', data=[]),
        dcc.Store(id='store-respuestas', data={}),
        dcc.Store(id='store-resultados', data={}),
    ])


STYLE_CONTROL_CARD = {
    'backgroundColor': THEME['card'],
    'borderRadius': '16px',
    'padding': '20px 22px',
    'border': '1px solid #1f2937',
    'boxShadow': '0 4px 14px rgba(0, 0, 0, 0.35)',
    'marginBottom': '18px',
}

STYLE_CONTROL_INPUT = {
    'width': '100%',
    'padding': '9px 12px',
    'backgroundColor': '#1e293b',
    'border': '1px solid #334155',
    'borderRadius': '8px',
    'color': '#e2e8f0',
    'fontSize': '14px',
    'fontFamily': '"Fira Code", "Consolas", monospace',
    'boxSizing': 'border-box',
}

STYLE_MINI_BUTTON = {
    'background': 'linear-gradient(135deg, #0369a1, #0284c7)',
    'color': '#ffffff',
    'border': 'none',
    'padding': '7px 14px',
    'borderRadius': '8px',
    'fontWeight': '700',
    'fontSize': '12px',
    'cursor': 'pointer',
}

STYLE_MINI_BUTTON_ALT = {**STYLE_MINI_BUTTON, 'background': 'linear-gradient(135deg, #7c3aed, #8b5cf6)'}
STYLE_MINI_BUTTON_GREEN = {**STYLE_MINI_BUTTON, 'background': 'linear-gradient(135deg, #065f46, #059669)'}
STYLE_MINI_BUTTON_AMBER = {**STYLE_MINI_BUTTON, 'background': 'linear-gradient(135deg, #b45309, #d97706)'}

STYLE_ERROR_BOX = {
    'backgroundColor': 'rgba(239,68,68,0.12)',
    'border': '1px solid rgba(239,68,68,0.4)',
    'borderRadius': '8px',
    'padding': '8px 12px',
    'color': '#fca5a5',
    'fontSize': '12.5px',
    'marginTop': '8px',
}

STYLE_ADVANCED_HEADER_BTN = {
    'width': '100%',
    'background': 'linear-gradient(90deg, #1e293b 0%, #0f172a 100%)',
    'border': '1px solid #334155',
    'borderRadius': '10px',
    'padding': '14px 18px',
    'color': '#e2e8f0',
    'fontWeight': '700',
    'fontSize': '15px',
    'cursor': 'pointer',
    'display': 'flex',
    'justifyContent': 'space-between',
    'alignItems': 'center',
}

STYLE_ADVANCED_PANEL_HIDDEN = {'display': 'none'}
STYLE_ADVANCED_PANEL_VISIBLE = {
    'display': 'block',
    'marginTop': '12px',
    'padding': '18px',
    'backgroundColor': '#0d1424',
    'border': '1px solid #1f2937',
    'borderRadius': '12px',
}

STYLE_MARGIN_PILL_OK = {
    'backgroundColor': 'rgba(16,185,129,0.15)', 'color': '#10b981',
    'padding': '4px 10px', 'borderRadius': '8px', 'fontWeight': '700', 'fontSize': '12px'
}
STYLE_MARGIN_PILL_BAD = {
    'backgroundColor': 'rgba(239,68,68,0.15)', 'color': '#f87171',
    'padding': '4px 10px', 'borderRadius': '8px', 'fontWeight': '700', 'fontSize': '12px'
}

# 1. PARSEO DE FUNCIONES DE TRANSFERENCIA
# =============================================================================

def parse_transfer_function(tf_str):
    """
    Parsea una función de transferencia escrita como texto, por ejemplo:
        '1/(s^2 + 0.5*s + 1)'
        '1/(s^3 + 2*s^2 + 2*s + 1)'
        '(s + 1)/(s^2 + 0.3*s + 1)'
        '1/(s^2 - 1)'
        '2*e^(-0.5*s)/(2*s + 1)'
        '1/((s^2 + 0.2*s + 4)*(s^2 + 0.1*s + 1))'

    Retorna un dict:
        {
          'num': [coefs numerador, mayor grado primero],
          'den': [coefs denominador, mayor grado primero],
          'delay': retardo T en segundos (float, 0 si no hay),
          'sys': scipy.signal.TransferFunction (sin el retardo, que se
                 aplica luego como desplazamiento temporal),
          'tf_str': el string original,
          'error': None si todo OK, o un mensaje de error legible
        }
    """
    result = {'num': None, 'den': None, 'delay': 0.0, 'sys': None,
              'tf_str': tf_str, 'error': None}

    if not tf_str or not tf_str.strip():
        result['error'] = "Ingresa una función de transferencia."
        return result

    if not _SYMPY_OK:
        result['error'] = "Falta el paquete 'sympy' (pip install sympy) para parsear la TF."
        return result

    try:
        s = sp.symbols('s')
        expr_str = tf_str.strip().replace('^', '**')

        # --- Extraer un factor de retardo puro tipo e^(-T*s) / exp(-T*s) ---
        delay = 0.0
        delay_pattern = re.compile(
            r'(?:e|exp)\s*\*\*\s*\(\s*-\s*([0-9]*\.?[0-9]+)\s*\*?\s*s\s*\)'
        )
        m = delay_pattern.search(expr_str)
        if m:
            delay = float(m.group(1))
            expr_str = delay_pattern.sub('1', expr_str)

        expr = sp.sympify(expr_str, locals={'s': s})
        expr = sp.together(sp.simplify(expr))
        numer, denom = sp.fraction(expr)

        num_poly = sp.Poly(sp.expand(numer), s)
        den_poly = sp.Poly(sp.expand(denom), s)

        num_coeffs = [float(c) for c in num_poly.all_coeffs()]
        den_coeffs = [float(c) for c in den_poly.all_coeffs()]

        if all(abs(c) < 1e-14 for c in den_coeffs):
            result['error'] = "El denominador de la función de transferencia es nulo."
            return result

        result['num'] = num_coeffs
        result['den'] = den_coeffs
        result['delay'] = delay
        result['sys'] = signal.TransferFunction(num_coeffs, den_coeffs)

    except Exception as e:
        result['error'] = f"No se pudo interpretar la función de transferencia: {e}"

    return result


def _poly_terms_str(coeffs):
    """Convierte coeficientes de polinomio (mayor grado primero) en texto legible: '1.00s² + 0.30s + 1.00'."""
    sup = {2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶'}
    n = len(coeffs)
    parts = []
    for i, c in enumerate(coeffs):
        power = n - 1 - i
        if abs(c) < 1e-10:
            continue
        sign = '-' if c < 0 else '+'
        mag = abs(c)
        if power == 0:
            body = f"{mag:.3g}"
        elif power == 1:
            body = "s" if abs(mag - 1) < 1e-9 else f"{mag:.3g}s"
        else:
            base = "s" if abs(mag - 1) < 1e-9 else f"{mag:.3g}s"
            body = base + sup.get(power, f"^{power}")
        parts.append((sign, body))
    if not parts:
        return "0"
    out = ("-" if parts[0][0] == '-' else "") + parts[0][1]
    for sign, body in parts[1:]:
        out += f" {sign} {body}"
    return out


def format_tf_display(num, den, delay=0.0, tf_str=""):
    """Devuelve un componente html que muestra la TF como una fracción matemática legible."""
    if num is None or den is None:
        return html.Div([
            html.Span("Sin función de transferencia válida cargada.", style={'color': '#94a3b8', 'fontSize': '13px'})
        ])

    num_str = _poly_terms_str(num)
    den_str = _poly_terms_str(den)
    delay_str = f" · e^(-{delay:g}s)" if delay and abs(delay) > 1e-12 else ""

    return html.Div(style={'textAlign': 'center', 'padding': '10px 6px'}, children=[
        html.Div(f"G(s){delay_str} =", style={'color': '#94a3b8', 'fontSize': '12.5px', 'marginBottom': '6px'}),
        html.Div(num_str, style={
            'color': '#38bdf8', 'fontSize': '18px', 'fontWeight': '700',
            'fontFamily': '"Fira Code", "Consolas", monospace',
            'paddingBottom': '6px', 'borderBottom': '2px solid #475569',
            'display': 'inline-block', 'minWidth': '140px'
        }),
        html.Div(den_str, style={
            'color': '#e2e8f0', 'fontSize': '18px', 'fontWeight': '700',
            'fontFamily': '"Fira Code", "Consolas", monospace',
            'paddingTop': '6px', 'display': 'block'
        }),
    ])

# 2. CONSTRUCCIÓN DEL LAZO CERRADO CON CONTROLADOR PID
# =============================================================================

def build_closed_loop(num_g, den_g, kp, ki, kd):
    """
    Dado el sistema G(s) = num_g/den_g y un controlador PID
    C(s) = Kp + Ki/s + Kd*s = (Kd*s^2 + Kp*s + Ki) / s,
    retorna (num_cl, den_cl) del lazo cerrado con realimentación unitaria:
        T(s) = C(s)G(s) / (1 + C(s)G(s))
    """
    num_c = [kd, kp, ki]     # Kd*s^2 + Kp*s + Ki
    den_c = [1.0, 0.0]       # s

    num_l = np.polymul(num_c, num_g)   # numerador del lazo abierto L(s)=C(s)G(s)
    den_l = np.polymul(den_c, den_g)   # denominador de L(s)

    den_cl = np.polyadd(den_l, num_l)  # 1 + L(s)  (con denominador común den_l)
    num_cl = num_l

    return num_cl, den_cl


def _apply_delay(t, y, delay):
    """Desplaza la respuesta y(t) en el tiempo por 'delay' segundos (retardo puro)."""
    if not delay or delay <= 0 or len(t) < 2:
        return y
    dt = t[1] - t[0]
    shift = int(round(delay / dt))
    if shift <= 0:
        return y
    shifted = np.zeros_like(y)
    if shift < len(y):
        shifted[shift:] = y[:len(y) - shift]
    return shifted


def _build_input(t, input_type, amplitude, omega=2.0):
    """Construye la señal de entrada u(t) según el tipo seleccionado."""
    if input_type == 'ramp':
        return amplitude * t
    elif input_type == 'sine':
        return amplitude * np.sin(omega * t)
    elif input_type == 'impulse':
        u = np.zeros_like(t)
        u[0] = amplitude / (t[1] - t[0]) if len(t) > 1 else amplitude
        return u
    return np.ones_like(t) * amplitude  # step por defecto


def simulate_tf_response(num, den, t, input_type='step', amplitude=1.0, delay=0.0):
    """Simula la respuesta temporal de un sistema num/den ante distintos tipos de entrada."""
    try:
        sys_ = signal.TransferFunction(num, den)
        if input_type == 'step':
            _, y = signal.step(sys_, T=t)
            y = y * amplitude
        elif input_type == 'impulse':
            _, y = signal.impulse(sys_, T=t)
            y = y * amplitude
        else:
            u = _build_input(t, input_type, amplitude)
            _, y, _ = signal.lsim(sys_, U=u, T=t)
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.clip(y, -1e6, 1e6)
        y = _apply_delay(t, y, delay)
        return y
    except Exception:
        return np.zeros_like(t)

# 3. MÉTRICAS DE RENDIMIENTO
# =============================================================================

def analyze_control_metrics(y, t, setpoint=1.0):
    """Calcula T. establecimiento (5%), overshoot, error en estado estacionario y T. de subida."""
    metrics = {}
    sp_ = setpoint if abs(setpoint) > 1e-9 else 1.0

    settle_threshold = 0.05 * abs(sp_)
    within = np.abs(y - sp_) <= settle_threshold
    settle_time = t[-1]
    for i in range(len(within) - 1, -1, -1):
        if not within[i]:
            settle_time = t[min(i + 1, len(t) - 1)]
            break
        settle_time = t[0]
    metrics['settle_time'] = float(settle_time)

    max_val = np.max(y) if len(y) else 0.0
    metrics['overshoot'] = float(max(0.0, (max_val - sp_) / sp_ * 100)) if sp_ != 0 else 0.0

    metrics['steady_state_error'] = float(abs(sp_ - y[-1]) / abs(sp_) * 100) if len(y) and sp_ != 0 else 0.0

    y10, y90 = 0.1 * sp_, 0.9 * sp_
    idx10 = np.where(y >= y10)[0] if sp_ > 0 else np.where(y <= y10)[0]
    idx90 = np.where(y >= y90)[0] if sp_ > 0 else np.where(y <= y90)[0]
    if len(idx10) and len(idx90):
        metrics['rise_time'] = float(t[idx90[0]] - t[idx10[0]])
    else:
        metrics['rise_time'] = 0.0

    return metrics

# 4. GRÁFICOS
# =============================================================================

_PLOTLY_KW = dict(template="plotly_dark", plot_bgcolor=THEME['card'], paper_bgcolor=THEME['card'])


def plot_time_response(t, y_open, y_closed, setpoint, input_type):
    labels = {'step': 'Escalón', 'ramp': 'Rampa', 'sine': 'Senoidal', 'impulse': 'Impulso'}
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y_open, mode='lines', name='Planta sin control (TF)',
                              line=dict(color='#64748b', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=t, y=y_closed, mode='lines', name='Planta con control PID',
                              line=dict(color='#38bdf8', width=2.5)))
    if input_type in ('step', 'ramp'):
        ref = np.ones_like(t) * setpoint if input_type == 'step' else setpoint * t
        fig.add_trace(go.Scatter(x=t, y=ref, mode='lines', name='Setpoint / Referencia',
                                  line=dict(color='#f59e0b', width=2, dash='dash')))
    fig.update_layout(
        title=f"<b>Respuesta Temporal</b> — Entrada: {labels.get(input_type, 'Escalón')}",
        xaxis_title="Tiempo (s)", yaxis_title="Amplitud",
        height=380, margin=dict(l=45, r=20, t=60, b=35),  # Aumentar t para dar espacio al título
        legend=dict(orientation='h', yanchor='top', y=-0.12, xanchor='center', x=0.5),  # Mover debajo del gráfico
        dragmode='pan', **_PLOTLY_KW
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def plot_control_signal(t, u):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=u, mode='lines', name='u(t)', line=dict(color='#a855f7', width=2)))
    fig.update_layout(
        title="<b>Señal de Control u(t)</b>", xaxis_title="Tiempo (s)", yaxis_title="u(t)",
        height=220, margin=dict(l=45, r=20, t=40, b=35), **_PLOTLY_KW
    )
    return fig


def plot_poles_zeros(num_g, den_g):
    try:
        z, p, _ = signal.tf2zpk(num_g, den_g)
    except Exception:
        z, p = np.array([]), np.array([])

    fig = go.Figure()

    # Región de estabilidad (semiplano izquierdo)
    max_re = max(1.0, np.max(np.abs(np.real(p))) if len(p) else 1.0, np.max(np.abs(np.real(z))) if len(z) else 1.0)
    max_im = max(1.0, np.max(np.abs(np.imag(p))) if len(p) else 1.0, np.max(np.abs(np.imag(z))) if len(z) else 1.0)
    max_re *= 1.4
    max_im *= 1.4

    fig.add_shape(type='rect', x0=-max_re, x1=0, y0=-max_im, y1=max_im,
                  fillcolor='rgba(16,185,129,0.08)', line=dict(width=0), layer='below')
    fig.add_vline(x=0, line_color='#475569', line_width=1)
    fig.add_hline(y=0, line_color='#475569', line_width=1)

    if len(p):
        fig.add_trace(go.Scatter(x=np.real(p), y=np.imag(p), mode='markers', name='Polos',
                                  marker=dict(symbol='x', size=13, color='#ef4444', line=dict(width=2))))
    if len(z):
        fig.add_trace(go.Scatter(x=np.real(z), y=np.imag(z), mode='markers', name='Ceros',
                                  marker=dict(symbol='circle-open', size=13, color='#38bdf8', line=dict(width=2))))

    fig.update_layout(
        title="<b>Plano S — Polos (x) y Ceros (o)</b>",
        xaxis_title="Re(s)", yaxis_title="Im(s)",
        height=380, margin=dict(l=45, r=20, t=45, b=35),
        xaxis=dict(range=[-max_re, max_re], zeroline=False),
        yaxis=dict(range=[-max_im, max_im], zeroline=False, scaleanchor='x', scaleratio=1),
        **_PLOTLY_KW
    )
    return fig


def plot_bode(num_g, den_g):
    try:
        sys_ = signal.TransferFunction(num_g, den_g)
        w, mag, phase = signal.bode(sys_, n=500)
    except Exception:
        w, mag, phase = np.logspace(-2, 2, 500), np.zeros(500), np.zeros(500)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=("<b>Magnitud</b>", "<b>Fase</b>"), vertical_spacing=0.12)
    fig.add_trace(go.Scatter(x=w, y=mag, mode='lines', line=dict(color='#38bdf8', width=2), name='Magnitud'), row=1, col=1)
    fig.add_hline(y=0, line_dash='dot', line_color='#64748b', row=1, col=1)
    fig.add_trace(go.Scatter(x=w, y=phase, mode='lines', line=dict(color='#f59e0b', width=2), name='Fase'), row=2, col=1)
    fig.add_hline(y=-180, line_dash='dot', line_color='#64748b', row=2, col=1)

    fig.update_xaxes(type='log', title_text='Frecuencia ω (rad/s)', row=2, col=1)
    fig.update_yaxes(title_text='Magnitud (dB)', row=1, col=1)
    fig.update_yaxes(title_text='Fase (°)', row=2, col=1)
    fig.update_layout(height=430, margin=dict(l=50, r=20, t=45, b=40), showlegend=False, **_PLOTLY_KW)
    return fig


def plot_root_locus(num_g, den_g, k_max=200.0, n_points=400):
    """Lugar de las raíces: barrido de ganancia K en 1 + K*G(s) = 0."""
    den_g = np.array(den_g, dtype=float)
    num_g = np.array(num_g, dtype=float)

    ks = np.concatenate([[0.0], np.logspace(-3, np.log10(k_max), n_points)])
    all_re, all_im, all_k = [], [], []

    for k in ks:
        char_poly = np.polyadd(den_g, k * num_g)
        roots = np.roots(char_poly)
        all_re.extend(np.real(roots))
        all_im.extend(np.imag(roots))
        all_k.extend([k] * len(roots))

    fig = go.Figure()
    fig.add_vline(x=0, line_color='#475569', line_width=1)
    fig.add_hline(y=0, line_color='#475569', line_width=1)
    fig.add_trace(go.Scatter(
        x=all_re, y=all_im, mode='markers',
        marker=dict(size=4, color=all_k, colorscale='Viridis', showscale=True,
                    colorbar=dict(title='K')),
        name='Lugar de las raíces'
    ))
    try:
        z0, p0, _ = signal.tf2zpk(num_g, den_g)
        fig.add_trace(go.Scatter(x=np.real(p0), y=np.imag(p0), mode='markers', name='Polos (K=0)',
                                  marker=dict(symbol='x', size=12, color='#ef4444')))
        fig.add_trace(go.Scatter(x=np.real(z0), y=np.imag(z0), mode='markers', name='Ceros',
                                  marker=dict(symbol='circle-open', size=12, color='#38bdf8')))
    except Exception:
        pass

    fig.update_layout(
        title="<b>Lugar de las Raíces (Root Locus)</b> — variando ganancia K de G(s)",
        xaxis_title="Re(s)", yaxis_title="Im(s)",
        height=430, margin=dict(l=45, r=20, t=45, b=35), **_PLOTLY_KW
    )
    return fig


def plot_nyquist(num_g, den_g):
    w = np.logspace(-3, 4, 3000)
    s_jw = 1j * w
    L = np.polyval(num_g, s_jw) / np.polyval(den_g, s_jw)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.real(L), y=np.imag(L), mode='lines', name='ω > 0',
                              line=dict(color='#38bdf8', width=2)))
    fig.add_trace(go.Scatter(x=np.real(L), y=-np.imag(L), mode='lines', name='ω < 0',
                              line=dict(color='#38bdf8', width=1.5, dash='dot')))
    fig.add_trace(go.Scatter(x=[-1], y=[0], mode='markers', name='Punto crítico (-1, 0)',
                              marker=dict(symbol='x', size=13, color='#ef4444')))
    fig.update_layout(
        title="<b>Diagrama de Nyquist</b>",
        xaxis_title="Re(L(jω))", yaxis_title="Im(L(jω))",
        height=430, margin=dict(l=45, r=20, t=45, b=35), **_PLOTLY_KW
    )
    return fig


def compute_stability_margins(num_g, den_g):
    """
    Calcula margen de ganancia y margen de fase de forma numérica
    (equivalente a control.margin, no disponible en scipy.signal).
    """
    try:
        sys_ = signal.TransferFunction(num_g, den_g)
        w, mag_db, phase = signal.bode(sys_, n=4000)
        mag_lin = 10 ** (mag_db / 20.0)

        # Margen de fase: en el cruce de ganancia (|L|=1 -> 0 dB)
        phase_margin, w_gc = None, None
        sign_change = np.where(np.diff(np.sign(mag_db)))[0]
        if len(sign_change):
            i = sign_change[0]
            w_gc = w[i]
            ph = np.interp(0.0, [mag_db[i], mag_db[i + 1]], [phase[i], phase[i + 1]])
            phase_margin = 180.0 + ph

        # Margen de ganancia: en el cruce de fase (-180°)
        gain_margin_db, w_pc = None, None
        sign_change2 = np.where(np.diff(np.sign(phase + 180.0)))[0]
        if len(sign_change2):
            i = sign_change2[0]
            w_pc = w[i]
            m = np.interp(-180.0, [phase[i], phase[i + 1]], [mag_db[i], mag_db[i + 1]])
            gain_margin_db = -m

        return {
            'phase_margin': phase_margin, 'w_gc': w_gc,
            'gain_margin_db': gain_margin_db, 'w_pc': w_pc,
        }
    except Exception:
        return {'phase_margin': None, 'w_gc': None, 'gain_margin_db': None, 'w_pc': None}


def build_stability_panel(margins, poles):
    unstable = any(np.real(p) > 1e-9 for p in poles) if len(poles) else False
    stability_pill = html.Span("INESTABLE" if unstable else "ESTABLE",
                                style=STYLE_MARGIN_PILL_BAD if unstable else STYLE_MARGIN_PILL_OK)

    def fmt(v, unit=""):
        return f"{v:.2f}{unit}" if v is not None else "N/D"

    pm = margins.get('phase_margin')
    gm = margins.get('gain_margin_db')
    pm_pill = STYLE_MARGIN_PILL_OK if (pm is not None and pm > 0) else STYLE_MARGIN_PILL_BAD
    gm_pill = STYLE_MARGIN_PILL_OK if (gm is not None and gm > 0) else STYLE_MARGIN_PILL_BAD

    return html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '14px', 'alignItems': 'center'}, children=[
        html.Div(["Estabilidad (polos en lazo abierto): ", stability_pill]),
        html.Div(["Margen de Fase: ", html.Span(fmt(pm, "°"), style=pm_pill),
                  html.Span(f"  (en ω={fmt(margins.get('w_gc'))} rad/s)" if margins.get('w_gc') else "",
                             style={'color': '#64748b', 'fontSize': '11px'})]),
        html.Div(["Margen de Ganancia: ", html.Span(fmt(gm, " dB"), style=gm_pill),
                  html.Span(f"  (en ω={fmt(margins.get('w_pc'))} rad/s)" if margins.get('w_pc') else "",
                             style={'color': '#64748b', 'fontSize': '11px'})]),
    ])

# 5. EXPORTAR CSV
# =============================================================================

def export_to_csv(t, y_open, y_closed, u):
    buf = io.StringIO()
    buf.write("tiempo_s,planta_sin_control,planta_con_control,senal_control_u\n")
    for i in range(len(t)):
        buf.write(f"{t[i]:.6f},{y_open[i]:.6f},{y_closed[i]:.6f},{u[i]:.6f}\n")
    return buf.getvalue()


# 6. LAYOUT DE LA PÁGINA
# =============================================================================

DEFAULT_TF = "1/(s^2 + 0.6*s + 1)"


def page_control():
    return html.Div(style=STYLE_CONTAINER, children=[
        page_header(
            "CONTROL DE SEÑALES",
            "Diseño de Controladores (PID) y Análisis de Sistemas",
            "Herramienta tipo MATLAB: ajusta parámetros PID, define funciones de transferencia reales "
            "(scipy.signal) y analiza estabilidad, polos/ceros, Bode, Root Locus y Nyquist."
        ),

        html.Div(style={'display': 'flex', 'gap': '16px', 'marginBottom': '18px', 'flexWrap': 'wrap'}, children=[

            html.Div(style={'flex': '3', 'minWidth': '250px'}, children=[
                html.Div(style=STYLE_CONTROL_CARD, children=[
                    html.H4("Parámetros de Control", style=STYLE_SECTION_TITLE),

                    html.Label("Kp (Ganancia Proporcional):", style=STYLE_LABEL),
                    dcc.Slider(id='slider-kp', min=0, max=10, step=0.1, value=1.0,
                               marks=make_slider_marks({0: '0', 5: '5', 10: '10'})),

                    html.Label("Ki (Ganancia Integral):", style=STYLE_LABEL),
                    dcc.Slider(id='slider-ki', min=0, max=5, step=0.1, value=0.0,
                               marks=make_slider_marks({0: '0', 2.5: '2.5', 5: '5'})),

                    html.Label("Kd (Ganancia Derivativa):", style=STYLE_LABEL),
                    dcc.Slider(id='slider-kd', min=0, max=2, step=0.05, value=0.0,
                               marks=make_slider_marks({0: '0', 1: '1', 2: '2'})),

                    html.Label("Setpoint:", style=STYLE_LABEL),
                    dcc.Slider(id='slider-setpoint', min=0.1, max=5, step=0.1, value=1.0,
                               marks=make_slider_marks({0.1: '0.1', 2.5: '2.5', 5: '5'})),

                    html.Label("Constante de tiempo τ (solo si no hay TF activa):", style=STYLE_LABEL),
                    dcc.Slider(id='slider-tau', min=0.1, max=5, step=0.1, value=1.0,
                               marks=make_slider_marks({0.1: '0.1', 2.5: '2.5', 5: '5'})),

                    html.Div(style={'marginTop': '12px'}, children=[
                        html.Label("Presets rápidos:", style=STYLE_LABEL),
                        html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '6px'}, children=[
                            html.Button("P", id='preset-p', n_clicks=0, style=STYLE_MINI_BUTTON_GREEN),
                            html.Button("PI", id='preset-pi', n_clicks=0, style=STYLE_MINI_BUTTON),
                            html.Button("PID", id='preset-pid', n_clicks=0, style=STYLE_MINI_BUTTON_ALT),
                            html.Button("Agresivo", id='preset-aggressive', n_clicks=0, style=STYLE_MINI_BUTTON_AMBER),
                        ])
                    ]),
                ])
            ]),

            # Columna 2 — Función de Transferencia 
            html.Div(style={'flex': '4.5', 'minWidth': '340px'}, children=[
                html.Div(style=STYLE_CONTROL_CARD, children=[
                    html.H4("Función de Transferencia G(s)", style=STYLE_SECTION_TITLE),

                    dcc.Input(
                        id='input-transfer-function',
                        type='text',
                        value=DEFAULT_TF,
                        placeholder='Ej: 1/(s**2 + 0.5*s + 1)  ó  (s+1)/(s**2+0.3*s+1)',
                        style=STYLE_CONTROL_INPUT
                    ),

                    # ===== BOTONES MOSTRAR Y APLICAR 
                    html.Div(style={'display': 'flex', 'gap': '8px', 'marginTop': '8px'}, children=[
                        html.Button("📊 Mostrar", id='btn-mostrar-tf', n_clicks=0, style=STYLE_MINI_BUTTON_GREEN),
                        html.Button("▶️ Aplicar PID", id='btn-aplicar-pid', n_clicks=0, style=STYLE_MINI_BUTTON),
                        html.Button("↺ Reset", id='btn-reset-tf', n_clicks=0, style=STYLE_MINI_BUTTON_ALT),
                    ]),

                    html.Div(id='tf-parse-error'),

                    html.Div(id='display-tf-formula', style={'marginTop': '10px'}),

                    dcc.Graph(id='graph-poles-zeros', config={'displayModeBar': False}, style={'marginTop': '6px'}),
                ])
            ]),
            # Columna 3 — Métricas y Herramientas (30%)
            html.Div(style={'flex': '2.5', 'minWidth': '230px'}, children=[
                html.Div(style=STYLE_CONTROL_CARD, children=[
                    html.H4("Métricas de Rendimiento", style=STYLE_SECTION_TITLE),
                    html.Div(id='control-metrics'),

                    html.Hr(style={'borderColor': '#374151', 'margin': '14px 0'}),
                    html.H4("Análisis de Estabilidad", style={**STYLE_SECTION_TITLE, 'fontSize': '16px'}),
                    html.Div(id='control-stability-panel'),

                    html.Hr(style={'borderColor': '#374151', 'margin': '14px 0'}),
                    html.Label("Tipo de Entrada:", style=STYLE_LABEL),
                    dcc.Dropdown(
                        id='dropdown-input-type', clearable=False, value='step',
                        options=[
                            {'label': 'Escalón', 'value': 'step'},
                            {'label': 'Rampa', 'value': 'ramp'},
                            {'label': 'Senoidal', 'value': 'sine'},
                            {'label': 'Impulso', 'value': 'impulse'},
                        ]
                    ),
                ])
            ]),
        ]),

        # ---------- HERRAMIENTAS AVANZADAS (DESPLEGABLE)
        html.Div(style=STYLE_CONTROL_CARD, children=[
            html.Button(
                id='btn-toggle-advanced', n_clicks=0, style=STYLE_ADVANCED_HEADER_BTN,
                children=[html.Span("🛠️ Herramientas Avanzadas (Bode · Root Locus · Nyquist · Exportar)"),
                          html.Span("▾", id='advanced-toggle-arrow')]
            ),
            html.Div(id='advanced-tools-panel', style=STYLE_ADVANCED_PANEL_HIDDEN, children=[
                html.Div(style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap'}, children=[
                    html.Div(style={'flex': '1', 'minWidth': '360px'}, children=[dcc.Graph(id='graph-bode', config={'displayModeBar': False})]),
                    html.Div(style={'flex': '1', 'minWidth': '360px'}, children=[dcc.Graph(id='graph-root-locus', config={'displayModeBar': False})]),
                ]),
                html.Div(style={'marginTop': '10px'}, children=[dcc.Graph(id='graph-nyquist', config={'displayModeBar': False})]),
                html.Div(style={'marginTop': '14px', 'display': 'flex', 'alignItems': 'center', 'gap': '12px'}, children=[
                    html.Button("📥 Exportar Datos (CSV)", id='btn-export-csv', n_clicks=0, style=STYLE_MINI_BUTTON_GREEN),
                    dcc.Download(id='download-control-csv'),
                ]),
            ]),
        ]),

        # ---------- GRÁFICOS PRINCIPALES ----------
        html.Div(style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap'}, children=[
            html.Div(style={'flex': '6', 'minWidth': '360px'}, children=[
                html.Div(style=STYLE_CONTROL_CARD, children=[
                    dcc.Graph(id='graph-control-response'),
                    html.Div("Respuesta temporal: setpoint (naranja), planta sin control (gris punteado) "
                              "y planta con control PID (cian). Usa scroll/arrastre para zoom y pan.",
                              style=STYLE_GRAPH_DESC)
                ])
            ]),
        ]),

        html.Div(style=STYLE_CONTROL_CARD, children=[
            dcc.Graph(id='graph-control-signal'),
            html.Div("Señal de control u(t) generada por el controlador PID.", style=STYLE_GRAPH_DESC)
        ]),

        dcc.Store(id='store-control-tf', data={}),
        dcc.Store(id='store-control-sim', data={}),
        dcc.Store(id='store-advanced-open', data=False),
    ])

# 7. CALLBACKS
# =============================================================================

def register_control_callbacks(app):

    # --- 7.1 Callback para MOSTRAR TF (botón "Mostrar") ---
    @app.callback(
        [Output('display-tf-formula', 'children'),
         Output('graph-poles-zeros', 'figure'),
         Output('tf-parse-error', 'children'),
         Output('store-control-tf', 'data')],
        [Input('btn-mostrar-tf', 'n_clicks'),
         Input('btn-reset-tf', 'n_clicks')],
        [State('input-transfer-function', 'value')],
        prevent_initial_call=True
    )
    def mostrar_tf(n_clicks_mostrar, n_clicks_reset, tf_str):
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Reset: volver a la TF por defecto
        if trigger_id == 'btn-reset-tf':
            tf_str = DEFAULT_TF
            # Actualizar el input con el valor por defecto
            # (esto se maneja con otro callback, pero guardamos el valor)
        
        if not tf_str or not tf_str.strip():
            error_box = html.Div("⚠️ Ingresa una función de transferencia.", style=STYLE_ERROR_BOX)
            return error_box, plot_poles_zeros([1.0], [1.0, 1.0]), "", {}
        
        parsed = parse_transfer_function(tf_str)
        
        if parsed['error']:
            error_box = html.Div(f"⚠️ {parsed['error']}", style=STYLE_ERROR_BOX)
            return error_box, plot_poles_zeros([1.0], [1.0, 1.0]), "", {}
        
        # IMPORTANTE: Eliminar 'sys' del dict antes de guardar (no serializable)
        tf_data = {
            'num': parsed['num'],
            'den': parsed['den'],
            'delay': parsed.get('delay', 0.0),
            'tf_str': parsed.get('tf_str', tf_str)
        }
        
        fig_pz = plot_poles_zeros(parsed['num'], parsed['den'])
        formula = format_tf_display(parsed['num'], parsed['den'], parsed.get('delay', 0.0))
        
        return formula, fig_pz, "", tf_data

    # --- 7.2 Callback para APLICAR PID y actualizar gráficas ---
    @app.callback(
        [Output('graph-control-response', 'figure'),
         Output('graph-control-signal', 'figure'),
         Output('control-metrics', 'children'),
         Output('control-stability-panel', 'children'),
         Output('slider-kp', 'value'),
         Output('slider-ki', 'value'),
         Output('slider-kd', 'value'),
         Output('store-control-sim', 'data')],
        [Input('slider-kp', 'value'),
         Input('slider-ki', 'value'),
         Input('slider-kd', 'value'),
         Input('slider-setpoint', 'value'),
         Input('slider-tau', 'value'),
         Input('dropdown-input-type', 'value'),
         Input('preset-p', 'n_clicks'),
         Input('preset-pi', 'n_clicks'),
         Input('preset-pid', 'n_clicks'),
         Input('preset-aggressive', 'n_clicks'),
         Input('btn-aplicar-pid', 'n_clicks')],
        [State('store-control-tf', 'data')],
        prevent_initial_call=False
    )
    def update_control_response(kp, ki, kd, setpoint, tau, input_type,
                                p_click, pi_click, pid_click, agg_click, aplicar_pid, tf_data):
        ctx = callback_context
        trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        # Aplicar presets
        if trigger == 'preset-p':
            kp, ki, kd = 2.0, 0.0, 0.0
        elif trigger == 'preset-pi':
            kp, ki, kd = 2.0, 0.5, 0.0
        elif trigger == 'preset-pid':
            kp, ki, kd = 3.0, 1.0, 0.5
        elif trigger == 'preset-aggressive':
            kp, ki, kd = 6.0, 2.5, 1.0

        kp = kp or 0.0
        ki = ki or 0.0
        kd = kd or 0.0
        setpoint = setpoint or 1.0
        tau = tau or 1.0

        # --- OBTENER LA PLANTA DESDE TF_DATA O USAR DEFAULT ---
        if tf_data and tf_data.get('num') is not None and tf_data.get('den') is not None:
            num_g = tf_data['num']
            den_g = tf_data['den']
            delay = tf_data.get('delay', 0.0)
        else:
            # Planta por defecto: sistema de primer orden
            num_g = [1.0]
            den_g = [tau, 1.0]
            delay = 0.0

        t = np.linspace(0, 15, 600)

        # --- Sin control (planta abierta) ---
        y_open = simulate_tf_response(num_g, den_g, t, input_type, setpoint, delay)

        # --- Con control PID (lazo cerrado) ---
        num_cl, den_cl = build_closed_loop(num_g, den_g, kp, ki, kd)
        y_closed = simulate_tf_response(num_cl, den_cl, t, input_type, setpoint, delay)

        # --- Señal de control u(t) ---
        if input_type == 'step':
            ref = np.ones_like(t) * setpoint
        elif input_type == 'ramp':
            ref = setpoint * t
        elif input_type == 'sine':
            ref = setpoint * np.sin(2.0 * t)
        else:
            ref = np.zeros_like(t)
        
        e = ref - y_closed
        dt = t[1] - t[0]
        integral_e = np.cumsum(e) * dt
        deriv_e = np.gradient(e, dt)
        u = kp * e + ki * integral_e + kd * deriv_e
        u = np.nan_to_num(u, nan=0.0, posinf=1e6, neginf=-1e6)
        u = np.clip(u, -1e6, 1e6)

        fig_response = plot_time_response(t, y_open, y_closed, setpoint, input_type)
        fig_signal = plot_control_signal(t, u)

        metrics = analyze_control_metrics(y_closed, t, setpoint if input_type == 'step' else 1.0)
        metrics_html = html.Div(style={'fontSize': '13px'}, children=[
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '4px'},
                    children=[html.Span("T. Establecimiento:", style={'color': '#94a3b8'}),
                            html.Span(f"{metrics['settle_time']:.2f} s", style={'color': '#38bdf8', 'fontWeight': '700'})]),
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '4px'},
                    children=[html.Span("Overshoot:", style={'color': '#94a3b8'}),
                            html.Span(f"{metrics['overshoot']:.1f} %", style={'color': '#38bdf8', 'fontWeight': '700'})]),
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '4px'},
                    children=[html.Span("Error E.E.:", style={'color': '#94a3b8'}),
                            html.Span(f"{metrics['steady_state_error']:.1f} %", style={'color': '#38bdf8', 'fontWeight': '700'})]),
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between'},
                    children=[html.Span("T. Subida:", style={'color': '#94a3b8'}),
                            html.Span(f"{metrics['rise_time']:.2f} s", style={'color': '#38bdf8', 'fontWeight': '700'})]),
        ])

        margins = compute_stability_margins(num_g, den_g)
        try:
            _, poles_open, _ = signal.tf2zpk(num_g, den_g)
        except Exception:
            poles_open = []
        stability_html = build_stability_panel(margins, poles_open)

        sim_store = {
            't': t.tolist(), 
            'y_open': y_open.tolist(), 
            'y_closed': y_closed.tolist(), 
            'u': u.tolist()
        }

        return fig_response, fig_signal, metrics_html, stability_html, kp, ki, kd, sim_store

    # --- 7.3 Herramientas avanzadas ---
    @app.callback(
        [Output('graph-bode', 'figure'),
         Output('graph-root-locus', 'figure'),
         Output('graph-nyquist', 'figure')],
        [Input('advanced-tools-panel', 'style'),
         Input('store-control-tf', 'data')],
        prevent_initial_call=True
    )
    def update_advanced_tools(panel_style, tf_data):
        if not panel_style or panel_style.get('display') == 'none':
            raise PreventUpdate
        
        if tf_data and tf_data.get('num') is not None and tf_data.get('den') is not None:
            num_g, den_g = tf_data['num'], tf_data['den']
        else:
            num_g, den_g = [1.0], [1.0, 1.0]
        
        return plot_bode(num_g, den_g), plot_root_locus(num_g, den_g), plot_nyquist(num_g, den_g)

    # --- 7.4 Toggle del botón desplegable ---
    @app.callback(
        [Output('advanced-tools-panel', 'style'),
         Output('advanced-toggle-arrow', 'children'),
         Output('store-advanced-open', 'data')],
        [Input('btn-toggle-advanced', 'n_clicks')],
        [State('store-advanced-open', 'data')],
        prevent_initial_call=True
    )
    def toggle_advanced_tools(n_clicks, is_open):
        new_state = not is_open
        style = STYLE_ADVANCED_PANEL_VISIBLE if new_state else STYLE_ADVANCED_PANEL_HIDDEN
        arrow = "▴" if new_state else "▾"
        return style, arrow, new_state

    # --- 7.5 Exportar datos a CSV ---
    @app.callback(
        Output('download-control-csv', 'data'),
        Input('btn-export-csv', 'n_clicks'),
        State('store-control-sim', 'data'),
        prevent_initial_call=True
    )
    def download_csv(n_clicks, sim_data):
        if not n_clicks or not sim_data:
            raise PreventUpdate
        csv_text = export_to_csv(
            np.array(sim_data['t']), np.array(sim_data['y_open']),
            np.array(sim_data['y_closed']), np.array(sim_data['u'])
        )
        return dict(content=csv_text, filename="respuesta_control.csv")

        # --- 7.6 Inicializar store-control-tf con la TF por defecto ---
    @app.callback(
        Output('store-control-tf', 'data', allow_duplicate=True),
        Input('url', 'pathname'),
        prevent_initial_call=True
    )
    def init_default_tf(pathname):
        if pathname == '/control':
            parsed = parse_transfer_function(DEFAULT_TF)
            if not parsed['error']:
                return {
                    'num': parsed['num'],
                    'den': parsed['den'],
                    'delay': parsed.get('delay', 0.0),
                    'tf_str': DEFAULT_TF
                }
        raise PreventUpdate

# SECCIÓN C APP PRINCIPAL: SIDEBAR, ROUTING Y MODAL

app = dash.Dash(__name__, title="MRI Libro Electrónico", suppress_callback_exceptions=True)

app.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
        {{%css%}}
        <style>
            {FULL_CSS}
        </style>
    </head>
    <body style="margin: 0; background-color: #0a0f1d;">
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
'''
#Barra de navegación
NAV_ITEMS = [
    ('fa-solid fa-house', 'Inicio', '/'),
    ('fa-solid fa-atom', 'Física de la MRI', '/fisica'),
    ('fa-solid fa-signal', 'Señal y Relajación', '/senal-relajacion'),
    ('fa-solid fa-palette', 'Contraste en MRI', '/contraste'),
    ('🌐', 'Espacio-k', '/espacio-k'),
    ('fa-solid fa-clock', 'Secuencias de Pulso', '/secuencias'),
    ('fa-solid fa-flask', 'Simulación', '/simulacion'),
    ('fa-solid fa-pencil', 'Actividades', '/actividades'),
    ('fa-solid fa-sliders-h', 'Control', '/control'),
]


def build_sidebar():
    links = [
        dcc.Link(
            html.Div([
                # Si es el emoji del planeta (Espacio-k), mostrarlo como texto
                html.Span(
                    icon if icon == '🌐' else None,
                    style={'marginRight': '12px', 'fontSize': '18px', 'width': '20px', 'textAlign': 'center'}
                ) if icon == '🌐' else html.I(
                    className=icon,
                    style={'marginRight': '12px', 'fontSize': '15px', 'width': '20px', 'textAlign': 'center', 'color': '#38bdf8'}
                ),
                html.Span(label, style={'fontSize': '13.5px', 'fontWeight': '600'})
            ], className='sidebar-link'),
            href=href, style={'textDecoration': 'none'}
        )
        for icon, label, href in NAV_ITEMS
    ]
    return html.Div(style=STYLE_SIDEBAR, children=[
        html.Div(style={'marginBottom': '28px', 'paddingLeft': '4px'}, children=[
            html.Div(
                html.I(className="fa-solid fa-magnet", style={'fontSize': '28px', 'color': '#38bdf8'}),
                style={'marginBottom': '4px'}
            ),
            html.Div("MRI Interactivo", style={'fontSize': '18px', 'fontWeight': '800', 'color': '#ffffff', 'marginTop': '4px'}),
            html.Div("Libro Electrónico de Resonancia Magnética", style={'fontSize': '11px', 'color': THEME['text_muted'], 'marginTop': '2px'})
        ]),
        html.Div(links, style={'display': 'flex', 'flexDirection': 'column'}),
        html.Div(style={'marginTop': '30px', 'paddingTop': '16px', 'borderTop': f"1px solid {THEME['card_border']}"}, children=[
            html.Div("Campo de referencia", style={'fontSize': '10.5px', 'color': THEME['text_muted']}),
            html.Div("1.5 T · 63.87 MHz", style={'fontSize': '13px', 'color': '#38bdf8', 'fontWeight': '700'})
        ])
    ])


modal = html.Div(id='modal-overlay', style=STYLE_MODAL_OVERLAY_HIDDEN, children=[
    html.Div(style=STYLE_MODAL_BOX, children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'flex-start', 'marginBottom': '14px'}, children=[
            html.H3(id='modal-title', style={'margin': '0', 'color': '#f8fafc', 'fontSize': '20px', 'fontWeight': '800', 'paddingRight': '20px'}),
            html.Button("✕", id='modal-close', n_clicks=0, className='modal-close-btn')
        ]),
        html.Div(id='modal-body')
    ])
])

app.layout = html.Div(style={
    'display': 'flex', 'minHeight': '100vh', 'backgroundColor': THEME['bg'],
    'fontFamily': '"Inter", "Segoe UI", -apple-system, sans-serif'
}, children=[
    dcc.Location(id='url', refresh=False),
    build_sidebar(),
    html.Div(id='page-content', style=STYLE_CONTENT_AREA),
    modal,
])

@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def render_page(pathname):
    if pathname == '/simulacion':
        return simulacion_layout
    elif pathname == '/fisica':
        return page_fisica()
    elif pathname == '/senal-relajacion':
        return page_senal_relajacion()
    elif pathname == '/contraste':
        return page_contraste()
    elif pathname == '/espacio-k':
        return page_kspace()
    elif pathname == '/secuencias':
        return page_secuencias()
    elif pathname == '/actividades':
        return page_actividades()
    elif pathname == '/control': 
        return page_control()
    else:
        return page_inicio()


@app.callback(
    Output('modal-overlay', 'style'),
    Output('modal-title', 'children'),
    Output('modal-body', 'children'),
    Input({'type': 'term-link', 'term': ALL}, 'n_clicks'),
    Input('modal-close', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_modal(term_clicks, close_clicks):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'modal-close':
        return STYLE_MODAL_OVERLAY_HIDDEN, no_update, no_update

    try:
        parsed = json.loads(trigger_id)
        term_key = parsed.get('term')
    except (ValueError, AttributeError):
        raise PreventUpdate

    if not any(term_clicks):
        raise PreventUpdate

    entry = GLOSSARY.get(term_key)
    if not entry:
        raise PreventUpdate

    return STYLE_MODAL_OVERLAY_VISIBLE, entry['title'], entry['body']
register_simulation_callbacks(app)

# ===== REGISTRAR CALLBACKS DE ACTIVIDADES ====
def register_activities_callbacks(app):
    
    # 1. GENERAR PREGUNTAS
    @app.callback(
        [Output('store-preguntas', 'data'),
         Output('store-respuestas', 'data'),
         Output('container-preguntas', 'children'),
         Output('modal-resultados', 'style')],
        [Input('btn-generar-preguntas', 'n_clicks')],
        prevent_initial_call=True
    )
    def generar_preguntas(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        
        preguntas = generar_preguntas_con_gemini(cantidad=5)
        if not preguntas:
            preguntas = generar_preguntas_fallback(5)
        
        children = []
        for idx, p in enumerate(preguntas):
            p['indice'] = idx
            opciones_html = []
            letras = ['A', 'B', 'C', 'D']
            for letra in letras:
                if letra in p['opciones']:
                    opciones_html.append(
                        html.Div(
                            html.Button(
                                f"{letra}. {p['opciones'][letra]}",
                                id={'type': 'opcion-respuesta', 'indice': idx, 'letra': letra},
                                n_clicks=0,
                                style=STYLE_OPTION_BUTTON
                            ),
                            style={'marginBottom': '8px'}
                        )
                    )
            
            children.append(
                html.Div(style=STYLE_QUIZ_CARD, children=[
                    html.H4(
                        f"Pregunta {idx+1}: {p['pregunta']}",
                        style={'color': '#f8fafc', 'fontSize': '16px', 'marginBottom': '12px'}
                    ),
                    html.Div(opciones_html),
                    html.Div(
                        id={'type': 'feedback-pregunta', 'indice': idx},
                        style={'marginTop': '10px', 'fontSize': '13px', 'color': '#94a3b8'}
                    )
                ])
            )
        
        children.append(
            html.Div(style={'display': 'flex', 'justifyContent': 'center', 'marginTop': '15px'}, children=[
                html.Button(
                    "✅ Verificar Respuestas",
                    id='btn-verificar-respuestas',
                    n_clicks=0,
                    className='nav-cta-btn',
                    style={'background': 'linear-gradient(135deg, #065f46, #059669)'}
                )
            ])
        )
        
        return preguntas, {}, children, STYLE_MODAL_OVERLAY_HIDDEN

    # 2. SELECCIONAR OPCIÓN
    @app.callback(
        [Output({'type': 'opcion-respuesta', 'indice': ALL, 'letra': ALL}, 'style'),
         Output('store-respuestas', 'data')],
        [Input({'type': 'opcion-respuesta', 'indice': ALL, 'letra': ALL}, 'n_clicks')],
        [State({'type': 'opcion-respuesta', 'indice': ALL, 'letra': ALL}, 'id'),
         State('store-respuestas', 'data')],
        prevent_initial_call=True
    )
    def seleccionar_opcion(n_clicks_list, id_list, respuestas_actuales):
        ctx = callback_context
        if not ctx.triggered or not any(n_clicks_list):
            raise PreventUpdate
        
        respuestas = dict(respuestas_actuales) if respuestas_actuales else {}
        trigger = ctx.triggered[0]
        trigger_id = json.loads(trigger['prop_id'].split('.')[0])
        idx_seleccionado = trigger_id['indice']
        letra_seleccionada = trigger_id['letra']
        
        respuestas[str(idx_seleccionado)] = letra_seleccionada
        
        nuevos_estilos = []
        for id_item in id_list:
            item_idx_str = str(id_item['indice'])
            if item_idx_str in respuestas and respuestas[item_idx_str] == id_item['letra']:
                nuevos_estilos.append(STYLE_OPTION_SELECTED)
            else:
                nuevos_estilos.append(STYLE_OPTION_BUTTON)
        
        return nuevos_estilos, respuestas

    # 3. VERIFICAR O ENVIAR RESULTADOS
    @app.callback(
        [Output('modal-resultados', 'style', allow_duplicate=True),
         Output('modal-resultados-body', 'children')],
        [Input('btn-verificar-respuestas', 'n_clicks'),
         Input('btn-enviar-resultados', 'n_clicks'),
         Input('btn-cerrar-resultados', 'n_clicks')],
        [State('store-preguntas', 'data'),
         State('store-respuestas', 'data'),
         State('input-nombre-estudiante', 'value'),
         State('input-correo-estudiante', 'value')],
        prevent_initial_call=True
    )
    def verificar_o_enviar(verificar_clicks, enviar_clicks, cerrar_clicks, preguntas, respuestas, nombre, correo):
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == 'btn-cerrar-resultados':
            return STYLE_MODAL_OVERLAY_HIDDEN, no_update
        
        if not preguntas:
            return STYLE_MODAL_OVERLAY_VISIBLE, html.P("⚠️ Primero debes generar las preguntas.", style={'color': '#f87171', 'fontWeight': 'bold'})
        
        respuestas = respuestas or {}
        total = len(preguntas)
        
        if len(respuestas) < total:
            faltantes = total - len(respuestas)
            return STYLE_MODAL_OVERLAY_VISIBLE, html.P(
                f"⚠️ Has respondido {len(respuestas)} de {total} preguntas. Te faltan {faltantes} por responder.",
                style={'color': '#f59e0b', 'fontSize': '15px', 'lineHeight': '1.6'}
            )
        
        if trigger_id == 'btn-enviar-resultados' and (not nombre or not correo):
            return STYLE_MODAL_OVERLAY_VISIBLE, html.P(
                "⚠️ Completa tu Nombre y Correo antes de enviar los resultados.",
                style={'color': '#f87171', 'fontSize': '15px', 'fontWeight': 'bold'}
            )
        
        correctas = 0
        for idx, p in enumerate(preguntas):
            if respuestas.get(str(idx)) == p.get('correcta'):
                correctas += 1
        
        porcentaje = round((correctas / total) * 100, 1)
        
        reporte = html.Div([
            html.P(f"👤 Estudiante: {nombre or 'No especificado'}", style={'color': '#e2e8f0', 'fontSize': '15px'}),
            html.P(f" Correo: {correo or 'No especificado'}", style={'color': '#e2e8f0', 'fontSize': '15px'}),
            html.Hr(style={'borderColor': '#334155', 'margin': '12px 0'}),
            html.P(f" Preguntas respondidas: {total} de {total}", style={'color': '#e2e8f0', 'fontSize': '15px'}),
            html.P(f"✅ Correctas: {correctas}", style={'color': '#10b981', 'fontSize': '18px', 'fontWeight': 'bold'}),
            html.P(f"❌ Incorrectas: {total - correctas}", style={'color': '#f87171', 'fontSize': '18px', 'fontWeight': 'bold'}),
            html.P(f" Calificación: {porcentaje}%", style={'color': '#38bdf8', 'fontSize': '20px', 'fontWeight': 'bold'}),
            html.Hr(style={'borderColor': '#334155', 'margin': '12px 0'}),
            html.P(" Revisa los temas del para reforzar conceptos.", style={'color': '#94a3b8', 'fontSize': '14px'}),
        ])
        
        if trigger_id == 'btn-enviar-resultados':
            try:
                with open('resultados_estudiantes.csv', 'a', encoding='utf-8') as f:
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"{timestamp},{nombre.strip()},{correo.strip()},{correctas},{total},{porcentaje}\n")
                reporte.children.append(
                    html.P("✅ Resultados guardados.", style={'color': '#10b981', 'fontSize': '14px', 'marginTop': '10px'})
                )
            except Exception as e:
                reporte.children.append(
                    html.P(f"❌ Error al guardar: {str(e)}", style={'color': '#f87171', 'fontSize': '14px', 'marginTop': '10px'})
                )
        
        return STYLE_MODAL_OVERLAY_VISIBLE, reporte

    # 4. SUBIR TAREA - MOSTRAR INFO DEL ARCHIVO
    @app.callback(
        Output('upload-tarea-info', 'children'),
        Input('upload-tarea-archivo', 'contents'),
        State('upload-tarea-archivo', 'filename'),
        prevent_initial_call=True
    )
    def mostrar_info_archivo(contents, filename):
        if contents is None:
            return ""
        size_bytes = len(contents)
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
        return f"✅ Archivo seleccionado: {filename} ({size_str})"

    # 5. GUARDAR TAREA CON ARCHIVO
    @app.callback(
        [Output('mensaje-tarea-estado', 'children'),
         Output('contador-tareas', 'children', allow_duplicate=True)],
        [Input('btn-enviar-tarea', 'n_clicks')],
        [State('upload-tarea-archivo', 'contents'),
         State('upload-tarea-archivo', 'filename'),
         State('input-nombre-tarea', 'value'),
         State('input-curso-tarea', 'value'),
         State('textarea-tarea-descripcion', 'value')],
        prevent_initial_call=True
    )
    def guardar_tarea_con_archivo(n_clicks, contents, filename, nombre, curso, descripcion):
        if n_clicks == 0:
            raise PreventUpdate
        
        if not nombre or not nombre.strip():
            return "⚠️ Ingresa tu nombre completo.", no_update
        if not contents:
            return "⚠️ Selecciona un archivo.", no_update
        if not descripcion or not descripcion.strip():
            return "⚠️ Escribe una descripción.", no_update
        
        try:
            carpeta_tareas = "tareas_subidas"
            if not os.path.exists(carpeta_tareas):
                os.makedirs(carpeta_tareas)
            
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            nombre_limpio = nombre.strip().replace(' ', '_')
            extension = filename.split('.')[-1] if '.' in filename else 'pdf'
            nombre_archivo = f"{timestamp}_{nombre_limpio}.{extension}"
            ruta_completa = os.path.join(carpeta_tareas, nombre_archivo)
            
            content_type, content_string = contents.split(',')
            with open(ruta_completa, 'wb') as f:
                f.write(base64.b64decode(content_string))
            
            with open('tareas_estudiantes.txt', 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f" Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f" Estudiante: {nombre}\n")
                f.write(f" Curso: {curso or 'No especificado'}\n")
                f.write(f"📁 Archivo: {nombre_archivo}\n")
                f.write(f" Descripción:\n{descripcion}\n")
                f.write(f"{'='*70}\n")
            
            try:
                with open('tareas_estudiantes.txt', 'r', encoding='utf-8') as f:
                    contenido = f.read()
                    num_tareas = contenido.count('='*70)
            except:
                num_tareas = 0
            
            return f"✅ ¡Tarea enviada! Archivo: {nombre_archivo}", f"{num_tareas} tareas"
            
        except Exception as e:
            return f"❌ Error: {str(e)}", no_update

    # 6. CERRAR MODAL AL CAMBIAR DE PÁGINA
    @app.callback(
        Output('modal-resultados', 'style', allow_duplicate=True),
        Input('url', 'pathname'),
        prevent_initial_call=True
    )
    def cerrar_modal_al_cambiar_pagina(pathname):
        return STYLE_MODAL_OVERLAY_HIDDEN

    # 7. CONTADOR DE TAREAS
    @app.callback(
        Output('contador-tareas', 'children'),
        Input('btn-enviar-tarea', 'n_clicks'),
        prevent_initial_call=True
    )
    def actualizar_contador_tareas(n_clicks):
        try:
            with open('tareas_estudiantes.txt', 'r', encoding='utf-8') as f:
                contenido = f.read()
                num_tareas = contenido.count('=' * 70)
            return f"{num_tareas} tareas"
        except:
            return "0 tareas"


# ===== REGISTRAR CALLBACKS =====
register_simulation_callbacks(app)
register_activities_callbacks(app)
register_control_callbacks(app)


# EJECUCIÓN
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050, debug=False)