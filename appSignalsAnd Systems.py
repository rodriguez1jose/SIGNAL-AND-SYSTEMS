
import os
import json
import numpy as np
_trapz = getattr(np, 'trapezoid', None) or np.trapz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal 
import traceback

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

# ---- 0. VERIFICACIÓN DE SYMPY ----
_SYMPY_OK = True
try:
    import sympy as sp
except ImportError:
    _SYMPY_OK = False

GEMINI_API_KEY = " "
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')



def generate_square_wave(t, T0, V=1.0):
    """Onda cuadrada bipolar con período T0."""
    phase = (t / T0) % 1.0
    return np.where(phase < 0.5, V, -V)

def generate_triangle_wave(t, T0, V=1.0):
    """Onda triangular con período T0."""
    phase = (t / T0) % 1.0
    return V * (4 * np.abs(phase - 0.5) - 1)

def generate_sawtooth_wave(t, T0, V=1.0):
    """Diente de sierra con período T0."""
    phase = (t / T0) % 1.0
    return V * (2 * phase - 1)

def generate_impulse_train(t, T0, V=1.0):
    """Tren de impulsos (aproximado con pulsos anchos)."""
    phase = (t / T0) % 1.0
    return np.where(phase < 0.05, V / 0.05, 0.0)

def fourier_coefficients_square(T0, V=1.0, N=50):
    """Coeficientes de Fourier para onda cuadrada bipolar."""
    coeffs = {}
    for k in range(-N, N + 1):
        if k == 0:
            coeffs[k] = 0.0
        elif k % 2 == 1:
            coeffs[k] = -1j * 2 * V / (k * np.pi)
        else:
            coeffs[k] = 0.0
    return coeffs

def fourier_coefficients_triangle(T0, V=1.0, N=50):
    """Coeficientes de Fourier para onda triangular."""
    coeffs = {}
    for k in range(-N, N + 1):
        if k == 0:
            coeffs[k] = 0.0
        elif k % 2 == 1:
            coeffs[k] = -2 * V / ((k * np.pi) ** 2)
        else:
            coeffs[k] = 0.0
    return coeffs

def fourier_coefficients_sawtooth(T0, V=1.0, N=50):
    """Coeficientes de Fourier para diente de sierra."""
    coeffs = {}
    for k in range(-N, N + 1):
        if k == 0:
            coeffs[k] = 0.0
        else:
            coeffs[k] = 1j * V / (k * np.pi)
    return coeffs

def reconstruct_signal_from_coeffs(t, coeffs, T0):
    """Reconstruye x(t) = sum Ck e^{jkω0t} truncada."""
    w0 = 2 * np.pi / T0
    x_approx = np.zeros_like(t, dtype=complex)
    for k, Ck in coeffs.items():
        x_approx += Ck * np.exp(1j * k * w0 * t)
    return np.real(x_approx)

def compute_spectrum(coeffs):
    """Devuelve arrays de frecuencias, magnitudes y fases para el espectro."""
    ks = sorted(coeffs.keys())
    freqs = np.array(ks)
    mags = np.array([np.abs(coeffs[k]) for k in ks])
    phases = np.array([np.angle(coeffs[k], deg=True) for k in ks])
    return freqs, mags, phases

def compute_mse(x_original, x_approx):
    """Error cuadrático medio."""
    return float(np.mean((x_original - x_approx) ** 2))

def compute_energy(coeffs):
    """Energía total de la señal (Parseval)."""
    return float(sum(np.abs(Ck) ** 2 for Ck in coeffs.values()))

# =============================================================================
# ===== NUEVAS FUNCIONES PARA LA SIMULACIÓN DE FOURIER MEJORADA =====
# =============================================================================

def parse_user_signal(expr_str, t_array):
    """
    Parsea una expresión matemática escrita por el usuario y la evalúa sobre t_array.
    Soporta: sin, cos, tan, exp, log, sqrt, pi, t, abs, sign, heaviside, etc.
    Retorna (x_array, error_msg).
    """
    if not expr_str or not expr_str.strip():
        return None, "Ingresa una expresión para x(t)."

    if not _SYMPY_OK:
        return None, "Falta sympy. Instala con: pip install sympy"

    try:
        # Limpieza
        expr_str = expr_str.strip()
        # Aceptar tanto ^ como **
        expr_str = expr_str.replace('^', '**')
        # Aceptar 'sen' como alias de 'sin'
        expr_str = re.sub(r'\bsen\b', 'sin', expr_str)

        t_sym = sp.symbols('t', real=True)
        # Diccionario de funciones permitidas
        local_dict = {
            't': t_sym,
            'pi': sp.pi,
            'e': sp.E,
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
            'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
            'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt,
            'abs': sp.Abs, 'sign': sp.sign,
            'heaviside': sp.Heaviside,
        }

        expr = sp.sympify(expr_str, locals=local_dict)

        # Lambdify para evaluación numérica rápida
        f = sp.lambdify(t_sym, expr, modules=['numpy'])

        with np.errstate(all='ignore'):
            x_vals = f(t_array)

        x_vals = np.asarray(x_vals, dtype=float)

        # Si devuelve un escalar, expandir
        if x_vals.ndim == 0:
            x_vals = np.full_like(t_array, float(x_vals), dtype=float)

        # Limpiar NaN/Inf
        x_vals = np.nan_to_num(x_vals, nan=0.0, posinf=0.0, neginf=0.0)

        return x_vals, None

    except Exception as e:
        return None, f"Error al interpretar la expresión: {e}"


def compute_fourier_coeffs_numeric(x_array, t_array, T0, N=50):
    """
    Calcula numéricamente los coeficientes de Fourier C_k de una señal
    periódica dada por muestras (x_array, t_array) sobre un período T0.

    C_k = (1/T0) * ∫_0^T0 x(t) e^{-j k ω0 t} dt
    """
    w0 = 2 * np.pi / T0
    coeffs = {}
    # Asegurar un período exacto para la integral
    # Usamos integración trapezoidal sobre un período
    mask = (t_array >= 0) & (t_array < T0)
    t_period = t_array[mask]
    x_period = x_array[mask]

    if len(t_period) < 2:
        # Fallback: usar todo el array asumiendo que cubre varios períodos
        t_period = t_array
        x_period = x_array

    for k in range(-N, N + 1):
        integrand = x_period * np.exp(-1j * k * w0 * t_period)
        Ck = _trapz(integrand, t_period) / T0
        coeffs[k] = complex(Ck)

    return coeffs


def compute_coeffs_table(coeffs, max_rows=15):
    """Devuelve un componente html con la tabla de coeficientes."""
    ks = sorted(coeffs.keys())
    # Filtrar los de magnitud despreciable y ordenar por magnitud
    nonzero = [(k, coeffs[k]) for k in ks if abs(coeffs[k]) > 1e-9]
    # Mezclar los más significativos (positivos y negativos)
    sorted_by_mag = sorted(nonzero, key=lambda x: -abs(x[1]))[:max_rows * 2]
    sorted_by_mag = sorted(sorted_by_mag, key=lambda x: x[0])

    rows = []
    for k, Ck in sorted_by_mag:
        mag = abs(Ck)
        phase = np.angle(Ck, deg=True)
        rows.append(html.Tr([
            html.Td(f"k = {k}", style={'padding': '6px 10px', 'color': '#e2e8f0',
                                        'fontFamily': 'monospace', 'fontSize': '12.5px'}),
            html.Td(f"{mag:.5f}", style={'padding': '6px 10px', 'color': '#38bdf8',
                                          'fontFamily': 'monospace', 'fontSize': '12.5px',
                                          'textAlign': 'right'}),
            html.Td(f"{phase:+.2f}°", style={'padding': '6px 10px', 'color': '#f59e0b',
                                              'fontFamily': 'monospace', 'fontSize': '12.5px',
                                              'textAlign': 'right'}),
        ]))

    return html.Div([
        html.H5(
            [html.I(className="fa-solid fa-clipboard-list", style={'marginRight': '8px', 'color': '#38bdf8'}),
            "Coeficientes más significativos"
            ],
            style={'color': '#f8fafc', 'marginBottom': '10px', 'fontSize': '15px'}
        ),
        html.Div(style={'maxHeight': '300px', 'overflowY': 'auto',
                     'border': '1px solid #334155', 'borderRadius': '8px'},
             children=[
            html.Table(style={'width': '100%', 'borderCollapse': 'collapse'},
                       children=[
                html.Thead(html.Tr([
                    html.Th("Armónico", style={'padding': '8px 10px', 'color': '#38bdf8',
                                                'textAlign': 'left', 'fontSize': '12px',
                                                'borderBottom': '1px solid #334155',
                                                'backgroundColor': '#0d1424'}),
                    html.Th("|Cₖ|", style={'padding': '8px 10px', 'color': '#38bdf8',
                                            'textAlign': 'right', 'fontSize': '12px',
                                            'borderBottom': '1px solid #334155',
                                            'backgroundColor': '#0d1424'}),
                    html.Th("∠Cₖ", style={'padding': '8px 10px', 'color': '#38bdf8',
                                           'textAlign': 'right', 'fontSize': '12px',
                                           'borderBottom': '1px solid #334155',
                                           'backgroundColor': '#0d1424'}),
                ])),
                html.Tbody(rows),
            ])
        ])
    ])


def export_fourier_to_csv(t, x_original, x_approx, coeffs):
    """Exporta señal y coeficientes a CSV."""
    buf = io.StringIO()
    buf.write("tiempo,senal_original,senal_aproximada,error\n")
    for i in range(len(t)):
        buf.write(f"{t[i]:.6f},{x_original[i]:.6f},{x_approx[i]:.6f},{x_original[i]-x_approx[i]:.6f}\n")
    buf.write("\n# Coeficientes de Fourier\n")
    buf.write("k,|Ck|,fase_deg\n")
    for k in sorted(coeffs.keys()):
        Ck = coeffs[k]
        buf.write(f"{k},{abs(Ck):.6f},{np.angle(Ck, deg=True):.4f}\n")
    return buf.getvalue()


def compute_convergence_curve(x_original, t, T0, N_max=50):
    """
    Calcula el MSE y la energía capturada en función de N
    para mostrar la convergencia de la serie.
    """
    Ns = list(range(1, N_max + 1, max(1, N_max // 25)))
    mses = []
    energies = []

    # Coeficientes hasta N_max
    all_coeffs = compute_fourier_coeffs_numeric(x_original, t, T0, N=N_max)
    total_energy = sum(abs(all_coeffs[k]) ** 2 for k in all_coeffs)

    for N in Ns:
        sub_coeffs = {k: all_coeffs[k] for k in all_coeffs if abs(k) <= N}
        x_approx = reconstruct_signal_from_coeffs(t, sub_coeffs, T0)
        mse = compute_mse(x_original, x_approx)
        mses.append(mse)
        partial_energy = sum(abs(sub_coeffs[k]) ** 2 for k in sub_coeffs)
        energies.append(partial_energy)

    return Ns, mses, energies, total_energy

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
    color: #d8b4fe !important;
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
    border: 2px solid #d8b4fe !important;
    background-color: #a78bfa !important;
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
# ===== ESTILOS DE BOTONES (definidos antes de usarse) =====
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
STYLE_MINI_BUTTON_ALT = {**STYLE_MINI_BUTTON, 'background': 'linear-gradient(135deg, #8b5cf6, #a78bfa)'}
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
    'fourier_periodica': {
        'title': 'Función Periódica',
        'body': _modal_body([
            "Una función x(t) es periódica con período T si x(t) = x(t + T) para todo t. El período fundamental T₀ es el menor valor positivo de T que cumple esta condición.",
            "La frecuencia fundamental se define como f₀ = 1/T₀ (en Hz) o ω₀ = 2π/T₀ (en rad/s).",
            "Ejemplos: cos(ωt), ondas cuadradas, triangulares, dientes de sierra, trenes de impulsos.",
        ])
    },
    'fourier_exponencial': {
        'title': 'Exponencial Compleja',
        'body': _modal_body([
            "Una exponencial compleja tiene la forma e^(jθ) = cos(θ) + j·sin(θ) (fórmula de Euler).",
            "En la serie de Fourier, cada armónico se escribe como Cₖ e^(jkω₀t), donde Cₖ es un número complejo que codifica amplitud y fase, y kω₀ es la frecuencia del armónico.",
            "La suma de exponenciales complejas conjugadas (k y -k) da como resultado una sinusoide real: C₋ₖ e^(-jkω₀t) + Cₖ e^(jkω₀t) = 2|Cₖ| cos(kω₀t + θₖ).",
        ])
    },
    'fourier_espectro': {
        'title': 'Espectro de Frecuencia',
        'body': _modal_body([
            "El espectro de frecuencia es una representación gráfica de las amplitudes (|Cₖ|) y fases (∠Cₖ) de cada armónico de una señal periódica.",
            "Para señales periódicas, el espectro es discreto (líneas verticales en frecuencias kω₀), a diferencia de las señales no periódicas, cuyo espectro es continuo.",
            "El espectro de magnitud muestra cuánta energía hay en cada frecuencia; el espectro de fase muestra el desfase de cada componente.",
        ])
    },
    'gibbs': {
        'title': 'Fenómeno de Gibbs',
        'body': _modal_body([
            "El fenómeno de Gibbs es el sobreimpulso que aparece en la serie de Fourier truncada cerca de una discontinuidad de la señal.",
            "A medida que se añaden más armónicos (N → ∞), el ancho de los sobreimpulsos disminuye, pero su amplitud no desaparece: se estabiliza en aproximadamente el 9% de la altura del salto.",
            "Es un efecto matemático inevitable de aproximar funciones discontinuas mediante sumas finitas de funciones continuas (senoides).",
        ])
    },

}


def term(text, key):
    """Botón de término clickeable que abre el modal de 'aprender más'."""
    return html.Button(text, id={'type': 'term-link', 'term': key}, n_clicks=0, className='term-link')


def edu_card(title, icon, children):
    # Si el icono empieza con "fa-", se renderiza como <i> de Font Awesome
    if isinstance(icon, str) and icon.startswith('fa-'):
        icon_element = html.I(className=icon, style={'fontSize': '20px', 'color': '#38bdf8'})
    else:
        icon_element = html.Span(icon, style={'fontSize': '22px'})
    
    return html.Div(className='edu-card', style=STYLE_EDU_CARD, children=[
        html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'marginBottom': '14px'}, children=[
            icon_element,
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

def next_section_link(href, label):
    return html.Div(style={'marginTop': '10px', 'marginBottom': '30px'}, children=[
        dcc.Link(f"Siguiente: {label} →", href=href, className='nav-cta-btn')
    ])

# ---- Página: Secuencias de Pulso 

def page_fourier():
    return html.Div(style=STYLE_CONTAINER, children=[
        page_header(
            "SERIES DE FOURIER",
            "Descomposición Espectral de Señales Periódicas",
            "Toda señal periódica puede expresarse como suma infinita de exponenciales complejas "
            "(o senoides) armónicamente relacionadas. Esta es la base del análisis espectral en ingeniería."
        ),

        # ===== PANEL 1: INTRODUCCIÓN =====
        edu_card("¿Qué es una Serie de Fourier?", "fa-solid fa-bullseye", [
            html.P([
                "Una ", term("función periódica", "fourier_periodica"),
                " x(t) con período fundamental T₀ puede descomponerse como una suma infinita de ",
                term("exponenciales complejas", "fourier_exponencial"),
                " cuyas frecuencias son múltiplos enteros de la frecuencia fundamental ω₀ = 2π/T₀."
            ], style={'color': '#e2e8f0', 'fontSize': '15px', 'lineHeight': '1.7'}),

            html.Div(style=STYLE_CALLOUT, children=[
                html.Strong("Idea central: "),
                "cualquier señal periódica, por complicada que sea, se construye sumando senoides simples."
            ]),

            html.P([
                "Existen tres formas equivalentes de escribir la serie:"
            ], style={'color': '#cbd5e1', 'fontSize': '14px', 'marginTop': '12px'}),

            html.Table(style={
                'width': '100%', 'borderCollapse': 'collapse', 'marginTop': '10px',
                'backgroundColor': '#0d1424', 'borderRadius': '10px', 'overflow': 'hidden'
            }, children=[
                html.Thead(html.Tr([
                    html.Th("Forma", style={'padding': '10px', 'color': '#38bdf8', 'borderBottom': '1px solid #334155', 'textAlign': 'left'}),
                    html.Th("Ecuación", style={'padding': '10px', 'color': '#38bdf8', 'borderBottom': '1px solid #334155', 'textAlign': 'left'}),
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td("Exponencial", style={'padding': '10px', 'color': '#e2e8f0', 'borderBottom': '1px solid #1f2937'}),
                        html.Td("x(t) = Σ Cₖ e^(jkω₀t)", style={'padding': '10px', 'color': '#fcd34d', 'fontFamily': 'monospace', 'borderBottom': '1px solid #1f2937'}),
                    ]),
                    html.Tr([
                        html.Td("Trigonométrica combinada", style={'padding': '10px', 'color': '#e2e8f0', 'borderBottom': '1px solid #1f2937'}),
                        html.Td("x(t) = C₀ + Σ 2|Cₖ| cos(kω₀t + θₖ)", style={'padding': '10px', 'color': '#fcd34d', 'fontFamily': 'monospace', 'borderBottom': '1px solid #1f2937'}),
                    ]),
                    html.Tr([
                        html.Td("Trigonométrica", style={'padding': '10px', 'color': '#e2e8f0'}),
                        html.Td("x(t) = A₀ + Σ (Aₖ cos kω₀t + Bₖ sin kω₀t)", style={'padding': '10px', 'color': '#fcd34d', 'fontFamily': 'monospace'}),
                    ]),
                ])
            ]),

            html.Div(className='edu-image-container', children=[
                html.Img(src='/assets/fourier_intro.png', className='edu-image',
                         style={'maxHeight': '200px', 'objectFit': 'contain'})
            ]),
            html.Div("Función periódica con período fundamental T₀", className='img-caption'),
        ]),

        # ===== PANEL 2: COEFICIENTES =====
        edu_card("Coeficientes de Fourier", "fa-solid fa-ruler-combined", [
            html.P([
                "Los coeficientes Cₖ se calculan mediante la integral de proyección sobre cada exponencial compleja:"
            ], style={'color': '#e2e8f0', 'fontSize': '15px', 'lineHeight': '1.7'}),

            html.Div(style={
                'background': '#0d1424', 'padding': '16px', 'borderRadius': '10px',
                'border': '1px solid #334155', 'textAlign': 'center', 'margin': '12px 0'
            }, children=[
                html.Div("Cₖ = (1/T₀) ∫₀ᵀ₀ x(t) e^(-jkω₀t) dt",
                         style={'color': '#38bdf8', 'fontSize': '18px', 'fontWeight': '700',
                                'fontFamily': 'monospace'})
            ]),

            html.Ul(style={'color': '#e2e8f0', 'fontSize': '14px', 'lineHeight': '1.9', 'paddingLeft': '22px'}, children=[
                html.Li([html.Strong("C₀", style={'color': '#f59e0b'}), " = valor promedio (componente DC) de la señal."]),
                html.Li([html.Strong("Cₖ", style={'color': '#38bdf8'}), " = amplitud y fase del k-ésimo armónico."]),
                html.Li([html.Strong("C₋ₖ = Cₖ*", style={'color': '#10b981'}), " (conjugado) porque x(t) es real."]),
                html.Li([html.Strong("2Cₖ = Aₖ − jBₖ", style={'color': '#a855f7'}), " relación con la forma trigonométrica."]),
            ]),

            html.Div(style=STYLE_CALLOUT, children=[
                html.Strong("Interpretación física: "),
                "Cₖ mide cuánta energía tiene la señal en la frecuencia kω₀."
            ]),
        ]),

        # ===== PANEL 3: ESPECTRO =====
        edu_card("Espectro de Frecuencia", "fa-solid fa-chart-bar", [
            html.P([
                "El ", term("espectro de frecuencia", "fourier_espectro"),
                " muestra las amplitudes (|Cₖ|) y fases (∠Cₖ) de cada armónico como líneas verticales "
                "en las frecuencias kω₀. Es discreto porque la señal es periódica."
            ], style={'color': '#e2e8f0', 'fontSize': '15px', 'lineHeight': '1.7'}),

            html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '14px', 'marginTop': '12px'}, children=[
                html.Div(style={'background': '#0d1424', 'padding': '14px', 'borderRadius': '10px', 'border': '1px solid #334155'}, children=[
                    html.H5("Espectro de Magnitud", style={'color': '#38bdf8', 'marginBottom': '8px'}),
                    html.P("|Cₖ| vs kω₀. Muestra cuánta energía hay en cada armónico.",
                           style={'color': '#94a3b8', 'fontSize': '13px'}),
                ]),
                html.Div(style={'background': '#0d1424', 'padding': '14px', 'borderRadius': '10px', 'border': '1px solid #334155'}, children=[
                    html.H5("Espectro de Fase", style={'color': '#f59e0b', 'marginBottom': '8px'}),
                    html.P("∠Cₖ vs kω₀. Muestra el desfase de cada armónico.",
                           style={'color': '#94a3b8', 'fontSize': '13px'}),
                ]),
            ]),

            html.Div(style=STYLE_CALLOUT, children=[
                html.Strong("Decaimiento de armónicos: "),
                "si x(t) tiene una discontinuidad, |Cₖ| ~ 1/k. Si es continua pero con derivada discontinua, "
                "|Cₖ| ~ 1/k². ¡Más suave la señal, más rápido decae su espectro!"
            ]),

            html.Div(className='edu-image-container', children=[
                html.Img(src='/assets/fourier_spectrum.gif', className='edu-image',
                         style={'maxHeight': '300px', 'objectFit': 'contain'})
            ]),
            html.Div("Espectro de magnitud y fase de una onda cuadrada", className='img-caption'),
        ]),

        # ===== PANEL 4: PROPIEDADES =====
        edu_card("Propiedades y Fenómeno de Gibbs", "fa-solid fa-bolt", [
            html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '14px'}, children=[
                html.Div(style={'background': '#0d1424', 'padding': '14px', 'borderRadius': '10px', 'border': '1px solid #334155'}, children=[
                    html.H5("Linealidad", style={'color': '#38bdf8', 'marginBottom': '8px'}),
                    html.P("Si x(t) ↔ Cₖ y y(t) ↔ Dₖ, entonces ax(t) + by(t) ↔ aCₖ + bDₖ.",
                           style={'color': '#e2e8f0', 'fontSize': '13px', 'fontFamily': 'monospace'}),
                ]),
                html.Div(style={'background': '#0d1424', 'padding': '14px', 'borderRadius': '10px', 'border': '1px solid #334155'}, children=[
                    html.H5("Transformación de amplitud", style={'color': '#10b981', 'marginBottom': '8px'}),
                    html.P("y(t) = Ax(t) + B → C₀ᵧ = AC₀ₓ + B, Cₖᵧ = ACₖₓ (k≠0)",
                           style={'color': '#e2e8f0', 'fontSize': '13px', 'fontFamily': 'monospace'}),
                ]),
                html.Div(style={'background': '#0d1424', 'padding': '14px', 'borderRadius': '10px', 'border': '1px solid #334155'}, children=[
                    html.H5("Desplazamiento temporal", style={'color': '#a855f7', 'marginBottom': '8px'}),
                    html.P("y(t) = x(t−t₀) → Cₖᵧ = Cₖₓ e^(−jkω₀t₀)",
                           style={'color': '#e2e8f0', 'fontSize': '13px', 'fontFamily': 'monospace'}),
                ]),
                html.Div(style={'background': '#0d1424', 'padding': '14px', 'borderRadius': '10px', 'border': '1px solid #334155'}, children=[
                    html.H5("Inversión temporal", style={'color': '#f59e0b', 'marginBottom': '8px'}),
                    html.P("y(t) = x(−t) → Cₖᵧ = Cₖₓ* (conjugado)",
                           style={'color': '#e2e8f0', 'fontSize': '13px', 'fontFamily': 'monospace'}),
                ]),
            ]),

            html.Div(style={'marginTop': '16px'}, children=[
                html.H5("Fenómeno de Gibbs", style={'color': '#ef4444', 'marginBottom': '8px'}),
                html.P([
                    "Cerca de una discontinuidad, la serie truncada presenta sobreimpulsos que ",
                    html.Strong("no desaparecen al aumentar N", style={'color': '#ef4444'}),
                    ", sino que se acercan al 9% de la altura del salto. El ancho de estos sobreimpulsos sí disminuye."
                ], style={'color': '#e2e8f0', 'fontSize': '14px', 'lineHeight': '1.7'}),
            ]),

            html.Div(className='edu-image-container', children=[
                html.Img(src='/assets/gibbs_phenomenon.png', className='edu-image',
                         style={'maxHeight': '280px', 'objectFit': 'contain'})
            ]),
            html.Div("Fenómeno de Gibbs: sobreimpulso cerca de discontinuidades", className='img-caption'),
        ]),

                # ===== PANEL 5: SIMULACIÓN =====
        html.Div(style=STYLE_EDU_CARD, children=[
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'marginBottom': '14px'}, children=[
                html.I(className="fa-solid fa-flask", style={'fontSize': '22px', 'color': '#38bdf8'}),
                html.H3("Laboratorio Interactivo de Series de Fourier",
                        style={'margin': '0', 'fontSize': '19px', 'fontWeight': '700', 'color': '#f8fafc'})
            ]),

            html.P([
                "Escribe tu propia señal x(t) o elige una predefinida, ajusta el número de armónicos ",
                "y explora cómo la serie truncada reconstruye la señal original."
            ], style={'color': '#cbd5e1', 'fontSize': '14px', 'marginBottom': '18px'}),

            # ===== BLOQUE 1: DEFINICIÓN DE LA SEÑAL =====
            html.Div(style={
                'background': '#0d1424', 'padding': '18px', 'borderRadius': '12px',
                'border': '1px solid #334155', 'marginBottom': '18px'
            }, children=[
                html.H5(
                    [
                        html.I(className="fa-solid fa-1", style={'marginRight': '8px', 'color': '#38bdf8'}),
                        "Define tu señal x(t)"
                    ],
                    style={'color': '#38bdf8', 'marginBottom': '12px', 'fontSize': '15px'}
                ),

html.Label("Modo de señal:", style=STYLE_LABEL),
dcc.RadioItems(
                    id='fourier-signal-mode',
                    options=[
                        {'label': '  Predefinida', 'value': 'predefined'},
                        {'label': '  Personalizada (escribe tu función)', 'value': 'custom'},
                    ],
                    value='predefined',
                    inline=True,
                    labelStyle={'marginRight': '20px', 'color': '#e2e8f0', 'fontSize': '14px', 'cursor': 'pointer'}
                ),

                # --- Subpanel: Predefinida ---
                html.Div(id='fourier-predefined-panel', children=[
                    html.Label("Tipo de Señal:", style=STYLE_LABEL),
                    dcc.Dropdown(
                        id='fourier-signal-type',
                        options=[
                            {'label': 'Onda Cuadrada', 'value': 'square'},
                            {'label': 'Onda Triangular', 'value': 'triangle'},
                            {'label': 'Diente de Sierra', 'value': 'sawtooth'},
                            {'label': 'Tren de Impulsos', 'value': 'impulse'},
                        ],
                        value='square', clearable=False
                    ),
                ]),

                # --- Subpanel: Personalizada ---
                html.Div(id='fourier-custom-panel', style={'display': 'none'}, children=[
                    html.Label("Función x(t):", style=STYLE_LABEL),
                    dcc.Input(
                        id='fourier-custom-fn',
                        type='text',
                        value='2*sin(2*pi*t) + 0.5*cos(6*pi*t)',
                        placeholder='Ej: sin(2*pi*t) + 0.3*cos(6*pi*t)',
                        style={
                            'width': '100%', 'padding': '12px 14px',
                            'backgroundColor': '#1e293b', 'border': '1px solid #334155',
                            'borderRadius': '8px', 'color': '#e2e8f0',
                            'fontFamily': '"Fira Code", "Consolas", monospace',
                            'fontSize': '14px', 'boxSizing': 'border-box'
                        }
                    ),
                    html.Div("Funciones disponibles: sin, cos, tan, exp, log, sqrt, abs, pi, e, heaviside",
                             style={'color': '#64748b', 'fontSize': '11.5px',
                                    'marginTop': '6px', 'fontStyle': 'italic'}),

                    html.Div(style={'display': 'flex', 'gap': '8px', 'marginTop': '10px'}, children=[
                        html.Button([
                            html.I(className="fa-solid fa-eye", style={'marginRight': '6px'}),
                            "Mostrar x(t)"
                        ], id='btn-mostrar-fn', n_clicks=0, style=STYLE_MINI_BUTTON_GREEN),
                        html.Button("↺ Reset", id='btn-reset-fn', n_clicks=0,
                                    style=STYLE_MINI_BUTTON_ALT),
                    ]),
                    html.Div(id='fourier-fn-status', style={'marginTop': '8px', 'fontSize': '13px'}),
                ]),
            ]),

            # ===== BLOQUE 2: PARÁMETROS =====
            html.Div(style={
                'background': '#0d1424', 'padding': '18px', 'borderRadius': '12px',
                'border': '1px solid #334155', 'marginBottom': '18px'
            }, children=[
                    html.H5(
                        [html.I(className="fa-solid fa-2", style={'marginRight': '8px', 'color': '#38bdf8'}),
                            "Parámetros de la Serie"],
                        style={'color': '#38bdf8', 'marginBottom': '12px', 'fontSize': '15px'}
                ),
                html.Div(style={'display': 'flex', 'gap': '18px', 'flexWrap': 'wrap'}, children=[
                    html.Div(style={'flex': '1', 'minWidth': '220px'}, children=[
                        html.Label("Número de Armónicos (N):", style=STYLE_LABEL),
                        dcc.Slider(id='fourier-n-harmonics', min=1, max=50, step=1, value=5,
                                   marks=make_slider_marks({1: '1', 10: '10', 25: '25', 50: '50'})),
                    ]),
                    html.Div(style={'flex': '1', 'minWidth': '220px'}, children=[
                        html.Label("Período T₀ (unidades arbitrarias):", style=STYLE_LABEL),
                        dcc.Slider(id='fourier-T0', min=0.5, max=5, step=0.5, value=2,
                                   marks=make_slider_marks({0.5: '0.5', 2: '2', 5: '5'})),
                    ]),
                    html.Div(style={'flex': '1', 'minWidth': '220px'}, children=[
                        html.Label("Períodos a mostrar:", style=STYLE_LABEL),
                        dcc.Slider(id='fourier-periods-view', min=1, max=5, step=1, value=3,
                                   marks=make_slider_marks({1: '1', 3: '3', 5: '5'})),
                    ]),
                    html.Div(style={'flex': '1', 'minWidth': '220px'}, children=[
                        html.Label("Escala del espectro:", style=STYLE_LABEL),
                        dcc.RadioItems(
                            id='fourier-spectrum-scale',
                            options=[
                                {'label': ' Lineal', 'value': 'linear'},
                                {'label': ' dB', 'value': 'db'},
                            ],
                            value='linear', inline=True,
                            labelStyle={'marginRight': '14px', 'color': '#e2e8f0',
                                         'fontSize': '13px', 'cursor': 'pointer'}
                        ),
                    ]),
                ]),
            ]),

            # ===== BLOQUE 3: BOTÓN PRINCIPAL =====
            html.Div(style={'display': 'flex', 'justifyContent': 'center', 'gap': '12px',
                             'marginBottom': '20px', 'flexWrap': 'wrap'}, children=[
                html.Button([
                    html.I(className="fa-solid fa-play", style={'marginRight': '8px'}),
                        "Ejecutar Simulación"], id='btn-simular-fourier', n_clicks=0,
                            className='nav-cta-btn',
                            style={'background': 'linear-gradient(135deg, #8b5cf6, #a78bfa)',
                                   'padding': '14px 32px', 'fontSize': '16px'}),
                html.Button([
                    html.I(className="fa-solid fa-download", style={'marginRight': '8px'}),
                    "Exportar CSV"], id='btn-exportar-fourier', n_clicks=0,
                            className='nav-cta-btn',
                            style={'background': 'linear-gradient(135deg, #065f46, #059669)',
                                   'padding': '14px 32px', 'fontSize': '16px'}),
                dcc.Download(id='download-fourier-csv'),
            ]),

            # ===== BLOQUE 4: GRÁFICOS PRINCIPALES =====
            html.Div(style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap'}, children=[
                html.Div(style={'flex': '1', 'minWidth': '340px'}, children=[
                    dcc.Graph(id='fourier-graph-approx', config={'displayModeBar': False}),
                ]),
                html.Div(style={'flex': '1', 'minWidth': '340px'}, children=[
                    dcc.Graph(id='fourier-graph-comparison', config={'displayModeBar': False}),
                ]),
            ]),

            html.Div(style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap', 'marginTop': '16px'}, children=[
                html.Div(style={'flex': '1', 'minWidth': '340px'}, children=[
                    dcc.Graph(id='fourier-graph-error', config={'displayModeBar': False}),
                ]),
                html.Div(style={'flex': '1', 'minWidth': '340px'}, children=[
                    dcc.Graph(id='fourier-graph-convergence', config={'displayModeBar': False}),
                ]),
            ]),

            html.Div(style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap', 'marginTop': '16px'}, children=[
                html.Div(style={'flex': '1', 'minWidth': '340px'}, children=[
                    dcc.Graph(id='fourier-graph-spectrum-mag', config={'displayModeBar': False}),
                ]),
                html.Div(style={'flex': '1', 'minWidth': '340px'}, children=[
                    dcc.Graph(id='fourier-graph-spectrum-phase', config={'displayModeBar': False}),
                ]),
            ]),

            # ===== BLOQUE 5: MÉTRICAS Y TABLA =====
            html.Div(style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap', 'marginTop': '20px'}, children=[
                html.Div(style={'flex': '1', 'minWidth': '320px'}, children=[
                    html.Div(id='fourier-metrics-panel'),
                ]),
                html.Div(style={'flex': '1', 'minWidth': '320px'}, children=[
                    html.Div(id='fourier-coeffs-table'),
                ]),
            ]),
        ]),

        # ===== PANEL 6: APLICACIÓN LTI =====
        edu_card("Aplicación: Respuesta de Sistemas LTI", "fa-solid fa-wrench", [
            html.P([
                "Si una señal periódica x(t) con coeficientes Cₖₓ entra a un sistema LTI con función de transferencia H(s), "
                "la salida en estado estacionario tiene coeficientes:"
            ], style={'color': '#e2e8f0', 'fontSize': '15px', 'lineHeight': '1.7'}),

            html.Div(style={
                'background': '#0d1424', 'padding': '16px', 'borderRadius': '10px',
                'border': '1px solid #334155', 'textAlign': 'center', 'margin': '12px 0'
            }, children=[
                html.Div("Cₖᵧ = H(jkω₀) · Cₖₓ",
                         style={'color': '#10b981', 'fontSize': '20px', 'fontWeight': '700',
                                'fontFamily': 'monospace'})
            ]),

            html.P([
                "Esto significa que el sistema ", html.Strong("filtra cada armónico por separado", style={'color': '#10b981'}),
                ": atenúa unos y amplifica otros según |H(jω)|. Por eso una onda cuadrada filtrada pasa-bajas "
                "se convierte en una senoide (el primer armónico domina)."
            ], style={'color': '#e2e8f0', 'fontSize': '14px', 'lineHeight': '1.7'}),

            html.Div(style=STYLE_CALLOUT, children=[
                html.Strong("Ejemplo clásico: "),
                "un circuito RL con H(s) = 1/(s+1) convierte una onda cuadrada de entrada en una señal "
                "casi senoidal, porque atenúa los armónicos 3ω₀, 5ω₀, ... mucho más que el fundamental ω₀."
            ]),

            html.Div(style={'marginTop': '14px'}, children=[
                dcc.Link("Ir al simulador de Control PID →", href='/control', className='nav-cta-btn')
            ]),
        ]),
        next_section_link('/control', 'Control de Señales'),
        dcc.Store(id='store-fourier-data', data={}),
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
# ===== FUNCIONES PARA GENERAR PREGUNTAS CON GEMINI (Signals & Systems) =====

def generar_preguntas_con_gemini(tema="señales y sistemas", cantidad=5):
    """Genera preguntas de opción múltiple usando Gemini API, basadas en Signals & Systems."""
    try:
        if model is None:
            return generar_preguntas_fallback(cantidad)
        prompt = f"""
        Eres un experto en Señales y Sistemas (Signals and Systems) y educación en ingeniería.
        Tu tarea es generar {cantidad} preguntas de opción múltiple para estudiantes universitarios.
        Las preguntas deben basarse en los siguientes conceptos clave:

        **CONCEPTOS FUNDAMENTALES (de un curso de Señales y Sistemas):**
        1. Series de Fourier: coeficientes, espectro de magnitud y fase, propiedades
        2. Fenómeno de Gibbs y convergencia de series de Fourier
        3. Transformada de Fourier y su relación con la serie
        4. Sistemas LTI: convolución, respuesta al impulso, función de transferencia
        5. Transformada de Laplace: polos, ceros, región de convergencia
        6. Estabilidad de sistemas: criterio de Routh-Hurwitz, ubicación de polos
        7. Controladores PID: acciones proporcional, integral y derivativa
        8. Diagramas de Bode: margen de ganancia y margen de fase
        9. Lugar de las raíces (Root Locus)
        10. Diagrama de Nyquist y criterio de estabilidad
        11. Respuesta temporal: tiempo de subida, tiempo de establecimiento, overshoot
        12. Teorema de Parseval y energía de señales
        13. Muestreo y teorema de Nyquist-Shannon
        14. Filtros: pasa-bajos, pasa-altos, pasa-banda

        **INSTRUCCIONES PARA LAS PREGUNTAS:**
        1. Cada pregunta debe ser clara, educativa y relevante para ingeniería
        2. Incluir 4 opciones (A, B, C, D) con una sola correcta
        3. Las opciones incorrectas deben ser plausibles pero claramente erróneas
        4. Incluir preguntas que conecten la teoría con aplicaciones prácticas
        5. Variar la dificultad: algunas fáciles, otras más desafiantes

        **FORMATO DE SALIDA (JSON):**
        [
            {{
                "pregunta": "¿Qué representa el coeficiente C₀ en la serie de Fourier de una señal periódica?",
                "opciones": {{
                    "A": "El valor promedio (componente DC) de la señal",
                    "B": "La frecuencia fundamental de la señal",
                    "C": "La potencia total de la señal",
                    "D": "El desfase inicial de la señal"
                }},
                "correcta": "A"
            }},
            {{
                "pregunta": "¿Cuál es el efecto de aumentar la ganancia proporcional Kp en un controlador PID?",
                "opciones": {{
                    "A": "Reduce el error en estado estacionario y puede aumentar el overshoot",
                    "B": "Elimina completamente el error en estado estacionario",
                    "C": "Aumenta el tiempo de establecimiento",
                    "D": "Reduce el ancho de banda del sistema"
                }},
                "correcta": "A"
            }}
        ]

        **IMPORTANTE:**
        - Devuelve SOLO el JSON, sin texto adicional
        - No uses markdown, solo el JSON puro
        - Asegúrate de que las preguntas sean precisas y estén bien redactadas
        - Incluye al menos 1 pregunta sobre Series de Fourier y 1 sobre Control PID
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
        print(f"⚠️ Error Gemini: {e}. Usando preguntas de respaldo.")
        return generar_preguntas_fallback(cantidad)


def generar_preguntas_fallback(cantidad=5):
    """Preguntas predefinidas de Signals & Systems como respaldo."""
    banco_preguntas = [
        {
            "pregunta": "¿Qué representa el coeficiente C₀ en la serie de Fourier de una señal periódica?",
            "opciones": {
                "A": "El valor promedio (componente DC) de la señal",
                "B": "La frecuencia fundamental",
                "C": "La potencia total",
                "D": "El desfase inicial"
            },
            "correcta": "A"
        },
        {
            "pregunta": "¿Cómo decae la magnitud de los coeficientes de Fourier |Cₖ| para una onda cuadrada?",
            "opciones": {
                "A": "Proporcional a 1/k",
                "B": "Proporcional a 1/k²",
                "C": "Exponencialmente",
                "D": "No decae"
            },
            "correcta": "A"
        },
        {
            "pregunta": "¿Qué establece el fenómeno de Gibbs?",
            "opciones": {
                "A": "Aparece un sobreimpulso cerca de discontinuidades que no desaparece al aumentar N",
                "B": "La serie de Fourier siempre converge uniformemente",
                "C": "Los coeficientes decaen exponencialmente",
                "D": "La energía se conserva"
            },
            "correcta": "A"
        },
        {
            "pregunta": "¿Qué efecto tiene aumentar la ganancia integral Ki en un PID?",
            "opciones": {
                "A": "Elimina el error en estado estacionario pero puede aumentar el overshoot",
                "B": "Reduce el tiempo de subida",
                "C": "Aumenta la estabilidad del sistema",
                "D": "No tiene efecto en el error"
            },
            "correcta": "A"
        },
        {
            "pregunta": "¿Qué mide el margen de fase de un sistema en lazo abierto?",
            "opciones": {
                "A": "Cuánto se puede aumentar el desfase antes de volver el sistema inestable",
                "B": "La ganancia máxima del sistema",
                "C": "La frecuencia de resonancia",
                "D": "El error en estado estacionario"
            },
            "correcta": "A"
        },
        {
            "pregunta": "¿Qué representa el lugar de las raíces (Root Locus)?",
            "opciones": {
                "A": "La trayectoria de los polos del lazo cerrado al variar la ganancia K",
                "B": "La respuesta temporal del sistema",
                "C": "La ganancia del controlador",
                "D": "La frecuencia de muestreo"
            },
            "correcta": "A"
        },
        {
            "pregunta": "¿Cuál es la condición para que un sistema LTI sea estable?",
            "opciones": {
                "A": "Todos los polos deben estar en el semiplano izquierdo del plano s",
                "B": "Todos los ceros deben estar en el semiplano izquierdo",
                "C": "La ganancia debe ser mayor que 1",
                "D": "El sistema debe tener al menos un polo en el origen"
            },
            "correcta": "A"
        },
        {
            "pregunta": "¿Qué establece el teorema de Parseval?",
            "opciones": {
                "A": "La energía de la señal es igual a la suma de las energías de sus armónicos",
                "B": "La transformada de Fourier es invertible",
                "C": "La convolución en tiempo es producto en frecuencia",
                "D": "La señal debe ser periódica"
            },
            "correcta": "A"
        },
        {
            "pregunta": "¿Qué relación existe entre la transformada de Fourier y la serie de Fourier?",
            "opciones": {
                "A": "La serie de Fourier es un caso particular de la transformada para señales periódicas",
                "B": "Son completamente independientes",
                "C": "La transformada solo aplica a señales periódicas",
                "D": "La serie solo aplica a señales no periódicas"
            },
            "correcta": "A"
        },
        {
            "pregunta": "En un diagrama de Bode, ¿qué indica el margen de ganancia?",
            "opciones": {
                "A": "Cuánto se puede aumentar la ganancia antes de la inestabilidad",
                "B": "La frecuencia de cruce de fase",
                "C": "El ancho de banda",
                "D": "El tiempo de establecimiento"
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
            "Prueba de conocimientos de Señales y Sistemas",
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
                        placeholder='Ej: Señales y Sistemas',
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
                    style={'background': 'linear-gradient(135deg, #8b5cf6, #a78bfa)'}
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

def page_inicio():
    return html.Div(style=STYLE_CONTAINER, children=[
        page_header(
            "SIGNALS AND SYSTEMS",
            "Procesamiento de Señales y Sistemas",
            "Un recorrido interactivo por el análisis de Fourier, el diseño de controladores PID y el análisis de sistemas lineales."
        ),
        
        html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(260px, 1fr))', 'gap': '18px'}, children=[
            edu_card("Series de Fourier", "fa-solid fa-chart-bar", [
                html.P("Descomposición espectral de señales periódicas: coeficientes, espectro, fenómeno de Gibbs.", 
                       style={'color': '#cbd5e1', 'fontSize': '14px'}),
                dcc.Link("Explorar →", href='/fourier', 
                        style={'color': THEME['accent_cyan'], 'fontWeight': '700', 'fontSize': '13px', 'textDecoration': 'none'})
            ]),
            edu_card("Control de Señales", "fa-solid fa-sliders", [
                html.P("Diseño de controladores PID, funciones de transferencia, Bode, Root Locus y Nyquist.", 
                       style={'color': '#cbd5e1', 'fontSize': '14px'}),
                dcc.Link("Explorar →", href='/control', 
                        style={'color': THEME['accent_cyan'], 'fontWeight': '700', 'fontSize': '13px', 'textDecoration': 'none'})
            ]),
        ]),
    ])

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

app = dash.Dash(__name__, title="SyS - Signals and Systems", suppress_callback_exceptions=True)

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
    ('fa-solid fa-chart-line', 'Series de Fourier', '/fourier'),
    ('fa-solid fa-sliders-h', 'Control de Señales', '/control'),
    ('fa-solid fa-pencil', 'Actividades', '/actividades'),  # Opcional
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
                html.I(className="fa-solid fa-infinity", style={'fontSize': '28px', 'color': '#38bdf8'}),
                style={'marginBottom': '4px'}
            ),
            html.Div("SyS", style={'fontSize': '18px', 'fontWeight': '800', 'color': '#ffffff', 'marginTop': '4px'}),
            html.Div("Signals and Systems", style={'fontSize': '11px', 'color': THEME['text_muted'], 'marginTop': '2px'})
        ]),
        html.Div(links, style={'display': 'flex', 'flexDirection': 'column'}),
        html.Div(style={'marginTop': '30px', 'paddingTop': '16px', 'borderTop': f"1px solid {THEME['card_border']}"}, children=[
            html.Div("Procesamiento de Señales", style={'fontSize': '10.5px', 'color': THEME['text_muted']}),
            html.Div("Fourier · Control · Sistemas", style={'fontSize': '13px', 'color': '#38bdf8', 'fontWeight': '700'})
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
    if pathname == '/fourier':
        return page_fourier()
    elif pathname == '/control':
        return page_control()
    elif pathname == '/actividades':
        return page_actividades()  # Solo si la conservas
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


# ===== CALLBACKS DE FOURIER =====
def register_fourier_callbacks(app):

    # -------------------------------------------------------------------------
    # CALLBACK 1: Toggle predefinida / personalizada
    # -------------------------------------------------------------------------
    @app.callback(
        [Output('fourier-predefined-panel', 'style'),
         Output('fourier-custom-panel', 'style')],
        [Input('fourier-signal-mode', 'value')],
        prevent_initial_call=True
    )
    def toggle_signal_mode(mode):
        if mode == 'custom':
            return {'display': 'none'}, {'display': 'block'}
        return {'display': 'block'}, {'display': 'none'}

    # -------------------------------------------------------------------------
    # CALLBACK 2: Mostrar x(t) personalizada (solo valida y da feedback)
    # -------------------------------------------------------------------------
    @app.callback(
        Output('fourier-fn-status', 'children'),
        [Input('btn-mostrar-fn', 'n_clicks'),
         Input('btn-reset-fn', 'n_clicks')],
        [State('fourier-custom-fn', 'value')],
        prevent_initial_call=True
    )
    def mostrar_fn_custom(n_mostrar, n_reset, fn_str):
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger == 'btn-reset-fn':
            fn_str = '2*sin(2*pi*t) + 0.5*cos(6*pi*t)'

        # Validación de prueba
        t_test = np.linspace(0, 4, 100)
        x_test, err = parse_user_signal(fn_str, t_test)
        if err:
            return html.Div(f"⚠️ {err}", style=STYLE_ERROR_BOX)
        return html.Div([
            html.Span("✅ Expresión válida. x(t) = ", style={'color': '#10b981'}),
            html.Code(fn_str, style={'color': '#38bdf8', 'fontFamily': 'monospace',
                                      'fontSize': '13px'})
        ])
    @app.callback(
        [Output('fourier-graph-approx', 'figure'),
         Output('fourier-graph-comparison', 'figure'),
         Output('fourier-graph-error', 'figure'),
         Output('fourier-graph-convergence', 'figure'),
         Output('fourier-graph-spectrum-mag', 'figure'),
         Output('fourier-graph-spectrum-phase', 'figure'),
         Output('fourier-metrics-panel', 'children'),
         Output('fourier-coeffs-table', 'children'),
         Output('store-fourier-data', 'data')],
        [Input('btn-simular-fourier', 'n_clicks')],
        [State('fourier-signal-mode', 'value'),
         State('fourier-signal-type', 'value'),
         State('fourier-custom-fn', 'value'),
         State('fourier-n-harmonics', 'value'),
         State('fourier-T0', 'value'),
         State('fourier-periods-view', 'value'),
         State('fourier-spectrum-scale', 'value')],
        prevent_initial_call=False
    )
    def simular_fourier(n_clicks, mode, signal_type, custom_fn, N, T0, n_periods, spec_scale):
        try:
            return _simular_fourier_impl(n_clicks, mode, signal_type, custom_fn, N, T0, n_periods, spec_scale)
        except Exception as e:
            print("=" * 70)
            print("❌ ERROR EN simular_fourier:")
            traceback.print_exc()
            print("=" * 70)
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title=f"<b>Error:</b> {str(e)[:100]}",
                template="plotly_dark",
                plot_bgcolor=THEME['card'], paper_bgcolor=THEME['card']
            )
            empty = go.Figure()
            empty.update_layout(template="plotly_dark",
                                plot_bgcolor=THEME['card'], paper_bgcolor=THEME['card'])
            return (empty_fig, empty, empty, empty, empty, empty,
                    html.Div(f"⚠️ Error: {str(e)}",
                             style={'color': '#f87171', 'padding': '12px'}),
                    html.Div(""),
                    {})

    # -------------------------------------------------------------------------
    # CALLBACK 4: Exportar datos a CSV
    # -------------------------------------------------------------------------
    @app.callback(
        Output('download-fourier-csv', 'data'),
        [Input('btn-exportar-fourier', 'n_clicks')],
        [State('store-fourier-data', 'data')],
        prevent_initial_call=True
    )
    def exportar_fourier_csv(n_clicks, store_data):
        if not n_clicks or not store_data:
            raise PreventUpdate

        try:
            t = np.array(store_data['t'])
            x_orig = np.array(store_data['x_original'])
            x_approx = np.array(store_data['x_approx'])

            coeffs = {}
            for i, k in enumerate(store_data['coeffs_keys']):
                coeffs[k] = complex(store_data['coeffs_vals_real'][i],
                                    store_data['coeffs_vals_imag'][i])

            csv_text = export_fourier_to_csv(t, x_orig, x_approx, coeffs)
            return dict(content=csv_text, filename="fourier_simulacion.csv")
        except Exception as e:
            buf = io.StringIO()
            buf.write("tiempo,senal_original,senal_aproximada\n")
            for i in range(len(store_data['t'])):
                buf.write(f"{store_data['t'][i]:.6f},"
                          f"{store_data['x_original'][i]:.6f},"
                          f"{store_data['x_approx'][i]:.6f}\n")
            return dict(content=buf.getvalue(), filename="fourier_simulacion.csv")


# ===== FUNCIÓN AUXILIAR (fuera de register_fourier_callbacks) =====
def _simular_fourier_impl(n_clicks, mode, signal_type, custom_fn, N, T0, n_periods, spec_scale):
    template = "plotly_dark"
    plot_bg = THEME['card']
    paper_bg = THEME['card']

    # --- Definir la señal original ---
    t = np.linspace(0, n_periods * T0, 2000)

    if mode == 'custom':
        x_original, err = parse_user_signal(custom_fn, t)
        if err:
            x_original = np.zeros_like(t)
            signal_name = "⚠️ Expresión inválida"
        else:
            signal_name = f"x(t) = {custom_fn[:40]}..." if len(custom_fn) > 40 else f"x(t) = {custom_fn}"
        coeffs = compute_fourier_coeffs_numeric(x_original, t, T0, N=N)
    else:
        if signal_type == 'square':
            x_original = generate_square_wave(t, T0, V=1.0)
            coeffs = fourier_coefficients_square(T0, V=1.0, N=N)
            signal_name = "Onda Cuadrada"
        elif signal_type == 'triangle':
            x_original = generate_triangle_wave(t, T0, V=1.0)
            coeffs = fourier_coefficients_triangle(T0, V=1.0, N=N)
            signal_name = "Onda Triangular"
        elif signal_type == 'sawtooth':
            x_original = generate_sawtooth_wave(t, T0, V=1.0)
            coeffs = fourier_coefficients_sawtooth(T0, V=1.0, N=N)
            signal_name = "Diente de Sierra"
        else:
            x_original = generate_impulse_train(t, T0, V=1.0)
            coeffs = fourier_coefficients_square(T0, V=1.0, N=N)
            signal_name = "Tren de Impulsos"

    # --- Reconstrucción ---
    x_approx = reconstruct_signal_from_coeffs(t, coeffs, T0)
    error = x_original - x_approx
    mse = compute_mse(x_original, x_approx)
    rmse = float(np.sqrt(mse))
    max_err = float(np.max(np.abs(error))) if len(error) > 0 else 0.0
    freqs, mags, phases = compute_spectrum(coeffs)

    # --- Energía capturada ---
    energy_total = compute_energy(coeffs)
    dt = t[1] - t[0]
    energy_signal = float(_trapz(x_original ** 2, t) / T0) if T0 > 0 else 0.0
    pct_energy = (energy_total / energy_signal * 100) if energy_signal > 0 else 0.0

    # ===== GRÁFICO 1: Aproximación =====
    fig_approx = go.Figure()
    fig_approx.add_trace(go.Scatter(x=t, y=x_original, mode='lines', name='x(t) original',
                                     line=dict(color='#64748b', width=2, dash='dot')))
    fig_approx.add_trace(go.Scatter(x=t, y=x_approx, mode='lines', name=f'xₙ(t) N={N}',
                                     line=dict(color='#38bdf8', width=2.5)))
    fig_approx.update_layout(
        title=f"<b>{signal_name}</b> — Aproximación con N={N}",
        xaxis_title="Tiempo (s)", yaxis_title="Amplitud",
        height=340, template=template, plot_bgcolor=plot_bg, paper_bgcolor=paper_bg,
        margin=dict(l=45, r=20, t=50, b=35),
        legend=dict(orientation='h', yanchor='top', y=-0.18, xanchor='center', x=0.5)
    )

    # ===== GRÁFICO 2: Comparación =====
    fig_comp = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              subplot_titles=("Original", f"Reconstrucción N={N}"),
                              vertical_spacing=0.15)
    fig_comp.add_trace(go.Scatter(x=t, y=x_original, mode='lines',
                                   line=dict(color='#10b981', width=2), name='Original'),
                        row=1, col=1)
    fig_comp.add_trace(go.Scatter(x=t, y=x_approx, mode='lines',
                                   line=dict(color='#38bdf8', width=2), name=f'N={N}'),
                        row=2, col=1)
    fig_comp.update_xaxes(title_text="Tiempo (s)", row=2, col=1)
    fig_comp.update_layout(height=340, template=template,
                            plot_bgcolor=plot_bg, paper_bgcolor=paper_bg,
                            showlegend=False, margin=dict(l=45, r=20, t=50, b=35))

    # ===== GRÁFICO 3: Error =====
    fig_error = go.Figure()
    fig_error.add_trace(go.Scatter(x=t, y=error, mode='lines', name='e(t)',
                                    line=dict(color='#ef4444', width=2)))
    fig_error.add_hline(y=0, line_dash='dash', line_color='#64748b')
    fig_error.update_layout(
        title=f"<b>Error de Aproximación</b> — MSE = {mse:.5f}",
        xaxis_title="Tiempo (s)", yaxis_title="e(t)",
        height=320, template=template, plot_bgcolor=plot_bg, paper_bgcolor=paper_bg,
        margin=dict(l=45, r=20, t=50, b=35)
    )

    # ===== GRÁFICO 4: Convergencia =====
    try:
        Ns, mses, energies, total_e = compute_convergence_curve(x_original, t, T0, N_max=min(50, N * 2 + 5))
        fig_conv = make_subplots(specs=[[{"secondary_y": True}]])
        fig_conv.add_trace(go.Scatter(x=Ns, y=mses, mode='lines+markers',
                                       name='MSE(N)',
                                       line=dict(color='#ef4444', width=2),
                                       marker=dict(size=5)),
                            secondary_y=False)
        if total_e > 0:
            pct_e = [100 * e / total_e for e in energies]
            fig_conv.add_trace(go.Scatter(x=Ns, y=pct_e, mode='lines+markers',
                                           name='Energía capturada (%)',
                                           line=dict(color='#10b981', width=2),
                                           marker=dict(size=5)),
                                secondary_y=True)
        fig_conv.update_xaxes(title_text="N (armónicos)")
        fig_conv.update_yaxes(title_text="MSE", secondary_y=False, color='#ef4444')
        fig_conv.update_yaxes(title_text="% Energía (Parseval)", secondary_y=True, color='#10b981')
        fig_conv.update_layout(
            title="<b>Convergencia de la Serie Truncada</b>",
            height=320, template=template, plot_bgcolor=plot_bg, paper_bgcolor=paper_bg,
            margin=dict(l=45, r=55, t=50, b=35),
            legend=dict(orientation='h', yanchor='top', y=-0.18, xanchor='center', x=0.5)
        )
    except Exception:
        fig_conv = go.Figure()
        fig_conv.update_layout(title="<b>Convergencia</b> — No disponible",
                                height=320, template=template,
                                plot_bgcolor=plot_bg, paper_bgcolor=paper_bg)

    # ===== GRÁFICO 5: Espectro de magnitud =====
    if spec_scale == 'db':
        mags_plot = 20 * np.log10(np.maximum(mags, 1e-12))
        y_title = "Magnitud (dB)"
    else:
        mags_plot = mags.copy()
        y_title = "|Cₖ|"

    fig_mag = go.Figure()
    fig_mag.add_trace(go.Bar(
        x=freqs, y=mags_plot, name='|Cₖ|',
        marker_color='#38bdf8',
        marker_line=dict(color='#0ea5e9', width=1)
    ))
    fig_mag.update_layout(
        title="<b>Espectro de Magnitud</b>",
        xaxis_title="Armónico k", yaxis_title=y_title,
        height=320, template=template, plot_bgcolor=plot_bg, paper_bgcolor=paper_bg,
        margin=dict(l=45, r=20, t=50, b=35)
    )

    # ===== GRÁFICO 6: Espectro de fase =====
    fig_phase = go.Figure()
    fig_phase.add_trace(go.Bar(
        x=freqs, y=phases, name='∠Cₖ',
        marker_color='#f59e0b',
        marker_line=dict(color='#d97706', width=1)
    ))
    fig_phase.update_layout(
        title="<b>Espectro de Fase</b>",
        xaxis_title="Armónico k", yaxis_title="Fase (°)",
        height=320, template=template, plot_bgcolor=plot_bg, paper_bgcolor=paper_bg,
        margin=dict(l=45, r=20, t=50, b=35)
    )

    # ===== MÉTRICAS =====
    num_armonicos_activos = sum(1 for k in coeffs if abs(coeffs[k]) > 1e-9)
    energia = compute_energy(coeffs)

    metrics_html = html.Div(style={
        'backgroundColor': '#1e293b', 'padding': '16px', 'borderRadius': '10px',
        'border': '1px solid #334155', 'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'
    }, children=[
        html.Div([
            html.Div("Señal", style={'color': '#94a3b8', 'fontSize': '11px'}),
            html.Div(signal_name, style={'color': '#38bdf8', 'fontWeight': '700', 'fontSize': '13px'})
        ]),
        html.Div([
            html.Div("Armónicos activos", style={'color': '#94a3b8', 'fontSize': '11px'}),
            html.Div(f"{num_armonicos_activos}", style={'color': '#10b981', 'fontWeight': '700', 'fontSize': '13px'})
        ]),
        html.Div([
            html.Div("MSE", style={'color': '#94a3b8', 'fontSize': '11px'}),
            html.Div(f"{mse:.5f}", style={'color': '#ef4444', 'fontWeight': '700', 'fontSize': '13px'})
        ]),
        html.Div([
            html.Div("RMSE", style={'color': '#94a3b8', 'fontSize': '11px'}),
            html.Div(f"{rmse:.5f}", style={'color': '#ef4444', 'fontWeight': '700', 'fontSize': '13px'})
        ]),
        html.Div([
            html.Div("Error máx.", style={'color': '#94a3b8', 'fontSize': '11px'}),
            html.Div(f"{max_err:.4f}", style={'color': '#f59e0b', 'fontWeight': '700', 'fontSize': '13px'})
        ]),
        html.Div([
            html.Div("Energía (Parseval)", style={'color': '#94a3b8', 'fontSize': '11px'}),
            html.Div(f"{energia:.4f}", style={'color': '#fcd34d', 'fontWeight': '700', 'fontSize': '13px'})
        ]),
        html.Div([
            html.Div("% Energía capturada", style={'color': '#94a3b8', 'fontSize': '11px'}),
            html.Div(f"{pct_energy:.1f}%", style={'color': '#10b981', 'fontWeight': '700', 'fontSize': '13px'})
        ]),
    ])

    # ===== TABLA DE COEFICIENTES =====
    coeffs_table = compute_coeffs_table(coeffs, max_rows=15)

    # ===== STORE para exportación =====
    store_data = {
        't': t.tolist(),
        'x_original': x_original.tolist(),
        'x_approx': x_approx.tolist(),
        'coeffs_keys': list(coeffs.keys()),
        'coeffs_vals_real': [float(np.real(coeffs[k])) for k in coeffs.keys()],
        'coeffs_vals_imag': [float(np.imag(coeffs[k])) for k in coeffs.keys()],
        'signal_name': signal_name,
    }

    return (fig_approx, fig_comp, fig_error, fig_conv, fig_mag, fig_phase,
            metrics_html, coeffs_table, store_data)

#REGISTRAR CALLBACKS DE ACTIVIDADES

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
register_fourier_callbacks(app)
register_activities_callbacks(app)
register_control_callbacks(app)


# EJECUCIÓN
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050, debug=False)