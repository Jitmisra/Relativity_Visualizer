import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as plotly_go
from plotly.subplots import make_subplots
from typing import List, Tuple
from .lorentz import FourVector

def set_dark_theme():
    """Applies a consistent dark theme for physics plots."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'axes.facecolor': '#111111',
        'figure.facecolor': '#111111',
        'grid.color': '#333333',
        'text.color': 'white',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
    })

def plot_time_dilation(beta_range: Tuple[float, float] = (0, 0.99)):
    """
    Plot proper time vs coordinate time as function of beta, showing the curve
    tau = t/gamma with annotations at specific beta values. Includes a simple
    moving clock visualization conceptual subplot.
    
    Physics:
        Moving clocks run slow. If a clock moves at velocity beta, proper time (tau)
        measured by that clock is less than coordinate time (t) measured by a stationary observer.
    """
    set_dark_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    betas = np.linspace(beta_range[0], beta_range[1], 500)
    gammas = 1.0 / np.sqrt(1 - betas**2)
    
    t_coord = 100.0  # arbitrary units of time
    taus = t_coord / gammas
    
    # Left subplot: tau vs beta
    ax1.plot(betas, taus, color='#00ffff', lw=2)
    ax1.set_xlabel('Velocity ($\\beta = v/c$)')
    ax1.set_ylabel('Proper Time ($\\tau$) for $t=100$')
    ax1.set_title('Time Dilation: $\\tau = t/\\gamma$')
    
    # Annotate key points
    key_betas = [0.5, 0.866, 0.99]
    for b in key_betas:
        g = 1 / np.sqrt(1 - b**2)
        tau = t_coord / g
        ax1.plot(b, tau, 'rx')
        ax1.annotate(f'$\\gamma \\approx {g:.1f}$', (b, tau), textcoords="offset points", xytext=(-10, 10), ha='center')
        
    # Right subplot: gamma vs beta
    ax2.plot(betas, gammas, color='#ff00ff', lw=2)
    ax2.set_xlabel('Velocity ($\\beta = v/c$)')
    ax2.set_ylabel('Lorentz Factor ($\\gamma$)')
    ax2.set_title('Asymptote at Speed of Light')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('time_dilation.png', dpi=150)
    return fig

def plot_length_contraction(beta_range: Tuple[float, float] = (0, 0.99)):
    """
    Plot contracted length vs rest length as function of beta.
    
    Physics:
        Moving objects are contracted along their direction of motion.
        L = L0 / gamma
    """
    set_dark_theme()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    betas = np.linspace(beta_range[0], beta_range[1], 500)
    gammas = 1.0 / np.sqrt(1 - betas**2)
    
    L0 = 100.0
    L = L0 / gammas
    
    ax.plot(betas, L, color='#00ff00', lw=2)
    ax.set_xlabel('Velocity ($\\beta = v/c$)')
    ax.set_ylabel('Contracted Length ($L$) for $L_0=100$')
    ax.set_title('Length Contraction: $L = L_0/\\gamma$')
    
    key_betas = [0.5, 0.866, 0.99]
    for b in key_betas:
        g = 1 / np.sqrt(1 - b**2)
        l_val = L0 / g
        ax.plot(b, l_val, 'rx')
        ax.annotate(f'$L \\approx {l_val:.1f}$', (b, l_val), textcoords="offset points", xytext=(-15, 10), ha='center')
        
    plt.tight_layout()
    plt.savefig('length_contraction.png', dpi=150)
    return fig

