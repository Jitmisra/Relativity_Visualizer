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

def plot_minkowski_diagram(events: List[Tuple[float, float]], beta: float):
    """
    Plot a Minkowski spacetime diagram showing world lines, rotated axes
    of boosted frame, and light cones.
    
    Args:
        events: List of (t, x) tuples representing events.
        beta: Velocity of the moving frame to show boosted axes.
    """
    set_dark_theme()
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Range
    limit = max([max(abs(t), abs(x)) for t, x in events] + [5]) * 1.2
    
    # Light cones (45 deg)
    x_vals = np.linspace(-limit, limit, 100)
    ax.plot(x_vals, x_vals, '--', color='#555555', alpha=0.5, label='Light cone')
    ax.plot(x_vals, -x_vals, '--', color='#555555', alpha=0.5)
    
    # Original axes (t, x)
    ax.axhline(0, color='white', lw=1)
    ax.axvline(0, color='white', lw=1)
    
    # Boosted axes (t', x')
    if abs(beta) > 0 and abs(beta) < 1:
        # ct' axis is x = beta * ct
        ax.plot(beta*x_vals, x_vals, color='#ff00ff', lw=1.5, label="$ct'$ axis")
        # x' axis is ct = beta * x
        ax.plot(x_vals, beta*x_vals, color='#00ffff', lw=1.5, label="$x'$ axis")
        
    colors = []
    # Plot events and classify interval from origin
    for t, x in events:
        ds2 = t**2 - x**2
        if ds2 > 0:
            c, label = '#00ff00', 'Timelike'
        elif ds2 < 0:
            c, label = '#ff0000', 'Spacelike'
        else:
            c, label = '#ffff00', 'Lightlike'
        ax.plot(x, t, 'o', color=c, markersize=8)
        
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel('Space ($x$)')
    ax.set_ylabel('Time ($ct$)')
    ax.set_title(f'Minkowski Diagram (Frame velocity $\\beta={beta}$)')
    ax.set_aspect('equal')
    ax.grid(color='#333333', linestyle=':', alpha=0.6)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('minkowski_diagram.png', dpi=150)
    return fig

def plot_velocity_addition(u: float, v_range: Tuple[float, float] = (-0.99, 0.99)):
    """
    Show relativistic velocity addition vs Newtonian addition.
    w = (u+v)/(1+uv)
    """
    set_dark_theme()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    v = np.linspace(v_range[0], v_range[1], 200)
    w_newt = u + v
    w_rel = (u + v) / (1 + u * v)
    
    ax.plot(v, w_newt, '--', color='#ff5555', label=f'Newtonian ($w = {u} + v$)')
    ax.plot(v, w_rel, '-', color='#55ff55', lw=2, label=f'Relativistic ($w = \\frac{{{u}+v}}{{1+{u}v}}$)')
    ax.axhline(1, color='w', linestyle=':', alpha=0.5, label='Speed of light ($c=1$)')
    ax.axhline(-1, color='w', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Frame velocity ($v$)')
    ax.set_ylabel('Observed velocity ($w$)')
    ax.set_title('Velocity Addition Paradox')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('velocity_addition.png', dpi=150)
    return fig

def plot_boost_visualization_3d(particles_before: List[FourVector], particles_after: List[FourVector], boost_vector: np.ndarray):
    """
    3D plotly scatter plot showing particle momenta before and after boost.
    Displays two subplots side by side.
    """
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                        subplot_titles=('Lab Frame (Before Boost)', 'Center of Mass Frame (After Boost)'))
    
    def add_particles_to_plot(particles, col, name_prefix):
        px = [p.px for p in particles]
        py = [p.py for p in particles]
        pz = [p.pz for p in particles]
        E = [p.E for p in particles]
        
        # Total momentum
        total_p = np.array([sum(px), sum(py), sum(pz)])
        
        fig.add_trace(plotly_go.Scatter3d(
            x=px, y=py, z=pz,
            mode='markers',
            marker=dict(size=[e*3 for e in E], color=E, colorscale='Viridis', showscale=True),
            name=f'{name_prefix} Particles',
            text=[f"E={e:.2f}, p=({ix:.2f}, {iy:.2f}, {iz:.2f})" for e, ix, iy, iz in zip(E, px, py, pz)],
            hoverinfo='text'
        ), row=1, col=col)
        
        # Draw total momentum vector as a line from origin
        fig.add_trace(plotly_go.Scatter3d(
            x=[0, total_p[0]], y=[0, total_p[1]], z=[0, total_p[2]],
            mode='lines+markers',
            line=dict(color='red', width=5),
            marker=dict(size=4, color='red', symbol='diamond'),
            name=f'{name_prefix} Total P'
        ), row=1, col=col)
        
    add_particles_to_plot(particles_before, 1, 'Lab')
    add_particles_to_plot(particles_after, 2, 'CoM')
    
    # Common layout
    fig.update_layout(
        title="3D Momentum Lorentz Boost Visualization",
        template="plotly_dark",
        scene=dict(xaxis_title="Px", yaxis_title="Py", zaxis_title="Pz"),
        scene2=dict(xaxis_title="Px", yaxis_title="Py", zaxis_title="Pz"),
        height=600, width=1100,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # Save as HTML and return JSON for streamlit
    fig.write_html('boost_visualization_3d.html')
    return fig

def plot_momentum_distribution_comparison(particles_lab: List[FourVector], particles_rest: List[FourVector]):
    """
    2x2 subplot comparing px, py, pz distributions and |p| distribution before and after.
    """
    set_dark_theme()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    comp_names = ['Px', 'Py', 'Pz']
    
    for i in range(3):
        ax = axes[i // 2, i % 2]
        lab_vals = [p.to_array()[i+1] for p in particles_lab]
        rest_vals = [p.to_array()[i+1] for p in particles_rest]
        
        ax.hist(lab_vals, bins=20, alpha=0.6, color='#4444ff', label='Lab')
        ax.hist(rest_vals, bins=20, alpha=0.6, color='#ff4444', label='Rest')
        ax.set_title(f'{comp_names[i]} Distribution')
        ax.legend()
        
    # |p| distribution
    ax = axes[1, 1]
    lab_p = [np.linalg.norm(p.to_array()[1:]) for p in particles_lab]
    rest_p = [np.linalg.norm(p.to_array()[1:]) for p in particles_rest]
    ax.hist(lab_p, bins=20, alpha=0.6, color='#4444ff', label='Lab')
    ax.hist(rest_p, bins=20, alpha=0.6, color='#ff4444', label='Rest')
    ax.set_title('|P| Distribution')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('momentum_distribution.png', dpi=150)
    return fig

def animate_boost(fourvectors: List[FourVector], beta_vector: np.ndarray, beta_steps: int = 50, filename: str = "boost_animation.gif"):
    """
    Create matplotlib animation showing constituents moving continuously 
    from lab frame to rest frame.
    """
    set_dark_theme()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Find max momentum for axis limits
    max_p = max([np.linalg.norm([p.px, p.py, p.pz]) for p in fourvectors]) * 1.5
    
    def init():
        ax.set_xlim(-max_p, max_p)
        ax.set_ylim(-max_p, max_p)
        ax.set_zlim(-max_p, max_p)
        ax.set_xlabel('Px')
        ax.set_ylabel('Py')
        ax.set_zlabel('Pz')
        ax.set_title('Lab Frame (beta=0)')
        return fig,
        
    # Store trajectory for each particle
    # betas go from 0 to actual beta_vector
    fractions = np.linspace(0, 1, beta_steps)
    
    from .lorentz import LorentzBoost
    
    # Initial states
    initial_arrays = np.array([fv.to_array() for fv in fourvectors])
    Es = np.array([fv.E for fv in fourvectors])
    
    sc = ax.scatter([], [], [], s=Es*10, c=Es, cmap='coolwarm', alpha=0.8)
    
    def update(frame):
        ax.clear()
        
        init()
        frac = fractions[frame]
        
        if frac == 0:
            boosted_arrays = initial_arrays
            ax.set_title(f'Boosting... beta = 0.00 |v|')
        else:
            current_beta = beta_vector * frac
            boost = LorentzBoost(current_beta)
            boosted_arrays = boost.boost_many(initial_arrays)
            ax.set_title(f'Boosting... beta = {frac:.2f} |v|')
            
        px = boosted_arrays[:, 1]
        py = boosted_arrays[:, 2]
        pz = boosted_arrays[:, 3]
        
        ax.scatter(px, py, pz, s=Es*10, c=Es, cmap='coolwarm', alpha=0.8)
        
        # Plot total momentum vector
        total_p = np.sum(boosted_arrays[:, 1:4], axis=0)
        ax.plot([0, total_p[0]], [0, total_p[1]], [0, total_p[2]], color='yellow', lw=2)
        ax.scatter([total_p[0]], [total_p[1]], [total_p[2]], color='yellow', s=50, marker='*')
        
        return fig,
        
    ani = animation.FuncAnimation(fig, update, frames=beta_steps, init_func=init, blit=False)
    ani.save(filename, writer='pillow', fps=15)
    plt.close(fig)
    return filename
