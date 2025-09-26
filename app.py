import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import io

from src.lorentz import FourVector, LorentzBoost, verify_boost
from src.visualization import plot_boost_visualization_3d, plot_minkowski_diagram

st.set_page_config(page_title="RelativityViz", page_icon="🌌", layout="wide")

st.sidebar.title("RelativityViz 🌌")
st.sidebar.markdown("Explore Special Relativity intuitively.")
page = st.sidebar.radio("Navigation", ["Special Relativity Calculator", "Lorentz Boost Explorer", "Minkowski Diagram Builder"])

def draw_minkowski(events, beta):
    # Wrapper to render matplotlib in streamlit
    fig = plot_minkowski_diagram(events, beta)
    st.pyplot(fig)

if page == "Special Relativity Calculator":
    st.title("Special Relativity Calculator")
    st.markdown("See how $\\gamma$ (Lorentz factor), time, and length change as you approach the speed of light.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Controls")
        beta = st.slider("Velocity $\\beta$ (v/c)", 0.0, 0.9999, 0.5, 0.0001, format="%.4f")
        
        st.markdown("**Famous Examples**:")
        if st.button("Electron in CRT TV (0.1c)"): beta = 0.1
        if st.button("GPS Satellite Effect (0.00001c)"): beta = 0.00001
        if st.button("Cosmic Ray Muon (0.998c)"): beta = 0.998
        if st.button("LHC Proton (0.999999c)"): beta = 0.999999 # Approximate
        
    with col2:
        st.subheader("Live Metrics")
        gamma = 1 / np.sqrt(1 - beta**2)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Lorentz Factor ($\\gamma$)", f"{gamma:,.4f}")
        m2.metric("Time Dilation Factor", f"{gamma:,.4f}x slower")
        m3.metric("Length Contraction", f"{1/gamma:,.4f}x shorter")
        
        st.markdown("---")
        st.subheader("Relativistic vs Newtonian Physics")
        
        # Suppose a 1kg object
        mass = 1.0
        p_newt = mass * beta  # taking c=1
        k_newt = 0.5 * mass * beta**2
        
        p_rel = gamma * mass * beta
        E_tot = gamma * mass
        k_rel = E_tot - mass
        
        df = pd.DataFrame({
            "Quantity (c=1 units)": ["Momentum ($p$)", "Kinetic Energy ($K$)"],
            "Newtonian": [f"{p_newt:.5f}", f"{k_newt:.5f}"],
            "Relativistic": [f"{p_rel:.5f}", f"{k_rel:.5f}"]
        })
        st.table(df)
        
        st.markdown("""
        > **Physics Note**: Newtonian kinetic energy $\\frac{1}{2}mv^2$ is only a low-velocity approximation of 
        > the true relativistic kinetic energy $K = (\\gamma - 1)mc^2$.
        """)

elif page == "Lorentz Boost Explorer":
    st.title("Lorentz Boost Explorer")
    st.markdown("Observe how 4-vectors transform when you boost into their Center of Mass (rest) frame.")
    
    if "particles" not in st.session_state:
        st.session_state.particles = []
        
    col_input, col_action = st.columns([2, 1])
    
    with col_input:
        if st.button("Generate random particle system"):
            # generate 3 random particles
            N = 3
            p = np.random.randn(N, 3) * 5
            m = np.ones(N) * 0.1
            E = np.sqrt(np.sum(p**2, axis=1) + m**2)
            st.session_state.particles = [FourVector(e, px, py, pz) for e, px, py, pz in zip(E, p[:,0], p[:,1], p[:,2])]
            
            # ensure total momentum is non-zero
            p_tot = np.sum(p, axis=0)
            st.session_state.particles.append(FourVector(np.sqrt(np.sum((p_tot*2)**2) + 1), p_tot[0]*2, p_tot[1]*2, p_tot[2]*2))
            
    with col_action:
        if len(st.session_state.particles) > 0:
            total_particle = FourVector(0,0,0,0)
            for p in st.session_state.particles:
                total_particle += p
            
            # To go to Center of Mass, we boost by -beta
            beta_vec = -total_particle.beta
            
            st.markdown(f"**System total momentum**: ({total_particle.px:.2f}, {total_particle.py:.2f}, {total_particle.pz:.2f})")
            st.markdown(f"**Required boost $\\beta$**: ({beta_vec[0]:.2f}, {beta_vec[1]:.2f}, {beta_vec[2]:.2f})")
            
            do_boost = st.button("Boost to Rest Frame", type="primary")

    if len(st.session_state.particles) > 0:
        if 'do_boost' in locals() and do_boost:
            boost = LorentzBoost(beta_vec)
            boosted_particles = [boost.boost(p) for p in st.session_state.particles]
            
            # verification
            v_res = verify_boost(st.session_state.particles, boost)
            
            st.subheader("Verification Panel")
            col_v1, col_v2, col_v3 = st.columns(3)
            col_v1.metric("Tot. Momentum After", f"{np.linalg.norm(v_res['residual_momentum']):.2e}")
            col_v2.metric("Invariant Mass Before", f"{v_res['mass_before']:.4f}")
            col_v3.metric("Invariant Mass After", f"{v_res['mass_after']:.4f}")
            
            st.success(f"Rest frame achieved: {v_res['is_rest_frame']}. Mass preserved: {v_res['is_mass_preserved']}.")
            
            st.plotly_chart(plot_boost_visualization_3d(st.session_state.particles, boosted_particles, beta_vec), use_container_width=True)
            
            # Download CSV
            csv_data = "E,px,py,pz\\n"
            for p in boosted_particles:
                csv_data += f"{p.E},{p.px},{p.py},{p.pz}\\n"
            st.download_button("Download Boosted Data as CSV", data=csv_data, file_name="boosted_particles.csv", mime="text/csv")
        else:
            # Only plot lab frame
            fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
            px = [p.px for p in st.session_state.particles]
            py = [p.py for p in st.session_state.particles]
            pz = [p.pz for p in st.session_state.particles]
            E = [p.E for p in st.session_state.particles]
            fig.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode='markers', marker=dict(size=[e*3 for e in E], color=E, colorscale='Viridis')))
            fig.update_layout(title="Lab Frame (Before Boost)", height=500, margin=dict(l=0,r=0,b=0,t=40), template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

elif page == "Minkowski Diagram Builder":
    st.title("Minkowski Diagram Builder")
    st.markdown("Visualize the relativity of space and time. Add events and adjust the reference frame velocity.")
    
    if "events" not in st.session_state:
        # Some default events (t, x)
        st.session_state.events = [(0, 0), (2, 4), (5, 2), (3, 3)]
        
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Reference Frame")
        mink_beta = st.slider("Frame Velocity $\\beta$", -0.99, 0.99, 0.5, 0.01)
        
        st.subheader("Add Event")
        new_t = st.number_input("Time (ct)", value=0.0)
        new_x = st.number_input("Space (x)", value=0.0)
        if st.button("Add Event"):
            st.session_state.events.append((new_t, new_x))
            
        if st.button("Clear Events"):
            st.session_state.events = []
            
        st.markdown("""
        **Legend**:
        - 🟢 **Timelike** ($ct > x$): Causally connected to origin.
        - 🔴 **Spacelike** ($x > ct$): Causally disconnected.
        - 🟡 **Lightlike** ($ct = x$): On the light cone.
        - 🟣 **Pink/Cyan Axes**: Rotated axes of the moving observer.
        """)
        
    with c2:
        if len(st.session_state.events) > 0:
            draw_minkowski(st.session_state.events, mink_beta)
        else:
            st.info("Add some events to see the diagram.")