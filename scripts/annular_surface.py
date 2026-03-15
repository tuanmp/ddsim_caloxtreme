import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils import transformation_matrices

# ── Parameters ───────────────────────────────────────────────────────────────
R1        = 1          # inner radius
R2        = 2          # outer radius
Z_MIN     = 1.38619        # bottom z
Z_MAX     =  1.59662         # top z
THETA_MIN = 5.654867  # start angle
THETA_MAX = 6.283185  # end angle

NTH = 10   # angular resolution
NR  = 4   # radial resolution (caps)
NZ  = 4   # vertical resolution (walls)

particle_direction = np.array(
    [0.97345626, 0.22640614, 0.03351366]
)
R, RT = transformation_matrices(particle_direction)

# RT = np.array(
#     [[-0.2265334,   0.03264242,  0.97345626],
#     [ 0.9740034,  0.00759196,  0.22640614],
#     [ 0. ,        -0.99943826 , 0.03351366]]
# )

COLORSCALE = [
    [0.0,  '#5b7fff'],
    [0.33, '#7ee8fa'],
    [0.66, '#b57bee'],
    [1.0,  '#ff6eb4'],
]
OPACITY = 0.6

# ── Helpers ──────────────────────────────────────────────────────────────────
def color_by_layer_id(layer_id, X):
    """Surfacecolor = radial distance, for consistent shading across surfaces."""
    return np.ones_like(X) * layer_id

def make_surface(X, Y, Z, RT=None, opacity=OPACITY, layer_id=0):
    X, Y, Z = np.asarray(X), np.asarray(Y), np.asarray(Z)
    if RT is not None:
        XYZ = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        XYZ_rot = (RT @ XYZ.T)
        X, Y, Z = XYZ_rot.reshape(3, *X.shape)
    # print(X)
    # print(Z)
    return go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=color_by_layer_id(layer_id, X),
        colorscale=COLORSCALE,
        cmin=0, cmax=1,
        showscale=False,
        opacity=opacity,
        # lighting=dict(ambient=0.6, diffuse=0.8, specular=0.4, roughness=0.3),
        # lightposition=dict(x=200, y=300, z=500),
    )

# ── Surface builders ─────────────────────────────────────────────────────────
def build_annular_surface(
    theta_range=(THETA_MIN, THETA_MAX),
    z_range=(Z_MIN, Z_MAX),
    r_range=(R1, R2),
):
    theta = np.linspace(THETA_MIN, THETA_MAX, NTH)
    z_arr = np.linspace(Z_MIN, Z_MAX, NZ)
    r_arr = np.linspace(R1, R2, NR)

    return theta, z_arr, r_arr

# theta, z_arr, r_arr = build_annular_surface()

# Cylindrical wall at fixed radius, varying theta & z
def cylindrical_wall(r, theta, z_arr, RT=None,
    layer_id=0):
    TH, ZZ = np.meshgrid(theta, z_arr)
    X = r * np.cos(TH)
    Y = r * np.sin(TH)
    return make_surface(X, Y, ZZ, RT)

# Annular cap at fixed z, varying r & theta
def annular_cap(z, r_arr, theta, RT=None,
    layer_id=0):
    RR, TH = np.meshgrid(r_arr, theta)
    X = RR * np.cos(TH)
    Y = RR * np.sin(TH)
    ZZ = np.full_like(X, z)
    return make_surface(X, Y, ZZ, RT)

# Flat radial side wall at fixed theta, varying r & z
def radial_wall(th, r_arr, z_arr, RT=None,
    layer_id=0):
    RR, ZZ = np.meshgrid(r_arr, z_arr)
    X = RR * np.cos(th)
    Y = RR * np.sin(th)
    return make_surface(X, Y, ZZ, RT)

def assemble_traces(
    theta_range=(THETA_MIN, THETA_MAX),
    z_range=(Z_MIN, Z_MAX),
    r_range=(R1, R2),
    RT = None,
    layer_id=0,
):
    theta, z_arr, r_arr = build_annular_surface(theta_range, z_range, r_range)
    theta_min, theta_max = theta_range
    z_min, z_max = z_range
    r_min, r_max = r_range
    
    surfaces = [
        cylindrical_wall(r_max, theta, z_arr, RT, layer_id),           # outer curved wall
        cylindrical_wall(r_min, theta, z_arr, RT, layer_id),           # inner curved wall
        annular_cap(z_max, r_arr, theta, RT, layer_id),             # top annular cap
        annular_cap(z_min, r_arr, theta, RT, layer_id),             # bottom annular cap
        radial_wall(theta_min, r_arr, z_arr, RT, layer_id),         # closing side wall at theta_min
        radial_wall(theta_max, r_arr, z_arr, RT, layer_id),         # closing side wall at theta_max
    ]

    return surfaces

def main():

    # ── Assemble traces ───────────────────────────────────────────────────────────
    traces = assemble_traces(RT=RT)

    # ── Layout ────────────────────────────────────────────────────────────────────
    layout = go.Layout(
        title=dict(
            text=f"Partial Annular Volume  ·  θ ∈ [{np.rad2deg(THETA_MIN):.0f}°, {np.rad2deg(THETA_MAX):.0f}°]",
            font=dict(size=16, color='#e8eaf0'),
            x=0.5,
        ),
        # paper_bgcolor='#080c14',
        scene=dict(
            bgcolor='#080c14',
            aspectmode='cube',
            xaxis=dict(title='x', color='#5b7fff', gridcolor='#1a2035', zerolinecolor='#3a4560'),
            yaxis=dict(title='y', color='#5b7fff', gridcolor='#1a2035', zerolinecolor='#3a4560'),
            zaxis=dict(title='z', color='#5b7fff', gridcolor='#1a2035', zerolinecolor='#3a4560'),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=600,
    )

    fig = go.Figure(layout=layout)

    endpoint = particle_direction * 2

    direction = go.Scatter3d(
        x=[0, endpoint[0]],
        y=[0, endpoint[1]],
        z=[0, endpoint[2]],
        mode='lines',
        line=dict(color='#ff6eb4', width=5),
        name='Particle Direction'
    )

    for i, ax in enumerate(R):
        _ = go.Scatter3d(
            x=[0, ax[0]],
            y=[0, ax[1]],
            z=[0, ax[2]],
            mode='lines',
            line=dict(color=COLORSCALE[i][1], width=3),
            name=f'R_{i}'
        )
        fig.add_trace(_)

    fig.add_trace(direction)

    fig.add_traces(traces)

    # fig.show()

    st.plotly_chart(fig, width="stretch", height="content")

if __name__ == "__main__":
    main()