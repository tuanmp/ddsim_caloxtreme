import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils import transformation_matrices

# ── Parameters ───────────────────────────────────────────────────────────────
R1        = 1
R2        = 2
Z_MIN     = 1.38619
Z_MAX     = 1.59662
THETA_MIN = 5.654867
THETA_MAX = 6.283185

NTH = 60   # angular resolution
NR  = 4    # radial resolution (caps)
NZ  = 4    # vertical resolution (walls)

particle_direction = np.array(
    [0.97345626, 0.22640614, 0.03351366]
)
R, RT = transformation_matrices(particle_direction)

COLORSCALE = [
    [0.0, '#7ee8fa'],
    [0.33, '#5b7fff'],
    [0.66, '#b57bee'],
    [1.0,  '#ff6eb4'],
]
OPACITY = 0.5

# ── Core triangulation helper ─────────────────────────────────────────────────
def quad_grid_to_triangles(idx_grid):
    """
    Convert a 2D grid of vertex indices into two triangles per quad cell.
        v00─v01
         │╲  │
        v10─v11
    Triangle 1: v00, v01, v10
    Triangle 2: v01, v11, v10
    """
    idx = np.asarray(idx_grid)
    v00 = idx[:-1, :-1].ravel()
    v01 = idx[:-1,  1:].ravel()
    v10 = idx[ 1:, :-1].ravel()
    v11 = idx[ 1:,  1:].ravel()
    i = np.concatenate([v00, v01])
    j = np.concatenate([v01, v11])
    k = np.concatenate([v10, v10])
    return i, j, k


# ── Patch vertex builders ─────────────────────────────────────────────────────
# Each returns (verts, i, j, k) with LOCAL face indices (caller applies offset).

def verts_cylindrical_wall(r, theta, z_arr):
    TH, ZZ = np.meshgrid(theta, z_arr)        # (NZ, NTH)
    X = (r * np.cos(TH)).ravel()
    Y = (r * np.sin(TH)).ravel()
    Z = ZZ.ravel()
    verts = np.column_stack([X, Y, Z])
    idx_grid = np.arange(len(verts)).reshape(len(z_arr), len(theta))
    return verts, *quad_grid_to_triangles(idx_grid)

def verts_annular_cap(z, r_arr, theta):
    RR, TH = np.meshgrid(r_arr, theta)        # (NTH, NR)
    X = (RR * np.cos(TH)).ravel()
    Y = (RR * np.sin(TH)).ravel()
    Z = np.full(len(X), z)
    verts = np.column_stack([X, Y, Z])
    idx_grid = np.arange(len(verts)).reshape(len(theta), len(r_arr))
    return verts, *quad_grid_to_triangles(idx_grid)

def verts_radial_wall(th, r_arr, z_arr):
    RR, ZZ = np.meshgrid(r_arr, z_arr)        # (NZ, NR)
    X = (RR * np.cos(th)).ravel()
    Y = (RR * np.sin(th)).ravel()
    Z = ZZ.ravel()
    verts = np.column_stack([X, Y, Z])
    idx_grid = np.arange(len(verts)).reshape(len(z_arr), len(r_arr))
    return verts, *quad_grid_to_triangles(idx_grid)


# ── Mesh assembly ─────────────────────────────────────────────────────────────
def assemble_mesh(
    theta_range=(THETA_MIN, THETA_MAX),
    z_range=(Z_MIN, Z_MAX),
    r_range=(R1, R2),
    RT=None,
    layer_id=0,
    opacity=OPACITY,
    color=None,
):
    theta_min, theta_max = theta_range
    z_min,     z_max     = z_range
    r_min,     r_max     = r_range

    theta = np.linspace(theta_min, theta_max, NTH)
    z_arr = np.linspace(z_min,     z_max,     NZ)
    r_arr = np.linspace(r_min,     r_max,     NR)

    patches = [
        verts_cylindrical_wall(r_max, theta, z_arr),   # outer wall
        verts_cylindrical_wall(r_min, theta, z_arr),   # inner wall
        verts_annular_cap(z_max, r_arr, theta),         # top cap
        verts_annular_cap(z_min, r_arr, theta),         # bottom cap
        verts_radial_wall(theta_min, r_arr, z_arr),     # side at theta_min
        verts_radial_wall(theta_max, r_arr, z_arr),     # side at theta_max
    ]

    all_verts = []
    all_i, all_j, all_k = [], [], []
    offset = 0

    for verts, fi, fj, fk in patches:
        all_verts.append(verts)
        all_i.append(fi + offset)
        all_j.append(fj + offset)
        all_k.append(fk + offset)
        offset += len(verts)

    V = np.vstack(all_verts)   # (N, 3)

    # Apply rotation if provided
    if RT is not None:
        V = (RT @ V.T).T

    I = np.concatenate(all_i)
    J = np.concatenate(all_j)
    K = np.concatenate(all_k)

    # Intensity: uniform per layer_id (matches original color_by_layer_id logic)
    intensity = np.full(len(V), float(layer_id))

    return go.Mesh3d(
        x=V[:, 0], y=V[:, 1], z=V[:, 2],
        i=I, j=J, k=K,
        intensity=intensity,
        colorscale=COLORSCALE,
        cmin=0, cmax=1,
        showscale=False,
        opacity=opacity,
    )


def main():

    # First annular surface — original parameters
    mesh1 = assemble_mesh(
        theta_range=(THETA_MIN, THETA_MAX),
        z_range=(Z_MIN, Z_MAX),
        r_range=(R1, R2),
        RT=RT,
        layer_id=0,
    )

    # Second annular surface — shares the top face (Z_MAX) of mesh1 as its bottom,
    # and shares the outer wall (R2) of mesh1 as its inner wall.
    # Effectively stacked radially outward AND vertically on top.
    Z2_MIN = Z_MAX                  # ← shared edge with mesh1 top cap
    Z2_MAX = Z_MAX + (Z_MAX - Z_MIN)
    R2_MIN = R2                     # ← shared edge with mesh1 outer wall
    R2_MAX = R2 + (R2 - R1)

    mesh2 = assemble_mesh(
        theta_range=(THETA_MIN, THETA_MAX),
        z_range=(Z2_MIN, Z2_MAX),
        r_range=(R2_MIN, R2_MAX),
        RT=RT,
        layer_id=1,                 # different layer_id → different color band
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    layout = go.Layout(
        title=dict(
            text=f"Two Adjacent Annular Volumes  ·  θ ∈ [{np.rad2deg(THETA_MIN):.0f}°, {np.rad2deg(THETA_MAX):.0f}°]",
            font=dict(size=16, color='#e8eaf0'),
            x=0.5,
        ),
        scene=dict(
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

    # Particle direction arrow
    endpoint = particle_direction * 2
    fig.add_trace(go.Scatter3d(
        x=[0, endpoint[0]],
        y=[0, endpoint[1]],
        z=[0, endpoint[2]],
        mode='lines',
        line=dict(color='#ff6eb4', width=5),
        name='Particle Direction'
    ))

    # Rotation frame axes
    for i, ax in enumerate(R):
        fig.add_trace(go.Scatter3d(
            x=[0, ax[0]],
            y=[0, ax[1]],
            z=[0, ax[2]],
            mode='lines',
            line=dict(color=COLORSCALE[i][1], width=3),
            name=f'R_{i}'
        ))

    fig.add_trace(mesh1)
    fig.add_trace(mesh2)

    st.plotly_chart(fig, width="stretch", height="content")


if __name__ == "__main__":
    main()