import logging
import os
from email.policy import default
from threading import local

import altair
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scripts.annular_mesh import assemble_mesh
from scripts.root_utils import (
    BARREL_CONTRIBUTION_KEY,
    BARREL_KEY,
    ENDCAP_CONTRIBUTION_KEY,
    ENDCAP_KEY,
    PARTICLE_KEY,
    compute_relative_position,
    extract_calo_showers,
    get_root_files,
    preprocess_calo_showers,
    preprocess_particles,
    transformation_matrices,
)
from scripts.voxelize import digitize_shower, get_voxels

PLOT_SIZE = (600, 600)
XY_LIMITS = (-1.7, 1.7)
R_LIMIT = (1.20, 1.550)
Z_LIMIT = (-4, 4)

legend_config = dict(
    x=0.02,
    y=0.98,
    xanchor='left',
    yanchor='top',
    bgcolor='rgba(255, 255, 255, 0.8)',
    bordercolor='gray',
    borderwidth=1
)

envelop_xml = "/global/cfs/cdirs/m3443/usr/pmtuan/ddsim_caloxtreme/detector/OpenDataDetectorEnvelopes.xml"
binning_xml = "/global/cfs/cdirs/m3443/usr/pmtuan/ddsim_caloxtreme/binning_dataset_1_pions.xml"

st.set_page_config(page_title="Calorimeter Visualization", layout="wide", page_icon="📈")
st.title("Calorimeter visualization")

# default_dir = os.getcwd()
default_dir = "./run"

root_files = get_root_files(default_dir)
# print(root_files)
if not root_files:
    st.info("No ROOT files found in the selected directory.")
    exit()

labels = [label for label, _ in root_files]
selected_file = st.selectbox("Select a ROOT file", labels, index=0)

selected_path = {f: p for f, p in root_files}[selected_file]

with st.spinner("Extracting calorimeter shower data..."):
    array = extract_calo_showers(selected_path)
    preprocessed_array = preprocess_calo_showers(array)
    preprocessed_array = preprocess_particles(preprocessed_array)
    logging.info(len(preprocessed_array))

st.write(f"Selected file: {selected_file} contains {len(preprocessed_array)} events.")

@st.fragment
@st.cache_data
def make_voxel_3d_chart(voxels, particle_df, barrel_df, selected_z_layer):
    voxel_3d_figure = go.Figure()                    

    selected_voxels = voxels[voxels["z_bin_index"].isin(selected_z_layer)]

    voxel_3d_chart = go.Scatter3d(
        x=selected_voxels["bin_centre_global_x"], 
        y=selected_voxels["bin_centre_global_y"], 
        z=selected_voxels["bin_centre_global_z"], 
        mode="markers",
        marker=dict(
            size=0.05,
        ),
        name="Voxel Centers"
    )

    particle_direction = particle_df[["MCParticles_direction_x", "MCParticles_direction_y", "MCParticles_direction_z"]].to_numpy()[0]
    origin = particle_direction
    direction_endpoint = origin + particle_direction
    direction_line = go.Scatter3d(
        x=[origin[0], direction_endpoint[0]],
        y=[origin[1], direction_endpoint[1]],
        z=[origin[2], direction_endpoint[2]],
        mode="lines",
        line=dict(color="red", width=2),
        name="Particle Direction"
    )

    voxel_3d_figure.add_trace(
        voxel_3d_chart
    )
    voxel_3d_figure.add_trace(
        direction_line
    )

    # voxel_surfaces = go.Figure()
    voxel_traces = []
    for i, vox in voxels.iterrows():
        if vox["z_bin_index"] not in selected_z_layer:
            continue
        if vox["binned_energy"] == 0:
            continue
        voxel_traces .append( assemble_mesh(
            (vox["phi_bin_min"] , vox["phi_bin_max"] ),
            (vox["z_bin_min"] , vox["z_bin_max"] ),
            (vox["r_bin_min"]  , vox["r_bin_max"] ),
            RT,
            vox["binned_energy"] # / voxels["binned_energy"].max()
        ))
        # voxel_3d_figure.add_trace(traces)
    return voxel_3d_chart, direction_line, voxel_traces

@st.cache_data
def make_3d_shower_chart(barrel_df, selected_z_layer=None):

    if selected_z_layer is not None:
        barrel_df = barrel_df[barrel_df["z_bin_index"].isin(selected_z_layer)]

    shower_chart = go.Scatter3d(
        x=barrel_df["x"],
        y=barrel_df["y"],
        z=barrel_df["z"],
        mode="markers",
        marker=dict(
            size=barrel_df["energy"] / barrel_df["energy"].max() * 60,  # Scale marker size by energy
            colorscale="Plotly3",
            color=barrel_df["energy"],  # Color by energy
            colorbar=dict(title="Energy"),
        ),
        name="Calorimeter Hits"
    )

    return shower_chart

if len(preprocessed_array) > 0:

    cols = st.columns(2)
    with cols[0]:
        event_number = st.number_input("Select event number", min_value=0, max_value=len(preprocessed_array)-1, value=0, step=1, format="%d")
    with cols[1]:
        deltaR = st.number_input("Select deltaR for hit selection", min_value=0.0, max_value=0.1, value=0.05, step=0.01)
    event_data = preprocessed_array[event_number]
    # event_data = pd.DataFrame(event_data)
    barrel_keys = [key for key in event_data.fields if key.startswith(f"{BARREL_KEY}.")]

    particle_keys = [key for key in event_data.fields if key.startswith(f"{PARTICLE_KEY}.")]

    barrel_contrib_keys = [key for key in event_data.fields if key.startswith(f"{BARREL_CONTRIBUTION_KEY}.")]


    barrel_df = pd.DataFrame(event_data[barrel_keys].to_list())


    particle_df = pd.DataFrame(event_data[particle_keys].to_list())
    barrel_contrib_df = pd.DataFrame(event_data[barrel_contrib_keys].to_list())
    # print(barrel_contrib_df)
    # print(barrel_contrib_df.columns)
    print("Total energy deposit: ", barrel_contrib_df["ECalBarrelCollectionContributions.energy"].sum())


    barrel_df = barrel_df.rename(columns=lambda col: col.replace(".", "_").split("_")[-1])

    particle_df = particle_df.rename(columns=lambda col: col.replace(".", "_"))
    # print(particle_df.columns)
    momentum = particle_df[["MCParticles_momentum_x", "MCParticles_momentum_y", "MCParticles_momentum_z"]].to_numpy()
    mominetum_at_endpoint = particle_df[["MCParticles_momentumAtEndpoint_x", "MCParticles_momentumAtEndpoint_y", "MCParticles_momentumAtEndpoint_z"]].to_numpy()
    print("particle momentum: ", np.linalg.norm(momentum, axis=1))
    print("particle momentum at endpoint: ", np.linalg.norm(mominetum_at_endpoint, axis=1))
    print("Particle mass: ", particle_df["MCParticles_mass"].to_numpy())
    # print(np.sqrt((particle_df["MCParticles_momentum_x"] ** 2 + particle_df["MCParticles_momentum_y"] ** 2 + particle_df["MCParticles_momentum_z"] ** 2).values[0]))

    barrel_contrib_df = barrel_contrib_df.rename(columns=lambda col: col.replace(".", "_").split("_", 1)[-1])

    barrel_contrib_df = barrel_contrib_df.astype({"PDG": int})




    truth_particle = particle_df.iloc[0]
    vertex = truth_particle[["MCParticles_vertex_x", "MCParticles_vertex_y", "MCParticles_vertex_z"]].to_numpy()
    direction = truth_particle[["MCParticles_momentum_x", "MCParticles_momentum_y", "MCParticles_momentum_z"]].to_numpy()
    direction = direction / np.linalg.norm(direction)
    particle_theta = np.arccos(direction[2])
    particle_eta = - np.log( np.tan(particle_theta / 2) )
    particle_phi = np.arctan2(direction[1], direction[0])
    direction_xy = direction[:2] / np.linalg.norm(direction[:2])
    direction_rz = np.array([direction[2], np.linalg.norm(direction[:2])])
    direction_rz = direction_rz / np.linalg.norm(direction_rz)
    vertex_rz = np.array([vertex[2], np.linalg.norm(vertex[:2])])

    endpoint_r = 10 # max(np.linalg.norm(endpoint[:2]), 1.8)
    endpoint = direction_xy * endpoint_r + vertex[:2]

    direction_rz_df = pd.DataFrame({
        "z": [vertex_rz[0], vertex_rz[0] + direction_rz[0] * endpoint_r],
        "r": [vertex_rz[1], vertex_rz[1] + direction_rz[1] * endpoint_r],
    })

    # barrel_df["delta_phi"] = barrel_df["phi"] - particle_phi

    barel_df = compute_relative_position(barrel_df, particle_df)
    # print(barrel_df.columns)


    # contrib_to_collection = []
    # for idx, row in barrel_df.iterrows():
    #     contrib_to_collection += [idx] * (row["end"] - row["begin"])#.astype(np.int64)
    # contrib_to_collection = np.array(contrib_to_collection)
    # for var in ["x", "y", "z", "r"]:
    #     if len(contrib_to_collection) > 0:
    #         barrel_contrib_df[f"cell_{var}"] = barrel_df[f"{var}"].to_numpy()[contrib_to_collection]

    voxels = get_voxels(particle_df, envelop_xml, binning_xml)

    c = voxels.copy()
    voxels /= 1000
    unitless_cols = ['z_bin_index', 'r_bin_index', 'phi_bin_index','phi_bin_centre', 'phi_bin_min', 'phi_bin_max', 'layer_id']
    voxels[unitless_cols] = c[unitless_cols]

    digitized_barrel_df, voxels = digitize_shower(barrel_df, voxels)

    with st.expander("Barrel Calorimeter Data", expanded=True):
        if len(barrel_df) == 0:
            st.info("No barrel calorimeter hits found for this event.")
        else:
            xy_barrel_fig = go.Figure()
            xy_barrel_fig.add_trace(
                go.Scatter(
                    x=barrel_df["x"],
                    y=barrel_df["y"],
                    mode="markers",
                    marker=dict(
                        size=np.clip(barrel_df["energy"] / barrel_df["energy"].max() * 15, 3, 20),
                        color=barrel_df["energy"],
                        colorscale="Viridis",
                        showscale=False,
                    ),
                    name="Barrel Hits",
                    text=[f"energy: {e}" for e in barrel_df["energy"]],
                    hoverinfo="text",
                )
            )
            xy_barrel_fig.add_trace(
                go.Scatter(
                    x=[vertex[0], endpoint[0]],
                    y=[vertex[1], endpoint[1]],
                    mode="lines",
                    line=dict(color="red", width=1),
                    name="Particle Direction",
                )
            )
            xy_barrel_fig.update_layout(
                title="Barrel Calorimeter Shower",
                width=PLOT_SIZE[0],
                height=PLOT_SIZE[1],
                xaxis=dict(range=[XY_LIMITS[0], XY_LIMITS[1]], title="x"),
                yaxis=dict(range=[XY_LIMITS[0], XY_LIMITS[1]], title="y"),
                margin=dict(l=40, r=20, b=40, t=40),
            )

            rz_barrel_fig = go.Figure()
            rz_barrel_fig.add_trace(
                go.Scatter(
                    x=barrel_df["z"],
                    y=barrel_df["r"],
                    mode="markers",
                    marker=dict(
                        size=np.clip(barrel_df["energy"] / barrel_df["energy"].max() * 15, 3, 20),
                        color=barrel_df["energy"],
                        colorscale="Viridis",
                        showscale=False,
                    ),
                    name="Barrel Hits",
                    text=[f"energy: {e}" for e in barrel_df["energy"]],
                    hoverinfo="text",
                )
            )
            rz_barrel_fig.add_trace(
                go.Scatter(
                    x=direction_rz_df["z"],
                    y=direction_rz_df["r"],
                    mode="lines",
                    line=dict(color="red", width=1),
                    name="Particle Direction",
                )
            )
            rz_barrel_fig.update_layout(
                title="Barrel Calorimeter Shower (Z-R)",
                width=PLOT_SIZE[0],
                height=PLOT_SIZE[1],
                xaxis=dict(range=[Z_LIMIT[0], Z_LIMIT[1]], title="z"),
                yaxis=dict(range=[R_LIMIT[0], R_LIMIT[1]], title="r"),
                margin=dict(l=40, r=20, b=40, t=40),
            )

            xy_barrel_contrib_chart = altair.Chart(barrel_contrib_df).mark_circle().encode(
                x=altair.X("cell_x", scale=altair.Scale(domain=[XY_LIMITS[0], XY_LIMITS[1]])),
                y=altair.Y("cell_y", scale=altair.Scale(domain=[XY_LIMITS[0], XY_LIMITS[1]])),
                size=altair.Size("energy", legend=None),
                color=altair.Color("PDG", legend=None).scale(scheme="category10", domain=sorted(barrel_contrib_df["PDG"].unique())),
                tooltip=["energy"]
            ).properties(title="Barrel Calorimeter Contributions", width=PLOT_SIZE[0], height=PLOT_SIZE[1]).interactive()

            rz_direcion_chart = altair.Chart(direction_rz_df).mark_line(color='red', opacity=0.5).encode(
                x="z",
                y="r"
            ).properties(title="Particle Direction", width=PLOT_SIZE[0], height=PLOT_SIZE[1]).interactive()

            rz_barrel_contrib_chart = altair.Chart(barrel_contrib_df).mark_circle().encode(
                x=altair.X("cell_z", scale=altair.Scale(domain=[Z_LIMIT[0], Z_LIMIT[1]])),
                y=altair.Y("cell_r", scale=altair.Scale(domain=[R_LIMIT[0], R_LIMIT[1]])),
                size="energy",
                color=altair.Color("PDG").scale(scheme="category10", domain=sorted(barrel_contrib_df["PDG"].unique())),
                tooltip=["energy"]
            ).properties(title="Barrel Calorimeter Contributions (Z-R)", width=PLOT_SIZE[0], height=PLOT_SIZE[1]).interactive()

            fig = go.Figure()
            shower_chart = make_3d_shower_chart(barrel_df)

            particle_direction = particle_df[["MCParticles_direction_x", "MCParticles_direction_y", "MCParticles_direction_z"]].to_numpy()[0]
            origin = np.array([0, 0, 0])
            direction_endpoint = origin + particle_direction * 1.8
            direction_line = go.Scatter3d(
                x=[origin[0], direction_endpoint[0]],
                y=[origin[1], direction_endpoint[1]],
                z=[origin[2], direction_endpoint[2]],
                mode="lines",
                line=dict(color="red", width=1),
                name="Particle Trajectory"
            )

            fig.add_trace(shower_chart)
            fig.add_trace(direction_line)
            fig.update_layout(
                title="3D Visualization of Barrel Calorimeter Shower",
                height=700,
                margin=dict(l=0, r=0, b=0, t=30),
                legend=legend_config,
                scene=dict(
                    xaxis_title="X (m)",
                    yaxis_title="Y (m)",
                    zaxis_title="Z (m)"
                )
            )


            st.info("The following charts show the barrel calorimeter showers as energy deposits on individual cells. The size of each point corresponds to the energy deposited on the cell.")
            with st.container():
                cols = st.columns(2)
                with cols[0]:
                    st.plotly_chart(xy_barrel_fig, width="content", height="content")
                with cols[1]:
                    st.plotly_chart(rz_barrel_fig, width="content", height="content")
                st.plotly_chart(fig, width="stretch", height="content")
            
            R, RT = transformation_matrices(particle_direction)

            with st.container():
                all_z_layers = np.sort(voxels["z_bin_index"].unique())
                selected_z_layer = st.multiselect("Select Z layers to display", options=all_z_layers, default=all_z_layers)
                show_voxels = st.checkbox("Show voxelized shower representation", value=True)
                show_shower = st.checkbox("Show original shower hits", value=False)
                with st.spinner("Rendering 3D voxel visualization...", show_time=True):
                    figure = go.Figure()
                    voxel_centers, direction_line, voxel_meshes = make_voxel_3d_chart(voxels, particle_df, barrel_df, selected_z_layer)
                    figure.add_trace(voxel_centers)
                    figure.add_trace(direction_line)
                    if show_voxels:
                        figure.add_traces(voxel_meshes)
                    if show_shower:
                        shower_chart = make_3d_shower_chart(digitized_barrel_df, selected_z_layer)
                        figure.add_trace(shower_chart)
                    figure.update_layout(
                        title=f"3D Voxel Visualization \n Fractional voxelized energy: {voxels['binned_energy'].sum() / barrel_df['energy'].sum():.2f}\n"
                        f"Total voxelized energy deposit: {voxels['binned_energy'].sum():.2f} MeV",
                        height=700,
                        margin=dict(l=0, r=0, b=0, t=30),
                        legend=legend_config
                    )
                    st.plotly_chart(figure, width="stretch", height="content", key="voxel_3d_figure")

else:
    st.warning("No events found in the selected file.")

