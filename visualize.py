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

from annular_mesh import assemble_mesh
from annular_surface import assemble_traces
from utils import (
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
from voxelize import digitize_shower, get_voxels

PLOT_SIZE = (600, 600)
XY_LIMITS = (-1.7, 1.7)
R_LIMIT = (1.20, 1.550)
Z_LIMIT = (-4, 4)

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

if len(preprocessed_array) > 0:

    cols = st.columns(3)
    with cols[0]:
        event_number = st.number_input("Select event number", min_value=0, max_value=len(preprocessed_array)-1, value=0, step=1, format="%d")
    with cols[1]:
        psi = st.number_input("Select cone angle (degrees)", min_value=0.0, max_value=45., value=10.0, step=0.5)
    with cols[2]:
        deltaR = st.number_input("Select deltaR for hit selection", min_value=0.0, max_value=0.1, value=0.05, step=0.01)
    event_data = preprocessed_array[event_number]
    # event_data = pd.DataFrame(event_data)
    barrel_keys = [key for key in event_data.fields if key.startswith(f"{BARREL_KEY}.")]
    endcap_keys = [key for key in event_data.fields if key.startswith(f"{ENDCAP_KEY}.")]
    particle_keys = [key for key in event_data.fields if key.startswith(f"{PARTICLE_KEY}.")]

    barrel_contrib_keys = [key for key in event_data.fields if key.startswith(f"{BARREL_CONTRIBUTION_KEY}.")]
    endcap_contrib_keys = [key for key in event_data.fields if key.startswith(f"{ENDCAP_CONTRIBUTION_KEY}.")]

    barrel_df = pd.DataFrame(event_data[barrel_keys].to_list())
    # print(barrel_df)
    # print(barrel_df.columns)
    endcap_df = pd.DataFrame(event_data[endcap_keys].to_list())

    particle_df = pd.DataFrame(event_data[particle_keys].to_list())
    barrel_contrib_df = pd.DataFrame(event_data[barrel_contrib_keys].to_list())
    # print(barrel_contrib_df)
    # print(barrel_contrib_df.columns)
    endcap_contrib_df = pd.DataFrame(event_data[endcap_contrib_keys].to_list())

    barrel_df = barrel_df.rename(columns=lambda col: col.replace(".", "_").split("_")[-1])
    endcap_df = endcap_df.rename(columns=lambda col: col.replace(".", "_").split("_")[-1])
    particle_df = particle_df.rename(columns=lambda col: col.replace(".", "_"))
    barrel_contrib_df = barrel_contrib_df.rename(columns=lambda col: col.replace(".", "_").split("_", 1)[-1])
    endcap_contrib_df = endcap_contrib_df.rename(columns=lambda col: col.replace(".", "_").split("_", 1)[-1])
    barrel_contrib_df = barrel_contrib_df.astype({"PDG": int})
    endcap_contrib_df = endcap_contrib_df.astype({"PDG": int})

    print(particle_df.columns)
    # print(particle_df.iloc[0])
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

    # theta = 10 # in degrees
    def rotation_matrix_2d(angle):
        rad = np.radians(angle)
        return np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])

    upper_line_direction = rotation_matrix_2d( psi) @ direction_xy
    upper_line = vertex[:2] + upper_line_direction * endpoint_r
    upper_line = (upper_line * np.abs(endpoint[0] / upper_line[0])) #if upper_line[0] != 0 else upper_line

    # print(upper_line_direction)
    lower_line_direction = rotation_matrix_2d(-psi) @ direction_xy
    lower_line = vertex[:2] + lower_line_direction * endpoint_r
    lower_line = (lower_line * np.abs(endpoint[0] / lower_line[0])) #if lower_line[0] != 0 else lower_line
    # print(lower_line_direction)

    cone = {
        "x": [vertex[0], endpoint[0]],
        "y": [vertex[1], endpoint[1]],
        "y1": [vertex[1], upper_line[1]],
        "y2": [vertex[1], lower_line[1]],
    }
    cone_xy = pd.DataFrame(cone)

    # print(cone_xy)

    upper_line_direction_rz = rotation_matrix_2d(psi) @ direction_rz
    upper_line_rz = vertex_rz + upper_line_direction_rz * endpoint_r
    lower_line_direction_rz = rotation_matrix_2d(-psi) @ direction_rz
    lower_line_rz = vertex_rz + lower_line_direction_rz * endpoint_r
    cone_rz  = {
        "z": [vertex_rz[0], vertex_rz[0] + direction_rz[0] * endpoint_r],
        "r": [vertex_rz[1], vertex_rz[1] + direction_rz[1] * endpoint_r],
        "z1": [vertex_rz[0], upper_line_rz[0]],
        "z2": [vertex_rz[0], lower_line_rz[0]],
    }
    cone_rz = pd.DataFrame(cone_rz)

    barrel_df["delta_phi"] = barrel_df["phi"] - particle_phi
    barrel_df["delta_eta"] = barrel_df["eta"] - particle_eta
    barrel_df["delta_r"] = np.sqrt(barrel_df["delta_phi"] ** 2 + barrel_df["delta_eta"] ** 2)
    barrel_df["contained_in_cone"] = barrel_df["delta_r"] < deltaR


    endcap_df["delta_phi"] = endcap_df["phi"] - particle_phi
    endcap_df["delta_eta"] = endcap_df["eta"] - particle_eta
    endcap_df["delta_r"] = np.sqrt(endcap_df["delta_phi"] ** 2 + endcap_df["delta_eta"] ** 2)
    endcap_df["contained_in_cone"] = endcap_df["delta_r"] < deltaR

    barel_df = compute_relative_position(barrel_df, particle_df)
    # print(barrel_df.columns)


    contrib_to_collection = []
    for idx, row in barrel_df.iterrows():
        contrib_to_collection += [idx] * (row["end"] - row["begin"])#.astype(np.int64)
    contrib_to_collection = np.array(contrib_to_collection)
    for var in ["x", "y", "z", "r"]:
        if len(contrib_to_collection) > 0:
            barrel_contrib_df[f"cell_{var}"] = barrel_df[f"{var}"].to_numpy()[contrib_to_collection]
    
    contrib_to_collection = []
    for idx, row in endcap_df.iterrows():
        contrib_to_collection += [idx] * (row["end"] - row["begin"])#.astype(np.int64)
    contrib_to_collection = np.array(contrib_to_collection)
    for var in ["x", "y", "z", "r"]:
        if len(contrib_to_collection) > 0:
            endcap_contrib_df[f"cell_{var}"] = endcap_df[f"{var}"].to_numpy()[contrib_to_collection]

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
            xy_barrel_chart = altair.Chart(barrel_df).mark_circle().encode(
                x=altair.X("x", scale=altair.Scale(domain=[XY_LIMITS[0], XY_LIMITS[1]])),
                y=altair.Y("y", scale=altair.Scale(domain=[XY_LIMITS[0], XY_LIMITS[1]]), title="y"),
                size=altair.Size("energy", legend=None),
                tooltip=["energy"]
            ).properties(title="Barrel Calorimeter Shower", width=PLOT_SIZE[0], height=PLOT_SIZE[1]).interactive()

            xy_direction_chart = altair.Chart(cone_xy).mark_line(color='red', opacity=0.5).encode(
                x="x",
                y="y"
            ).properties(title="Particle Direction", width=PLOT_SIZE[0], height=PLOT_SIZE[1]).interactive()

            xy_cone = xy_direction_chart.mark_area(opacity=0.2, color='#57A44C').encode(
                y=altair.Y("y1", title="y"),
                y2=altair.Y2("y2", title=None)
            ).properties(title="Particle Direction Cone", width=PLOT_SIZE[0], height=PLOT_SIZE[1]).interactive()

            cone_plot_xy = altair.layer(xy_direction_chart, xy_cone).resolve_scale(x="shared", y="shared")

            rz_barrel_chart = altair.Chart(barrel_df).mark_circle().encode(
                x=altair.X("z", scale=altair.Scale(domain=[Z_LIMIT[0], Z_LIMIT[1]])),
                y=altair.Y("r", scale=altair.Scale(domain=[R_LIMIT[0], R_LIMIT[1]])),
                size="energy",
                tooltip=[
                    "energy",
                    "x",
                    "y",
                ]
            ).properties(title="Barrel Calorimeter Shower (Z-R)", width=PLOT_SIZE[0], height=PLOT_SIZE[1]).interactive()

            xy_barrel_contrib_chart = altair.Chart(barrel_contrib_df).mark_circle().encode(
                x=altair.X("cell_x", scale=altair.Scale(domain=[XY_LIMITS[0], XY_LIMITS[1]])),
                y=altair.Y("cell_y", scale=altair.Scale(domain=[XY_LIMITS[0], XY_LIMITS[1]])),
                size=altair.Size("energy", legend=None),
                color=altair.Color("PDG", legend=None).scale(scheme="category10", domain=sorted(barrel_contrib_df["PDG"].unique())),
                tooltip=["energy"]
            ).properties(title="Barrel Calorimeter Contributions", width=PLOT_SIZE[0], height=PLOT_SIZE[1]).interactive()

            rz_direcion_chart = altair.Chart(cone_rz).mark_line(color='red', opacity=0.5).encode(
                x="z",
                y="r"
            ).properties(title="Particle Direction", width=PLOT_SIZE[0], height=PLOT_SIZE[1]).interactive()

            rz_cone = rz_direcion_chart.mark_area(opacity=0.2, color='#57A44C').encode(
                x=altair.X("z1", title="z"),
                x2=altair.X2("z2", title=None)
            ).properties(title="Particle Direction Cone", width=PLOT_SIZE[0], height=PLOT_SIZE[1]).interactive()   

            cone_plot_rz = altair.layer(rz_direcion_chart, rz_cone).resolve_scale(x="shared", y="shared")

            rz_barrel_contrib_chart = altair.Chart(barrel_contrib_df).mark_circle().encode(
                x=altair.X("cell_z", scale=altair.Scale(domain=[Z_LIMIT[0], Z_LIMIT[1]])),
                y=altair.Y("cell_r", scale=altair.Scale(domain=[R_LIMIT[0], R_LIMIT[1]])),
                size="energy",
                color=altair.Color("PDG").scale(scheme="category10", domain=sorted(barrel_contrib_df["PDG"].unique())),
                tooltip=["energy"]
            ).properties(title="Barrel Calorimeter Contributions (Z-R)", width=PLOT_SIZE[0], height=PLOT_SIZE[1]).interactive()

            # print(barrel_df)
            cone_view = altair.Chart(barrel_df).mark_circle().encode(
                x=altair.X("delta_phi", scale=altair.Scale(domain=[(-deltaR) * 1.1, deltaR * 1.1])),
                y=altair.Y("delta_eta", scale=altair.Scale(domain=[(-deltaR) * 1.1, deltaR * 1.1])),
                size=altair.Size("energy", legend=None),
                color=altair.Color("contained_in_cone", legend=None).scale(scheme="category10"),
                tooltip=["energy", "contained_in_cone"]
            ).properties(title="Barrel Calorimeter Shower with Cone Highlighting", width=PLOT_SIZE[0], height=PLOT_SIZE[1]).interactive()

            fig = go.Figure()
            shower_chart = go.Scatter3d(
                x=barrel_df["x"],
                y=barrel_df["y"],
                z=barrel_df["z"],
                mode="markers",
                marker=dict(
                    size=barrel_df["energy"] / barrel_df["energy"].max() * 40,  # Scale marker size by energy
                    colorscale="Plotly3",
                    color=barrel_df["energy"],  # Color by energy
                    colorbar=dict(title="Energy"),
                ),
                name="Calorimeter Hits"
            )

            particle_direction = particle_df[["MCParticles_direction_x", "MCParticles_direction_y", "MCParticles_direction_z"]].to_numpy()[0]
            origin = np.array([0, 0, 0])
            direction_endpoint = origin + particle_direction * 1.8
            direction_line = go.Scatter3d(
                x=[origin[0], direction_endpoint[0]],
                y=[origin[1], direction_endpoint[1]],
                z=[origin[2], direction_endpoint[2]],
                mode="lines",
                line=dict(color="red", width=2),
                name="Particle Trajectory"
            )

            fig.add_trace(shower_chart)
            fig.add_trace(direction_line)
            fig.update_layout(
                title="3D Visualization of Barrel Calorimeter Shower",
                height=700,
                margin=dict(l=0, r=0, b=0, t=30),
                legend=dict(
                    x=0.02,
                    y=0.98,
                    xanchor='left',
                    yanchor='top',
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='gray',
                    borderwidth=1
                ),
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
                    st.altair_chart(xy_barrel_chart + cone_plot_xy, width="content", height="content")
                with cols[1]:
                    st.altair_chart(rz_barrel_chart + cone_plot_rz, width="content", height="content")
                st.plotly_chart(fig, width="stretch", height="content")
            
            # local_xy_voxel_chart = altair.Chart(voxels).mark_circle().encode(
            #     x=altair.X("rphi_bin_centre_x", scale=altair.Scale(domain=[XY_LIMITS[0] , XY_LIMITS[1] ])),
            #     y=altair.Y("rphi_bin_centre_y", scale=altair.Scale(domain=[XY_LIMITS[0] , XY_LIMITS[1] ]), title="y"),
            #     color="layer_id:N",
            #     # size=1,
            #     # tooltip=["energy"]
            # ).properties(title="Voxels (Local)", width=PLOT_SIZE[0], height=PLOT_SIZE[1]).interactive()

            # local_rz_voxel_chart = altair.Chart(voxels).mark_circle().encode(
            #     x=altair.X("rphi_bin_centre_x", scale=altair.Scale(domain=[Z_LIMIT[0] , Z_LIMIT[1] ])),
            #     y=altair.Y("z_bin_centre", scale=altair.Scale(domain=[R_LIMIT[0] , R_LIMIT[1] ])),
            #     color="layer_id:N"
            # ).properties(title="Voxels (Local, Z-R)", width=PLOT_SIZE[0], height=PLOT_SIZE[1]).interactive()

            R, RT = transformation_matrices(particle_direction)

            
            # with st.container():
            #     cols = st.columns(2)
            #     with cols[0]:
            #         st.altair_chart(local_xy_voxel_chart, width="content", height="content")
            #     with cols[1]:
            #         st.altair_chart(local_rz_voxel_chart, width="content", height="content")

            with st.container():
                voxel_3d_figure = go.Figure()                    

                all_z_layers = np.sort(voxels["z_bin_index"].unique())
                selected_z_layer = st.multiselect("Select Z layers to display", options=all_z_layers, default=all_z_layers)

                selected_voxels = voxels[voxels["z_bin_index"].isin(selected_z_layer)]

                voxel_3d_chart = go.Scatter3d(
                    x=selected_voxels["bin_centre_global_x"], 
                    y=selected_voxels["bin_centre_global_y"], 
                    z=selected_voxels["bin_centre_global_z"], 
                    mode="markers",
                    marker=dict(
                        size=1,
                        # color=selected_voxels["layer_id"],
                        # colorscale=px.colors.qualitative.G10,
                        # colorbar={"title": "Layer ID"},
                    ),
                    name="Voxel Centers"
                )

                particle_direction = particle_df[["MCParticles_direction_x", "MCParticles_direction_y", "MCParticles_direction_z"]].to_numpy()[0]
                origin = particle_direction * 1.05
                direction_endpoint = origin + particle_direction * 1.06
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
                for i, vox in voxels.iterrows():
                    if vox["z_bin_index"] not in selected_z_layer:
                        continue
                    if vox["binned_energy"] == 0:
                        continue
                    traces = assemble_mesh(
                        (vox["phi_bin_min"] , vox["phi_bin_max"] ),
                        (vox["z_bin_min"] , vox["z_bin_max"] ),
                        (vox["r_bin_min"]  , vox["r_bin_max"] ),
                        RT,
                        vox["binned_energy"] / voxels["binned_energy"].max()
                    )
                    voxel_3d_figure.add_trace(traces)

                voxel_3d_figure.update_layout(
                    title=f"3D Voxel Visualization \n Fractional voxelized energy: {voxels['binned_energy'].sum() / barrel_df['energy'].sum():.2f}",
                    scene=dict(
                        xaxis=dict(
                        zeroline=True,
                        zerolinecolor='black',
                        zerolinewidth=3,
                        ),
                    yaxis=dict(
                        zeroline=True,
                        zerolinecolor='black',
                        zerolinewidth=3,
                        ),
                    zaxis=dict(
                        zeroline=True,
                        zerolinecolor='black',
                        zerolinewidth=3,
                        ),
                    ),
                    height=700,
                    margin=dict(l=0, r=0, b=0, t=30),
                    legend=dict(
                        x=0.02,
                        y=0.98,
                        xanchor='left',
                        yanchor='top',
                        bgcolor='rgba(255, 255, 255, 0.8)',
                        bordercolor='gray',
                        borderwidth=1
                    )
                )
                st.plotly_chart(voxel_3d_figure, width="stretch", height="content")

            # print(voxels)
            # print(voxels.columns)
            # print(digitized_barrel_df)
            # print(digitized_barrel_df.columns)

            # st.info("The following charts show each particle's contribution to the cellular energy deposit. The size of each point corresponds to the energy contribution, and the color is categorically encoded by particle PDG ID.")
            # with st.container():
            #     cols = st.columns(2)
            #     with cols[0]:
            #         st.altair_chart(xy_barrel_contrib_chart + cone_plot_xy, width="content", height="content")
            #     with cols[1]:
            #         st.altair_chart(rz_barrel_contrib_chart + cone_plot_rz, width="content", height="content")
    
    # with st.expander("Endcap Calorimeter Data", expanded=True):
    #     if len(endcap_df) == 0:
    #         st.info("No endcap calorimeter hits found for this event.")
    #     else:
    #         xy_endcap_chart = altair.Chart(endcap_df).mark_circle().encode(
    #             x=altair.X("x", scale=altair.Scale(domain=[XY_LIMITS[0], XY_LIMITS[1]])),
    #             y=altair.Y("y", scale=altair.Scale(domain=[XY_LIMITS[0], XY_LIMITS[1]])),
    #             size=altair.Size("energy", legend=None),
    #             tooltip=["energy"]
    #         ).properties(title="Endcap Calorimeter Shower", width=PLOT_SIZE[0], height=PLOT_SIZE[1]).interactive()

    #         rz_endcap_chart = altair.Chart(endcap_df).mark_circle().encode(
    #             x=altair.X("z", scale=altair.Scale(domain=[Z_LIMIT[0], Z_LIMIT[1]])),
    #             y=altair.Y("r", scale=altair.Scale(domain=[R_LIMIT[0], R_LIMIT[1]])),
    #             size="energy",
    #             tooltip=[
    #                 "energy"
    #             ]
    #         ).properties(title="Endcap Calorimeter Shower (R-Z)", width=PLOT_SIZE[0], height=PLOT_SIZE[1]).interactive()

    #         xy_endcap_contrib_chart = altair.Chart(endcap_contrib_df).mark_circle().encode(
    #             x=altair.X("cell_x", scale=altair.Scale(domain=[XY_LIMITS[0], XY_LIMITS[1]])),
    #             y=altair.Y("cell_y", scale=altair.Scale(domain=[XY_LIMITS[0], XY_LIMITS[1]])),
    #             size=altair.Size("energy", legend=None),
    #             color=altair.Color("PDG", legend=None).scale(scheme="category10", domain=sorted(endcap_contrib_df["PDG"].unique())),
    #             tooltip=["energy"]
    #         ).properties(title="Endcap Calorimeter Contributions", width=PLOT_SIZE[0], height=PLOT_SIZE[1]).interactive()

    #         rz_endcap_contrib_chart = altair.Chart(endcap_contrib_df).mark_circle().encode(
    #             x=altair.X("cell_z", scale=altair.Scale(domain=[Z_LIMIT[0], Z_LIMIT[1]])),
    #             y=altair.Y("cell_r", scale=altair.Scale(domain=[R_LIMIT[0], R_LIMIT[1]])),
    #             size="energy",
    #             color=altair.Color("PDG").scale(scheme="category10", domain=sorted(endcap_contrib_df["PDG"].unique())),
    #             tooltip=["energy"]
    #         ).properties(title="Endcap Calorimeter Contributions (R-Z)", width=PLOT_SIZE[0], height=PLOT_SIZE[1]).interactive()

    #         st.info("The following charts show the endcap calorimeter showers as energy deposits on individual cells. The size of each point corresponds to the energy deposited on the cell.")
            
    #         with st.container():
    #             cols = st.columns(2)
    #             with cols[0]:
    #                 st.altair_chart(xy_endcap_chart, width="content", height="stretch")
    #             with cols[1]:
    #                 st.altair_chart(rz_endcap_chart, width="content", height="stretch")
            
    #         st.info("The following charts show each particle's contribution to the cellular energy deposit. The size of each point corresponds to the energy contribution, and the color is categorically encoded by particle PDG ID.")
    #         with st.container():
    #             cols = st.columns(2)
    #             with cols[0]:
    #                 st.altair_chart(xy_endcap_contrib_chart, width="content", height="stretch")
    #             with cols[1]:
    #                 st.altair_chart(rz_endcap_contrib_chart, width="content", height="stretch")


else:
    st.warning("No events found in the selected file.")

