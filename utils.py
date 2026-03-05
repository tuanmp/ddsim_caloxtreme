import os
from threading import local

import awkward as ak
import numpy as np
import pandas as pd
import uproot

BARREL_KEY = "ECalBarrelCollection"
ENDCAP_KEY = "ECalEndcapCollection"
BARREL_CONTRIBUTION_KEY = "ECalBarrelCollectionContributions"
ENDCAP_CONTRIBUTION_KEY = "ECalEndcapCollectionContributions"
PARTICLE_KEY = "MCParticles"

def transformation_matrices(r_hat):
    """
    Construct local spherical basis (phi, theta, r)
    from a given unit radial vector r_hat.
    
    Returns:
        R  : 3x3 rotation matrix (global -> local)
        RT : 3x3 inverse rotation (local -> global)
    """

    r_hat = np.asarray(r_hat, dtype=float)
    r_hat = r_hat / np.linalg.norm(r_hat)

    # Choose reference vector to avoid singularity
    if abs(r_hat[2]) < 0.999:
        a = np.array([0.0, 0.0, 1.0])
    else:
        a = np.array([1.0, 0.0, 0.0])

    # phi_hat
    phi_hat = np.cross(a, r_hat)
    phi_hat /= np.linalg.norm(phi_hat)

    # theta_hat
    theta_hat = np.cross(phi_hat, r_hat)

    # Assemble rotation matrix (global -> local)
    R = np.vstack([phi_hat, theta_hat, r_hat])

    # Inverse (local -> global)
    RT = R.T

    return R, RT


def get_root_files(directory: str) -> list[tuple[str, str]]:
    root_files: list[tuple[str, str]] = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".root"):
                full_path = os.path.join(root, file)
                display_path = os.path.relpath(full_path, directory)
                root_files.append((display_path, full_path))
    return sorted(root_files)

def extract_calo_showers(file_path, tree_name="events"):
    """
    Extracts calorimeter shower data from a ROOT file and returns it as a pandas DataFrame.

    Parameters:
    - file_path: str, path to the ROOT file
    - tree_name: str, name of the TTree containing the calorimeter shower data

    Returns:
    - DataFrame containing the calorimeter shower data
    """
    with uproot.open(file_path) as file:
        tree = file[tree_name]
        array = tree.arrays(library="ak")
    
    return array

def preprocess_calo_showers(array: ak.Array):
    """
    Preprocesses the calorimeter shower DataFrame by applying necessary transformations.

    Parameters:
    - array: ak.Array, raw calorimeter shower data, containing showers, contributions, and particle information

    Returns:
    - array: ak.Array, preprocessed calorimeter shower data with additional features and transformations applied
    """
    # Example preprocessing steps (these can be modified based on specific requirements)

    array['ECalBarrelCollection.position.x'] = array['ECalBarrelCollection.position.x'] / 1000  # Convert from mm to m
    array['ECalBarrelCollection.position.y'] = array['ECalBarrelCollection.position.y'] / 1000  # Convert from mm to m
    array['ECalBarrelCollection.position.z'] = array['ECalBarrelCollection.position.z'] / 1000  # Convert from mm to m
    
    array['ECalBarrelCollection.position.r'] = np.sqrt( 
        array['ECalBarrelCollection.position.x'] ** 2 + \
        array['ECalBarrelCollection.position.y'] ** 2 
    )

    array['ECalBarrelCollection.position.phi'] = np.arctan2( 
        array['ECalBarrelCollection.position.y'],
        array['ECalBarrelCollection.position.x'] 
    )

    theta = np.arctan2( 
        array['ECalBarrelCollection.position.r'],
        array['ECalBarrelCollection.position.z'] 
    )

    array['ECalBarrelCollection.position.eta'] = - np.log( 
        np.tan(theta / 2)
    )

    array["ECalBarrelCollection.energy"] = array["ECalBarrelCollection.energy"] * 1000

    array['ECalEndcapCollection.position.x'] = array['ECalEndcapCollection.position.x'] / 1000  # Convert from mm to m
    array['ECalEndcapCollection.position.y'] = array['ECalEndcapCollection.position.y'] / 1000  # Convert from mm to m
    array['ECalEndcapCollection.position.z'] = array['ECalEndcapCollection.position.z'] / 1000  # Convert from mm to m
    
    array['ECalEndcapCollection.position.r'] = np.sqrt( 
        array['ECalEndcapCollection.position.x'] ** 2 + \
        array['ECalEndcapCollection.position.y'] ** 2 
    )

    array['ECalEndcapCollection.position.phi'] = np.arctan2( 
        array['ECalEndcapCollection.position.y'],
        array['ECalEndcapCollection.position.x'] 
    )

    theta = np.arctan2( 
        array['ECalEndcapCollection.position.r'],
        array['ECalEndcapCollection.position.z'] 
    )

    array['ECalEndcapCollection.position.eta'] = - np.log( 
        np.tan(theta / 2)
    )

    array["ECalEndcapCollection.energy"] = array["ECalEndcapCollection.energy"] * 1000

    array["ECalBarrelCollectionContributions.stepPosition.x"] = array["ECalBarrelCollectionContributions.stepPosition.x"] 
    array["ECalBarrelCollectionContributions.stepPosition.y"] = array["ECalBarrelCollectionContributions.stepPosition.y"] 
    array["ECalBarrelCollectionContributions.stepPosition.z"] = array["ECalBarrelCollectionContributions.stepPosition.z"] 
    array["ECalBarrelCollectionContributions.energy"] = array["ECalBarrelCollectionContributions.energy"] * 1000

    array["ECalBarrelCollectionContributions.stepPosition.r"] = np.sqrt(
        array["ECalBarrelCollectionContributions.stepPosition.x"] ** 2 + \
        array["ECalBarrelCollectionContributions.stepPosition.y"] ** 2
    )

    array["ECalEndcapCollectionContributions.stepPosition.x"] = array["ECalEndcapCollectionContributions.stepPosition.x"] / 1000
    array["ECalEndcapCollectionContributions.stepPosition.y"] = array["ECalEndcapCollectionContributions.stepPosition.y"] / 1000
    array["ECalEndcapCollectionContributions.stepPosition.z"] = array["ECalEndcapCollectionContributions.stepPosition.z"] / 1000
    array["ECalEndcapCollectionContributions.energy"] = array["ECalEndcapCollectionContributions.energy"] * 1000

    array["ECalEndcapCollectionContributions.stepPosition.r"] = np.sqrt(
        array["ECalEndcapCollectionContributions.stepPosition.x"] ** 2 + \
        array["ECalEndcapCollectionContributions.stepPosition.y"] ** 2
    )

    return array

def preprocess_particles(array: ak.Array):
    # Example processing steps for particles (these can be modified based on specific requirements)
    array['MCParticles.endpoint.x'] = array['MCParticles.endpoint.x'] / 1000  # Convert from mm to m
    array['MCParticles.endpoint.y'] = array['MCParticles.endpoint.y'] / 1000  # Convert from mm to m
    array['MCParticles.endpoint.z'] = array['MCParticles.endpoint.z'] / 1000  # Convert from mm to m

    array['MCParticles.vertex.x'] = array['MCParticles.vertex.x'] / 1000  # Convert from mm to m
    array['MCParticles.vertex.y'] = array['MCParticles.vertex.y'] / 1000  # Convert from mm to m
    array['MCParticles.vertex.z'] = array['MCParticles.vertex.z'] / 1000  # Convert from mm to m

    array["MCParticles.endpoint.r"] = np.sqrt(
        array["MCParticles.endpoint.x"] ** 2 + \
        array["MCParticles.endpoint.y"] ** 2 
    )

    array["MCParticles.vertex.r"] = np.sqrt(
        array["MCParticles.vertex.x"] ** 2 + \
        array["MCParticles.vertex.y"] ** 2 
    )

    momentum_magnitude = np.sqrt(
        array["MCParticles.momentum.x"] ** 2 + \
        array["MCParticles.momentum.y"] ** 2 + \
        array["MCParticles.momentum.z"] ** 2
    )

    array["MCParticles.direction.x"] = array["MCParticles.momentum.x"] / momentum_magnitude
    array["MCParticles.direction.y"] = array["MCParticles.momentum.y"] / momentum_magnitude
    array["MCParticles.direction.z"] = array["MCParticles.momentum.z"] / momentum_magnitude

    return array

def compute_relative_position(shower_df: pd.DataFrame, particle_df: pd.DataFrame) -> pd.DataFrame:

    assert len(particle_df) == 1, "Expected exactly one particle per event for relative position computation"
    # Compute relative position of shower pixels to the direction of propagation of the particle
    position_keys = ["x", "y", "z"]
    global_position = shower_df[position_keys].to_numpy()  # Convert to NumPy array for vectorized operations


    direction_keys = ["MCParticles_direction_x", "MCParticles_direction_y", "MCParticles_direction_z"]
    direction = particle_df[direction_keys].to_numpy().flatten()

    # # compute local position of shower pixel relative to particle direction
    # local_z = np.dot(global_position, direction)
    # # parallel component of position along particle direction
    # local_position_parallel = local_z[:, np.newaxis] * direction
    # # perpendicular component of position relative to particle direction
    # local_position_perpendicular = global_position - local_position_parallel

    # # get phi_hat and theta_hat unit vectors for the particle direction
    # phi_hat = np.cross(np.array([0, 0, 1]), direction)
    # phi_hat /= np.linalg.norm(phi_hat)

    # theta_hat = np.cross(phi_hat, direction)
    # theta_hat /= np.linalg.norm(theta_hat)

    # # get local coordinates of shower pixel in particle's local frame
    # local_x = np.dot(local_position_perpendicular, phi_hat)
    # local_y = np.dot(local_position_perpendicular, theta_hat)

    # # verify that the magnitude of perpendicular component is consistent with local_x and local_y
    # local_r_perpendicular = np.sqrt(local_x ** 2 + local_y ** 2)
    # local_r_perpendicular_check = np.linalg.norm(local_position_perpendicular, axis=1)
    # assert np.allclose(local_r_perpendicular, local_r_perpendicular_check), "Inconsistent perpendicular distance computation"

    R, RT = transformation_matrices(direction)

    # local_coordinates_check = (R @ global_position.T).T
    # print(local_coordinates_check[:5])
    # print(np.stack([local_x, local_y, local_z], axis=1)[:5])
    # assert np.allclose(local_coordinates_check, np.stack([local_x, local_y, local_z], axis=1)), "Inconsistent local coordinate transformation"

    # local_coordinates = np.stack([local_x, local_y, local_z], axis=1)
    local_coordinates = (R @ global_position.T).T
    shower_df[["local_x", "local_y", "local_z"]] = local_coordinates

    shower_df["local_r"] = np.linalg.norm(local_coordinates[:, :2], axis=1)  # Radial distance in the local frame (perpendicular to particle direction)
    shower_df["local_phi"] = np.arctan2(local_coordinates[:, 1], local_coordinates[:, 0])  # Azimuthal angle in the local frame

    return shower_df

