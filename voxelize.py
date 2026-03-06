import re
import xml.etree.ElementTree as ET
from threading import local

import awkward as ak
import numpy as np
import pandas as pd
import vector

from utils import transformation_matrices


def get_ecal_barrel_dimensions(xml_file):
    """
    Read the OpenDataDetectorEnvelopes.xml file and extract ECal barrel dimensions.
    
    Args:
        xml_file: Path to the XML file (e.g., 'detector/OpenDataDetectorEnvelopes.xml')
        
    Returns:
        tuple: (ecal_b_rmin, ecal_b_rmax) in millimeters
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Define unit conversions to millimeters
    unit_conversions = {
        'mm': 1.0,
        'cm': 10.0,
        'm': 1000.0,
        'km': 1000000.0,
    }
    
    def parse_value(value_str):
        """Parse a value string like '1250.*mm' and convert to millimeters."""
        # Remove whitespace
        value_str = value_str.strip()
        
        # Extract the numeric part and unit
        # Pattern matches expressions like "1250.*mm", "1.5*m", "100*cm", etc.
        match = re.match(r'([\d.]+)\s*\*?\s*([a-zA-Z]+)', value_str)
        if match:
            numeric_value = float(match.group(1))
            unit = match.group(2)
            
            # Convert to millimeters
            if unit in unit_conversions:
                return numeric_value * unit_conversions[unit]
            else:
                raise ValueError(f"Unknown unit: {unit}")
        else:
            raise ValueError(f"Cannot parse value: {value_str}")
    
    # Find the constants
    ecal_b_rmin = None
    ecal_b_rmax = None
    
    for constant in root.findall('.//constant'):
        name = constant.get('name')
        value = constant.get('value')
        
        if name == 'ecal_b_rmin':
            ecal_b_rmin = parse_value(value)
        elif name == 'ecal_b_rmax':
            ecal_b_rmax = parse_value(value)
    
    if ecal_b_rmin is None or ecal_b_rmax is None:
        raise ValueError("Could not find ecal_b_rmin or ecal_b_rmax in XML file")
    
    return ecal_b_rmin, ecal_b_rmax


def digitize_in_z(shower_df: pd.DataFrame, voxel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Digitize the ECal barrel shower contributions into radial bins.
    
    Args:
        shower_df: DataFrame containing the shower pixels with their positions and energies
        zmin: Minimum radius (in millimeters)
        zmax: Maximum radius (in millimeters)
        num_bins: Number of radial bins
        
    Returns:
        Awkward Array with an additional field 'ECalBarrelCollection.energy_binned' containing the binned energies.

    ### TODO: This is currently implemented to take the entire trajectory in z
    In calochallenge, only the portion of the shower contained within (delta z) * (N z bins) is considered for voxelization.
    """
    z_bins = (voxel_df[['z_bin_index', "z_bin_min", "z_bin_max"]]).drop_duplicates()

    # zmin = z_bins["z_bin_min"].min()
    # zmax = z_bins["z_bin_max"].max()
    
    # num_bins = len(z_bins)

    shower_df['z_bin_index'] = -1  # Initialize with -1 for pixels that don't fall into any bin

    for z_bin_index, z_bin_min, z_bin_max in z_bins.itertuples(index=False):
        in_bin = (shower_df["local_z"] >= z_bin_min) & (shower_df["local_z"] < z_bin_max)
        shower_df.loc[in_bin, 'z_bin_index'] = z_bin_index
    
    return shower_df

def digitize_in_r_phi(shower_df: pd.DataFrame, voxel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Digitize the ECal barrel shower contributions into radial and azimuthal bins.
    
    Args:
        shower_df: DataFrame containing the shower pixels with their positions and energies
        z_bin: The z-bin index for which to perform the r-phi digitization
        r_edges: List of radial bin edges (in millimeters)
        n_bin_alpha: Number of azimuthal bins
    Returns: DataFrame with additional fields 'r_bin' and 'phi_bin' containing the bin indices for radial and azimuthal bins.
    """

    if "r_bin_index" not in shower_df.columns:
        shower_df["r_bin_index"] = -1  # Initialize with -1 for pixels that don't fall into any bin
    if "phi_bin_index" not in shower_df.columns:
        shower_df["phi_bin_index"] = -1  # Initialize with -1 for pixels that don't fall into any bin

    for z_bin_index in voxel_df['z_bin_index'].unique():
        layer = voxel_df[voxel_df['z_bin_index'] == z_bin_index]

        for _, row in layer.iterrows():
            r_bin_index = row['r_bin_index']
            r_bin_min = row['r_bin_min']
            r_bin_max = row['r_bin_max']
            phi_bin_index = row['phi_bin_index']
            phi_bin_min = row['phi_bin_min']
            phi_bin_max = row['phi_bin_max']

            local_phi = np.mod(shower_df["local_phi"], 2 * np.pi) # Ensure phi is in the range [0, 2*pi]

            in_bin = (
                (shower_df["local_r"] >= r_bin_min) & (shower_df["local_r"] < r_bin_max) & \
                (local_phi >= phi_bin_min) & (local_phi < phi_bin_max) & \
                (shower_df["z_bin_index"] == z_bin_index)
            )

            shower_df.loc[in_bin, 'r_bin_index'] = r_bin_index
            shower_df.loc[in_bin, 'phi_bin_index'] = phi_bin_index

    # # Filter the shower pixels for the specified z-bin
    # shower_df_z_bin = shower_df[shower_df['z_bin'] == z_bin]
    
    # # Digitize the radial positions to get radial bin indices
    # local_r = shower_df_z_bin["local_r"].to_numpy()  # Convert to NumPy array for digitization
    # r_bin_indices = np.digitize(local_r, bins=r_edges) - 1  # Convert to 0-based index
    
    # # Compute azimuthal angles (phi) and digitize to get azimuthal bin indices
    # local_phi = shower_df_z_bin["local_phi"].to_numpy()  # Convert to NumPy array for digitization
    # # phi[phi < 0] += 2 * np.pi  # Convert to range [0, 2*pi]
    # local_phi = np.mod(local_phi, 2 * np.pi)  # Ensure phi is in the range [0, 2*pi]
    
    # phi_bin_indices = np.floor(local_phi / (2 * np.pi / n_bin_alpha)).astype(int)
    
    # # Assign the computed bin indices back to the DataFrame
    # shower_df[ shower_df['z_bin'] == z_bin, 'r_bin'] = r_bin_indices
    # shower_df[ shower_df['z_bin'] == z_bin, 'phi_bin'] = phi_bin_indices
    
    return shower_df

def digitize_shower(shower_df: pd.DataFrame, voxel_df: pd.DataFrame):
    
    shower_df = digitize_in_z(shower_df, voxel_df)
    
    shower_df = digitize_in_r_phi(shower_df, voxel_df)

    energy = shower_df.groupby(['z_bin_index', 'r_bin_index', 'phi_bin_index'])['energy'].sum().reset_index()
    energy = energy.rename(columns={'energy': 'binned_energy'})
    energy = energy[energy['z_bin_index'] != -1]  # Filter out pixels that were not assigned to any z-bin
    energy = energy[energy['r_bin_index'] != -1]  # Filter out pixels that were not assigned to any r-bin
    energy = energy[energy['phi_bin_index'] != -1]  # Filter out pixels that were not assigned to any phi-bin
    # print(energy)

    energized_voxels = voxel_df.merge(energy, on=['z_bin_index', 'r_bin_index', 'phi_bin_index'], how='left').fillna(0)

    digitized_shower = shower_df.merge(energy, on=['z_bin_index', 'r_bin_index', 'phi_bin_index'], how='inner').dropna()
    
    return digitized_shower, energized_voxels

def read_binning_structure(xml_file):
    """
    Read an XML binning file and return the binning structure.
    
    Args:
        xml_file: Path to the binning XML file (e.g., 'binning_dataset_1_photons.xml')
        
    Returns:
        list: List of dictionaries, each containing:
            - id: Layer ID (int)
            - r_edges: List of radial bin edges (list of floats)
            - n_bin_alpha: Number of azimuthal bins (int)
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    binning_structure = []
    
    # Find all Layer elements (they can be under Bin elements)
    for layer in root.findall('.//Layer'):
        layer_id = int(layer.get('id'))
        r_edges_str = layer.get('r_edges')
        n_bin_alpha = int(layer.get('n_bin_alpha'))
        
        # Parse r_edges string into a list of floats
        # Handle case where r_edges is just "0" or comma-separated values
        if r_edges_str:
            r_edges = [float(x.strip()) for x in r_edges_str.split(',')]
        else:
            r_edges = []
        
        binning_structure.append({
            'id': layer_id,
            'r_edges': r_edges,
            'n_bin_alpha': n_bin_alpha
        })
    
    return binning_structure

def get_voxels(particle_df: pd.DataFrame, envelope_xml: str, binning_xml: str) -> pd.DataFrame:
    """
    Get the voxelized shower data for a given particle.
    
    Args:
        particle_df: DataFrame containing the particle information
        envelope_xml: Path to the OpenDataDetectorEnvelopes.xml file
        binning_xml: Path to the binning structure XML file
    Returns: DataFrame containing the voxelized shower data with columns for particle ID, z-bin, r-bin, phi-bin, and energy.
    """

    particle_direction = particle_df[["MCParticles_direction_x", "MCParticles_direction_y", "MCParticles_direction_z"]].to_numpy().flatten()
    p_hat = vector.obj(x=particle_direction[0], y=particle_direction[1], z=particle_direction[2])
    theta = p_hat.theta

    # Get ECal barrel dimensions
    ecal_b_rmin, ecal_b_rmax = get_ecal_barrel_dimensions(envelope_xml)
    
    # Read binning structure
    binning_structure = read_binning_structure(binning_xml)

    # create z bin edges
    z_bin_edges = np.linspace(ecal_b_rmin, ecal_b_rmax, len(binning_structure) + 1) / np.sin(theta)
    # print(z_bin_edges)
    upper_edges = z_bin_edges[1:]
    non_empty_bins = [(len(b["r_edges"]) > 1) for b in binning_structure]
    selected_upper_edges = upper_edges[non_empty_bins]
    z_bin_edges = np.concatenate(([z_bin_edges[0]], selected_upper_edges[:-1], [z_bin_edges[-1]]))
    z_bins = np.stack([z_bin_edges[:-1], z_bin_edges[1:]], axis=1)  # shape (num_z_bins, 2)

    # print(z_bins)

    # print(z_bins)

    # z_bin_edges = np.sort(np.unique(z_bins.flatten()))
    # z_bins = np.stack([z_bin_edges[:-1], z_bin_edges[1:]], axis=1)[:-1]  # shape (num_z_bins, 2)
    assert (len(z_bins)) > 0, "No valid z bins found. Check the binning structure and ECal barrel dimensions."
    assert (len(z_bins) == sum(non_empty_bins)), "Number of z bins must match the number of layers in the binning structure."

    non_empty_bins = [b for b in binning_structure if len(b["r_edges"]) > 1]
    
    # create a list to hold the voxelized shower data
    # each entry contains z-bin index, r-bin index, phi-bin index, bin center in (z,r,phi), bin center in global xyz
    voxels = pd.DataFrame()
    
    R, RT = transformation_matrices(particle_direction)

    for i, layer in enumerate(non_empty_bins):

        # bin center in local coordinates
        z_bin_edges = z_bins[i]
        z_bin_centre = 0.5 * (z_bin_edges[0] + z_bin_edges[1])
        # print(z_bin_centre)
        z_bin_index = np.array([i])  # shape (1,)

        layer_id = layer['id']
        r_edges = layer['r_edges']
        n_bin_alpha = layer['n_bin_alpha']
        
        # create r bin edges
        r_bin_edges = np.array(r_edges)
        r_bins = np.stack([r_bin_edges[:-1], r_bin_edges[1:]], axis=1)  # shape (num_r_bins, 2)
        r_bin_centre = 0.5 * (r_bins[:, 0] + r_bins[:, 1]).reshape(-1, 1)  # shape (num_r_bins,1)
        r_bin_index = np.arange(len(r_bin_centre))  # shape (num_r_bins,)
        # print(r_bin_index.reshape(-1,1), r_bin_centre.reshape(-1,1), r_bins)
        r_block = np.concatenate([r_bin_index.reshape(-1,1), r_bin_centre.reshape(-1,1), r_bins], axis=1)  # shape (num_r_bins, 4)

        # create phi bin edges
        phi_bin_edges = np.linspace(0, 2 * np.pi, n_bin_alpha + 1)
        phi_bins = np.stack([phi_bin_edges[:-1], phi_bin_edges[1:]], axis=1)  # shape (num_phi_bins, 2)
        phi_bin_centre = 0.5 * (phi_bins[:, 0] + phi_bins[:, 1])  # shape (num_phi_bins,)
        phi_bin_index = np.arange(len(phi_bin_centre)).reshape(-1,1)  # shape (num_phi_bins,1)
        phi_block = np.concatenate([phi_bin_index, phi_bin_centre.reshape(-1, 1), phi_bins], axis=1)  # shape (num_phi_bins, 4)

        local_bin_centre = np.array(np.meshgrid(z_bin_index, r_bin_index, phi_bin_index, indexing='ij')).reshape(3, -1).T  # shape (num_bins, 3)
        # print(local_bin_centre, local_bin_centre.shape)
        bin_df = pd.DataFrame(local_bin_centre, columns=['z_bin_index', 'r_bin_index', 'phi_bin_index'])

        bin_df["z_bin_min"] = z_bin_edges[0]
        bin_df["z_bin_max"] = z_bin_edges[1]
        bin_df["z_bin_centre"] = z_bin_centre

        r_df = pd.DataFrame(r_block, columns=['r_bin_index', 'r_bin_centre', 'r_bin_min', 'r_bin_max'])
        phi_df = pd.DataFrame(phi_block, columns=['phi_bin_index', 'phi_bin_centre', 'phi_bin_min', 'phi_bin_max'])
        bin_df = bin_df.merge(r_df, on="r_bin_index").merge(phi_df, on='phi_bin_index')  

        bin_df["rphi_bin_centre_x"] = bin_df["r_bin_centre"] * np.cos(bin_df["phi_bin_centre"])
        bin_df["rphi_bin_centre_y"] = bin_df["r_bin_centre"] * np.sin(bin_df["phi_bin_centre"])

        bin_centre_local = bin_df[["rphi_bin_centre_x", "rphi_bin_centre_y", "z_bin_centre"]].to_numpy()  # shape (num_bins, 3)
        bin_centre_global = (RT @ bin_centre_local.T).T  # shape (num_bins, 3)
        bin_df[["bin_centre_global_x", "bin_centre_global_y", "bin_centre_global_z"]] = bin_centre_global
        bin_df["bin_centre_global_r"] = np.sqrt(bin_df["bin_centre_global_x"]**2 + bin_df["bin_centre_global_y"]**2)
        bin_df["layer_id"] = layer_id

        voxels = pd.concat([voxels, bin_df], ignore_index=True)
    
    # print(voxels, voxels.columns)
    return voxels

def main():
    """Test the get_ecal_barrel_dimensions and read_binning_structure functions."""
    xml_path = 'detector/OpenDataDetectorEnvelopes.xml'
    
    try:
        print(f"Reading ECal barrel dimensions from: {xml_path}")
        rmin, rmax = get_ecal_barrel_dimensions(xml_path)
        
        print(f"\nECal Barrel Dimensions:")
        print(f"  ecal_b_rmin = {rmin:.1f} mm")
        print(f"  ecal_b_rmax = {rmax:.1f} mm")
        print(f"  Thickness   = {rmax - rmin:.1f} mm")
        print(f"\nConverted to meters:")
        print(f"  ecal_b_rmin = {rmin / 1000:.3f} m")
        print(f"  ecal_b_rmax = {rmax / 1000:.3f} m")
        
    except FileNotFoundError:
        print(f"Error: File not found at {xml_path}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test the binning structure reader
    print("\n" + "="*60)
    binning_path = 'binning_dataset_1_photons.xml'
    
    try:
        print(f"\nReading binning structure from: {binning_path}")
        binning = read_binning_structure(binning_path)
        
        print(f"\nFound {len(binning)} layers:")
        for layer in binning:
            print(f"\n  Layer {layer['id']}:")
            print(f"    n_bin_alpha: {layer['n_bin_alpha']}")
            print(f"    r_edges: {layer['r_edges']}")
            if len(layer['r_edges']) > 1:
                print(f"    Number of radial bins: {len(layer['r_edges']) - 1}")
        
    except FileNotFoundError:
        print(f"Error: File not found at {binning_path}")
    except Exception as e:
        print(f"Error: {e}")

    


if __name__ == "__main__":
    main()

