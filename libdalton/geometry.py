
"""Fuctions for geometry calculations used in the library."""

import numpy as np

from libdalton import constants

def get_r2_ij(p_i, p_j):
    """Calculate squared distance between two 3D cartesian points.
    
    Args:
        p_i (float*): 3 cartesian coordinates [Angstrom] of point i.
        p_j (float*): 3 cartesian coordinates [Angstrom] of point j.
    
    Returns:
        r2_ij (float): Squared distance [Angstrom] between points i and j.
    """
    v_out = np.subtract(p_i, p_j)
    r2_ij = np.dot(v_out, v_out)
    return r2_ij

def get_r_ij(p_i, p_j):
    """Calculate distance between two 3D cartesian points.
    
    Args:
        p_i (float*): 3 cartesian coordinates [Angstrom] of point i.
        p_j (float*): 3 cartesian coordinates [Angstrom] of point j.
    
    Returns:
        r_ij (float): Distance [Angstrom] between points i and j.
    """
    r_ij = np.sqrt(get_r2_ij(p_i, p_j))
    return r_ij

def get_u_ij(p_i, p_j, r_ij=None):
    """Calculate unit vector between two 3D cartesian points.
    
    Args:
        p_i (float*): 3 cartesian coordinates [Angstrom] of point i.
        p_j (float*): 3 cartesian coordinates [Angstrom] of point j.
    
    Returns:
        u_ij (float): Distance [Angstrom] between points i and j.
    """
    if not r_ij:
        r_ij = get_r_ij(p_i, p_j)
        
    u_ij = np.zeros(3)
    if r_ij > 0.0:
        u_ij = np.subtract(p_j,p_i) / r_ij
        
    return u_ij

def get_udp(uvec_i, uvec_j):
    udp = np.dot(uvec_i, uvec_j)
    udp = max(-1.0, min(1.0, udp))
    return udp

def get_a_ijk(p_i, p_j, p_k, r_ij=None, r_jk=None):
    """Calculate angle formed between three 3D cartesian points.
    
    Args:
        p_i (float*): 3 cartesian coordinates [Angstrom] of point i.
        p_j (float*): 3 cartesian coordinates [Angstrom] of (root) point j.
        p_k (float*): 3 cartesian coordinates [Angstrom] of point k.
        r_ij (float): Distance [Angstrom] between points i and j.
        r_jk (float): Distance [Angstrom] between points j and k.
    
    Returns:
        a_ijk (float): angle [degrees] between points i and j.
    """
    u_ji = get_u_ij(p_j, p_i, r_ij)
    u_jk = get_u_ij(p_j, p_k, r_jk)
    
    a_ijk = constants.RAD2DEG * np.arccos(get_udp(u_ji, u_jk))
    return a_ijk

def get_t_ijkl(p_i, p_j, p_k, p_l, r_ij=None, r_jk=None, r_kl=None):
    """Calculate torsion formed between four 3D cartesian points.
    
    Args:
        p_i (float*): 3 cartesian coordinates [Angstrom] of point i.
        p_j (float*): 3 cartesian coordinates [Angstrom] of (root) point j.
        p_k (float*): 3 cartesian coordinates [Angstrom] of (root) point k.
        p_l (float*): 3 cartesian coordinates [Angstrom] of point l.
        r_ij (float): Distance [Angstrom] between points i and j.
        r_jk (float): Distance [Angstrom] between points j and k.
        r_kl (float): Distance [Angstrom] between points k and l.
    
    Returns:
        t_ijkl (float): torsion signed angle [degrees] between points i, j, k and l.
    """
    u_ji = get_u_ij(p_j, p_i, r_ij)
    u_jk = get_u_ij(p_j, p_k, r_jk)
    u_kl = get_u_ij(p_k, p_l, r_kl)
    
    ucp_jijk = np.cross(u_ji, u_jk)
    ucp_jijk /= np.linalg.norm(ucp_jijk)
    ucp_kjkl = np.cross(-u_jk, u_kl)
    ucp_kjkl /= np.linalg.norm(ucp_kjkl)
    
    dp_jijk_kjkl = get_udp(ucp_jijk, ucp_kjkl)

    sign = np.sign(get_udp(ucp_jijk, u_kl))
    t_ijkl = -1.0 * constants.RAD2DEG * sign * np.arccos(dp_jijk_kjkl)
    return t_ijkl

def get_o_ijkl(p_i, p_j, p_k, p_l, r_ki=None, r_kj=None, r_kl=None):
    """Calculate torsion formed between four 3D cartesian points.
    
    Args:
        p_i (float*): 3 cartesian coordinates [Angstrom] of point i.
        p_j (float*): 3 cartesian coordinates [Angstrom] of point j.
        p_k (float*): 3 cartesian coordinates [Angstrom] of (root) point k.
        p_l (float*): 3 cartesian coordinates [Angstrom] of point l.
        r_ki (float): Distance [Angstrom] between points i and j.
        r_kj (float): Distance [Angstrom] between points j and k.
        r_kl (float): Distance [Angstrom] between points k and l.
    
    Returns:
        o_ijkl (float): torsion signed angle [degrees] between points i, j, k and l.
    """
    u_ki = get_u_ij(p_k, p_i, r_ki)
    u_kj = get_u_ij(p_k, p_j, r_kj)
    u_kl = get_u_ij(p_k, p_l, r_kl)
    
    ucp_kikj = np.cross(u_ki, u_kj)
    ucp_kikj /= np.linalg.norm(ucp_kikj)
    dp_kjkl_kl = get_udp(ucp_kikj, u_kl)
    
    o_ijkl = constants.RAD2DEG * np.arcsin(dp_kjkl_kl)
    return o_ijkl

def get_volume(boundary, boundary_type):
    """Calculate volume of molecular system based on boundary type.
    
    Boundary types: 'cube', 'sphere'.
    
    Args:
        boundary (float): Maximum extent of a system away from origin.
        boundary_type (str): Type of boundary shape.
        
    Returns:
        volume (float): Unrestricted volume of the system.
    
    Raises:
        ValueError: If boundary_type is not 'cube' or 'sphere'.
    """
    if boundary_type == 'cube':
        return 8.0 * boundary**3
    elif boundary_type == 'sphere':
        return 4.0 * np.pi * boundary**3 / 3.0
    else:
        raise ValueError('Unexpected boundary type: %s.\n'
                         "Use 'cube' or 'sphere'.\n" % boundary_type)
