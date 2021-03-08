################################################################################
# Sample                                                                       #
#                                                                              #
"""Sample functions to be run on clusters."""
################################################################################


import os
import sys
import math
import chemfiles as cf

import porems as pms
import poreana.utils as utils
import poreana.geometry as geometry


def density(link_pore, link_traj, link_out, mol, atoms=[], masses=[], bin_num=150, entry=0.5, is_force=False):
    """This function samples the density inside and outside of the pore. This
    function is to be run on the cluster due to a high time and resource
    consumption. The output, a data object, is then used to calculate the
    density using further calculation functions.

    All atoms are sampled each frame if they are inside or outside the bounds of
    the pore minus an entry length on both sides.

    Inside the pore the atom instances will be added to radial cylindric slices
    :math:`r_i-r_{i-1}` and outside to rectangular slices :math:`z_j-z_{j-1}`
    with pore radius :math:`r_i` of radial slice :math:`i` and length
    :math:`z_j` of slice :math:`j`.

    Parameters
    ----------
    link_pore : string
        Link to poresystem object file
    link_traj : string
        Link to trajectory file (trr or xtc)
    link_out : string
        Link to output object file
    mol : Molecule
        Molecule to calculate the density for
    atoms : list, optional
        List of atom names, leave empty for whole molecule
    masses : list, optional
        List of atom masses, leave empty to read molecule object masses
    bin_num : integer, optional
        Number of bins to be used
    entry : float, optional
        Remove pore entrance from calculation
    is_force : bool, optional
        True to force re-extraction of data
    """
    # Get molecule ids
    atoms = [atom.get_name() for atom in mol.get_atom_list()] if not atoms else atoms
    atoms = [atom_id for atom_id in range(mol.get_num()) if mol.get_atom_list()[atom_id].get_name() in atoms]
    num_atoms = len(atoms)

    # Check masses
    if not masses:
        if len(atoms)==mol.get_num():
            masses = mol.get_masses()
        elif num_atoms == 1:
            masses = [1]

    # Check consistency
    if atoms and not len(masses) == len(atoms):
        print("Length of variables *atoms* and *masses* do not match!")
        return

    # Get pore properties
    pore = utils.load(link_pore)
    if isinstance(pore, pms.PoreCylinder):
        res = pore.reservoir()
        diam = pore.diameter()
        focal = pore.centroid()
        box = pore.box()
        box[2] += 2*res

        # Define bins intern
        bin_in = [[diam/2/bin_num*x for x in range(bin_num+2)], [0 for x in range(bin_num+1)]]

    # Define bins extern
    bin_out = [[res/bin_num*x for x in range(bin_num+1)], [0 for x in range(bin_num+1)]]

    # Check if already calculated
    if not os.path.exists(link_out) or is_force:
        # Load trajectory
        traj = cf.Trajectory(link_traj)
        num_frame = traj.nsteps
        res_list = {}

        # Run through frames
        for frame_id in range(num_frame):
            # Read frame
            frame = traj.read()
            positions = frame.positions

            # Create list of relevant atom ids
            if not res_list:
                # Get number of residues in system
                num_res = len(frame.topology.atoms)/mol.get_num()

                # Check number of residues
                if abs(int(num_res)-num_res) >= 1e-5:
                    print("Number of atoms is inconsistent with number of residues.")
                    return

                # Check relevant atoms
                for res_id in range(int(num_res)):
                    res_list[res_id] = [res_id*mol.get_num()+atom for atom in range(mol.get_num()) if atom in atoms]

            # Run through residues
            for res_id in res_list:
                # Get position vectors
                pos = [[positions[res_list[res_id][atom_id]][i]/10 for i in range(3)] for atom_id in range(num_atoms)]

                # Calculate centre of mass
                com = [sum([pos[atom_id][i]*masses[atom_id] for atom_id in range(num_atoms)])/sum(masses) for i in range(3)]

                # Remove edge molecules
                is_edge = False
                for i in range(3):
                    is_edge = True if abs(com[i]-pos[0][i])>res else is_edge

                # Check if com was calculated on edges
                if not is_edge:
                    # Calculate radial distance towards center axis
                    radial = geometry.length(geometry.vector([focal[0], focal[1], com[2]], com))

                    # Inside pore
                    if com[2] > res+entry and com[2] < box[2]-res-entry:
                        index = math.floor(radial/bin_in[0][1])

                        if index <= bin_num:
                            bin_in[1][index] += 1

                    # Outside Pore
                    elif com[2] <= res or com[2] > box[2]-res:
                        # Calculate distance to crystobalit and apply perodicity
                        dist = com[2] if com[2] < focal[2] else abs(com[2]-box[2])
                        index = math.floor(dist/bin_out[0][1])

                        # Out
                        if radial > diam/2 and index <= bin_num:
                            bin_out[1][index] += 1

            # Progress
            sys.stdout.write("Finished frame "+"%4i" % (frame_id+1)+"/"+"%4i" % num_frame+"...\r")
            sys.stdout.flush()

        print()

        # Define output dictionary
        inp = {"frame": num_frame, "mass": mol.get_mass(),
               "entry": entry, "res": res, "diam": diam, "box": box}
        output = {"in": bin_in, "out": bin_out, "inp": inp}

        # Save data
        utils.save(output, link_out)

    # File already exists
    else:
        print("Object file already exists. If you wish to overwrite the file set the input *is_force* to True.")


def diffusion_bin(link_pore, link_traj, link_out, mol, atoms=[], masses=[], bin_num=50, entry=0.5, len_obs=16e-12, len_frame=2e-12, len_step=2, bin_step_size=1, is_force=False):
    """This function samples the mean square displacement of a molecule group
    in a pore in both axial and radial direction separated in radial bins. This
    function is to be run on the cluster due to a high time and resource
    consumption. The output, a data object, is then used to calculate the
    self-diffusion using further calculation functions.

    First a centre of mass-list is filled with :math:`w\\cdot s`
    frames with window length :math:`w` and stepsize :math:`s`. Each following
    frame removes the first com of the list and a new added to the end of it.
    This way only one loop over the frames is needed, since each frame is
    only needed for :math:`w\\cdot s` frames in total.

    All molecule com's are sampled each window if they are inside the bounds of
    the pore minus an entry length on both sides. Once the com leaves the
    boundary, it is no longer sampled for this specific window. Additionally,
    the radial bin index is checked for each frame. If the molecule com stays in
    the pore for the whole window length and in the same starting bin plus an
    allowed offset, the msd is added to it is added to a the corresponding
    window starting radial bin. The sub volumes, or rather bins, of the radial
    distance are calculated by

    .. math::

        V_i^\\text{radial}=\\pi z_\\text{pore}(r_i^2-r_{i-1}^2)

    with :math:`r` of bin :math:`i` and pore length :math:`z_\\text{pore}`.

    Once the first com-list is filled, the mean square displacement (msd)
    is sampled each frame. The axial diffusion only considers the
    deviation in :math:`z` direction

    .. math::

        \\text{msd}_\\text{axial}=\\langle\\left[z(0)-z(t_i)\\right]^2\\rangle
        =\\frac{\\left[z(0)-z(t_i)\\right]^2}{M_i}

    with time :math:`t` and normalization :math:`M` at bin :math:`i`.
    The radial diffusion only considers the radial components :math:`x` and
    :math:`y`

    .. math::

        \\text{msd}_\\text{radial}=\\langle\\left[r(0)-r(t_i)\\right]^2\\rangle
        =\\frac{\\left[\\sqrt{x^2(0)+y^2(0)}-\\sqrt{x^2(t_i)+y^2(t_i)}\\right]^2}{M_i}.

    Parameters
    ----------
    link_pore : string
        Link to poresystem object file
    link_traj : string
        Link to trajectory file (trr or xtc)
    link_out : string
        Link to output object file
    mol : Molecule
        Molecule object to calculate the density for
    atoms : list, optional
        List of atom names, leave empty for whole molecule
    masses : list, optional
        List of atom masses, leave empty to read molecule object masses
    bin_num : integer, optional
        Number of radial bins
    entry : float, optional
        Remove pore entrance from calculation
    len_obs : float, optional
        Observation length of a window in seconds
    len_frame : float, optional
        Length of a frame in seconds
    len_step : integer, optional
        Length of the step size between frames
    bin_step_size : integer, optional
        Number of allowed bins for the molecule to leave
    is_force : bool, optional
        True to force re-extraction of data
    """
    # Get molecule ids
    atoms = [atom.get_name() for atom in mol.get_atom_list()] if not atoms else atoms
    atoms = [atom_id for atom_id in range(mol.get_num()) if mol.get_atom_list()[atom_id].get_name() in atoms]
    num_atoms = len(atoms)

    # Check masses
    if not masses:
        if len(atoms)==mol.get_num():
            masses = mol.get_masses()
        elif num_atoms == 1:
            masses = [1]

    # Check consistency
    if atoms and not len(masses) == len(atoms):
        print("Length of variables *atoms* and *masses* do not match!")
        return

    # Get pore properties
    pore = utils.load(link_pore)
    if isinstance(pore, pms.PoreCylinder):
        res = pore.reservoir()
        diam = pore.diameter()
        focal = pore.centroid()
        box = pore.box()
        box[2] += 2*res

        # Define bins
        bins = [diam/2/bin_num*x for x in range(bin_num+2)]

    # Define window length
    len_window = len_obs/len_step/len_frame+1
    if not len_window == int(len_window):
        obs_u = (math.ceil(len_window)-1)*len_step*len_frame
        obs_d = (math.floor(len_window)-1)*len_step*len_frame
        print("Observation length not possible with current inputs. Alternatively use len_obs="+"%.1e" % obs_u+" or len_obs="+"%.1e" % obs_d+".")
        return
    else:
        len_window = int(len_window)

    # Define allowed bin step list
    def bin_step(idx):
        out_list = [idx+x for x in range(bin_step_size, 0, -1)]
        out_list += [idx]
        out_list += [idx-x for x in range(1, bin_step_size+1)]
        return out_list

    # Check if calculated
    if not os.path.exists(link_out) or is_force:
        # Load trajectory
        traj = cf.Trajectory(link_traj)
        num_frame = traj.nsteps
        res_list = {}

        # Initialize bin lists
        bin_z = [[0 for y in range(len_window)] for x in range(bin_num+1)]
        bin_r = [[0 for y in range(len_window)] for x in range(bin_num+1)]
        bin_n = [[0 for y in range(len_window)] for x in range(bin_num+1)]

        bin_tot_z = [[0 for y in range(len_window)] for x in range(bin_num+1)]
        bin_tot_r = [[0 for y in range(len_window)] for x in range(bin_num+1)]
        bin_tot_n = [[0 for y in range(len_window)] for x in range(bin_num+1)]

        # Run through frames
        com_list = []
        idx_list = []
        for frame_id in range(num_frame):
            # Read frame
            frame = traj.read()
            positions = frame.positions

            # Add new dictionaries and remove unneeded references
            if frame_id >= (len_window*len_step):
                com_list.pop(0)
                idx_list.pop(0)
            com_list.append({})
            idx_list.append({})

            # Create list of relevant atom ids
            if not res_list:
                # Get number of residues in system
                num_res = len(frame.topology.atoms)/mol.get_num()

                # Check number of residues
                if abs(int(num_res)-num_res) >= 1e-5:
                    print("Number of atoms is inconsistent with number of residues.")
                    return

                # Check relevant atoms
                for res_id in range(int(num_res)):
                    res_list[res_id] = [res_id*mol.get_num()+atom for atom in range(mol.get_num()) if atom in atoms]

            # Run through residues
            for res_id in res_list:
                # Get position vectors
                pos = [[positions[res_list[res_id][atom_id]][i]/10 for i in range(3)] for atom_id in range(num_atoms)]

                # Calculate centre of mass
                com = [sum([pos[atom_id][i]*masses[atom_id] for atom_id in range(num_atoms)])/sum(masses) for i in range(3)]

                # Remove edge molecules
                is_edge = False
                for i in range(3):
                    is_edge = True if abs(com[i]-pos[0][i])>res else is_edge

                # Check if reference is inside pore
                if not is_edge and com[2] > res+entry and com[2] < box[2]-res-entry:
                    # Calculate radial bin index
                    radial = geometry.length(geometry.vector([focal[0], focal[1], com[2]], com))
                    index = math.floor(radial/bins[1])

                    # Add com and bin index to global lists
                    com_list[-1][res_id] = com
                    idx_list[-1][res_id] = index

                    # Start sampling when initial window is filled
                    if len(com_list) == len_window*len_step and res_id in com_list[0]:
                        # Set reference position
                        pos_ref = com_list[0][res_id]
                        idx_ref = idx_list[0][res_id]

                        # Create temporary msd lists
                        msd_z = [0 for x in range(len_window)]
                        msd_r = [0 for x in range(len_window)]
                        norm = [0 for x in range(len_window)]
                        len_msd = 0

                        # Run through position list to sample msd
                        for step in range(0, len_window*len_step, len_step):
                            # Check if com is inside pore
                            if res_id in com_list[step]:
                                # Initialize step information
                                pos_step = com_list[step][res_id]
                                idx_step = idx_list[step][res_id]

                                # Get window index
                                win_idx = int(step/len_step)

                                # Add to msd
                                msd_z[win_idx] += (pos_ref[2]-pos_step[2])**2
                                msd_r[win_idx] += geometry.length(geometry.vector(pos_ref, [pos_step[0], pos_step[1], pos_ref[2]]))**2

                                # Add to normalize
                                norm[win_idx] += 1

                                # Check if com is within range of reference bin
                                if idx_step in bin_step(idx_ref):
                                    len_msd += 1

                                # COM left radial bin
                                else:
                                    break

                            # COM left the boundary
                            else:
                                break

                        # Save msd
                        if idx_ref <= bin_num:
                            for i in range(len_window):
                                # Add to total list
                                bin_tot_z[idx_ref][i] += msd_z[i]
                                bin_tot_r[idx_ref][i] += msd_r[i]
                                bin_tot_n[idx_ref][i] += norm[i]

                                # Add to bin calculation list if msd is permissible
                                if len_msd == len_window:
                                    bin_z[idx_ref][i] += msd_z[i]
                                    bin_r[idx_ref][i] += msd_r[i]
                                    bin_n[idx_ref][i] += norm[i]

            sys.stdout.write("Finished frame "+"%3i" % (frame_id+1)+"/"+"%3i" % num_frame+"...\r")
            sys.stdout.flush()
        print()

        # Define output dictionary
        inp = {"window": len_window, "step": len_step, "frame": len_frame,
                 "bins": bin_num, "entry": entry, "diam": diam}
        output = {"inp": inp, "bins": bins,
                  "axial": bin_z, "radial": bin_r, "norm": bin_n,
                  "axial_tot": bin_tot_z, "radial_tot": bin_tot_r, "norm_tot": bin_tot_n,}

        # Save data
        utils.save(output, link_out)

    # File already existing
    else:
        print("Object file already exists. If you wish to overwrite the file set the input *is_force* to True.")
