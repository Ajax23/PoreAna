################################################################################
# Density                                                                      #
#                                                                              #
"""Analyse density in a pore."""
################################################################################


import os
import sys
import math
import numpy as np
import chemfiles as cf

import seaborn as sns
import matplotlib.pyplot as plt

import poreana.utils as utils
import poreana.geometry as geometry

from porems import *


def sample(link_pore, link_pdb, link_trr, link_out, mol, atoms=[], masses=[], bin_num=150, entry=0.5, is_force=False):
    """This function samples the density inside and outside of the pore. This
    function is to be run on the cluster due to a high time and resource
    consumption. The output, a data object, is then used to calculate the
    density using function :func:`calculate`.

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
    link_pdb : string
        Link to pdb file
    link_trr : string
        Link to trr file
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

    # Get pore information
    pore = utils.load(link_pore)
    res = pore.reservoir()
    diam = pore.diameter()
    focal = pore.centroid()
    box = pore.box()
    box[2] += 2*res

    # Define bins
    bin_in = [[diam/2/bin_num*x for x in range(bin_num+2)], [0 for x in range(bin_num+1)]]
    bin_out = [[res/bin_num*x for x in range(bin_num+1)], [0 for x in range(bin_num+1)]]

    # Check if already calculated
    if not os.path.exists(link_out) or is_force:
        # Get number of atoms in system
        num_res = int(len(cf.Trajectory(link_pdb).read().topology.atoms)/mol.get_num())

        # Create list of relevant atom ids
        res_list = {}
        for res_id in range(num_res):
            res_list[res_id] = [res_id*mol.get_num()+atom for atom in range(mol.get_num()) if atom in atoms]

        # Load trajectory
        traj = cf.Trajectory(link_trr)
        num_frame = traj.nsteps

        # Run through frames
        # com_list = []
        for frame_id in range(num_frame):
            frame = traj.read()
            positions = frame.positions

            # Run through residues
            for res_id in res_list:
                # Get position vectors
                pos = [[positions[res_list[res_id][atom_id]][i]/10 for i in range(3)] for atom_id in range(num_atoms)]

                # Calculate centre of mass
                com = [sum([pos[atom_id][i]*masses[atom_id] for atom_id in range(num_atoms)])/sum(masses) for i in range(3)]

                # Remove edge molecules
                is_edge = False
                for i in range(3):
                    if abs(com[i]-pos[0][i])>res:
                        is_edge = True

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


def calculate(link_data, area=[[10, 90], [10, 90]], target_dens=None, is_print=True):
    """This function calculates the density inside and outside of the pore.
    This is done by calculating the number density :math:`\\rho_n` and using the
    molar mass :math:`M` of the molecule to determine the mass density
    :math:`\\rho`.

    The basic idea is counting the number of molecules :math:`N_i` in volume
    slices :math:`V_i`, thus getting the number density :math:`\\rho_{n,i}` in
    these sub volumes. Inside the pore this is done by creating a radial slicing
    like the radial distribution function. These sub volumes are calculated by

    .. math::

        V_i^\\text{radial}=\\pi z_\\text{pore}(r_i^2-r_{i-1}^2).

    with pore length :math:`z_\\text{pore}` and radius :math:`r_i` of sub volume
    :math:`i`. This yields

    .. math::

        \\rho_{n,i}^\\text{radial}=\\frac{N_i}{V_i^\\text{radial}}=\\frac{N_i}{\\pi z_\\text{pore}}\\frac{1}{r_i^2-r_{i-1}^2}.

    Outside the pore, the sub volumes are given by

    .. math::

        V_j^\\text{out}=(x_\\text{pore}\\cdot y_\\text{pore}-\\pi r^2)z_j

    with pore width :math:`x_\\text{pore}`, height :math:`y_\\text{pore}`, pore
    radius :math:`r` and slice width :math:`z_j`. Thus

    .. math::

        \\rho_{n,j}^\\text{out}=\\frac{N_j}{V_j^\\text{out}}=\\frac{N_j}{x_\\text{pore}\\cdot y_\\text{pore}-\\pi r^2}\\frac{1}{z_j}.

    Note that the outside refers to the reservoirs of the pore simulation.
    Therefore the slices add up to the reservoir length :math:`z_{res}`.
    Since there is a reservoir on each side, they are brought together
    by translating the atom coordinates to one of the reservoirs. Since the
    outside density refers to the density of the outside surface, it does
    not contain the cylindrical extension of the pore inside the reservoirs.

    Finally, the mass density is calculated by

    .. math::

        \\rho=\\frac M{N_A}\\rho_n

    with Avogadro constant :math:`N_A`. The units are then transformed to
    :math:`\\frac{\\text{kg}}{\\text m^3}` by

    .. math::

        [\\rho]=\\frac{[M]\\frac{\\text{g}}{\\text{mol}}}{[N_A]10^{23}\\frac{\\#}{\\text{mol}}}[\\rho_n]\\frac{\\#}{\\text{nm}^3}
               =\\frac{[M]}{[N_A]}[\\rho_n]\\cdot10\\frac{\\text{kg}}{\\text m^3}

    where the square brackets mean, that only the variables value is taken.
    Since finding full molecules in a sub volume is difficult, the atoms
    of the specified molecule are counted in the sub volumes and the result
    is then divided by the number of atoms the molecule consists of.

    Parameters
    ----------
    link_data : string
        Link to data object generated by the sample routine :func:`sample`
    area : list,  optional
        Bin areas to calculate the mean number density from (pore,out)
    target_dens : float, None, optional
        Target density in :math:`\\frac{\\text{kg}}{\\text{m}^3}`
    is_print : bool, optional
        True to print output
    """
    # Load data object
    sample = utils.load(link_data)

    # Load bins
    bin_in = sample["in"]
    bin_out = sample["out"]
    bin_num = len(bin_out[0])-1

    # Load input data
    inp = sample["inp"]
    num_frame = inp["frame"]
    entry = inp["entry"]
    res = inp["res"]
    diam = inp["diam"]
    box = inp["box"]
    mass = inp["mass"]

    # Calculate bin volume
    vol_in = [math.pi*(box[2]-2*res-2*entry)*(bin_in[0][i+1]**2-bin_in[0][i]**2) for i in range(0, bin_num+1)]
    vol_out = [2*bin_out[0][1]*(box[0]*box[1]-math.pi*(diam/2)**2) for i in range(bin_num+1)]

    # Calculate the number density
    num_dens_in = [bin_in[1][i]/vol_in[i]/num_frame for i in range(bin_num+1)]
    num_dens_out = [bin_out[1][i]/vol_out[i]/num_frame for i in range(bin_num+1)]

    # Calculate the mean in the selected area
    mean_in = np.mean(num_dens_in[area[0][0]:area[0][1]])
    mean_out = np.mean(num_dens_out[area[1][0]:area[1][1]])

    # Calculate Density
    dens_in = mass*10/6.022*mean_in
    dens_out = mass*10/6.022*mean_out

    # Calculate difference to target density
    num_diff = (target_dens/mass/10*6.022-mean_out)*box[0]*box[1]*res*2 if target_dens is not None else None

    # Output
    if is_print:
        print("Density inside  Pore = "+"%5.3f"%mean_in+" #/nm^3 ; "+"%7.3f"%dens_in+" kg/m^3")
        print("Density outside Pore = "+"%5.3f"%mean_out+" #/nm^3 ; "+"%7.3f"%dens_out+" kg/m^3")
        if target_dens is not None:
            print("Density difference   = "+"%5.3f" % (target_dens-dens_out)+" kg/m^3 ; "
                  +"%4.2f" % ((1-dens_out/target_dens)*100)+" % ; "
                  +"%3i" % num_diff+" #")

    # Return output
    return  {"in": [bin_in, num_dens_in, mean_in, dens_in], "out": [bin_out, num_dens_out, mean_out, dens_out], "inp": inp, "diff": num_diff}


def adsorption(link_pore, link_data, res_cutoff=1, is_normalize=True):
    """This function calculates the values for the adsorption isotherms. This is
    done by counting the number of molecules inside the reservoir and within the
    pore over the whole simulation.

    By normalizing the summation by the number of frames, the resulting value
    is converted to a surface specific concentration inside the pore and volume
    specific concentration within the reservoir.

    The resulting value pair is a point in the adsorption isotherm.

    Parameters
    ----------
    link_pore : string
        Link to poresystem object file
    link_data : string
        Link to data object generated by the sample routine :func:`sample`
    res_cutoff : float, optional
        Area of the reservoir to remove from counting on both sides of the
        reservoir
    is_normalize : bool, optional
        True to normalize the number of atoms with the number of frames

    Returns
    -------
    adsorption : dictionary
        Normalized number of molecules outside and insidem and value pair of a
        point on the adsorption isotherm
        :math:`\\left[\\frac{\\text{mmol}}{\\text{l}}\\ ,\\frac{\\mu\\text{mol}}{\\text{m}^2}\\right]`
    """
    # Load pore properties
    pore = utils.load(link_pore)
    res = pore.reservoir()
    diam = pore.diameter()
    box = pore.box()

    # Load data object
    sample = utils.load(link_data)

    # Load bins
    bin_in = sample["in"]
    bin_out = sample["out"]

    # Load input data
    inp = sample["inp"]
    num_frames = inp["frame"]
    entry = inp["entry"]

    # Calculate number of molecules
    num_in = sum(bin_in[1])
    num_out = sum([num_mol for i, num_mol in enumerate(bin_out[1]) if bin_out[0][i] <= res-res_cutoff and bin_out[0][i] >= res_cutoff])

    # Normalize number of instances by the number of frames
    num_in /= num_frames if is_normalize else 1
    num_out /= num_frames if is_normalize else 1

    # Calculate surface and volume
    surface = 2*np.pi*(diam)/2*(box[2]-2*entry)
    volume = 2*(res-2*res_cutoff)*box[0]*box[1]

    # Convert to concentrations
    mumol_m2 = utils.mols_to_mumol_m2(num_in, surface)
    mmol_l = utils.mols_to_mmol_l(num_out, volume)

    return {"c": [mmol_l, mumol_m2], "n": [num_out, num_in]}


def plot(density, intent=None, target_dens=None, is_mean=False):
    """This function plots the density of the given object. If an intent is
    given instead of None, only a plot-function will be called. Available
    options for ``intent`` are

    * **None** - Create subplots for the density inside and outside the pore
    * **in** - Create plot for the density inside pore
    * **out** - Create plot for the density outside pore

    Parameters
    ----------
    density : dictionary
        Density object from the density calculation :func:`calculate`
    intent : string, None, optional
        Intent for plotting
    target_dens : float, None, optional
        Target density for plot, None to disable
    is_mean : bool, optional
        True to plot mean values
    """
    # Full plot
    if intent is None:
        # Plot
        plt.figure(figsize=(10, 7))

        sns.set(style="whitegrid")
        sns.set_palette(sns.color_palette("deep"))

        plt.subplot(211)
        sns.lineplot(x=density["in"][0][0][:-1], y=density["in"][1], linewidth=2.5)
        if is_mean:
            sns.lineplot(x=density["in"][0][0][:-1], y=[density["in"][2] for x in density["in"][0][0][:-1]])

        plt.xlim([0, density["in"][0][0][-1]])
        plt.xlabel("Distance from pore center (nm)")
        plt.ylabel(r"Density (atoms nm$^{-3}$)")
        plt.legend(["Density", "Mean"])

        plt.subplot(212)
        #
        sns.lineplot(x=density["out"][0][0], y=density["out"][1], linewidth=2.5)
        if is_mean:
            sns.lineplot(x=density["out"][0][0], y=[density["out"][2] for x in density["out"][0][0]])

        if target_dens is not None:
            sns.lineplot(x=density["out"][0][0], y=[target_dens]*len(density["out"][0][0]))

        plt.xlim([0, density["out"][0][0][-1]])
        plt.xlabel("Distance from reservoir end (nm)")
        plt.ylabel(r"Density in (atoms nm$^{-3}$)")
        if is_mean:
            plt.legend(["Density Out", "Mean Out", "Target density"])
        else:
            plt.legend(["Density Out", "Target density"])

    # Intent plots
    else:
        if intent == "in":
            bins = density[intent][0][0][:-1]
        elif intent == "out":
            bins = density[intent][0][0]
        else:
            print("Wrong intent. Check documentation for available options.")
            return

        sns.lineplot(x=bins, y=density[intent][1])
        plt.xlim([0, density[intent][0][0][-1]])
