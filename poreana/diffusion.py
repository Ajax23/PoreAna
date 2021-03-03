################################################################################
# Diffusion                                                                    #
#                                                                              #
"""Analyse diffusion in a pore."""
################################################################################


import os
import sys
import math
import warnings

import scipy as sp
import numpy as np
import chemfiles as cf

import seaborn as sns
import matplotlib.pyplot as plt

import poreana.utils as utils
import poreana.geometry as geometry
import poreana.density as density

from porems import *


def sample(link_pore, link_traj, link_out, mol, atoms=[], masses=[], bin_num=50, entry=0.5, len_obs=16e-12, len_frame=2e-12, len_step=2, bin_step_size=1, is_force=False):
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
    if isinstance(pore, PoreCylinder):
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
                    if abs(com[i]-pos[0][i])>res:
                        is_edge = True

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


def cui(data_link, z_dist=None, ax_area=[0.2, 0.8], intent=None, is_fit=False, is_plot=True):
    """This function samples and calculates the diffusion coefficient of a
    molecule group in a pore in both axial and radial direction, as described
    in the paper of `Cui <https://doi.org/10.1063/1.1989314>`_.

    The mean square displacement is sampled in function :func:`sample`.

    The axial diffusion is given by the Einstein relation

    .. math::

        \\langle\\left[z(0)-z(t)\\right]^2\\rangle=2D_\\text{axial}t

    with axial diffusion coefficient :math:`D_\\text{axial}`. Thus the
    coefficient corresponds to the slope of the axial msd

    .. math::

        D_\\text{axial}=\\frac12\\frac{\\text{msd}_\\text{axial}[i]-\\text{msd}_\\text{axial}[j]}{t_i-t_j}

    with bin :math:`i>j`. The radial diffusion is given by

    .. math::

        \\langle\\left[r(0)-r(t)\\right]^2\\rangle=R^2\\left[1-\\sum_{n=1}^\\infty\\frac{8}{\\lambda_{1n}^2(\\lambda_{1n}^2-1)}\\exp\\left(-\\frac{\\lambda_{1n}^2}{R^2}D_\\text{radial}t\\right)\\right].

    with radial diffusion coefficient :math:`D_\\text{radial}`, the maximal
    accessible radial position :math:`R` by an atom

    .. math::

        R = \\frac12d-0.2

    with pore diameter :math:`d`, and the zeros :math:`\\lambda_{1n}^2` of
    the derivative

    .. math::

        \\frac{dJ_1}{dx}

    of the first order Bessel function :math:`J_1`. Therefore the coefficient
    has to be fitted to the sampled radial msd. The author observed that
    the first 20 Bessel function zeros were sufficient for the function fit.

    The final unit transformation is done by

    .. math::

        \\frac{\\text{nm}^2}{\\text{ps}}=10^{-2}\\frac{\\text{cm}^2}{\\text{s}}.

    **Note that the function for the radial diffusion is obtained under the
    assumption that the density inside the pore is uniform.**

    Parameters
    ----------
    data_link : string
        Link to data object generated by the sample routine :func:`sample`
    z_dist : float, None, optional
        Distance from pore centre to calculate the mean
    ax_area : list, optional
        Bin area percentage to calculate the axial diffusion coefficient
    intent : string, None, optional
        axial, radial or None for both
    is_fit : bool, optional
        True to plot the fitted function
    is_plot : bool, optional
        True to create plot in this function
    """
    # Load data object
    sample = utils.load(data_link)

    # Load data
    inp = sample["inp"]
    bins = inp["bins"] if z_dist is None else math.floor(z_dist/sample["bins"][1])
    msd_z = [0 for x in range(inp["window"])]
    msd_r = [0 for x in range(inp["window"])]
    norm_z = [0 for x in range(inp["window"])]
    norm_r = [0 for x in range(inp["window"])]

    # Sum up all bins
    for i in range(bins):
        for j in range(inp["window"]):
            msd_z[j] += sample["axial_tot"][i][j]
            norm_z[j] += sample["norm_tot"][i][j]

    for i in range(inp["bins"]):
        for j in range(inp["window"]):
            msd_r[j] += sample["radial_tot"][i][j]
            norm_r[j] += sample["norm_tot"][i][j]

    # Normalize
    msd_z_n = [msd_z[i]/norm_z[i] if norm_z[i] > 0 else 0 for i in range(inp["window"])]
    msd_r_n = [msd_r[i]/norm_r[i] if norm_r[i] > 0 else 0 for i in range(inp["window"])]

    # Define time axis and range
    time_ax = [x*inp["step"]*inp["frame"] for x in range(inp["window"])]
    t_range = (inp["window"]-1)*inp["step"]*inp["frame"]

    # Calculate axial coefficient
    if intent is None or intent == "axial":
        dz = (msd_z_n[int(ax_area[1]*inp["window"])]-msd_z_n[int(ax_area[0]*inp["window"])])*1e-9**2/((ax_area[1]-ax_area[0])*t_range)/2*1e2**2*1e5  # 10^-9 m^2s^-1

        print("Diffusion axial:  "+"%.3f" % dz+" 10^-9 m^2s^-1")

    # Calculate radial coefficient
    if intent is None or intent == "radial":
        def diff_rad(x, a, b, c):
            # Process input
            if not isinstance(x, list) and not isinstance(x, np.ndarray):
                x = [x]
            # Get bessel function zeros
            jz = sp.special.jnp_zeros(1, math.ceil(b))
            # Calculate sum
            sm = [[8/(z**2*(z**2-1))*math.exp(-(z/c)**2*a*t) for z in jz] for t in x]
            # Final equation
            return [c**2*(1-sum(s)) for s in sm]

        # Fit function
        popt, pcov = sp.optimize.curve_fit(diff_rad, [x*1e12 for x in time_ax], msd_r_n, p0=[1, 20, inp["diam"]/2-0.2], bounds=(0, np.inf))

        print("Diffusion radial: "+"%.3f" % (popt[0]*1e3)+" 10^-9 m^2 s^-1; Number of zeros: "+"%2i" % (math.ceil(popt[1]))+"; Radius: "+"%5.2f" % popt[2])

    # Plot
    if is_plot:
        # plt.figure(figsize=(10, 7))
        sns.set(style="whitegrid")
        sns.set_palette(sns.color_palette("deep"))
        legend = []

    if intent is None or intent == "axial":
        sns.lineplot(x=[x*1e12 for x in time_ax], y=msd_z_n)
        if is_plot:
            legend += ["Axial"]
        if is_fit:
            sns.lineplot(x=[x*1e12 for x in time_ax], y=[dz*2*time_ax[x]/1e5/1e-7**2 for x in range(inp["window"])])
            legend += ["Fitted Axial"]

    if intent is None or intent == "radial":
        sns.lineplot(x=[x*1e12 for x in time_ax], y=msd_r_n)
        if is_plot:
            legend += ["Radial"]
        if is_fit:
            sns.lineplot(x=[x*1e12 for x in time_ax], y=diff_rad([x*1e12 for x in time_ax], *popt))
            legend += ["Fitted Radial"]

    if is_plot:
        plt.xlabel("Time (ps)")
        plt.ylabel(r"Mean square displacement (nm$^2$)")
        plt.legend(legend)


def bins(data_link, ax_area=[0.2, 0.8], intent="plot", is_norm=False):
    """This function calculates the axial (z-axis) diffusion coefficient as a
    function of the radial distance. This is done by sampling the mean square
    displacement for all molecules in a radial sub volume.

    The mean square displacement is sampled in function :func:`sample`.

    For each bin, the msd is summed up, resulting into a msd slope for each
    bin. Thus, the axial diffusion coefficient can be calculated using

    .. math::

        D_\\text{axial}=\\frac12\\frac{\\text{msd}_\\text{axial}[i]-\\text{msd}_\\text{axial}[j]}{t_i-t_j}.

    Note that the msd is evaluated in the area, where the slope is uniform,
    which means that the first twenty and last twenty percent should be
    neglected.

    If ``is_norm`` is set to **True**, the radius will be normalized in respect
    to the effective radius which means, the last radius that has a
    Diffusion greater than zero is taken

    .. math::

        r_\\text{norm}=\\frac{1}{r_\\text{eff}}r.

    Parameters
    ----------
    data_link : string
        Link to data object generated by the sample routine :func:`sample`
    ax_area : list, optional
        Bin area percentage to calculate the axial diffusion coefficient
    intent : string, optional
        Set to **plot**, for plotting or set to **line** to only return the
        lineplot, leave empty for nothing
    is_norm : bool, optional
        True to normalize x-axis

    Returns
    -------
    diffusion : list
        List of the slope of the non-normalized diffusion coefficient
    """
    # Load data object
    sample = utils.load(data_link)

    # Load data
    inp = sample["inp"]
    bins = sample["bins"]
    msd_z = sample["axial"]
    norm = sample["norm"]

    # Normalize
    msd_norm = [[msd_z[i][j]/norm[i][j] if norm[i][j] > 0 else 0 for j in range(inp["window"])] for i in range(inp["bins"]+1)]

    # Calculate slope
    f_start = int(ax_area[0]*inp["window"])
    f_end = int(ax_area[1]*inp["window"])
    time_ax = [x*inp["step"]*inp["frame"] for x in range(inp["window"])]
    slope = [(msd_norm[i][f_end]-msd_norm[i][f_start])/(time_ax[f_end]-time_ax[f_start]) for i in range(inp["bins"]+1)]

    # Calculate diffusion coefficient
    diff = [msd*1e-9**2/2*1e2**2*1e5 for msd in slope]  # 10^-9 m^2s^-1

    # Normalize x-axis
    if is_norm:
        for i in range(len(diff)-1, 0, -1):
            if diff[i] > 0:
                x_max = bins[i+1]
                break

        bins_norm = [x/x_max for x in bins]

    # Plot
    if intent == "plot":
        # plt.figure(figsize=(10, 7))
        sns.set(style="whitegrid")
        sns.set_palette(sns.color_palette("deep"))

    if intent == "plot" or intent == "line":
        x_axis = bins_norm if is_norm else bins
        sns.lineplot(x=x_axis[:-1], y=diff)

    if intent == "plot":
        if is_norm:
            plt.xlabel("Normalized distance from pore center")
        else:
            plt.xlabel("Distance from pore center (nm)")
        plt.ylabel(r"Diffusion coefficient ($10^{-9}$ m${^2}$ s$^{-1}$)")

    return {"bins": bins, "diff": diff}


def mean(data_link_diff, data_link_dens, ax_area=[0.2, 0.8], is_norm=False, is_check=False):
    """This function uses the diffusion coefficient slope obtained from
    function :func:`bins` and the density slope of function
    :func:`poreana.density.calculate` to calculate a weighted diffusion
    coefficient inside the pore

    .. math::

        \\langle D_\\text{axial}\\rangle
        =\\frac{\\int\\rho(r)D_\\text{axial}(r)dA(r)}{\\int\\rho(r)dA(r)}.

    In a discrete form, following formula is evaluated

    .. math::

        \\langle D_\\text{axial}\\rangle=\\frac{\\sum_{i=1}^n\\rho(r_i)D_\\text{axial}(r_i)A(r_i)}{\\sum_{i=1}^n\\rho(r_i)A(r_i)}

    with the partial area

    .. math::

        A(r_i)=\\pi(r_i^2-r_{i-1}^2)

    of radial bin :math:`i`.

    Parameters
    ----------
    data_link_dens : string
        Link to density data object generated by the sample rountine
        :func:`poreana.density.sample`
    data_link_diff : string
        Link to diffusion data object generated by the sample routine
        :func:`sample`
    ax_area : list, optional
        Bin area percentage to calculate the axial diffusion coefficient
    is_norm : bool, optional
        True to normalize x-axis
    is_check : bool, optional
        True to show density function fit
    """
    # Load data
    dens = density.calculate(data_link_dens, is_print=False)
    diff = bins(data_link_diff, ax_area=ax_area, intent="", is_norm=is_norm)

    # Get number of bins
    bin_num = len(diff["bins"][:-1])

    # Set diffusion functions
    bins_f = diff["bins"][:-1]
    diff_f = diff["diff"]

    # Fit density function
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Polyfit may be poorly conditioned')
        param = np.polyfit(dens["in"][0][0][:-1], dens["in"][1], 100)
        dens_f = np.poly1d(param)(bins_f)

    # Check results
    if is_check:
        # Plot check
        plt.plot(dens["in"][0][0][:-1], dens["in"][1], diff["bins"][:-1], dens_f)
        plt.show()

        # Output data as excel
        df = pd.DataFrame({"bins": bins_f, "dens": dens_f, "diff": diff_f})
        df.to_excel("C:/Users/Ajax/Desktop/"+data_link_diff.split("/")[-1].split(".")[0]+".xlsx")

    # Integrate density
    dens_int = sum([dens_f[i]*(bins_f[i+1]**2-bins_f[i]**2) for i in range(bin_num-1)])

    # Calculate weighted diffusion
    diff_int = sum([dens_f[i]*diff_f[i]*(bins_f[i+1]**2-bins_f[i]**2) for i in range(bin_num-1)])

    # Normalize
    diff_weight = diff_int/dens_int

    print("Mean Diffusion axial: "+"%.3f" % diff_weight+" 10^-9 m^2s^-1")
