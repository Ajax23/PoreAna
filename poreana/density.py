################################################################################
# Density                                                                      #
#                                                                              #
"""Analyse density in a pore."""
################################################################################


import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import poreana.utils as utils


def bins(link_data, area=[[10, 90], [10, 90]], target_dens=0, is_print=True):
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

        V_j^\\text{ex}=(x_\\text{pore}\\cdot y_\\text{pore}-\\pi r^2)z_j

    with pore width :math:`x_\\text{pore}`, height :math:`y_\\text{pore}`, pore
    radius :math:`r` and slice width :math:`z_j`. Thus

    .. math::

        \\rho_{n,j}^\\text{ex}=\\frac{N_j}{V_j^\\text{ex}}=\\frac{N_j}{x_\\text{pore}\\cdot y_\\text{pore}-\\pi r^2}\\frac{1}{z_j}.

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
        Link to data object generated by the sample routine :func:`poreana.sample.Sample.init_density`
    area : list,  optional
        Bin areas to calculate the mean number density from (pore, exout)
    target_dens : float, optional
        Target density in :math:`\\frac{\\text{kg}}{\\text{m}^3}`
    is_print : bool, optional
        True to print output
    """
    # Load data object
    sample = utils.load(link_data)
    is_pore = "pore" in sample

    # Load bins
    bins = {}
    bins["in"] = sample["data"]["in"] if is_pore else []
    bins["ex"] = sample["data"]["ex"]

    # Load width
    width = {}
    width["in"] = sample["data"]["in_width"] if is_pore else []
    width["ex"] = sample["data"]["ex_width"]

    # Load input data
    inp = sample["inp"]
    bin_num = inp["bin_num"]
    num_frame = inp["num_frame"]
    entry = inp["entry"]
    mass = inp["mass"]

    # Load pore data
    if is_pore:
        pore = sample["pore"]
        pore_type = pore["type"]
        res = pore["res"]
        diam = pore["diam"]
        box = pore["box"]
    else:
        box = sample["box"]

    # Calculate bin volume
    volume = {}
    if is_pore and pore_type=="CYLINDER":
        volume["in"] = [math.pi*(box[2]-2*res-2*entry)*(width["in"][i+1]**2-width["in"][i]**2) for i in range(0, bin_num+1)]
        volume["ex"] = [2*width["ex"][1]*(box[0]*box[1]-math.pi*(diam/2)**2) for i in range(bin_num+1)]
    elif is_pore and pore_type=="SLIT":
        volume["in"] = [box[0]*(box[2]-2*res-2*entry)*(width["in"][i+1]-width["in"][i])*2 for i in range(0, bin_num+1)]
        volume["ex"] = [2*width["ex"][1]*box[0]*(box[1]-diam) for i in range(bin_num+1)]
    else:
        volume["ex"] = [width["ex"][1]*box[0]*box[1] for i in range(bin_num+1)]

    # Calculate the number density
    num_dens = {}
    num_dens["in"] = [bins["in"][i]/volume["in"][i]/num_frame for i in range(bin_num+1)] if is_pore else []
    num_dens["ex"] = [bins["ex"][i]/volume["ex"][i]/num_frame for i in range(bin_num+1)]

    # Calculate the mean in the selected area
    mean = {}
    mean["in"] = np.mean(num_dens["in"][area[0][0]:area[0][1]]) if is_pore else []
    mean["ex"] = np.mean(num_dens["ex"][area[1][0]:area[1][1]])

    # Calculate Density
    dens = {}
    dens["in"] = mass*10/6.022*mean["in"] if is_pore else []
    dens["ex"] = mass*10/6.022*mean["ex"]

    # Calculate difference to target density
    num_diff = (target_dens/mass/10*6.022-mean["ex"])*box[0]*box[1]*res*2 if target_dens else 0

    # Output
    if is_print:
        if is_pore:
            print("Density inside  Pore = "+"%5.3f"%mean["in"]+" #/nm^3 ; "+"%7.3f"%dens["in"]+" kg/m^3")
            print("Density outside Pore = "+"%5.3f"%mean["ex"]+" #/nm^3 ; "+"%7.3f"%dens["ex"]+" kg/m^3")
        else:
            print("Density = "+"%5.3f"%mean["ex"]+" #/nm^3 ; "+"%7.3f"%dens["ex"]+" kg/m^3")
        if target_dens:
            print("Density difference   = "+"%5.3f" % (target_dens-dens["ex"])+" kg/m^3 ; "
                  +"%4.2f" % ((1-dens["ex"]/target_dens)*100)+" % ; "
                  +"%3i" % num_diff+" #")

    # Return output
    return  {"sample": sample, "num_dens": num_dens, "mean": mean, "dens": dens, "diff": num_diff}


def bins_plot(density, intent="", target_dens=0, is_mean=False, kwargs={}):
    """This function plots the density of the given object. If an intent is
    given instead, only a plot-function will be called. Available options
    for ``intent`` are

    * empty string - Create subplots for the density inside and outside the pore
    * **in** - Create plot for the density inside pore
    * **ex** - Create plot for the density outside pore

    Parameters
    ----------
    density : dictionary
        Density object from the density calculation :func:`bins`
    intent : string, optional
        Intent for plotting
    target_dens : float, optional
        Target density for plot
    is_mean : bool, optional
        True to plot mean values
    kwargs: dict, optional
        Dictionary with plotting parameters (only for given intent)
    """
    # Define bins
    width = {}
    width["in"] = density["sample"]["data"]["in_width"][:-1] if "pore" in density["sample"] else []
    width["ex"] = density["sample"]["data"]["ex_width"]

    # Full plot
    if not intent:
        # Plot
        plt.figure(figsize=(10, 7))

        plt.subplot(211)
        sns.lineplot(x=width["in"], y=density["num_dens"]["in"])
        if is_mean:
            sns.lineplot(x=width["in"], y=[density["mean"]["in"] for x in width["in"]])

        plt.xlim([0, width["in"][-1]])
        plt.xlabel("Distance from pore center (nm)")
        plt.ylabel(r"Density (atoms nm$^{-3}$)")
        plt.legend(["Density", "Mean"])

        plt.subplot(212)
        #
        sns.lineplot(x=width["ex"], y=density["num_dens"]["ex"])
        if is_mean:
            sns.lineplot(x=width["ex"], y=[density["mean"]["ex"] for x in width["ex"]])

        if target_dens:
            sns.lineplot(x=width["ex"], y=[target_dens]*len(width["ex"]))

        plt.xlim([0, width["ex"][-1]])
        plt.xlabel("Distance from reservoir end (nm)")
        plt.ylabel(r"Density in (atoms nm$^{-3}$)")
        if is_mean:
            plt.legend(["Density Ex", "Mean Ex", "Target density"])
        else:
            plt.legend(["Density Ex", "Target density"])

    # Intent plots
    else:
        if intent not in ["in", "ex"]:
            print("Invalid intent. Check documentation for available options.")
            return

        sns.lineplot(x=width[intent], y=density["num_dens"][intent], **kwargs)
        plt.xlim([0, width[intent][-1]])
