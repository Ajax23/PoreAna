################################################################################
# Free Energy                                                                  #
#                                                                              #
"""Analyse free energy in a pore."""
################################################################################


import seaborn as sns
import matplotlib.pyplot as plt
import poreana.utils as utils


####################
# Free Energy - MC #
####################
def mc_profile(link, len_step=[], is_plot=True, kwargs={}):
    """This function plots the free energy profile over the box for the
    calculated lag times. In contrast to the diffusion profile the diffusion
    profile has not a dependency on the lag time. If the free energy profiles
    are not close to equal the calculation is incorrect.

    Parameters
    ----------
    link : string
        Link to the diffusion hdf5 data file generated by the :func:`poreana.mc.MC.do_mc_cycles`
    len_step: integer list, optional
        List of the different step length, if it is [] all free energy profiles
        depending on the lag time are shown
    is_plot : bool, optional
        Show free energy profile
    kwargs: dict, optional
        Dictionary with plotting parameters

    Returns
    -------
    df_bin: dictionary
        free energy profile for every calculated lag time
    bins : list
        bins over the box length
    """

    # Load Results from the output object file
    data = utils.load_hdf(link)

    df_bin = {}

    # Load results
    results = data["output"]
    for i in results["df_profile"]:
        df_bin[int(i)] = results["df_profile"][i][:]

    # Load model inputs
    model = data["model"]
    dt = float(model["len_frame"][0])
    bins = model["bins"]

    # If no specific step length is chosen take the step length from the object file
    if not len_step:
        len_step = model["len_step"][:]

    # Set legend
    legend = ["$\\Delta t_{\\alpha}$ = " + str(len_step[i] * dt) + " ps" for i in range(len(len_step))]

    # Plot the free energy profiles
    if is_plot:
        for i in len_step:
            sns.lineplot(x=bins, y=(df_bin[i]), **kwargs)

        # Plot options
        plt.xlabel("Box length (nm)")
        plt.ylabel("Free energy (-)")
        plt.legend(legend)
        plt.xlim([0,max(bins)])

    return df_bin, bins
