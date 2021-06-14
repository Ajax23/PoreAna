################################################################################
# Tables                                                                       #
#                                                                              #
"""Print analysis information in table form."""
################################################################################


import numpy as np
import pandas as pd

import poreana.utils as utils


###############################
# MC - Diffusion, Free Energy #
###############################
def mc_statistics(link_out, print_con=False):
    """This function prints the statistic of the MC Run for every lag time.

    Parameters
    ----------
    link_out: string
        Link to the diffusion data object generated by the MC Algorithm function
        :func:`poreana.mc.MC.do_mc_cycles`

    Returns
    -------
    df_results: obj
        Data frame of the MC Alogrithm statistics
    print_con: bool, optional
        True for printing in console
    """

    # Load data from obj file
    results = utils.load(link_out)
    model = results["model"]
    len_step = model["len_step"]
    inp = results["inp"]
    nmc_eq = inp["MC steps eq"]
    nmc = inp["MC steps"]

    # Read MC statistic
    nacc_df_mean = results["nacc_df"]
    nacc_diff_mean = results["nacc_diff"]
    list_diff_fluc = results["fluc_diff"]
    list_df_fluc = results["fluc_df"]

    # Table for MC Statistics
    data = [[str("%.4e" % list_df_fluc[i]) for i in len_step],[str("%.4e" % list_diff_fluc[i]) for i in len_step],[str("%.0f" % nacc_df_mean[i]) for i in len_step],[str("%.0f" % nacc_diff_mean[i]) for i in len_step],[str("%.2f" % (nacc_df_mean[i]*100/(nmc+nmc_eq))) for i in len_step],[str("%.2f" % (nacc_diff_mean[i]*100/(nmc+nmc_eq))) for i in len_step]]

    df_results = pd.DataFrame(data,index=list(['fluctuation df','fluctuation diff','acc df steps','acc diff steps','acc df steps (%)','acc diff steps (%)']),columns=list(len_step))

    # If the table has to print in console
    if print_con:
        print('\nStatistics of the MC Algorithm')
        print(df_results)

    # Set styler for pandas table in jupyter
    df_results = pd.DataFrame(df_results.rename_axis('Step Length', axis=1))
    styler = df_results.style.set_caption('Statistics of the MC Algorithm')
    df_results = styler.set_properties(**{'text-align': 'right'})
    df_results = df_results.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])

    return df_results


def mc_lag_time(link_out, print_con=False):
    """This function prints the final coefficients of the profile for
    every lag time profile.

    Parameters
    ----------
    link_out: string
        Link to the diffusion data object generated by the MC Algorithm function
        :func:`poreana.mc.MC.do_mc_cycles`

    Returns
    -------
    df_results: obj
        Data frame of profile coefficients
    print_con: bool, optional
        True for printing in console
    """

    # Load data from obj file
    results = utils.load(link_out)
    model = results["model"]
    len_step = model["len_step"]
    diff_coeff = results["list_diff_coeff"]
    df_coeff = results["list_df_coeff"]
    nD = model["nD"]
    nF = model["nF"]

    # Initialize data dictionary for the diffusion profile coefficients
    data = {}

    # Save diffusion profile coefficients on data dictionary
    for i in len_step:
        data[i] = [str("%.4e" % diff_coeff[i][j]) for j in range(nD)]

    # Pandas table
    diff_coeff = pd.DataFrame(data, index=list(np.arange(1, nD+1)), columns=list(len_step))
    diff_coeff = pd.DataFrame(diff_coeff.rename_axis('Step Length', axis=1))

    # If the table has to print in console
    if print_con:
        print('\nDiffusion coefficients')
        print(diff_coeff)

    # Set styler for pandas table in jupyter
    styler = diff_coeff.style.set_caption('Diffusion coefficients')
    diff_coeff = styler.set_properties(**{'text-align': 'right'})
    diff_coeff = diff_coeff.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])

    # Initialize data dictionary for the diffusion profile coefficients
    data = {}

    # Save free energy profile coefficients on data dictionary
    for i in len_step:
        data[i] = [str("%.4e" % df_coeff[i][j]) for j in range(nF)]

    # Pandas table
    df_coeff = pd.DataFrame(data, index=list(np.arange(1, nF+1)), columns=list(len_step))
    df_coeff = pd.DataFrame(df_coeff.rename_axis('Step Length', axis=1))

    # If the table has to print in console and not in a jupyter notebook
    if print_con:
        print('\nFree energy coefficients')
        print(df_coeff)

    # Set styler for pandas table in jupyter
    styler = df_coeff.style.set_caption('Free energy coefficients')
    df_coeff = styler.set_properties(**{'text-align': 'right'})
    df_coeff = df_coeff.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])

    return diff_coeff, df_coeff


def mc_model(link_out, print_con=False):
    """This function prints the model inputs of the calculation.

    Parameters
    ----------
    link_out: string
        Link to the diffusion data object generated by the MC Algorithm function
        :func:`poreana.mc.MC.do_mc_cycles`

    Returns
    -------
    df_model: obj
        Data frame of the model inputs
    print_con: bool, optional
        True for printing in console
    """
    # Read model inputs
    results = utils.load(link_out)
    model = results["model"]
    bin_number = model["bin number"]
    len_step = model["len_step"]
    len_frame = model["len_frame"]
    frame_num = model["num_frame"]
    nD = model["nD"]
    nF = model["nF"]
    nDrad = model["nDrad"]
    d = model["guess"]
    model = model["model"]

    if "pore" in results:
        system = "pore"
    if "box" in results:
        system = "box"


    # String which contains all lag times
    len_step_string = ', '.join(str(step) for step in len_step)

    # Dictionary for model inputs
    data = [str("%.f" % bin_number), len_step_string, str("%.2e" % (len_frame * 10**(-12))), str("%.f" % frame_num), str("%.f" % nD), str("%.f" % nF), str("%.f" % nDrad), model, str("%.2e" % (d * 10**(-6))), system]
    df_model = pd.DataFrame(data, index=list(['Bin number', 'step length', 'frame length (s)', 'frame number', 'nD', 'nF', 'nDrad', 'model', 'guess diffusion (m2/s-1)', 'system']), columns=list(['Input']))

    # If the table has to print in console and not in a jupyter notebook
    if print_con:
        print('\nModel Inputs')
        print(df_model)

    # Set styler for pandas table in jupyter
    styler = df_model.style.set_caption('Model Inputs')
    df_model = styler.set_properties(**{'text-align': 'right'})
    df_model = df_model.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])

    return df_model


def mc_inputs(link_out, print_con=False):
    """This function prints the MC Algorithm inputs of the calculation.

    Parameters
    ----------
    link_out: string
        Link to the diffusion data object generated by the MC Algorithm function
        :func:`poreana.mc.MC.do_mc_cycles`

    Returns
    -------
    df_mc: obj
        Data frame of the MC Alogrithm inputs
    print_con: bool, optional
        True for printing in console
    """
    # Load Results from the output object file
    results = utils.load(link_out)

    # Read MC inputs
    inp = results["inp"]
    nmc_eq = inp["MC steps eq"]
    nmc = inp["MC steps"]
    num_mc_update = inp["step width update"]
    print_freq = inp["print freq"]

    # Table for MC Inputs
    data = [nmc_eq, nmc, num_mc_update, print_freq]
    df_mc = pd.DataFrame(data, index=list(['MC steps (Equilibrium)', 'MC steps (Production)', 'movewidth update frequency', 'print frequency']), columns=list(['Input']))

    # If the table has to print in console and not in a jupyter notebook
    if print_con:
        print('\nMC Inputs')
        print(df_mc)

    # Set style for the pandas table
    styler = df_mc.style.set_caption('MC Inputs')
    df_mc = styler.set_properties(**{'text-align': 'right'})
    df_mc = df_mc.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])

    return df_mc
