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
        Link to the diffusion hdf5 data file generated by the MC Algorithm function
        :func:`poreana.mc.MC.do_mc_cycles`

    Returns
    -------
    df_results: obj
        Data frame of the MC Alogrithm statistics
    print_con: bool, optional
        True for printing in console
    """

    # Load Results from the output object file
    data = utils.load_hdf(link_out)

    # Load results
    results = data["output"]

    # Load model inputs
    model = data["model"]
    len_step = model["len_step"][:]
    inp = data["inp"]
    nmc_eq = int(inp["MC steps eq"][0])
    nmc = int(inp["MC steps"][0])

    # Read MC statistic
    nacc_df_mean = {}
    nacc_diff_mean = {}
    list_diff_fluc = {}
    list_df_fluc = {}

    for i in results["nacc_df"]:
        nacc_df_mean[int(i)] = float(results["nacc_df"][i][0])
    for i in results["nacc_diff"]:
        nacc_diff_mean[int(i)] = float(results["nacc_diff"][i][0])
    for i in results["fluc_diff"]:
        list_diff_fluc[int(i)] = float(results["fluc_diff"][i][0])
    for i in results["fluc_df"]:
        list_df_fluc[int(i)] = float(results["fluc_df"][i][0])

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
        Link to the diffusion hdf5 data file generated by the MC Algorithm function
        :func:`poreana.mc.MC.do_mc_cycles`

    Returns
    -------
    df_results: obj
        Data frame of profile coefficients
    print_con: bool, optional
        True for printing in console
    """

    # Load Results from the output object file
    data = utils.load_hdf(link_out)

    # Load results
    results = data["output"]
    diff_coeff = {}
    df_coeff = {}
    for i in results["diff_coeff"]:
        diff_coeff[float(i)] = results["diff_coeff"][i][:]
    for i in results["df_coeff"]:
        df_coeff[float(i)] = results["df_coeff"][i][:]

    # Load model inputs
    model = data["model"]
    len_step = model["len_step"][:]
    nD = int(model["nD"][0])
    nF = int(model["nF"][0])

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
        Link to the diffusion hdf5 data file generated by the MC Algorithm function
        :func:`poreana.mc.MC.do_mc_cycles`

    Returns
    -------
    df_model: obj
        Data frame of the model inputs
    print_con: bool, optional
        True for printing in console
    """
    # Load Results from the output object file
    #data = utils.load(link_out)
    data = utils.load_hdf(link_out)

    # Load model inputs
    model = data["model"]
    bin_number = model["bin number"][0]
    len_step = model["len_step"][:]
    len_frame = model["len_frame"][0]
    frame_num = model["num_frame"][0]
    nD = model["nD"][0]
    nF = model["nF"][0]
    nDrad = model["nDrad"][0]
    d = model["guess"][0]
    model_string = model["model"][0].decode("utf-8")
    pbc = model["pbc"][0]
    direction = model["direction"][0]

    # String for pbc
    if int(pbc[0]) == 1:
        pbc = "True"
    else:
        pbc = "False"

    # String for system
    if "pore" in data:
        system = "pore"
    if "box" in data:
        system = "box"

    if direction == 0:
        direction = "x"
    elif direction == 1:
        direction = "y"
    elif direction == 2:
        direction = "z"

    # Len step string
    len_step_string = ', '.join(str(step) for step in len_step)

    # Dictionary for model inputs
    data = [str("%.i" % bin_number), len_step_string, str("%.2e" % (len_frame * 10**(-12))), str("%.i" % frame_num), str("%.i" % nD), str("%.i" % nF), str("%.i" % nDrad), model_string, str("%.2e" % (d * 10**(-6))), system, pbc, direction]
    df_model = pd.DataFrame(data, index=list(['Bin number', 'step length', 'frame length (s)', 'frame number', 'nD', 'nF', 'nDrad', 'model', 'guess diffusion (m2/s-1)', 'system',"pbc", "direction"]), columns=list(['Input']))

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
        Link to the diffusion hdf5 data file generated by the MC Algorithm function
        :func:`poreana.mc.MC.do_mc_cycles`

    Returns
    -------
    df_mc: obj
        Data frame of the MC Alogrithm inputs
    print_con: bool, optional
        True for printing in console
    """
    # Load Results from the output object file
    #data = utils.load(link_out)
    data = utils.load_hdf(link_out)

    # Read MC inputs
    inp = data["inp"]
    nmc_eq = int(inp["MC steps eq"][0])
    nmc = int(inp["MC steps"][0])
    num_mc_update = int(inp["step width update"][0])
    print_freq = int(inp["print freq"][0])
    temp = float(inp["temperature"][0])

    # Table for MC Inputs
    data = [str("%.i" % nmc_eq), str("%.i" % nmc), temp, str("%.i" % num_mc_update), str("%.i" % print_freq)]
    df_mc = pd.DataFrame(data, index=list(['MC steps (Equilibrium)', 'MC steps (Production)','temperature (MC)', 'movewidth update frequency', 'print frequency']), columns=list(['Input']))

    # If the table has to print in console and not in a jupyter notebook
    if print_con:
        print('\nMC Inputs')
        print(df_mc)

    # Set style for the pandas table
    styler = df_mc.style.set_caption('MC Inputs')
    df_mc = styler.set_properties(**{'text-align': 'right'})
    df_mc = df_mc.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])

    return df_mc
