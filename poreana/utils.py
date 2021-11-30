################################################################################
# Utils                                                                        #
#                                                                              #
"""Here popular basic methods are noted."""
################################################################################


import os
import time
import pickle
import h5py
import numpy as np
import pandas as pd
import datetime
import poreana.diffusion as diffusion
import poreana.tables as tables


def mkdirp(directory):
    """Create directory if it does not exist.

    Parameters
    ----------
    directory : string
        Directory name
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def column(data):
    """Convert given row list matrix into column list matrix

    Parameters
    ----------
    data : list
        Row data matrix

    Returns
    -------
    data_col : list
        column data matrix
    """
    num_row = len(data)
    num_col = len(data[0])

    data_col = [[] for i in range(num_col)]

    for i in range(num_row):
        for j in range(num_col):
            data_col[j].append(data[i][j])

    return data_col


def tic():
    """MATLAB tic replica - return current time.

    Returns
    -------
    time : float
        Current time in seconds
    """
    return time.time()


def toc(t, message="", is_print=True):
    """MATLAB toc replica - return time difference to tic and alternatively
    print a message.

    Parameters
    ----------
    t : float
        Starting time - given by tic function
    message : string, optional
        Custom output message
    is_print : bool, optional
        True for printing an output message

    Returns
    -------
    time : float
        Time difference
    """
    if message:
        message += " - runtime = "

    t_diff = time.time()-t

    if is_print:
        print(message+"%6.3f" % t_diff+" s")

    return t_diff


def save(obj, link):
    """Save an object or hdf5 file using pickle in the specified link.

    Parameters
    ----------
    obj : Object
        Object to be saved
    link : string
        Specific link to save object
    """

    if link[-3:]=="obj":
        with open(link, "wb") as f:
            pickle.dump(obj, f)

    elif link[-2:]=="h5":
        # Save results in a hdf5 file
        f = h5py.File(link, 'w')

        # Create input groupe
        groups = {}
        data_base = {}

        for key in obj.keys():
            groups[key] = f.create_group(key)
            if not key =="box":
                data_base[key] = obj[key].keys()
        #print(groups)
        for gkey in groups:
            if not gkey =="box":
                for base in data_base[gkey]:
                    if isinstance(obj[gkey][base],dict):
                        data = groups[gkey].create_group(base)
                        for base2 in obj[gkey][base]:
                            if isinstance(obj[gkey][base][base2],(list, np.ndarray)) :
                                data.create_dataset(str(base2),data = obj[gkey][base][base2])
                            else:
                                value = data.create_dataset(str(base2), shape=(1,1),dtype=type(obj[gkey][base][base2]))
                                value[0] = obj[gkey][base][base2]

                    elif isinstance(obj[gkey][base],str):
                        dt = h5py.special_dtype(vlen=str)
                        string = groups[gkey].create_dataset(str(base), (1), dtype=dt)
                        string[0] = obj[gkey][base]
                    elif isinstance(obj[gkey][base],(list, np.ndarray)):
                        data = groups[gkey].create_dataset(str(base), data = obj[gkey][base])
                    else:
                        data = groups[gkey].create_dataset(str(base), shape=(1,1))
                        data[0] = obj[gkey][base]
            else:
                groups[gkey].create_dataset("length",data = obj[gkey])


def load(link):
    """Load pickled object or a hdf5 file from the specified folder.

    Parameters
    ----------
    link : string
        Specific link to load object

    Returns
    -------
    obj : Object
        Loaded object
    """
    if link[-3:]=="obj":
        with open(link, 'rb') as f:
            return pickle.load(f)
    elif link[-2:]=="h5":
        data = h5py.File(link, 'r')
        data_load = {}

        for keys in data.keys():
            data_load[keys] = {}
            if keys=="data":
                for keys2 in data[keys].keys():
                    try:
                        data_load[keys][int(keys2)] = data[keys][keys2][:]
                    except:
                        data_load[keys][keys2] = data[keys][keys2][:]
            else:
                for keys2 in data[keys].keys():
                    data_load[keys][keys2] = {}
                    try:
                        for keys3 in data[keys][keys2].keys():
                            data_load[keys][keys2][int(keys3)] = data[keys][keys2][keys3][:]
                    except:
                        if len(data[keys][keys2][:])==1:
                            try:
                                data_load[keys][keys2] = float(data[keys][keys2][0])
                            except:
                                data_load[keys][keys2] = data[keys][keys2][0].decode("utf-8")
                        else:
                            data_load[keys][keys2] = data[keys][keys2][:]
        return data_load

def check_filetype(link):
    if not link[-2:]=="h5" or not link[-3:]=="obj":
        print("Wrong data type. Please select .obj or .h5 as the data type")
        return


def file_to_text(link):
    """ This function converts an output directory in txt file.

    Parameters
    ----------
    link : string
        Link to output hdf5 file
    pickle : dict
        dictionary which should be saved
    """

    # Load data
    data = load(link)

    # Output
    if link[-2:]=="h5":
        link_txt = link[:-2] + "txt"
    elif link[-3:]=="obj":
        link_txt = link[:-3] + "txt"

    if "pore" in data:
        system = "pore"
        pore = data["pore"]
        res = float(pore["res"])
        diam = float(pore["diam"])
        box = pore["box"]
        type = pore["type"]
        data = [[box],[diam],[res],[type]]
        df_system = pd.DataFrame(data, index=list(["Box dimension (nm)","Pore diameter (nm)","reservoir (nm)", "type"]),columns=list(["Value"]))
    if "box" in data:
        system = "box"
        box_group = data["box"]
        box = box_group["length"]
        data = [[box]]
        df_system = pd.DataFrame(data, index=list(["Box dimension (nm)"]), columns=list(["Value"]))


    # # Save txt file
    # # Calculated diffusion coefficient
    #diff_fit = diffusion.mc_fit(link, is_print=False)
    df_inputs = tables.mc_inputs(link, print_con=False)
    df_model = tables.mc_model(link, print_con=False)
    df_results = tables.mc_results(link, print_con=False)
    df_model_string = df_model.to_string(header=True, index=True)
    df_inputs_string = df_inputs.to_string(header=True, index=True)
    df_results_string = df_results.to_string(header=True, index=True)
    df_system_string = df_system.to_string(header=True, index=True)
    with open(link_txt, 'w') as file:
        file.write("# This file was created " + str(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "\n")
        file.write("# Created by:\n")
        file.write("\t\t\t\t\t PoreAna\n\n")
        file.write("[System]\n")
        file.write(df_system_string)
        file.write("\n\n\n[Model Inputs]\n")
        file.write(df_model_string)
        file.write("\n\n\n")
        file.write("[MC Inputs]\n")
        file.write(df_inputs_string)
        file.write("\n\n\n")
        file.write("[MC Results]\n")
        file.write(df_results_string)

        file.write("\n\n[Diffusion profile]\n\n")
        file.write("\tBins [nm] \t \t \t \t Diffusion coefficient [10^-9 m^2s^-1] \n")
        diff = diffusion.mc_fit(link, is_print=False, is_plot = False)
        for i in range(len(diff[2])):
            file.write("\t%.2f\t\t\t\t" % diff_prof[2][i] + "%.2f" % diff_prof[0][i] + "\n")
        file.close()

def mumol_m2_to_mols(c, A):
    """Convert the concentration in :math:`\\frac{\\mu\\text{mol}}{\\text{m}^2}`
    to number of molecules.

    The concentration is given as

    .. math::

        c=\\frac{N}{N_A\\cdot A}

    with surface :math:`A` and avogadro constant
    :math:`N_A=6.022\\cdot10^{23}\\,\\text{mol}^{-1}`. Assuming that the unit of
    the concentration is :math:`\\mu\\text{mol}\\cdot\\text{m}^{-2}` and of the
    surface is :math:`\\text{nm}^2`, the number of molecules is given by

    .. math::

        N=c\\cdot N_A\\cdot A
        =[c]\\cdot10^{-24}\\,\\frac{\\text{mol}}{\\text{nm}^2}\\cdot6.022\\cdot10^{23}\\,\\frac{1}{\\text{mol}}\\cdot[A]\\,\\text{nm}^2

    and thus

    .. math::

        \\boxed{N=0.6022\\cdot[c]\\cdot[A]}\\ .

    Parameters
    ----------
    c : float
        Concentration in :math:`\\frac{\\mu\\text{mol}}{\\text{m}^2}`
    A : float
        Surface in :math:`\\text{nm}^2`

    Returns
    -------
    N : float
        Number of molecules
    """
    return 0.6022*c*A


def mols_to_mumol_m2(N, A):
    """Convert the number of molecules to concentration in
    :math:`\\frac{\\mu\\text{mol}}{\\text{m}^2}`.

    The concentration is given as

    .. math::

        c=\\frac{N}{N_A\\cdot A}

    with surface :math:`A` and avogadro constant
    :math:`N_A=6.022\\cdot10^{23}\\,\\text{mol}^{-1}`. Assuming that the unit of
    the concentration is :math:`\\mu\\text{mol}\\cdot\\text{m}^{-2}` and of the
    surface is :math:`\\text{nm}^2`, the number of molecules is given by

    .. math::

        c=\\frac{[N]}{6.022\\cdot10^{23}\\,\\text{mol}^{-1}\\cdot[A]\\,\\text{nm}^2}

    and thus

    .. math::

        \\boxed{c=\\frac{N}{0.6022\\cdot[A]}\\cdot\\frac{\\mu\\text{mol}}{\\text{m}^2}}\\ .

    Parameters
    ----------
    N : int
        Number of molecules
    A : float
        Surface in :math:`\\text{nm}^2`

    Returns
    -------
    c : float
        Concentration in :math:`\\frac{\\mu\\text{mol}}{\\text{m}^2}`
    """
    return N/0.6022/A


def mmol_g_to_mumol_m2(c, SBET):
    """Convert the concentration :math:`\\frac{\\text{mmol}}{\\text{g}}`
    to :math:`\\frac{\\mu\\text{mol}}{\\text{m}^2}`.

    This is done by dividing the concentration per gram by the material surface
    per gram property :math:`S_\\text{BET}`

    .. math::

        \\boxed{c_A=\\frac{c_g}{S_\\text{BET}}\\cdot10^3}\\ .

    Parameters
    ----------
    c : float
        Concentration in :math:`\\frac{\\text{mmol}}{\\text{g}}`
    SBET : float
        Material surface in :math:`\\frac{\\text{m}^2}{\\text{g}}`

    Returns
    -------
    c : float
        Concentration in :math:`\\frac{\\mu\\text{mol}}{\\text{m}^2}`
    """
    return c/SBET*1e3


def mmol_l_to_mols(c, V):
    """Convert the concentration in :math:`\\frac{\\text{mmol}}{\\text{l}}`
    to number of molecules.

    The concentration in regard to volume is calculated by

    .. math::

        c=\\frac{N}{N_A\\cdot V}

    with volume :math:`V`. Assuming that the unit of the concentration is
    :math:`\\text{mmol}\\cdot\\text{l}` and of the volume is
    :math:`\\text{nm}^3`, the number of molecules is given by

    .. math::

        N=c\\cdot N_A\\cdot V
        =[c]\\cdot10^{-27}\\,\\frac{\\text{mol}}{\\text{nm}^3}\\cdot6.022\\cdot10^{23}\\,\\frac{1}{\\text{mol}}\\cdot[V]\\,\\text{nm}^3

    and thus

    .. math::

        \\boxed{N=6.022\\cdot10^{-4}\\cdot[c]\\cdot[V]}\\ .

    Parameters
    ----------
    c : float
        Concentration in :math:`\\frac{\\text{mmol}}{\\text{l}}`
    V : float
        Surface in :math:`\\text{nm}^3`

    Returns
    -------
    N : float
        Number of molecules
    """
    return 6.022e-4*c*V


def mols_to_mmol_l(N, V):
    """Convert the number of molecules to concentration in
    :math:`\\frac{\\text{mmol}}{\\text{l}}`.

    The concentration in regard to volume is calculated by

    .. math::

        c=\\frac{N}{N_A\\cdot V}

    with volume :math:`V`. Assuming that the unit of the concentration is
    :math:`\\text{mmol}\\cdot\\text{l}` and of the volume is
    :math:`\\text{nm}^3`, the number of molecules is given by

    .. math::

        c=\\frac{N}{6.022\\cdot10^{23}\\cdot[V]\\,\\text{nm}^3}

    and thus

    .. math::

        \\boxed{c=\\frac{N}{6.022\\times10^{-4}\\cdot[V]}\\cdot\\frac{\\text{mmol}}{\\text{l}}}\\ .

    Parameters
    ----------
    N : int
        Number of molecules
    V : float
        Surface in :math:`\\text{nm}^3`

    Returns
    -------
    c : float
        Concentration in :math:`\\frac{\\text{mmol}}{\\text{l}}`
    """
    return N/6.022e-4/V
