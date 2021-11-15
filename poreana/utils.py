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
import poreana.diffusion as diffusion


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
    """Save an object using pickle in the specified link.

    Parameters
    ----------
    obj : Object
        Object to be saved
    link : string
        Specific link to save object
    """
    with open(link, "wb") as f:
        pickle.dump(obj, f)


def load(link):
    """Load pickled object from the specified folder.

    Parameters
    ----------
    link : string
        Specific link to load object

    Returns
    -------
    obj : Object
        Loaded object
    """
    with open(link, 'rb') as f:
        return pickle.load(f)

def load_hdf(link):
    """Load hdf5 file from the specified folder.

    Parameters
    ----------
    link : string
        Specific link to hdf5 file

    Returns
    -------
    f : Object
        Loaded hdf5 file
    """
    f = h5py.File(link, 'r')
    return f

def save_dict_to_hdf(link, pickle):
    """ This function saves the output directory in a hdf5 file.

    Parameters
    ----------
    link : string
        Link to output hdf5 file
    pickle : dict
        dictionary which should be saved
    """
    # Save results in a hdf5 file
    f = h5py.File(link, 'w')

    # Create input groupe
    keys = pickle.keys()
    groups = {}
    data_base = {}

    for i in keys:
        groups[i] = f.create_group(i)
        if not i =="box":
            data_base[i] = pickle[i].keys()

    for i in groups:
        if not i =="box":
            for j in data_base[i]:
                if isinstance(pickle[i][j],dict):
                    data = groups[i].create_group(j)
                    for z in pickle[i][j]:
                        if isinstance(pickle[i][j][z],(list, np.ndarray)) :
                            data.create_dataset(str(z),data = pickle[i][j][z])
                        else:
                            value = data.create_dataset(str(z), shape=(1,1))
                            value[0] = pickle[i][j][z]

                elif isinstance(pickle[i][j],str):
                    dt = h5py.special_dtype(vlen=str)
                    string = groups[i].create_dataset(str(j), (1), dtype=dt)
                    string[0] = pickle[i][j]
                elif isinstance(pickle[i][j],(list, np.ndarray)):
                    data = groups[i].create_dataset(str(j), data = pickle[i][j])
                else:
                    data = groups[i].create_dataset(str(j), shape=(1,1))
                    data[0] = pickle[i][j]
        else:
            groups[i].create_dataset("length",data = pickle[i])

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
    # Step Länge etc
    if link[-2:]=="h5":
        data = load_hdf(link)
        link_txt = link[:-2] + "txt"
        if "pore" in data:
            system = "pore"
            pore = data["pore"]
            res = float(pore["res"][0])
            diam = float(pore["diam"][0])
            box = pore["box"][:]
            type = pore["type"][0].decode("utf-8")
            diff_fit_pore = diffusion.mc_fit(link, section = "pore", is_print=False)
            diff_fit_res = diffusion.mc_fit(link, section = "reservoir", is_print=False)
        if "box" in data:
            system = "box"
            box_group = data["box"]
            box = box_group["length"]
    elif link[-3:]=="obj":
        data = load(link)
        link_txt = link[:-3] + "txt"
        if "pore" in data:
            system = "pore"
            pore = data["pore"]
            res = float(pore["res"])
            diam = float(pore["diam"])
            box = pore["box"]
            type = pore["type"]
            diff_fit_pore = diffusion.mc_fit(link, section = "pore", is_print=False)
            diff_fit_res = diffusion.mc_fit(link, section = "reservoir", is_print=False)
        if "box" in data:
            system = "box"
            box_group = data["box"]
            box = box_group["length"]

    # # Save txt file
    # # Calculated diffusion coefficient
    diff_fit = diffusion.mc_fit(link, is_print=False)
    with open(link_txt, 'w') as file:
        file.write("This file was created by PoreAna Package\n\n")
        file.write("Analyzed system: " + system + "\n\n")
        file.write("\tBox: " + str(box) + "\n\n")
        if system == "pore":
            file.write("\tPore Type: " + type + "\n")
            file.write("\tReservoir: " + str(res) + "\n")
            file.write("\tDiameter: " + str(diam) + "\n\n")
        file.write("Diffusion analysis for the whole system:\n\n")
        file.write("\tDiffusion axial: "+"%.4e" % (diff_fit[0] * 10 **-9) + " m^2/s\n")
        file.write("\tResidual: "+"%.4e" % (diff_fit[3] * 10 **-9) + " m^2/s\n\n")
        if system == "pore":
            file.write("Diffusion analysis for the pore:\n\n")
            file.write("\tDiffusion axial: "+"%.4e" % (diff_fit_pore[0] * 10 **-9) + " m^2/s\n")
            file.write("\tResidual: "+"%.4e" % (diff_fit_pore[3] * 10 **-9) + " m^2/s\n\n")
            file.write("Diffusion analysis for the reservoir:\n\n")
            file.write("\tDiffusion axial: "+"%.4e" % (diff_fit_res[0] * 10 **-9) + " m^2/s\n")
            file.write("\tResidual: "+"%.4e" % (diff_fit_res[3] * 10 **-9) + " m^2/s\n\n")

        # file.write("Diffusion profile\n\n")
        # file.write("\tBins [nm] \t \t \t \t Diffusion coefficient [10^-9 m^2s^-1] \n")
        # for i in range(len(diff_prof[2])):
        #     file.write("\t%.2f\t\t\t\t" % diff_prof[2][i] + "%.2f" % diff_prof[0][i] + "\n")
        # file.close()

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
