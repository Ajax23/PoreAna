################################################################################
# Utils                                                                        #
#                                                                              #
"""Here popular basic methods are noted."""
################################################################################


import os
import time
import h5py
import yaml
import pickle
import datetime
import numpy as np
import pandas as pd
import poreana as pa


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
    """Save an object files, yaml files, and hd5 files using pickle in the specified link.

    Parameters
    ----------
    obj : Object
        Object to be saved
    link : string
        Specific link to save object
    """
    # Pickle object file


    # Hd5 file
    if link.split(".")[-1]=="h5":
        # Save results in a hdf5 file
        f = h5py.File(link, 'w')

        # Create input groupe
        groups = {}
        data_base = {}

        # Create groups for the keys in the obj file
        for key in obj.keys():
            groups[key] = f.create_group(key)

            # If key "box" only a value is on frist key level
            if not key in ["box","type"]:
            # For all other key read the second level keys
                data_base[key] = obj[key].keys()

        for gkey in groups:
            # Write the box length on data base
            if gkey =="box":
                groups[gkey].create_dataset("length",data=obj[gkey]["length"])
            # Write the type of calculation on data base
            elif gkey =="type":
                dt = h5py.special_dtype(vlen=str)
                type = groups[gkey].create_dataset("type", (1), dtype=dt)
                type[0] = obj[gkey]
            # Loop over second level keys
            else:
                for base in data_base[gkey]:
                    # Check if a third level exists
                    if isinstance(obj[gkey][base], dict):
                        # If a third level exists create new group in first level groups
                        data = groups[gkey].create_group(base)
                        # Loop over thrid level keys
                        for base2 in obj[gkey][base]:
                            # Check if third level key is a matrix
                            if isinstance(obj[gkey][base][base2], (list, np.ndarray)):
                                data.create_dataset(str(base2), data=obj[gkey][base][base2])
                            # Else is a vlaue (int/float)
                            else:
                                value = data.create_dataset(str(base2), shape=(1,1))
                                value[0] = obj[gkey][base][base2]
                    # Check If second level is value/list or string
                    # If string
                    elif isinstance(obj[gkey][base],str):
                        dt = h5py.special_dtype(vlen=str)
                        string = groups[gkey].create_dataset(str(base), (1), dtype=dt)
                        string[0] = obj[gkey][base]
                    # If list
                    elif isinstance(obj[gkey][base], (list, np.ndarray)):
                        data = groups[gkey].create_dataset(str(base), data=obj[gkey][base])
                    # If value
                    else:
                        data = groups[gkey].create_dataset(str(base), shape=(1,1))
                        data[0] = obj[gkey][base]

    elif link.split(".")[-1]=="yml":
        with open(link, "w") as file_out:
            file_out.write(yaml.dump(obj))
    else:
        if link.split(".")[-1]!="obj":
            print("Wrong data type")
            return

        with open(link, "wb") as f:
            pickle.dump(obj, f)

def load(link, file_type=""):
    """Load pickled object files, yaml files, and hd5 files.

    Parameters
    ----------
    link : string
        Specific link to load object
    file_type : string, optional
        Specify filetype - **obj**, **yml**, **h5**  leave empty for automatic
        determination

    Returns
    -------
    obj : Object
        Loaded object
    """
    # YAML file
    if file_type=="yml" or (not file_type and link.split(".")[-1]=="yml"):
        with open(link, "r") as file_in:
            return yaml.load(file_in, Loader=yaml.UnsafeLoader)

    # Hd5 file
    elif file_type=="h5" or (not file_type and link.split(".")[-1]=="h5"):
        data = h5py.File(link, 'r')
        data_load = {}

        # Check keys in hdf5 file
        for keys in data.keys():
            data_load[keys] = {}
            # If sample file is load
            if keys=="data":
                for keys2 in data[keys].keys():
                    try:
                        data_load[keys][int(keys2)] = data[keys][keys2][:]
                    except(ValueError):
                        data_load[keys][keys2] = data[keys][keys2][:]
            # If key type only a string has to be load
            elif keys=="type":
                data_load[keys] = str(data[keys][keys][0], 'utf-8') #data[keys][keys][0].encode().decode("utf-8")
            # other keys
            else:
                # Load second level keys
                for keys2 in data[keys].keys():
                    data_load[keys][keys2] = {}
                    # Try if a thrid level exist
                    try:
                        # load thrid level
                        for keys3 in data[keys][keys2].keys():
                            data_load[keys][keys2][int(keys3)] = data[keys][keys2][keys3][:]
                    # Save second level data
                    except(AttributeError):
                        if len(data[keys][keys2][:])==1:
                            try:
                                data_load[keys][keys2] = float(data[keys][keys2][0])
                            except(ValueError):
                                data_load[keys][keys2] = str(data[keys][keys2][0], 'utf-8') #.encode().decode("utf-8")
                        else:
                            data_load[keys][keys2] = data[keys][keys2][:]
        return data_load
    else:
        if link.split(".")[-1]!="obj":
            print("Wrong data type")
            return
        with open(link, 'rb') as f:
            return pickle.load(f)



def file_to_text(link, link_output, link_dens=[]):
    """This function converts an output directory in txt file. For the bin diffusion
    result text file the density sampling file is additionally necessary.

    Parameters
    ----------
    link : string
        Link to output hdf5 or obj file
    link_output : String
        Link to an output text file
    link_dens : optional
        Link to density output hdf5 or obj file. Only necessary for the bin
        diffusion and gyration text ouput.
    """

    # Load data
    data = load(link)

    ###############################
    # Further properties Function #
    ###############################
    if data["type"] == "gyr_bin":
        if not link_dens:
            print("Gyration calculation needs a density sampling file. Check documentation and set a link_dens.")
            return

        # If pore system
        if "pore" in data:
            # Read data
            pore = data["pore"]
            res = float(pore["res"])
            diam = float(pore["diam"])
            box = pore["box"]
            type = pore["type"]

            # Set panda tables for pore properties
            data_system = [[["%.2f" % i for i in box]],[diam],[res],[type]]
            df_system = pd.DataFrame(data_system, index=list(["Box dimension (nm)","Pore diameter (nm)","reservoir (nm)", "type"]),columns=list(["Value"]))
            df_system = df_system.rename_axis('# Identifier', axis=1)
        # If box system
        elif "box" in data:
            box_group = data["box"]
            box = box_group["length"]
            data_box = [[["%.2f" % i for i in box]]]
            df_system = pd.DataFrame(data_box, index=list(["Box dimension (nm)"]), columns=list(["Value"]))
            df_system = df_system.rename_axis('# Identifier', axis=1)

        # Set pandas table for input
        data = [[data["inp"]["bin_num"]],[data["inp"]["entry"]],[data["inp"]["num_frame"]], [data["inp"]["mass"]]]
        df_inputs = pd.DataFrame(data,index = list(["Bin number","Entry","Frame number", "Mass"]), columns=list(["Value"]))
        df_inputs = df_inputs.rename_axis('# Identifier', axis=1)

        # Calculate gyration and set pandas table
        gyr_in = pa.gyration.bins_plot(link, link_dens, intent="ex")
        gyr_pd = pd.DataFrame(gyr_in, index=(["gyration"]))
        gyr_pd = gyr_pd.rename_axis('# Identifier', axis=1)

        # Convert pandas table to string
        df_inputs_string = df_inputs.to_string(header=True, index=True)
        df_system_string = df_system.to_string(header=True, index=True)
        gyr_pd_string = gyr_pd.to_string(header=True, index=True)

        # Write file
        with open(link_output, 'w') as file:
            # Write file header
            file.write("# This file was created " + str(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "\n")
            file.write("# Analysis created using  PoreAna\n")
            file.write("# Object file : " + os.path.dirname(os.path.abspath(__file__)) + link + "\n\n")

            # Write System table
            file.write("[System]\n")
            file.write(df_system_string)

            # Write Inputs
            file.write("[Input]\n")
            file.write(df_inputs_string)

            # Write gyration table
            file.write("\n\n[Gyration]\n")
            file.write(gyr_pd_string)
        # Close file
        file.close()


    ##########################
    # Bin Diffusion Function #
    ##########################
    # If diffusion bin sample file was loaded
    elif data["type"] == "diff_bin":
        # Check inputs (for bin diffusion is density sample file necessary)
        if not link_dens:
            print("Bin diffusion needs a density sampling file. Check documentation and set a link_dens.")
            return

        # Calculate bin diffusion
        bins = pa.diffusion.bins(link, is_norm=True)
        dens = pa.density.bins(link_dens)
        mean = pa.diffusion.mean(bins, dens)

        # If pore system
        if "pore" in data:
            # Read data
            pore = data["pore"]
            res = float(pore["res"])
            diam = float(pore["diam"])
            box = pore["box"]
            type = pore["type"]

            # Set panda tables for pore properties
            data_system = [[["%.2f" % i for i in box]],[diam],[res],[type]]
            df_system = pd.DataFrame(data_system, index=list(["Box dimension (nm)","Pore diameter (nm)","reservoir (nm)", "type"]),columns=list(["Value"]))
            df_system = df_system.rename_axis('# Identifier', axis=1)


        # Set pandas table for input
        data_inp = [[data["inp"]["bin_num"]],[data["inp"]["entry"]],[data["inp"]["num_frame"]], [data["inp"]["mass"]], [bins["is_norm"]]]
        df_inputs = pd.DataFrame(data_inp,index = list(["Bin number","Entry","Frame number", "Mass", "is_norm"]), columns=list(["Value"]))
        df_inputs = df_inputs.rename_axis('# Identifier', axis=1)

        # Set pandas table for bin diffusion profiles
        data = {}
        data["# Width"] = bins["width"][1:]
        data["D"] = bins["diff"]
        df_data = pd.DataFrame(data)

        # Pandas table to string
        df_system_string = df_system.to_string(header=True, index=True)
        df_inputs_string = df_inputs.to_string(header=True, index=True)
        df_data_string = df_data.to_string(header=True, index=False)

        # Write file
        with open(link_output, 'w') as file:
            # Write file header
            file.write("# This file was created " + str(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "\n")
            file.write("# Analysis created using  PoreAna\n")
            file.write("# Object file : " + os.path.dirname(os.path.abspath(__file__)) + link + "\n")
            file.write("# Units\n")
            file.write("# Diffusion D(10^-9 m^2s^-1)\n\n")

            # Write System table
            file.write("[System]\n")
            file.write(df_system_string)

            # Wirte input table
            file.write("\n\n[Inputs]\n")
            file.write(df_inputs_string)

            # Write diffusion coefficent
            file.write("\n\n[Diffusion]\n")
            file.write(str("%.2f " % mean ) + "* 10^-9 m^2s^-1")

            # Write diffusion profile
            file.write("\n\n[Bin Diffusion Profiles]\n")
            file.write(df_data_string)

        # Close file
        file.close()

    ####################
    # Density Function #
    ####################
    # If density sample file was loaded
    elif data["type"] == "dens_bin":
        # If pore system
        if "pore" in data:
            pore = data["pore"]
            res = float(pore["res"])
            diam = float(pore["diam"])
            box = pore["box"]
            type = pore["type"]

            # Set panda table for pore properties
            data_system = [[["%.2f" % i for i in box]],[diam],[res],[type]]
            df_system = pd.DataFrame(data_system, index=list(["Box dimension (nm)","Pore diameter (nm)","reservoir (nm)", "type"]),columns=list(["Value"]))
            df_system = df_system.rename_axis('# Identifier', axis=1)
        # If box system
        elif "box" in data:
            box_group = data["box"]
            box = box_group["length"]
            data_box = [[["%.2f" % i for i in box]]]
            df_system = pd.DataFrame(data_box, index=list(["Box dimension (nm)"]), columns=list(["Value"]))
            df_system = df_system.rename_axis('# Identifier', axis=1)

        # Set panda table for inputs
        data_inp = [[data["inp"]["bin_num"]],[data["inp"]["entry"]],[data["inp"]["num_frame"]],[True if data["inp"]['remove_pore_from_res'] else False], [str("%.2f" % data["inp"]["mass"])]]
        df_inputs = pd.DataFrame(data_inp,index = list(["Bin number","Entry","Frame number", "Remove pore from reservoir", "Mass"]), columns=list(["Value"]))
        df_inputs = df_inputs.rename_axis('# Identifier', axis=1)

        # Calculated adsorption and set pandas table
        if "pore" in data:
            ads = pa.adsorption.calculate(link)
            df_ads = pd.DataFrame(ads)
            df_ads = df_ads.rename_axis('# Identifier', axis=1)

        # Calculated density and set pandas table
        dens = pa.density.bins(link)
        if  "pore" in data:
            df_mean = pd.DataFrame([[dens["mean"]["in"], dens["dens"]["in"]],[dens["mean"]["ex"], dens["dens"]["ex"]]],index = list(["Density inside pore","Density outside pore"]), columns=list(["Density (#/nm^3)", "Density (kg/m^3)"]))
        else:
            df_mean = pd.DataFrame([[dens["mean"]["ex"], dens["dens"]["ex"]]],index = list(["Density box"]), columns=list(["Density (#/nm^3)", "Density (kg/m^3)"]))
        df_mean = df_mean.rename_axis('# Identifier', axis=1)

        # Set pandas table for density profile
        data_dens = {}
        data_dens["# Ex width"] = dens["sample"]["data"]["ex_width"]
        data_dens["Density (Ex)"] = dens["num_dens"]["ex"]
        if "pore" in data:
            data_dens["In width"] = dens["sample"]["data"]["in_width"][1:]
            data_dens["Density (In)"] = dens["num_dens"]["in"]
        df_data = pd.DataFrame(data_dens)

        # Convert pandas table to string
        if "pore" in data:
            df_ads_string = df_ads.to_string(header=True, index=True, na_rep="")
        df_data_string = df_data.to_string(header=True, index=False)
        df_mean_string = df_mean.to_string(header=True, index=True)
        df_system_string = df_system.to_string(header=True, index=True)
        df_inputs_string = df_inputs.to_string(header=True, index=True)

        # Write file
        with open(link_output, 'w') as file:
            # Write tables in file
            file.write("# This file was created " + str(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "\n")
            file.write("# Analysis created using  PoreAna\n")
            file.write("# Object file : " + os.path.dirname(os.path.abspath(__file__)) + link + "\n\n")

            # Write system table
            file.write("[System]\n")
            file.write(df_system_string)

            # Write input table
            file.write("\n\n[Inputs]\n")
            file.write(df_inputs_string)

            # Write adsorption table
            if "pore" in data:
                file.write("\n\n[Adsorption]\n")
                file.write(df_ads_string)

            # Write density
            file.write("\n\n[Density]\n")
            file.write(df_mean_string)

            # Write density profile
            file.write("\n\n[Density Profiles]\n")
            file.write(df_data_string)

        # Close file
        file.close()

    ###############
    # MC Function #
    ###############
    # If mc file was loaded
    elif data["type"] == "mc":
        # If pore system
        if "pore" in data:
            pore = data["pore"]
            t = data["model"]["len_frame"]
            res = float(pore["res"])
            diam = float(pore["diam"])
            box = pore["box"]
            type = pore["type"]
            data = [[["%.2f" % i for i in box]],[diam],[res],[type]]
            df_system = pd.DataFrame(data, index=list(["Box dimension (nm)","Pore diameter (nm)","reservoir (nm)", "type"]),columns=list(["Value"]))
            df_system = df_system.rename_axis('# Identifier', axis=1)

        # If box system
        elif "box" in data:
            t = data["model"]["len_frame"]
            box_group = data["box"]
            box = box_group["length"]
            data = [[["%.2f" % i for i in box]]]
            df_system = pd.DataFrame(data, index=list(["Box dimension (nm)"]), columns=list(["Value"]))
            df_system = df_system.rename_axis('# Identifier', axis=1)


        # Get profiles
        diff = pa.diffusion.mc_profile(link, is_plot = False, infty_profile=True)
        free_energy= pa.freeenergy.mc_profile(link, is_plot = False)

        # Set pandas table for profiles
        data = {}
        data["# Bins [nm]"] = diff[2]
        for i in diff[1]:
            data["D (t={})".format(t*i)] = diff[1][i]
        data["   D (t=∞)"]=diff[0]
        for i in free_energy[0].keys():
            data["Free energy [-]"]= free_energy[0][i][1:]
        df_data = pd.DataFrame(data)


        # Set pandas tables
        df_inputs = pa.tables.mc_inputs(link, print_con=False)
        df_inputs = df_inputs.rename_axis('# Identifier', axis=1)
        df_model = pa.tables.mc_model(link, print_con=False)
        df_model = df_model.rename_axis('# Identifier', axis=1)
        df_results = pa.tables.mc_results(link, print_con=False)
        df_results = df_results.rename_axis('# Identifier', axis=1)

        # Convert pandas table to strings
        df_model_string = df_model.to_string(header=True, index=True)
        df_inputs_string = df_inputs.to_string(header=True, index=True)
        df_results_string = df_results.to_string(header=True, index=True)
        df_system_string = df_system.to_string(header=True, index=True)
        df_data_string = df_data.to_string(header=True, index=False)

        # Write output file
        with open(link_output, 'w') as file:
            # Write tables in file
            file.write("# This file was created " + str(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "\n")
            file.write("# Analysis created using  PoreAna\n")
            file.write("# Object file : " + os.path.dirname(os.path.abspath(__file__)) + "/" + link + "\n")
            file.write("# Units\n")
            file.write("# Diffusion D(10^-9 m^2s^-1)\n")
            file.write("# Lag time  t(ps)\n\n")

            # Write system table
            file.write("[System]\n")
            file.write(df_system_string)

            # Write model input table
            file.write("\n\n\n[Model Inputs]\n")
            file.write(df_model_string)

            # Write mc input table
            file.write("\n\n\n[MC Inputs]\n")
            file.write(df_inputs_string)

            # Write MC results (Diffusion coefficients)
            file.write("\n\n\n[MC Results]\n")
            file.write(df_results_string)

            # Write diffusion and free energy profile in file
            file.write("\n\n[Profiles]\n")
            file.write(df_data_string)

            # Close file
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
