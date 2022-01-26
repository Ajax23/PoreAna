import numpy as np
import pandas as pd
import poreana.utils as utils


class Model:
    """This class sets the general parameters which are used to initialize
    a model.

    Parameters
    ----------
    link : string
        Data link to the hdf5, obj or yml data file from :func:`poreana.sample.Sample.init_diffusion_mc`

    """

    def __init__(self, link):
        # Load hdf5 data file
        sample = utils.load(link)

        inp = sample["inp"]
        # Read the inputs
        self._bin_num = int(inp["bin_num"])               # number of bins z-direction
        self._frame_num = int(inp["num_frame"])             # number of frames
        self._len_step = inp["len_step"]                 # step length
        self._dt = float(inp["len_frame"]) * 10**12             # frame length [ps]
        self._bins = inp["bins"]                      # bins [nm]
        self._bin_width = self._bins[1] - self._bins[0]  # bin width [nm]
        self._direction = int(inp["direction"])
        self._trans_mat= sample["data"]                # transition matrix
        self._pbc = inp["is_pbc"]                    # pbc or nopbc
        self._type = sample["type"]

        self._sys_props = {}
        if "pore" in sample:
            self._sys_props["type"] = sample["pore"]["type"]
            self._sys_props["res"] = float(sample["pore"]["res"])
            self._sys_props["focal"] = sample["pore"]["focal"]
            self._sys_props["box"] = sample["pore"]["box"]
            self._sys_props["diam"] = float(sample["pore"]["diam"])
            self._system = "pore"
        if "box" in sample:
            self._system = "box"
            self._sys_props["length"] = sample["box"]["length"]


        # Initialize units of diffusion and free energy unit
        self._df_unit = 1.                                 # in kBT
        self._diff_unit = np.log(self._bin_width**2 / 1.)  # in m^2/s

    def _init_profiles(self):
        """This function initializes the normal diffusion, radial diffusion and
        free energy profile over the bins.
        """
        # Initialize the diffusion and free energy profile
        self._df_bin = np.float64(np.zeros(self._bin_num))    # in kBT
        self._diff_bin = np.float64(np.zeros(self._bin_num))  # in dz**2/dt

        # Initialize the diffusion profile
        self._diff_bin += (np.log(self._d0) - self._diff_unit)

    def _calc_profile(self, coeff, basis):
        """This function calculates the diffusion and free energy profile over
        the bins. It is used to initialize the system at the beginning of the
        calculation/MC run. Additionally, it is needed to update the profiles
        in the Monte Carlo part after the adjustment of a profile coefficient.

        The profile is determining with the basis and the coefficients for the
        diffusion with

        .. math::

            \\ln D_i = a_0 + \\sum_{k} a_{k} \\cdot \\mathrm{basis}_{\\text{F}},

        and for the free energy with

        .. math::

            F_i = \\sum_{k} a_{k} \\cdot \\mathrm{basis}_{\\text{D}}.

        The diffusion is calculated between to bins on the bin border and the free energy
        in the center of a bin.
        The basis is received by the create basis function for the selected model:

        **CosineModel**
         * :func:`CosineModel._create_basis_center`
         * :func:`CosineModel._create_basis_border`

        **StepModel**
         * :func:`StepModel._create_basis_center`
         * :func:`StepModel._create_basis_border`

        Parameters
        ----------
        coeff : list
            list of coefficients
        basis : list
            list of the basis part of model
        """

        # Calculate a matrix with the bin_num x ncos
        # Columns contains the n-summand in every bin
        # Line contains the summand of the series
        a = coeff * basis

        return np.sum(a, axis=1)


class CosineModel(Model):
    """This class sets the Cosine Model to calculate the free energy profile
    and the diffusion profile. The profiles have the typical cosine oscillation.
    These profiles over the bins are expressed by the following Fourier
    series. The diffusion profile is calculated between bin i and i+1  over
    the bin border with

    .. math::

        \\ln \\left(D_{i+ \\frac{1}{2}}\\right)=a_{0}+\\sum_{k=1}^{n_b} a_{k} \\cdot \\cos \\left(\\frac{2\\pi ki}{n}\\right).

    Similarly, the free energy Fourier series can be written:

    .. math::

        F_{i} = a_{0}+\\sum_{k=1}^{n_b} a_{k} \\cdot \\cos \\left(\\frac{2\\pi ki}{n} \\right),

    with the number of bins :math:`n`, the index of a bin :math:`i`,
    the coefficients :math:`a_{k}` of the Fourier series and :math:`k` as the
    number of coefficents. The free energy is calculated in the bin center.

    For the free energy the coefficient it is assumed that :math:`a_{0} = 0`.

    Parameters
    ----------
    data_link : string
        Data link to the pickle data from :func:`poreana.sample.Sample.init_diffusion_mc`
    n_diff : integer, optional
        Number of the Fourier coefficients for the diffusion profile
    n_df : integer, optional
        Number of the Fourier coefficients for the free energy profile
    n_diff_radial : integer, optional
        Number of the Fourier coefficients for the radial diffusion profile
    d0 : double, optional
        Initial guess of diffusion coefficient :math:`\\left( 10^{-9} \\frac{m^2}{s}\\right)`
    is_print : bool, optional
        True to print output
    """

    def __init__(self, data_link, n_diff=6, n_df=10, n_diff_radial=6, d0=1, is_print=False):

        # Inherit the variables from Model class
        super(CosineModel, self).__init__(data_link)

        # Set the model type
        self._model = "CosineModel"

        self._n_df = n_df
        self._n_diff = n_diff
        self._n_diff_radial = n_diff_radial
        self._print_output = is_print
        self._d0 = d0 * (10**9)/(10**12)                # guess init profile [A^2/ps]

        self._init_model()     # Initial model
        self._init_profiles()  # Initial Profiles
        self._cosine_model()   # Set basis of Fourier series

    def _init_model(self):
        """This function initializes the coefficient list for the Fourier series
        and the profile list for the free energy and diffusion profile.
        It is used to reinitialize these lists at the beginning of a MC run.
        """

        # Initialize the diffusion and free energy coefficient
        self._df_coeff = np.float64(np.zeros(self._n_df))      # in dz**2/dt
        self._diff_coeff = np.float64(np.zeros(self._n_diff))  # in dz**2/dt

        # Set start diffusion profile
        self._diff_coeff = np.zeros((self._n_diff), float)

        # Initialize diffusion profile with the guess value [A^2/ps]
        self._diff_coeff[0] += (np.log(self._d0) - self._diff_unit)

    def _cosine_model(self):
        """This function sets a Fourier Cosine Series Model for the MC Diffusion
        Calculation and determines the initialize profiles.
        """

        # Create basis (for the free energy)
        self._create_basis_center()

        # Create basis (for the free energy)
        self._create_basis_border()

        # Update diffusion profile
        self._diff_bin = self._calc_profile(self._diff_coeff, self._diff_basis)

        # Update free energy profile
        self._df_bin = self._calc_profile(self._df_coeff, self._df_basis)

        # Print for console
        if self._print_output == True:
            print("--------------------------------------------------------------------------------")
            print("-------------------------Initalize Cosine Model---------------------------------")
            print("--------------------------------------------------------------------------------\n")
            print("Model Inputs")

            # Set data list for panda table
            len_step_string = ', '.join(str(step) for step in self._len_step)
            data = [str("%.f" % self._bin_num),  len_step_string, str("%.2e" % (self._dt * 10**(-12))), str("%.f" % self._n_diff),
                    str("%.f" % self._n_df), self._model, self._pbc, str("%.2e" % (self._d0 * (10**(-18))/(10**(-12)))), self._system]

            # Set pandas table
            df_model = pd.DataFrame(data, index=list(
                ['Bin number', 'step length', 'frame length', 'nD', 'nF', 'model', 'pbc', 'guess diffusion (m2/s-1)', 'system']), columns=list(['Input']))

            # Print panda table with model inputs
            print(df_model)

    def _create_basis_center(self):
        """This function creates the basis part of the Fourier series for the
        free energy and the radial diffusion profile.
        For a bin the basis is calculated with

        .. math::

            \\mathrm{basis} = \\cos \\left(\\frac{2\\pi k(i+0.5)}{n}\\right)

        hereby :math:`k` is the number of coefficients, :math:`i` is the bin
        index and :math:`n` is the number of the bins.
        """

        # Allocate a vector with bin_num entries
        x = np.arange(self._bin_num)

        # Calculate basis for Fourier cosine series
        basis_df = [np.cos(2 * k * np.pi * (x + 0.5) / self._bin_num) / (k + 1) for k in range(self._n_df)]                      # basis for the free energy profile
        basis_diff_radial = [np.cos(2 * k * np.pi * (x + 0.5) / self._bin_num) / (k + 1) for k in range(self._n_diff_radial)]    # basis for the radial energy profile

        # Transpose basis (is now a bin_num x ncos Matrix)
        self._df_basis = np.array(basis_df).transpose()
        self._diff_radial_basis = np.array(basis_diff_radial).transpose()

    def _create_basis_border(self):
        """This function creates the basis part in every bin of the Fourier
        series for the diffusion :math:`\\ln \\ (D)`.
        At the bin border the basis is calculated with

        .. math::

            \\mathrm{basis} = \\cos \\left(\\frac{2\\pi ki}{n}\\right)

        hereby :math:`k` is the number of coefficients, :math:`i` is the bin
        index and :math:`n` is the number of the bins.
        """

        # Allocate a vector with bin_num entries
        x = np.arange(self._bin_num)

        # Calculate basis for Fourier cosine series
        basis = [np.cos(2 * k * np.pi * (x + 1.) / self._bin_num) / (k + 1) for k in range(self._n_diff)]

        # Transpose basis (is now a bin_num x ncos Matrix)
        self._diff_basis = np.array(basis).transpose()


class StepModel(Model):
    """This class sets the Step Model to calculate the free energy profile and
    the diffusion profile. This model based on a spline calculation.
    In contrast to the Cosine Model the determined profile has not the typical
    oscillation and receives a profile which is better interpretable.
    The profile will be determined with the following equations

    .. math::

        \\ln \\left(D_{i+\\frac{1}{2}}\\right) = a_{k} \\cdot \\mathrm{basis}_{\\mathrm{diff}}

    .. math::

        F_i = a_{k} \\cdot \\mathrm{basis}_{\\mathrm{df}}

    with :math:`a_k` as the coefficients and the :math:`\\mathrm{basis}`. The
    basis of the modell is calculated with

    **StepModel**

    * :func:`StepModel._create_basis_center`
    * :func:`StepModel._create_basis_border`

    Parameters
    ----------
    data_link : string
        Data link to the pickle data from :func:`poreana.sample.Sample.init_diffusion_mc`
    n_diff : integer, optional
        Number of the Fourier coefficients for the diffusion profile
    n_df : integer, optional
        Number of the Fourier coefficients for the free energy profile
    n_diff_radial : integer, optional
        Number of the Fourier coefficients for the radial diffusion profile
    d0 : double, optional
        Initial guess of diffusion coefficient :math:`\\left( 10^{-9} \\frac{m^2}{s}\\right)`
    is_print : bool, optional
        True to print output
    """

    def __init__(self, data_link, n_diff=6, n_df=10, n_diff_radial=6, d0=1, is_print=False):

        # Inherit the variables from Model class
        super(StepModel, self).__init__(data_link)

        self._model = "Step Model"

        self._n_diff = n_diff
        self._n_df = n_df
        self._n_diff_radial = n_diff_radial
        self._print_output = is_print
        self._d0 = d0 * (10**9)/(10**12)

        self._init_model()     # Initial model
        self._init_profiles()  # Initial Profiles
        self._step_model()     # Set basis of Step Model

    def _init_model(self):
        """This function initializes the coefficient list for the Step Model and
        the profile list for the free energy and diffusion profile. It is used
        to reinitializes these lists for every lag time calculation.
        """

        # Initialize the diffusion and free energy coefficient
        self._df_coeff = np.float64(np.zeros(self._n_df))
        self._diff_coeff = np.float64(np.zeros(self._n_diff))

        # Calculate dz
        dx_df = self._bin_num / 2. / self._n_df
        dx_diff = self._bin_num / 2. / self._n_diff

        self._df_x0 = np.arange(0, self._n_df * dx_df, dx_df)
        self._diff_x0 = np.arange(0, self._n_diff * dx_diff, dx_diff)

        # Initialize diffusion profile with the guess value [A^2/ps]
        self._diff_coeff[0] += (np.log(self._d0) - self._diff_unit)

    def _step_model(self):
        """This function set a Step Model for the MC Diffusion Calculation and
        determine the initialize profiles.
        """

        # create basis (for the free energy)
        self._create_basis_center()

        # create basis (for the free energy)
        self._create_basis_border()

        # Update diffusion profile
        self._diff_bin = self._calc_profile(self._diff_coeff, self._diff_basis)

        # Update free energy profile
        self._df_bin = self._calc_profile(self._df_coeff, self._df_basis)

        # Print for console
        if self._print_output == True:
            print("\n")
            print("--------------------------------------------------------------------------------")
            print("-------------------------Initalize Step Model-----------------------------------")
            print("--------------------------------------------------------------------------------\n")
            print("Model Inputs")

            # Set data list for panda table
            len_step_string = ', '.join(str(step) for step in self._len_step)
            data = [str("%.f" % self._bin_num),  len_step_string, str("%.2e" % (self._dt * 10**(-12))), str("%.f" % self._n_diff),
                    str("%.f" % self._n_df), self._model, self._pbc, str("%.2e" % (self._d0 * (10**(-18))/(10**(-12)))), self._system]

            # Set pandas table
            df_model = pd.DataFrame(data, index=list(
                ['Bin number', 'step length', 'frame length', 'nD', 'nF', 'model', 'pbc', 'guess diffusion (m2/s-1)','system']), columns=list(['Input']))

            # Print panda table with model inputs
            print(df_model)
            print("\n")

    def _create_basis_center(self):
        """This function creates the basis part in every bin of the Step model
        for the free energy and the radial diffusion profile. The following
        explanation is for the free energy profile. For the radial diffusion
        profile the number of free energy coefficients :math:`n_{\\mathrm{df}}` has to
        exchange with :math:`n_{\\mathrm{diff\_radial}}`. The Dimension of the basis matrix
        is :math:`n_{\\mathrm{bin}} \\times n_{\\mathrm{df}}`. For a bin the
        basis is calculated with

        .. math::

            \\mathrm{basis} = \\begin{cases}
                            1 & (\\mathrm{bin}+0.5)\\geq \\Delta x\ \\& \ (\\mathrm{bin}+0.5)\\leq n_{\\mathrm{bin}}-\\Delta x \\\\
                            0 & \\mathrm{else}                                 \\
                    \\end{cases}

        hereby is :math:`\\mathrm{bin} = [0,...,n_{bin}]` with
        :math:`n_{\\mathrm{bin}}` as the number of the bins. The variable
        :math:`\\Delta x` is define by

        .. math::

            \\Delta x = \\left [ 0,i \\cdot \\frac{n_{\\mathrm{bin}}}{n_{\\mathrm{df}} \\cdot 2},\\frac{n_{\\mathrm{bin}}}{2} \\right ]

        with :math:`i = [1,...,n_{\\mathrm{df}}-1]`.
        """
        # Calculated the basis in the center of a bin
        x = np.arange(self._bin_num)+0.5
        basis = [np.where((x >= i) & (x <= self._bin_num-i), 1., 0.) for i in self._df_x0]

        # Transpose basis (is now a bin_num x ncos Matrix)
        self._df_basis = np.array(basis).transpose()

    def _create_basis_border(self):
        """This function creates the basis part in every bin of the Step model
        for the diffusion profile. The dimension of the basis matrix is
        :math:`n_{bin} \\times n_{diff}`. At the bin border the basis is
        calculated with

        .. math::

            \\mathrm{basis} = \\begin{cases}
                            1 & (\\mathrm{bin}+1)\\geq \\Delta x\ \\& \ (\\mathrm{bin}+1)\\leq n_{\\mathrm{bin}}-\\Delta x \\\\
                            0 &  \\mathrm{else}                                 \\
                    \\end{cases}

        hereby is :math:`bin = [0,...,n_{\\mathrm{bin}}]` with
        :math:`n_{\\mathrm{bin}}` as the number of the bins. The variable
        :math:`\\Delta x` is define by

        .. math::

            \\Delta x = \\left [ 0,i \\cdot \\frac{n_{\\mathrm{bin}}}{n_{\\mathrm{diff}} \\cdot 2},\\frac{n_{\\mathrm{bin}}}{2} \\right ]

        with :math:`i = [1,...,n_{\\mathrm{diff}}-1]`.
        """
        # Calculated the basis in the border of a bin
        x = np.arange(self._bin_num)+1.
        basis = [np.where((x >= i) & (x <= self._bin_num-i), 1., 0.) for i in self._diff_x0]

        # Transpose basis (is now a bin_num x ncos Matrix)
        self._diff_basis = np.array(basis).transpose()
