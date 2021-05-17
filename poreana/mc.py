import numpy as np
import scipy
import scipy.linalg
import numpy.linalg
import copy

import poreana.utils as utils
import pandas as pd
from numpy import array
from statistics import mean
from .model import Model
from collections import deque

class MC:
    """
    This class contains the Monte Carlo part of the diffusion calculation. The
    class initializes the MC (Monte Carlo) parameters and contains all functions
    to execute the MC cycle part of the diffusion calculation.
    The MC Algorithm is divided in two parts. Frist a equilibrium MC run starts
    to adjust the profile. The number of the equilibrium runs can be set with
    :math:`nmc_eq`. After the equilibrium phase the sampling of the diffusion
    profile can begin. In this part the MC alogrithm calculates in every step
    an diffusion profile. The average over all profiles which are yields in MC
    production will be determined. The received profile gives the final results.
    Here you have to set also step the move width of a MC step.

    *More information about a MC step can be found in*
     * :func:`mcmove_diffusion`
     * :func:`mcmove_diffusion_radial`
     * :func:`mcmove_df`

    The MC calculation can be started with :func:`do_mc_cycles`.

    Parameters
    ----------
    model : Model
        Model object which sets with the model class
    nmc_eq : integer, optional
        number of equilibrium MC steps
    nmc : integer, optional
        number of prdocution MC steps
    nmc_eq_radial : integer, optional
        number of equilibrium MC steps
    nmc_radial : integer, optional
        number of prdocution MC steps
    delta_df : float, optional
        potential MC move width
    delta_diff : float, optional
        ln(diffusion) MC move width
    delta_diff_radial : float, optional
        ln(radial diffusion) MC move width
    num_mc_update : integer, optional
        number of moves between MC step width adjustments ( 0 if no adjustment)
    temp : float, optional
        temperature in Monte Carlo acceptance criterium
    lmax : integer, optional
        number of Bessel functions
    print_output : bool, optional
        if it is true the output will be printed
    print_freq : integer, optional
        every print_freq MC step is printed
    """

    def __init__(self, model, nmc_eq=50000, nmc = 100000,nmc_eq_radial=50000, nmc_radial=100000, delta_df=0.05, delta_diff=0.05, delta_diff_radial=0.05, num_mc_update=10, temp=1, lmax=50,print_output=True,print_freq=100):

        ########### Initialize MC ###########


        # Set MC step width
        self._delta_df = delta_df                            # MC Move width free energy
        self._delta_diff = delta_diff                        # MC Move width Diffusion
        self._delta_diff_radial = delta_diff_radial          # MC Move width radial Diffusion

        # Save beginning step width (to initialize it every mc run)
        self._delta_df_start = delta_df                      # MC Move width free energy
        self._delta_diff_start = delta_diff                  # MC Move width Diffusion
        self._delta_diff_radial_start = delta_diff_radial

        # Set MC options
        self._nmc = nmc                                      # Number of MC steps
        self._nmc_eq = nmc_eq                                # Number of MC steps
        self._nmc_eq_radial = nmc_eq_radial                  # Number of MC steps
        self._nmc_radial = nmc_radial                        # Number of MC steps
        self._num_mc_update = num_mc_update                  # MC steps before update delta
        self._temp = temp                                    # Temperature for acceptance criterion
        self._lmax = lmax                                    # Number of bessel functions

        # Set output/print options
        self._print_output = print_output                    # Bool (If False nothing will be printed in the konsole)
        self._print_freq = print_freq                        # print frequency for MC steps (default every 100 steps)



    def do_mc_cycles(self, model, link_out, do_radial=False):
        """
        This function do the MC Cycle to calculate the diffusion and free energy
        profil over the bins and save the results in a output object file.
        This happen with the adjustment of the coefficient from the model which
        is set with the appropriate model class. The code determine the
        results for all lagtimes for which a transition matrix was caclulated.
        The results can be displayed with the post process functions.

        The MC accepted criterium for a MC step is define as

        .. math::

            r < \\exp \\left ( \\frac{L_\\text{new} - L_\\text{old}}{T}  \\right)

        with :math:`T` as the temperature, :math:`r` as a random number between
        :math:`0` and :math:`1` and :math:`L_\\text{new}` and
        :math:`L_\\text{old}` as the likelihood for the current and the last
        step. The likelihood is calculated with :func:`log_likelihood_z`
        function.

        Information about a single MC step can be find in
        :func:`mcmove_diffusion` or :func:`mcmove_df`

        Parameters
        ----------
        model : Model
            Model object which set before with the model class
        link_out : string
            Link to output object file
        do_radial : bool, optional
            if it's True the code calculate the radial diffusion too
        """
        # Print that MC Calculation starts
        if self._print_output==False:
            print("MC Calculation Start")
            print("...")

        # Print to see that the MC options
        if self._print_output==True:
            print("\n-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
            print("----------------------------------------------------------------------Start MC-----------------------------------------------------------------------------------")
            print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------\n")
            print("MC Inputs")

            # Table for MC Inputs (set data structure)
            data = [self._nmc_eq,self._nmc,self._nmc_eq_radial,self._nmc_radial,self._num_mc_update,self._print_freq]
            df_input = pd.DataFrame(data,index=list(['MC step (Equilibrium)','MC step (Production)','MC step radial (Equilibrium)','MC step radial (Production)','movewidth update','print frequency']),columns=list(['Input']))

            # Table for MC Inputs
            print(df_input)


        # Initialize result lists (profiles, coefficients, fluctuation and accepted MC steps)
        # Diffusion
        list_diff_profile = {}
        list_diff_coeff = {}
        list_diff_fluc = {}
        nacc_diff_mean = {}

        # Radial diffusion
        list_diff_radial_profile = {}
        list_diff_radial_coeff = {}
        list_diff_radial_fluc = {}
        nacc_diff_radial_mean = {}

        # Free energy
        list_df_profile = {}
        list_df_coeff = {}
        list_df_fluc = {}
        nacc_df_mean = {}


        # Loop over the different step_length (lag times) (-> for every lag time a MC Claculation have to run)
        for self._len_step in model._len_step:

            # Print that a new calculation with a new lag time starts
            lagtime_string = "Lagtime: " + str(self._len_step * model._dt) + " ps"
            if self._print_output==True:
                print('\n=============================================================='+lagtime_string+'====================================================================================')


            # Initialize lists for the profiles
            results_diff_profile = {}
            results_df_profile = {}
            results_diff_coeff = {}
            results_df_coeff = {}
            nacc_diff = {}
            nacc_df = {}

            # Initialize Model for every MC Run
            model.init_model()
            model.init_profiles()


            # Start calucalation for the normal diffusion and free energy Profile
            if self._print_output==True:
                print("\n---------------------------------------------------------Calculate normal diffusion------------------------------------------------------------------------------")

            # Calculated the initalize likelihood
            self._log_like = self.log_likelihood_z(model)

            # Print first likelihood
            if self._print_output==True:
                print("likelihood init", self._log_like, "\n")
                print("Start equilibration")

            # Initialize a new statistic
            self.init_stats(model)

            # Set step width for every MC run on the input values
            self._delta_df = copy.deepcopy(self._delta_df_start)
            self._delta_diff = copy.deepcopy(self._delta_diff_start)
            self._delta_diff_radial = copy.deepcopy(self._delta_diff_radial_start)

            # Initialize the fluction every MC run
            diff_profile_flk = np.float64(np.zeros(model._bin_num))
            diff_radial_profile_flk = np.float64(np.zeros(model._bin_num))
            df_profile_flk = np.float64(np.zeros(model._bin_num))
            mean_diff_radial_profile_flk = np.float64(np.zeros(model._bin_num))
            self._fluctuation_diff = 0
            self._fluctuation_diff_radial = 0
            self._fluctuation_df = 0

            ##############################################
            ########## Start MC Alogrithm ################
            ##############################################
            # Loop over the MC steps
            for imc in range (self._nmc + self._nmc_eq ):

                # Random number to decide between diffusion or free energy
                self._choice = np.random.rand()

                # Decide with choice which MC move will be execute
                if self._choice < 0.5:
                    # Do a MC move in the free energy profile
                    self.mcmove_df(model)

                else:
                    # Do a MC move in the diffusion profile
                    self.mcmove_diffusion(model)

                # Update the MC movewidth
                self.update_movewidth_mc(imc)

                # Calculate the fluktuation and start the produktion
                if imc >= self._nmc_eq:

                    # Add all profiles
                    diff_profile_flk += copy.deepcopy(model._diff_bin)
                    df_profile_flk += copy.deepcopy(model._df_bin)

                    #Calculate the mean profile over all runs
                    mean_diff_profile_flk = [diff_profile_flk[i]/((imc-self._nmc_eq)+1) for i in range(model._bin_num)]
                    mean_df_profile_flk = [df_profile_flk[i]/((imc-self._nmc_eq)+1) for i in range(model._bin_num)]

                    # Calculate the difference between the current profile and the mean of all profiles
                    delta_diff = ((model._diff_bin) - mean_diff_profile_flk)**2
                    delta_df = ((model._df_bin) - mean_df_profile_flk)**2

                    # Determine the fluctuation
                    self._fluctuation_diff = np.sqrt((self._fluctuation_diff + np.mean(delta_diff))/(imc+1))
                    self._fluctuation_df = np.sqrt((self._fluctuation_df + np.mean(delta_df))/(imc+1))

                    # Start to print the output after the Equilibrium phase and if _print_output is true
                    if imc == self._nmc_eq and self._print_output==True:
                        print("\nStart production\n")
                        print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------\n")
                        print("imc","\t", "likelihood","\t", "accepted_df (%)", "\t","accepted_diff (%)", "\t","df_step_width", "\t","diff_step_width","\t","fluktuation_df","\t","fluktuation_diff")
                    if  (imc%self._print_freq == 0) and imc > self._nmc_eq and self._print_output==True:
                        print(imc,"\t","%.3f" %self._log_like,"\t", "%.2f" % (float(self._nacc_df)*100/(imc+1)),"\t\t\t", "%.2f" % (float(self._nacc_diff)*100/(imc+1)), "\t\t\t", "%.5f" % self._delta_df,"\t", "%.5f" %  self._delta_diff,"\t","\t", "%.4e" %  self._fluctuation_df, "\t","\t","%.4e" %  self._fluctuation_diff)
                        print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------\n")

            # Save the profiles over the bins for the current lag time in a list
            list_diff_profile[self._len_step] = mean_diff_profile_flk #copy.deepcopy(model._diff_bin) #mean_diff_profile_flk
            list_df_profile[self._len_step] = mean_df_profile_flk # copy.deepcopy(model._df_bin)

            # Save the coefficents of the model for the current lag time in a list
            list_diff_coeff[self._len_step] = model._diff_coeff
            list_df_coeff[self._len_step] = model._df_coeff

            # Save the fluctuation of the current lag time run in a list
            list_diff_fluc[self._len_step] = self._fluctuation_diff
            list_df_fluc[self._len_step] = self._fluctuation_df

            # Mean over all lag times calculations
            nacc_df_mean[self._len_step] = copy.deepcopy(self._nacc_df)
            nacc_diff_mean[self._len_step] = copy.deepcopy(self._nacc_diff)




            #####################################################
            ########## Start Radial MC Alogrithm ################
            #####################################################

            # Radial diffusion
            if do_radial == True:
                # Start MC calucalation for the radial diffusion
                print("\n-------------------------------------------------------------Calculate radial diffusion--------------------------------------------------------------------------")

                # Calculated the initalize likelihood and bessel function
                self.setup_bessel_box(model)
                self._log_like_radial = self.log_likelihood_radial(model,model._diff_radial_bin)

                # Print first likelihood
                print("likelihood init", self._log_like_radial, "\n")
                print("Start equilibration")

                # Start MC Alogrithm for the radial diffusion
                for imc in range (self._nmc_radial+self._nmc_eq_radial):

                    # Do a MC move in the radial diffusion profile
                    self.mcmove_diffusion_radial(model)

                    # Update the MC movewidth
                    self.update_movewidth_mc(imc,True)

                    # Calculate the fluktuation and start the produktion
                    if imc >= self._nmc_eq_radial:

                        # Add all profiles
                        diff_radial_profile_flk += copy.deepcopy(model._diff_radial_bin)

                        #Calculate the mean profile over all runs
                        mean_diff_radial_profile_flk = [diff_radial_profile_flk[i]/((imc-self._nmc_eq_radial)+1) for i in range(model._bin_num)]

                        # Calculate the difference between the current profile and the mean of all profiles
                        delta_diff_radial = ((model._diff_radial_bin) - mean_diff_radial_profile_flk)**2

                        # Determine the fluctuation
                        self._fluctuation_diff_radial = np.sqrt((self._fluctuation_diff_radial + np.mean(delta_diff_radial))/(imc+1))

                        # Print output in production phase
                        if imc == self._nmc_eq_radial and self._print_output==True:
                            print("\nStart production\n")
                            print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------\n")
                            print("imc","\t", "likelihood","\t", "\t","accepted_diff_rad (%)", "\t","diff_rad_step_width","\t","fluktuation_diff_rad")
                        if  (imc%self._print_freq == 0) and imc > self._nmc_eq_radial and self._print_output==True:
                            print(imc,"\t","%.6f" %self._log_like_radial,"\t", "%.2f" % (float(self._nacc_diff_radial)*100/(imc+1)),"\t\t\t", "%.5f" %  self._delta_diff_radial,"\t","\t", "%.4e" %  self._fluctuation_diff_radial)
                            print(self._nacc_diff_radial)
                            print(np.mean(np.exp(model._diff_radial_bin + model._diff_radial_unit)) * 10**-6 )
                print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------\n")

                #Save results for the current lag time
                list_diff_radial_profile[self._len_step] = mean_diff_radial_profile_flk #copy.deepcopy(model._diff_radial_bin)
                list_diff_radial_coeff[self._len_step] = copy.deepcopy(model._diff_radial_coeff)
                list_diff_radial_fluc[self._len_step] = self._fluctuation_diff_radial

                # Mean over all lag times calculations
                nacc_diff_radial_mean[self._len_step] = copy.deepcopy(self._nacc_diff_radial)

            # If no radial diffusion is calculated set the initalize condition on the result lists
            if do_radial == False:
                #Save results for the current lag time
                list_diff_radial_profile[self._len_step] = np.float64(np.zeros(model._bin_num))
                list_diff_radial_coeff[self._len_step] = np.float64(np.zeros(model._n_df))
                list_diff_radial_fluc[self._len_step] = 0

                # Mean over all lag times calculations
                nacc_diff_radial_mean[self._len_step] = 0


        # Print MC statistics
        if self._print_output==True:
            print("\n-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
            print("----------------------------------------------------------------------MC Statistics------------------------------------------------------------------------------")
            print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------\n")

            # Set data structure fpr pandas table
            data = [[str("%.4e" % list_df_fluc[i]) for i in model._len_step],[str("%.4e" % list_diff_fluc[i]) for i in model._len_step],[str("%.4e" % list_diff_radial_fluc[i]) for i in model._len_step],[str("%.0f" % nacc_df_mean[i]) for i in model._len_step],[str("%.0f" % nacc_diff_mean[i]) for i in model._len_step],[str("%.0f" % nacc_diff_radial_mean[i]) for i in model._len_step],[str("%.2f" % (nacc_df_mean[i]*100/(self._nmc_eq+self._nmc))) for i in model._len_step],[str("%.2f" % (nacc_diff_mean[i]*100/(self._nmc_eq+self._nmc))) for i in model._len_step],[str("%.2f" % float(nacc_diff_radial_mean[i]*100/(self._nmc_eq_radial+self._nmc_radial))) for i in model._len_step]]

            # Set options for pandas table
            df = pd.DataFrame(data,index=list(['fluctuation df','fluctuation diff','fluctuation rad. diff','acc df steps','acc diff steps','acc rad. diff steps','acc df steps (%)','acc diff steps (%)','acc rad. diff steps (%)']),columns=list(model._len_step))
            df = pd.DataFrame(df.rename_axis('Step Length', axis=1))

            # Print pandas table for the MC statistics
            print(df)
            print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------\n")

            #Print coefficients
            print("\n-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
            print("----------------------------------------------------------------------Coefficients-------------------------------------------------------------------------------")
            print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------\n")

            # Set data dictionary for the diffusion profile coefficients
            print("Diffusion Model Coefficients")
            data = {}
            for i in model._len_step:
                data[i] = [str("%.4e" % list_diff_coeff[i][j]) for j in range(model._n_diff)]
            diff_coeff = pd.DataFrame(data,index=list(np.arange(1,model._n_diff+1)),columns=list(model._len_step))

            # Set options for pandas table
            diff_coeff = pd.DataFrame(diff_coeff.rename_axis('Step Length', axis=1))

            # Print pandas table with diffusion profile coefficients
            print(diff_coeff)

            # Set data dictionary for the free energy profile coefficients
            print("\nFree Energy Model Coefficients")
            data = {}
            for i in model._len_step:
                data[i] = [str("%.4e" % list_df_coeff[i][j]) for j in range(model._n_df)]
            df_coeff = pd.DataFrame(data,index=list(np.arange(1,model._n_df+1)),columns=list(model._len_step))

            # Set options for pandas table
            df_coeff = pd.DataFrame(df_coeff.rename_axis('Step Length', axis=1))

            # Print pandas table with free energy profile coefficients
            print(df_coeff)
            print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------\n")

        # Set inp data MC algorithm
        inp = {"MC steps eq": self._nmc_eq,"MC steps radial eq": self._nmc_eq_radial,"MC steps": self._nmc,"MC steps radial": self._nmc_radial, "step width update": self._num_mc_update,  "temperature": self._temp, "print freq":self._print_freq}

        # Set inp data for model
        model = {"bin number": model._bin_num, "bins": model._bins[:-1] , "diffusion unit": model._diff_unit, "len_frame": model._dt,"len_step": model._len_step, "model": model._model, "nD": model._n_diff,"nF":model._n_df,"nDrad": model._n_diff_radial, "guess": model._d0, "pbc": model._pbc, "num_frame": model._frame_num,"data": model._trans_mat}

        # Set output data
        output = {"inp": inp, "model":  model, "diff_profile": list_diff_profile, "df_profile": list_df_profile,"diff_coeff": list_diff_coeff,  "df_coeff": list_df_coeff, "nacc_df": nacc_df_mean, "nacc_diff": nacc_diff_mean, "fluc_df": list_df_fluc,"fluc_diff": list_diff_fluc, "list_diff_coeff": list_diff_profile, "list_df_coeff":  list_df_profile}

        # Print MC Calculation is done
        print("MC Calculation Done.")


        # Save inp and output data
        utils.save(output, link_out)

    def init_stats(self,model):
        """ This function sets the MC statistic counters to zero after
        every MC run.

        Parameters
        ----------
        model : Model
            Model object which set before with the model class

        """

        # Initialize MC statistic parameters

        # Number of accepted moves
        self._nacc_df = 0                                                       # number accepted free energy moves
        self._nacc_diff = 0                                                     # number accepted diffusion moves
        self._nacc_diff_radial = 0                                              # number accepted diffusion moves

        # Number of accepted moves between move width adjusts
        self._nacc_df_update = 0                                                # number accepted free energy moves between adjusts
        self._nacc_diff_update = 0                                              # number accepted diffsuion moves between adjusts
        self._nacc_diff_radial_update = 0                                       # number accepted diffsuion moves between adjusts

        # Number of accepted coefficient update
        self._nacc_diff_coeff = np.zeros(model._n_diff,int)                     # number accepted for different coefficient update (diffusion)
        self._nacc_df_coeff = np.zeros(model._n_df,int)                         # number accepted for different coefficient update (free energy)
        self._nacc_diff_radial_coeff = np.zeros(model._n_diff_radial,int)       # number accepted for different coefficient update (free energy)


    def mcmove_diffusion(self,model):
        """
        This function does the MC move for the diffusion profile and adjust
        one coefficent of the model. A MC move in the parameter space is defined
        by

        .. math::

            a_{k,\\mathrm{new}}= a_{k} + \\Delta_\\text{MC} \\cdot (R - 0.5)

        with :math:`\\Delta_\\text{MC}` as the MC step movewidth and :math:`R`
        as a random number between :math:`0` and :math:`1`. The choice of the
        coefficient is also made by determining a random number.

        Parameters
        ----------
        model : Model
            Model object which set with the model class
        """

        # Caclulate a random number to choose a random coefficient
        # Attion the first coefficent of the diffusion profile is fixed 0 all the time
        idx = np.random.randint(0,model._n_diff)

        # Calculate one new coefficient
        diff_coeff_temp = copy.deepcopy(model._diff_coeff)
        diff_coeff_temp[idx] += self._delta_diff * (np.random.random() - 0.5)

        # Use the new coeff to calculate a new diffusion profile
        diff_bin_temp = model.calc_profile(diff_coeff_temp, model._diff_basis)

        #Calculate a new likelihood to check acceptance of the step
        log_like_try = self.log_likelihood_z(model, diff_bin_temp)

        # # propagtor behavior
        if log_like_try is not None and not np.isnan(log_like_try):   # propagator is well behaved  TODO implement

            # Calculate different between new and old likelihood
            dlog = log_like_try - self._log_like

            # Calculate a random number for the acceptance criterion
            r = np.random.random()  #in [0,1[

            # acceptance criterion
            if r < np.exp(dlog / self._temp): # warum geht hier eig die Temperatur mit ein?

                # Save new diffusion profile (after MC step)
                model._diff_bin[:] = diff_bin_temp[:]

                # Save new coefficent vector
                model._diff_coeff[:] = diff_coeff_temp[:]

                #Save new likelihood
                self._log_like = log_like_try

                # Update MC statistics
                self._nacc_diff_coeff[idx] += 1
                self._nacc_diff += 1
                self._nacc_diff_update += 1



    def mcmove_df(self,model):
        """This function does the MC move for the free energy profile and adjust
        the coefficents of the model. A MC move in the parameter space is defined
        by

        .. math::

            a_{k,\\mathrm{new}}= a_{k} + \\Delta_\\mathrm{MC} \\cdot (R - 0.5)

        with :math:`\\Delta_\\mathrm{MC}` as the MC step movewidth and :math:`R`
        as a random number between :math:`0` and :math:`1`. The choice of the
        coefficient is also made by determining a random number.

        Parameters
        ----------
        model : Model
            Model object which set with the model class
        """

        # Caclulate a random number to choose a random coefficient
        # Attion the first coefficent of the df profile is fixed 0 all the time
        idx = np.random.randint(1,model._n_df)

        # Calculate one new coefficient
        df_coeff_temp = copy.deepcopy(model._df_coeff)
        df_coeff_temp[idx] += self._delta_df * (np.random.random() - 0.5)

        # Use the new coeff to calculate a new diffusion profile
        df_bin_temp = model.calc_profile(df_coeff_temp, model._df_basis)

        #Calculate a new likelihood to check acceptance of the step
        log_like_try = self.log_likelihood_z(model, df_bin_temp)

        # # propagtor behavior
        if log_like_try is not None and not np.isnan(log_like_try):   # propagator is well behaved  TODO implement

            # Calculate different between new and old likelihood
            dlog = log_like_try - self._log_like

            # Calculate a random number for the acceptance criterion
            r = np.random.random()  #in [0,1[

            # acceptance criterion
            if r < np.exp(dlog / self._temp): # warum geht hier eig die Temperatur mit ein?

                # Save new diffusion profile (after MC step)
                model._df_bin[:] = df_bin_temp[:]

                # Save new coefficent vector
                model._df_coeff[:] = df_coeff_temp[:]

                #Save new likelihood
                self._log_like = log_like_try

                # Update MC statistics
                self._nacc_df_coeff[idx] += 1
                self._nacc_df += 1
                self._nacc_df_update += 1


    def mcmove_diffusion_radial(self,model):
        """
        This function do the MC move for the radial diffusion profile and adjust
        the coefficents of the model. A MC move in the parameter space is defined by

        .. math::

            a_{k,\\mathrm{new}}= a_{k} + \\Delta_\\text{MC} \\cdot (R - 0.5)

        with :math:`\\Delta_\\text{MC}` as the MC step movewidth and :math:`R`
        as a random number between :math:`0` and :math:`1`. The choice of the
        coefficient is also made by determining a random number.

        Parameters
        ----------
        model : Model
            Model object which set before with the model class
        """
        # Caclulate a random number to choose a random coefficient
        # Attion the first coefficent of the diffusion profile is fixed 0 all the time
        idx = np.random.randint(0,model._n_diff_radial)

        # Calculate one new coefficient
        diff_radial_coeff_temp = copy.deepcopy(model._diff_radial_coeff)
        diff_radial_coeff_temp[idx] += self._delta_diff_radial * (np.random.random() - 0.5)

        # Use the new coeff to calculate a new diffusion profile
        diff_radial_bin_temp = model.calc_profile(diff_radial_coeff_temp, model._diff_radial_basis)

        #Calculate a new likelihood to check acceptance of the step
        log_like_try = self.log_likelihood_radial(model,diff_radial_bin_temp)

        # # propagtor behavior
        if log_like_try is not None and not np.isnan(log_like_try):   # propagator is well behaved  TODO implement

            # Calculate different between new and old likelihood
            dlog = log_like_try - self._log_like_radial

            # Calculate a random number for the acceptance criterion
            r = np.random.random()  #in [0,1[

            # acceptance criterion
            if r < np.exp(dlog / self._temp): # warum geht hier eig die Temperatur mit ein?

                # Save new diffusion profile (after MC step)
                model._diff_radial_bin[:] = diff_radial_bin_temp[:]

                # Save new coefficent vector
                model._diff_radial_coeff[:] = diff_radial_coeff_temp[:]

                #Save new likelihood
                self._log_like_radial = log_like_try

                # Update MC statistics
                self._nacc_diff_radial_coeff[idx] += 1
                self._nacc_diff_radial += 1
                self._nacc_diff_radial_update += 1



    def update_movewidth_mc(self,imc,radial=False):
        """
        This function sets a new MC movewidth after a define number of MC steps
        :math:`n_\\text{MC,update}`. The new step width is estimate with

        .. math::

            \\Delta_\\text{MC} = \\exp \\left ( 0.1 \\cdot \\frac{n_\\text{accept}}{n_\\text{MC,update}} \\right)

        Parameters
        ----------
        imc : integer
            current number of MC steps
        do_radial : bool
            if it's True the code calculate the radial diffusion too
        """

        if self._num_mc_update > 0:

            if ( (imc+1) % self._num_mc_update == 0 ) and radial==False:
                self._delta_df *= np.exp ( 0.1 * ( float(self._nacc_df_update) / self._num_mc_update - 0.3 ) )
                self._delta_diff *= np.exp ( 0.1 * ( float(self._nacc_diff_update) / self._num_mc_update - 0.3 ) )


                self._nacc_df_update = 0
                self._nacc_diff_update = 0

            if ( (imc+1) % self._num_mc_update == 0 ) and radial==True:
                self._delta_diff_radial *= np.exp ( 0.1 * ( float(self._nacc_diff_radial_update) / self._num_mc_update - 0.3 ) )
                self._nacc_diff_radial_update = 0



    def init_rate_matrix_pbc(self, n, diff_bin, df_bin):
        """
        This function estimates the rate Matrix R for the current free energy
        and log diffusion profiles over the bins for periodic boundary
        conditions. The dimension of the matrix is :math:`n \\times n`
        with :math:`n` as number of the bins.
        The calculation of the secondary diagonal elements in the rate
        matrix :math:`R` happen with the following equations

        .. math::

            R_{i+1,i} = \\exp  \\underbrace{\\left( \\ln \\left( \\frac{D_{i+\\frac{1}{2}}}{\\Delta z^2}\\right) \\right)}_{\\mathrm{diff}_\\mathrm{bin}} - 0.5(\\beta(F(\\Delta z_{i+1})-F(\\Delta z_{i})

        .. math::

            R_{i,i+1} = \\exp \\left( \\ln \\left( \\frac{D_{i+\\frac{1}{2}}}{\Delta z^2}\\right) \\right) + 0.5(\\beta(F(\\Delta z_{i+1})-F(\\Delta z_{i})

        with :math:`\\Delta z` as the bin width, :math:`D_{i+\\frac{1}{2}}`
        as the diffusion between to bins and :math:`F_i` as free energy in
        the bin center. The diagonal elements can be calculated with the
        secondary elements determine with the equations above.

        .. math::

            R_{i,i} = -R_{i-1,i}-R_{i+1,i}

        The corner of the rate matrix is set with:

        .. math::

            R_{1,1} = - R_{2,1} - R_{N,1}

        .. math::

            R_{N,N} = - R_{N-1,N} - R_{1,N}

        The periodic boundary conditions are implemeted with

        .. math::

            R_{1,N} = \\exp \\left( \\ln \\left( \\frac{D_{N+\\frac{1}{2}}}{\\Delta z^2}\\right) \\right) - 0.5(\\beta(F(\\Delta z_{1})-F(\\Delta z_{N}))

        .. math::

            R_{N,1} = \\exp \\left( \\ln \\left( \\frac{D_{N+\\frac{1}{2}}}{\\Delta z^2}\\right) \\right) + 0.5(\\beta(F(\\Delta z_{1})-F(\\Delta z_{N}))

        Parameters
        ----------
        n : integer
            Link to poresystem object file
        diff_bin : list
            log diffusion profile over the bins
        df_bin : list
            free energy profile over the bins

        Returns
        -------
        rate matrix :
            rate matrix for the current free energy and log diffusion profile
        """
        # Initialize rate matrix
        rate = np.float64(np.zeros((n,n)))

        # off-diagonal elements
        delta_df_bin = df_bin[1:]-df_bin[:-1]
        exp1 = diff_bin[:n-1] - 0.5 * delta_df_bin
        exp2 = diff_bin[:n-1] + 0.5 * delta_df_bin
        rate.ravel()[n::n+1] = np.exp(exp1)[:n-1]
        rate.ravel()[1::n+1] = np.exp(exp2)[:n-1]

        # corners and periodic boundary conditions
        rate[0,-1]  = np.exp(diff_bin[-1]-0.5*(df_bin[0]-df_bin[-1]))
        rate[-1,0]  = np.exp(diff_bin[-1]-0.5*(df_bin[-1]-df_bin[0]))
        rate[0,0]   = - rate[1,0] - rate[-1,0]
        rate[-1,-1] = - rate[-2,-1] - rate[0,-1]


        # diagonal elements
        for i in range(1,n-1):
            rate[i,i] = - rate[i-1,i] - rate[i+1,i]

        return rate

    def init_rate_matrix_nopbc(self, n, diff_bin, df_bin):
        """
        This function estimate the rate Matrix R for the current free energy
        and log diffusion profiles over the bins for a reflected wall. The
        dimension of the matrix is :math:`n \\times n` with n as number of
        the bins. The calculation of the secondary diagonal elements in the
        rate matrix :math:`R` happen with the following equations

        .. math::

            R_{i+1,i} = \\exp  \\underbrace{\\left( \\ln \\left( \\frac{D_{i+\\frac{1}{2}}}{\\Delta z^2}\\right) \\right)}_{\\mathrm{diff}_\\mathrm{bin}}  - 0.5(\\beta(F(\\Delta z_{i+1})-F(\\Delta z_{i})

        .. math::

            R_{i,i+1} = \\exp \\left( \\ln \\left( \\frac{D_{i+\\frac{1}{2}}}{\Delta z^2}\\right) \\right) + 0.5(\\beta(F(\\Delta z_{i+1})-F(\\Delta z_{i})

        with :math:`\\Delta z` as the bin width, :math:`D_{i+\\frac{1}{2}}`
        as the diffusion between to bins and :math:`F_i` as free energy in
        the bin center.
        The diagonal elements can be calculated with the secondary elements
        determine with the equations above.

        .. math::

            R_{i,i} = -R_{i-1,i}-R_{i+1,i}

        The corner of the rate matrix is set with:

        .. math::

            R_{1,1} = - R_{2,1} - R_{N,1}

        .. math::

            R_{N,N} = - R_{N-1,N} - R_{1,N}


        Parameters
        ----------
        n : integer
            Link to poresystem object file
        diff_bin : list
            log diffusion profile over the bins
        df_bin : list
            free energy profile over the bins

        Returns
        -------
        rate matrix :
            rate matrix for the current free energy and log diffusion profile
        """

        # Initialize rate matrix
        rate = np.float64(np.zeros((n,n)))

        # off-diagonal elements
        delta_df_bin = df_bin[1:]-df_bin[:-1]
        exp1 = diff_bin[:n-1] - 0.5 * delta_df_bin
        exp2 = diff_bin[:n-1] + 0.5 * delta_df_bin
        rate.ravel()[n::n+1] = np.exp(exp1)[:n-1]
        rate.ravel()[1::n+1] = np.exp(exp2)[:n-1]

        # corners for a reflected wall
        rate[0,0]   = - rate[1,0]
        rate[-1,-1] = - rate[-2,-1]

        # diagonal elements
        for i in range(1,n-1):
            rate[i,i] = - rate[i-1,i] - rate[i+1,i]

        return rate

    def log_likelihood_z(self, model,  temp = None):
        """
        This function estimate the likelihood of the current free energy or diffusion profile over the bins in a simulation box. It is used to calculated the diffusion in z-direction over z-coordinate. This likelihood is necessary to decide whether the MC step will be accepted.

        .. math::

            \\ln \ L = \\sum_{j \\rightarrow i} \\ln \\left ( \\left[ \\left( e^{\\mathbf{R}\\Delta_{ij}t_{\\alpha}} \\right)_{ij}\\right]^{N_{ij}(\\Delta_{ij}t_{\\alpha})} \\right)

        with :math:`R` as the rate matrix from :func:`init_rate_matrix_pbc` , :math:`\\Delta_{ij}t_{\\alpha}` as the current lag time and :math:`N_{ij}(\\Delta_{ij}t_{\\alpha})` as the transition matrix. The transition matrix contains the numbers of all observed transition :math:`j \\rightarrow i` at a given lag time in a simulated trajectory. This matrix is sampled with the function :func:`init_diffusion_mc` from a simulated trajectory at the beginning and depends on the lag time.

        Parameters
        ----------
        model : integer
            model class set with the model function
        temp : list
            profile which was adapted in the current MC (can be the free energy or diffusion profile)

        Returns
        -------
        log_like : float
            likelihood for the current profiles in a simulation box
        """
        # Initialize log_like
        log_like = np.float64(0.0)
        tiny = 1e-10

        # Calculate the current rate matrix for a trajectory with periodic boundary condition
        if model._pbc==True:
            if temp is None:
                rate = self.init_rate_matrix_pbc(model._bin_num, model._diff_bin, model._df_bin)
            else :
                if self._choice > 0.5:
                    rate = self.init_rate_matrix_pbc(model._bin_num,  temp , model._df_bin)
                else:
                    rate = self.init_rate_matrix_pbc(model._bin_num, model._diff_bin, temp)

        # Calculate the current rate matrix for a trajectory with a reflected wall in z direction
        elif model._pbc==False:
            if temp is None:
                rate = self.init_rate_matrix_nopbc(model._bin_num, model._diff_bin, model._df_bin)
            else :
                if self._choice > 0.5:
                    rate = self.init_rate_matrix_nopbc(model._bin_num,  temp , model._df_bin)
                else:
                    rate = self.init_rate_matrix_nopbc(model._bin_num, model._diff_bin, temp)

        # Calculate the propagtor
        propagator = scipy.linalg.expm(self._len_step * model._dt * rate)

        # Calculate likelihood
        mat = model._trans_mat[self._len_step] * np.log(propagator.clip(tiny))
        log_like = np.sum(mat)

        return log_like



    def log_likelihood_radial(self, model, wrad):
        """
        This function estimate the likelihood of the current radial diffusion profile over the bins in a simulation box. This likelihood is necessary to decide whether the MC step will be accepted.

        .. math::
             \\ln \\ L(M) = \\sum_{m,i \\leftarrow j} \\ln \\left( \\left[ \\Delta r \\sum_{\\alpha_k} 2 r_m \\frac{J_0(\\alpha_kr_m)}{s^2J_1^2(x_k)} [e^{(\\mathbf{R}-\\alpha_k^2D)\\Delta_{ij}t_{\\alpha}}]_{ij} \\right] ^{N_{m,ij}(\\Delta_{ij}t_{\\alpha})} \\right)


        with :math:`t_{i}` as the current lag time and :math:`N_{m,ij}(t_i)` as the radial transition matrix, :math:`J_0` as the roots of the Bessel function and :math:`J_1` as the 1st order Bessel first type in those zeros. The variable :math:`s` is the length where the bessel function going to zero and :math:`r_m` is the middle of the radial bin. The lag time is :math:`t_{i}` and :math:`D` is the current radial diffusion profile.
        The rate matrix :math:`R` from :func:`init_rate_matrix_pbc` which constant over the radial diffuison determination and is calculated with the normal diffusion and the free energy profile is need. This the reason why the normal diffusion calculation is necessary for the radial diffusion calculation.

        The transition matrix :math:`N_{m,ij}(t_i)` contains all observed number of transition of a particle being in z-bin i and radial bin m after a lag time t, given that it was in z-bin j and r = 0 at the start.
        This matrix is sampled with the function :func:`init_diffusion_mc` from a simulated trajectory at the beginning and depends on the lag time.

        The bessel part

        .. math::
            \\mathrm{bessel} = \\sum_{\\alpha_k} 2r_m \\frac{J_0(\\alpha_kr_m)}{s^2J_1^2(x_k)}

        of the likelihood does not depend on the radial diffusion profile and is determined with the function :func:`setup_bessel_box`

        The function :func:`log_likelihood_radial` determine the rate matrix part of the Likelihood

        .. math::
            \\mathrm{rate}_{\\mathrm{radial}} = \\left [e^{\\left(\\mathbf{R}-\\alpha_k^2D\\right)\\Delta_{ij}t_{\\alpha}}\\right]_{ij}

        multiple it with the results from the :func:`setup_bessel_box` and logarithmize the result. Afterwards the results are multipe with the transition matrix :math:`N_{m,ij}(\\Delta_{ij}t_{\\alpha})`. The likelihood thus obtained provides the acceptance criteria for the MC alogirthm and is returned by the function .

        Parameters
        ----------
        model : obj
            model class set with the model function
        wrad : list
            radial diffusion profile which was adapted in the current MC

        Returns
        -------
        log_like : float
            likelihood for the current radial diffusion profile in a simulation box
        """
        # Initialize log_like
        log_like = np.float64(0.0)
        tiny = np.empty((model._bin_num_rad,model._bin_num,model._bin_num))
        tiny.fill(1.e-32) # lower bound of propagator (to avoid NaN's)

        # Calculate the current rate matrix
        if model._pbc==True:
            rate = self.init_rate_matrix_pbc(model._bin_num, model._diff_bin, model._df_bin)

        elif model._pbc==False:
            rate = self.init_rate_matrix_nopbc(model._bin_num, model._diff_bin, model._df_bin)

        # Calculate propagtor
        rmax = 5 #max(model._bins_radial) # in units [dr]

        # initialize Arrays
        rate_l = np.zeros((model._bin_num,model._bin_num),dtype=np.float64)
        propagator = np.zeros((model._bin_num_rad,model._bin_num,model._bin_num),dtype=np.float64)
        trans = np.zeros((model._bin_num_rad,model._bin_num,model._bin_num),dtype=np.float64)

        # set up sink term
        sink = np.zeros((model._bin_num),dtype=np.float64)

        # loop over l (index of Bessel function)
        for l in range(self._lmax):

            # Calculate sink term
            sink = np.exp(wrad) * self._bessel0_zeros[l]**2 / rmax**2           # sink term for on bessel index (sink term has dimension r=z)

            # Calculate the rate matrix with sink term
            rate_l[:,:] = rate[:,:]
            rate_l.ravel()[::model._bin_num+1] -= sink                          # Sink term substract from the diagonal of the rate matrix

            # Calculate the exp of the rate - sink term
            mat_exp = scipy.linalg.expm(self._len_step * model._dt * rate_l)

            # Calculate the propagator
            for k in range(model._bin_num_rad):
                propagator[k,:,:] += self._bessels[l,k] * mat_exp[:,:]          # Likelihood per r bin and per z bin (rxzxz)

        # Log of the propagator
        ln_prop = np.log(model._bin_radial_width * np.maximum(propagator,tiny))

        # Set trans matrix
        for i in range(model._bin_num_rad):
            trans[i,:,:] = model._trans_mat_radial[self._len_step][i][:,:]

        # Calculate likelihood
        log_like = np.sum(trans[:,:,:] * ln_prop[:,:,:])                        # every raidal Transition matrix (zxz) are multiplying the every radial propergator (zxz)

        return log_like


    def setup_bessel_box(self,model):
        """
        This function set the zeros of the 0th order Bessel first type and the bessel function for the radial likelihood.

        .. math::
            \\mathrm{bessel} = \\sum_{\\alpha_k} 2r_m \\frac{J_0(\\alpha_kr_m)}{s^2J_1^2(x_k)}

        The bessel matrix has the dimension of lxr and contains the constant part of the radial likelihood


        Parameters
        ----------
        model : Model
            Model object which set with the model class
        """

        # Zeros of the bessel funktions
        self._bessel0_zeros = scipy.special.jn_zeros(0,self._lmax)              # zeropoints of the 0th bessel function (x-values)

        # 1st order Bessel function zeropoints
        bessel1_inzeros = scipy.special.j1(self._bessel0_zeros)                 # Calculate the y value of the 1th bessel function at zeropoints of 0th bessel function (x-values above)

        # Set max r and r_m vector
        rmax = max(model._bins_radial)                                                                                  # maximal radius -> propagtor becomes zero
        r = np.arange(model._bin_radial_width/2,max(model._bins_radial),model._bin_radial_width,dtype=np.float64)       # Vektors in the middle of the radial bins

        # set up Bessel functions
        self._bessels = np.zeros((self._lmax,model._bin_num_rad),dtype=np.float64)

        # Determine the bessel part of the propogator
        for l in range(self._lmax):
            self._bessels[l,:] = 2 * r * scipy.special.j0(r / rmax * self._bessel0_zeros[l]) / bessel1_inzeros[l]**2 / rmax**2  #Set bessel part of propagtor
