import sys
import copy
import scipy as sc
import numpy as np
import pandas as pd
import multiprocessing as mp
import numpy as numpy

import poreana.utils as utils


class MC:
    """This class contains the Monte Carlo part of the diffusion calculation.
    The class initializes the MC (Monte Carlo) and contains all
    functions to execute the MC cycle part of the diffusion calculation.

    The MC calculation can be started with :func:`run`.

    """
    def __init__(self):
        return

        # Save for radial diffusion
        #self._delta_diff_radial = delta_diff_radial          # MC Move width radial Diffusion
        #self._delta_diff_radial_start = delta_diff_radial
        #self._nmc_eq_radial = nmc_eq_radial                  # Number of MC steps
        #self._nmc_radial = nmc_radial                        # Number of MC steps
        #self._lmax = lmax                                    # Number of bessel functions

    ##############
    # MC - Cylce #
    ##############
    def run(self, model, link_out, nmc_eq=50000, nmc=100000, delta_df=0.05, delta_diff=0.05,  num_mc_update=10, temp=1, np=0, print_freq=100, is_print=False, do_radial=False, is_parallel=True):
        """This function do the MC Cycle to calculate the diffusion and free
        energy profile over the bins and save the results in an output hdf5
        file. This happens with the adjustment of the coefficient from the model
        which is set with the appropriate model class. The code determines the
        results for all lag times for which a transition matrix was calculated.
        The results can be displayed with the post process functions.
        The MC Algorithm is divided in two parts. Frist ab equilibrium MC run starts
        to adjust the profile. The number of the equilibrium runs can be set with
        :math:`\\text{nmc\_eq}`. After the equilibrium phase the sampling of the diffusion
        profile can start. In this part the MC Algorithm calculates in every step
        a new diffusion profile. The average over all profiles which are yields in
        MC production will be determined. The received profile gives the final
        result. The MC algorithm calculates a diffusion profile for each step length
        previously specified in the sampling
        (:func:`poreana.sample.Sample.init_diffusion_mc`) and thus, for each lag
        time. Here you have to set also step the move width of a MC step.

        **More information about a MC step can be found in**

        * :func:`_mcmove_diffusion`
        * :func:`_mcmove_df`

        The MC accepted criterium for a MC step is define as

        .. math::

            r < \\exp \\left ( \\frac{L_\\text{new} - L_\\text{old}}{T}  \\right)

        with :math:`T` as the temperature, :math:`r` as a random number between
        :math:`0` and :math:`1` and :math:`L_\\text{new}` and
        :math:`L_\\text{old}` as the likelihood for the current and the last
        step. The likelihood is calculated with :func:`_log_likelihood_z`
        function.


        Parameters
        ----------
        model : class
            Model object which set before with the model class
        link_out : string
            Link to output hdf5 data file
        nmc_eq : integer, optional
            Number of equilibrium MC steps
        nmc : integer, optional
            Number of production MC steps
        delta_df : float, optional
            Potential MC move width
        delta_diff : float, optional
            ln(diffusion) MC move width
        num_mc_update : integer, optional
            Number of moves between MC step width adjustments (=0 if no adjustment
            is required)
        temp : float, optional
            Temperature in Monte Carlo acceptance criterium
        np : integer, optional
            Number of cores to use
        print_freq : integer, optional
            Print MC step every print_freq
        is_print : bool, optional
            True to print output
        do_radial : bool, optional
            True to calculate the radial diffusion
        is_parallel : bool, optional
            True to run parallelized sampling
        """

        # Set MC step width
        self._delta_df = delta_df                            # MC Move width free energy
        self._delta_diff = delta_diff                        # MC Move width Diffusion

        # Save beginning step width (to initialize it every mc run)
        self._delta_df_start = delta_df                      # MC Move width free energy
        self._delta_diff_start = delta_diff                  # MC Move width Diffusion

        # Set MC options
        self._nmc = nmc + 1                                  # Number of MC steps
        self._nmc_eq = nmc_eq                                # Number of MC steps
        self._num_mc_update = num_mc_update                  # MC steps before update delta
        self._temp = temp                                    # Temperature for acceptance criterion

        # Set output/print options
        # Bool (If False nothing will be printed in the konsole)
        self._print_output = is_print

        # print frequency for MC steps (default every 100 steps)
        self._print_freq = print_freq


        # Set dictionary for model and input informations
        # Set inp data MC algorithm
        inp = {"MC steps": self._nmc, "MC steps eq": self._nmc_eq, "step width update": self._num_mc_update,  "temperature": self._temp, "print freq": self._print_freq}

        # Set inp data for model
        model_inp = {"bin number": model._bin_num, "bins": model._bins[:-1], "diffusion unit": model._diff_unit, "len_frame": model._dt, "len_step": model._len_step, "model": model._model, "nD": model._n_diff, "nF": model._n_df, "nDrad": model._n_diff_radial, "guess": model._d0, "pbc": model._pbc, "num_frame": model._frame_num, "data": model._trans_mat, "direction": model._direction}

        # Print that MC Calculation starts
        if not self._print_output:
            print("MC Calculation Start")
            print("...")
        else:
            print("\n")
            print("--------------------------------------------------------------------------------")
            print("--------------------------------MC Starts---------------------------------------")
            print("--------------------------------------------------------------------------------\n")
            print("MC Inputs")

            # Table for MC Inputs (set data structure)
            data = [self._nmc_eq, self._nmc, self._num_mc_update, self._print_freq]
            df_input = pd.DataFrame(data, index=list(['MC step (Equilibrium)', 'MC step (Production)', 'movewidth update', 'print frequency']), columns=list(['Input']))

            # Table for MC Inputs (set data structure) (Save for radial)
            #data = [self._nmc_eq, self._nmc, self._nmc_eq_radial, self._nmc_radial, self._num_mc_update, self._print_freq]
            #df_input = pd.DataFrame(data, index=list(['MC step (Equilibrium)', 'MC step (Production)', 'MC step radial (Equilibrium)', 'MC step radial (Production)', 'movewidth update', 'print frequency']), columns=list(['Input']))

            # Table for MC Inputs
            print(df_input)


        # Get number of cores
        np = np if np and np<=mp.cpu_count() else mp.cpu_count()

        # List for step times per cpu
        len_step_cpu = numpy.array_split(model._len_step,np);


        # If is parallel is true, each lag time MC run is calculated on one CPU
        if is_parallel:
            pool = mp.Pool(processes=np)
            results = [pool.apply_async(self._run_helper, args=(model,list(step), do_radial)) for step in len_step_cpu]
            pool.close()
            pool.join()
            output_para = [x.get() for x in results]

            # Destroy object
            del results

            # Concatenate output
            output = output_para[0]
            for out,len_step in zip(output_para[1:],len_step_cpu[1:]):
                for step in len_step:
                    output["diff_profile"][step] = out["diff_profile"][step]
                    output["df_profile"][step] = out["df_profile"][step]
                    output["diff_coeff"][step] = out["diff_coeff"][step]
                    output["df_coeff"][step] = out["df_coeff"][step]
                    output["nacc_df"][step] = out["nacc_df"][step]
                    output["nacc_diff"][step] = out["nacc_diff"][step]
                    output["fluc_diff"][step] = out["fluc_diff"][step]
                    output["fluc_df"][step] = out["fluc_df"][step]
                    output["list_df_coeff"][step] = out["list_df_coeff"][step]
                    output["list_diff_coeff"][step] = out["list_diff_coeff"][step]
        else:
            output = self._run_helper(model, model._len_step, do_radial)


        # Print MC statistics
        if self._print_output:
            print("--------------------------------------------------------------------------------")
            print("--------------------------------MC Statistics-----------------------------------")
            print("--------------------------------------------------------------------------------\n")

            # Set data structure fpr pandas table
            data = [[str("%.4e" % output["fluc_df"][i]) for i in model._len_step], [str("%.4e" % output["fluc_diff"][i]) for i in model._len_step], [str("%.0f" % output["nacc_df"][i]) for i in model._len_step], [str("%.0f" % output["nacc_diff"][i]) for i in model._len_step], [str("%.2f" % (output["nacc_df"][i]*100/(self._nmc_eq+self._nmc))) for i in model._len_step], [str("%.2f" % (output["nacc_diff"][i]*100/(self._nmc_eq+self._nmc))) for i in model._len_step]]

            # Set options for pandas table
            df = pd.DataFrame(data, index=list(['fluctuation df', 'fluctuation diff', 'acc df steps', 'acc diff steps', 'acc df steps (%)', 'acc diff steps (%)']), columns=list(model._len_step))
            df = pd.DataFrame(df.rename_axis('Step Length', axis=1))

            # Print pandas table for the MC statistics
            print(df)
            print("--------------------------------------------------------------------------------\n\n")

        # Print MC Calculation is done
        print("MC Calculation Done.")

        # Save inp and output data (remove if hdf5 is ready)
        #utils.save({"inp": inp, "model": model_inp, model._system: model._sys_props, "output": output}, link_out)

        # pickle directory to save late in hdf5
        results = {"inp": inp, "model": model_inp, model._system: model._sys_props, "output": output, "type": "mc"}

        utils.save(results,link_out)


        # # Save txt file
        # # Calculated diffusion coefficient
        # diff_fit = diffusion.mc_fit(link_out)
        # diff_prof = diffusion.mc_profile(link_out, infty_profile=True)
        #
        # if model._system == "pore":
        #     diff_fit_pore = diffusion.mc_fit(link_out, section = "pore")
        #     diff_fit_res = diffusion.mc_fit(link_out, section = "reservoir")
        #
        # if is_txt:

        return

    def _run_helper(self, model, len_step, do_radial):
        """Helper function for MC run

        Parameters
        ----------
        model : class
            Model object which set before with the model class
        len_step : list
            List of step length
        do_radial : bool, optional
            True to calculate the radial diffusion

        Returns : dictionary
            Dictionary containing all MC results
        """

        # Initialize result lists (profiles, coefficients, fluctuation and accepted MC steps)
        ## Diffusion
        list_diff_profile = {}
        list_diff_coeff = {}
        list_diff_fluc = {}
        nacc_diff_mean = {}

        # ## Radial diffusion
        list_diff_radial_profile = {}
        list_diff_radial_coeff = {}
        list_diff_radial_fluc = {}
        nacc_diff_radial_mean = {}

        ## Free energy
        list_df_profile = {}
        list_df_coeff = {}
        list_df_fluc = {}
        nacc_df_mean = {}

        # Loop over the different step_length (lag times) (-> for every lag time a MC Calculation have to run)
        for self._len_step in len_step:
            # Print that a new calculation with a new lag time starts
            lagtime_string = "Lagtime: " + str(self._len_step * model._dt) + " ps"
            if self._print_output:
                print("\n")
                print("# "+lagtime_string+"\n")


            # Initialize Model for every MC Run
            model._init_model()
            model._init_profiles()

            # Start calculation for the normal diffusion and free energy Profile
            if self._print_output:
                print("## Calculate normal diffusion\n")
                print("### Start equilibration\n")

            # Calculated the initialize likelihood
            self._log_like = self._log_likelihood_z(model)

            # Initialize a new statistic
            self._init_stats(model)

            # Set step width for every MC run on the input values
            self._delta_df = copy.deepcopy(self._delta_df_start)
            self._delta_diff = copy.deepcopy(self._delta_diff_start)
            #self._delta_diff_radial = copy.deepcopy(self._delta_diff_radial_start)

            # Initialize the fluctuation every MC run
            diff_profile_flk = np.float64(np.zeros(model._bin_num))
            # diff_radial_profile_flk = np.float64(np.zeros(model._bin_num))
            df_profile_flk = np.float64(np.zeros(model._bin_num))
            # mean_diff_radial_profile_flk = np.float64(np.zeros(model._bin_num))
            self._fluctuation_diff = 0
            self._fluctuation_diff_radial = 0
            self._fluctuation_df = 0

            ######################
            # Start MC Algorithm #
            ######################
            # Loop over the MC steps
            for imc in range(self._nmc + self._nmc_eq):
                # Random number to decide between diffusion or free energy
                self._choice = np.random.rand()

                # Decide with choice which MC move will be execute
                if self._choice < 0.5:
                    # Do a MC move in the free energy profile
                    self._mcmove_df(model)
                else:
                    # Do a MC move in the diffusion profile
                    self._mcmove_diffusion(model)

                # Update the MC movewidth
                self._update_movewidth_mc(imc)

                # Calculate the fluctuation and start the production
                if imc >= self._nmc_eq:

                    # Add all profiles
                    diff_profile_flk += copy.deepcopy(model._diff_bin)
                    df_profile_flk += copy.deepcopy(model._df_bin)

                    # Calculate the mean profile over all runs
                    mean_diff_profile_flk = [diff_profile_flk[i] / ((imc-self._nmc_eq)+1) for i in range(model._bin_num)]
                    mean_df_profile_flk = [df_profile_flk[i] / ((imc-self._nmc_eq)+1) for i in range(model._bin_num)]

                    # Calculate the difference between the current profile and the mean of all profiles
                    delta_diff = ((model._diff_bin) - mean_diff_profile_flk)**2
                    delta_df = ((model._df_bin) - mean_df_profile_flk)**2

                    # Determine the fluctuation
                    self._fluctuation_diff = np.sqrt((self._fluctuation_diff + np.mean(delta_diff))/(imc+1))
                    self._fluctuation_df = np.sqrt((self._fluctuation_df + np.mean(delta_df))/(imc+1))

                    # Start to print the output after the Equilibrium phase and if _print_output is true
                    if imc == self._nmc_eq and self._print_output:
                        print("### Start production\n")
                        print("--------------------------------------------------------------------------------")
                        print("imc | accepted_df(%) | accepted_diff(%) | fluktuation_df | fluktuation_diff    |" )
                        print("--------------------------------------------------------------------------------")
                    if (imc % self._print_freq == 0) and imc > self._nmc_eq and self._print_output:
                        sys.stdout.write(str(imc)+" "+ str("%.2f"%(self._nacc_df*100/(imc+1))) +" "+ str("%.2f"%(self._nacc_diff*100/(imc+1)))+" "+str("%.2e"%self._fluctuation_df) +" "+ str("%.2e"%self._fluctuation_diff) +"\r")
                        sys.stdout.flush()
                        if imc%2000==0:
                            print(str(imc)+" "+ str("%.2f"%(self._nacc_df*100/(imc+1))) +" "+ str("%.2f"%(self._nacc_diff*100/(imc+1)))+" "+str("%.2e"%self._fluctuation_df) +" "+ str("%.2e"%self._fluctuation_diff))

            if self._print_output:
                print("--------------------------------------------------------------------------------\n\n")

            # Save the profiles over the bins for the current lag time in a list
            # copy.deepcopy(model._diff_bin) #mean_diff_profile_flk
            list_diff_profile[self._len_step] = mean_diff_profile_flk
            list_df_profile[self._len_step] = mean_df_profile_flk  # copy.deepcopy(model._df_bin)

            # Save the coefficients of the model for the current lag time in a list
            list_diff_coeff[self._len_step] = model._diff_coeff
            list_df_coeff[self._len_step] = model._df_coeff

            # Save the fluctuation of the current lag time run in a list
            list_diff_fluc[self._len_step] = self._fluctuation_diff
            list_df_fluc[self._len_step] = self._fluctuation_df

            # Mean over all lag times calculations
            nacc_df_mean[self._len_step] = copy.deepcopy(self._nacc_df)
            nacc_diff_mean[self._len_step] = copy.deepcopy(self._nacc_diff)

            #############################
            # Start Radial MC Algorithm #
            #############################
            # # Radial diffusion
            # if do_radial:
            #     # Start MC calculation for the radial diffusion
            #     print("## Calculate radial diffusion")
            #
            #     # Calculated the initialize likelihood and bessel function
            #     self.setup_bessel_box(model)
            #     self._log_like_radial = self.log_likelihood_radial(model, model._diff_radial_bin)
            #
            #     # Print first likelihood
            #     if self._print_output:
            #         if self._log_like==0:
            #                 print("likelihood init", self._log_like_radial, "\n")
            #         print("### Start equilibration\n")
            #
            #     # Start MC Algorithm for the radial diffusion
            #     for imc in range(self._nmc_radial+self._nmc_eq_radial):
            #
            #         # Do a MC move in the radial diffusion profile
            #         self._mcmove_diffusion_radial(model)
            #
            #         # Update the MC movewidth
            #         self._update_movewidth_mc(imc, True)
            #
            #         # Calculate the fluctuation and start the production
            #         if imc >= self._nmc_eq_radial:
            #
            #             # Add all profiles
            #             diff_radial_profile_flk += copy.deepcopy(model._diff_radial_bin)
            #
            #             # Calculate the mean profile over all runs
            #             mean_diff_radial_profile_flk = [diff_radial_profile_flk[i]/((imc-self._nmc_eq_radial)+1) for i in range(model._bin_num)]
            #
            #             # Calculate the difference between the current profile and the mean of all profiles
            #             delta_diff_radial = ((model._diff_radial_bin) -
            #                                  mean_diff_radial_profile_flk)**2
            #
            #             # Determine the fluctuation
            #             self._fluctuation_diff_radial = np.sqrt(
            #                 (self._fluctuation_diff_radial + np.mean(delta_diff_radial))/(imc+1))
            #
            #             # Print output in production phase
            #             if imc == self._nmc_eq_radial and self._print_output:
            #                 print("### Start production\n")
            #                 print("------------------------------------------------")
            #                 print("imc | accepted_rdiff(%) | fluktuation_rdiff    |" )
            #                 print("------------------------------------------------")
            #                 mjmlll
            #             if (imc % self._print_freq == 0) and imc > self._nmc_eq_radial and self._print_output:
            #                 sys.stdout.write(str(imc)+" "+ str("%.2f"%(self._nacc_diff*100/(imc+1)))+" "+ str("%.2e"%self._fluctuation_diff) +"\r")
            #                 sys.stdout.flush()
            #                 if imc%2000==0:
            #                     print(str(imc)+" "+ str("%.2f"%(self._nacc_df*100/(imc+1))) +" "+ str("%.2f"%(self._nacc_diff*100/(imc+1)))+" "+str("%.2e"%self._fluctuation_df) +" "+ str("%.2e"%self._fluctuation_diff))
            #
            #                 print(imc, "\t", "%.6f" % self._log_like_radial, "\t", "%.2f" % (float(self._nacc_diff_radial)*100/(
            #                     imc+1)), "\t\t\t", "%.5f" % self._delta_diff_radial, "\t", "\t", "%.4e" % self._fluctuation_diff_radial)
            #                 print(self._nacc_diff_radial)
            #                 print(np.mean(np.exp(model._diff_radial_bin + model._diff_radial_unit)) * 10**-6)
            #     if self._print_output:
            #         print("--------------------------------------------------------------------------------\n")
            #
            #     # Save results for the current lag time
            #     # copy.deepcopy(model._diff_radial_bin)
            #     list_diff_radial_profile[self._len_step] = mean_diff_radial_profile_flk
            #     list_diff_radial_coeff[self._len_step] = copy.deepcopy(model._diff_radial_coeff)
            #     list_diff_radial_fluc[self._len_step] = self._fluctuation_diff_radial
            #
            #     # Mean over all lag times calculations
            #     nacc_diff_radial_mean[self._len_step] = copy.deepcopy(self._nacc_diff_radial)

            # If no radial diffusion is calculated set the initialize condition on the result lists
            if not do_radial:
                # Save results for the current lag time
                list_diff_radial_profile[self._len_step] = np.float64(np.zeros(model._bin_num))
                list_diff_radial_coeff[self._len_step] = np.float64(np.zeros(model._n_df))
                list_diff_radial_fluc[self._len_step] = 0

                # Mean over all lag times calculations
                nacc_diff_radial_mean[self._len_step] = 0


        # Set output data
        output = {"diff_profile": list_diff_profile, "df_profile": list_df_profile, "diff_coeff": list_diff_coeff,  "df_coeff": list_df_coeff, "nacc_df": nacc_df_mean, "nacc_diff": nacc_diff_mean, "fluc_df": list_df_fluc, "fluc_diff": list_diff_fluc, "list_diff_coeff": list_diff_profile, "list_df_coeff":  list_df_profile}

        return output


    ###########################
    # Helper functions for MC #
    ###########################
    def _init_stats(self, model):
        """ This function sets the MC statistic counters to zero after every MC
        run.

        Parameters
        ----------
        model : class
            Model object which set before with the model class
        """

        # Number of accepted moves
        self._nacc_df = 0          # Free energy
        self._nacc_diff = 0        # Diffusion
        self._nacc_diff_radial = 0 # Radial diffsuion

        # Number of accepted moves between adjustments
        self._nacc_df_update = 0          # Free energy
        self._nacc_diff_update = 0        # Diffsuion
        self._nacc_diff_radial_update = 0 # Radial diffsuion

        # Number accepted for different coefficient update
        self._nacc_df_coeff = np.zeros(model._n_df, int)                   # Free energy
        self._nacc_diff_coeff = np.zeros(model._n_diff, int)               # Diffusion
        self._nacc_diff_radial_coeff = np.zeros(model._n_diff_radial, int) # Radial diffusion

    def _mcmove_diffusion(self, model):
        """This function does the MC move for the diffusion profile and adjust
        one coefficient of the model. A MC move in the parameter space is defined
        by

        .. math::

            a_{k,\\mathrm{new}}= a_{k} + \\Delta_\\text{MC} \\cdot (R - 0.5)

        with :math:`\\Delta_\\text{MC}` as the MC step move width and :math:`R`
        as a random number between :math:`0` and :math:`1`. The choice of the
        coefficient is also made by determining a random number.

        Parameters
        ----------
        model : class
            Model object which set with the model class
        """

        # Caclulate a random number to choose a random coefficient
        ## The first coefficient of the diffusion profile is fixed 0 all the time
        idx = np.random.randint(0, model._n_diff)

        # Calculate one new coefficient
        diff_coeff_temp = copy.deepcopy(model._diff_coeff)
        diff_coeff_temp[idx] += self._delta_diff * (np.random.random() - 0.5)

        # Use the new coefficient to calculate a new diffusion profile
        diff_bin_temp = model._calc_profile(diff_coeff_temp, model._diff_basis)

        # Calculate a new likelihood to check acceptance of the step
        log_like_try = self._log_likelihood_z(model, diff_bin_temp)

        # Propagator behavior
        if log_like_try is not None and not np.isnan(log_like_try):
            # Calculate different between new and old likelihood
            dlog = log_like_try - self._log_like

            # Calculate a random number for the acceptance criterion
            r = np.random.random()  # in [0,1[

            # Acceptance criterion
            if r < np.exp(dlog / self._temp):
                # Save new diffusion profile (after MC step)
                model._diff_bin[:] = diff_bin_temp[:]

                # Save new coefficient vector
                model._diff_coeff[:] = diff_coeff_temp[:]

                # Save new likelihood
                self._log_like = log_like_try

                # Update MC statistics
                self._nacc_diff_coeff[idx] += 1
                self._nacc_diff += 1
                self._nacc_diff_update += 1


    ############
    # MC Moves #
    ############
    def _mcmove_df(self, model):
        """This function does the MC move for the free energy profile and adjust
        the coefficents of the model. A MC move in the parameter space is
        defined by

        .. math::

            a_{k,\\mathrm{new}}= a_{k} + \\Delta_\\mathrm{MC} \\cdot (R - 0.5)

        with :math:`\\Delta_\\mathrm{MC}` as the MC step move width and :math:`R`
        as a random number between :math:`0` and :math:`1`. The choice of the
        coefficient is also made by determining a random number.

        Parameters
        ----------
        model : class
            Model object which set with the model class
        """
        # Caclulate a random number to choose a random coefficient
        ## The first coefficient of the df profile is fixed 0 all the time
        idx = np.random.randint(1, model._n_df)

        # Calculate one new coefficient
        df_coeff_temp = copy.deepcopy(model._df_coeff)
        df_coeff_temp[idx] += self._delta_df * (np.random.random() - 0.5)

        # Use the new coeff to calculate a new diffusion profile
        df_bin_temp = model._calc_profile(df_coeff_temp, model._df_basis)

        # Calculate a new likelihood to check acceptance of the step
        log_like_try = self._log_likelihood_z(model, df_bin_temp)

        # Propagtor behavior
        if log_like_try is not None and not np.isnan(log_like_try):
            # Calculate different between new and old likelihood
            dlog = log_like_try - self._log_like

            # Calculate a random number for the acceptance criterion
            r = np.random.random()  # in [0,1[

            # acceptance criterion
            if r < np.exp(dlog / self._temp):
                # Save new diffusion profile (after MC step)
                model._df_bin[:] = df_bin_temp[:]

                # Save new coefficent vector
                model._df_coeff[:] = df_coeff_temp[:]

                # Save new likelihood
                self._log_like = log_like_try

                # Update MC statistics
                self._nacc_df_coeff[idx] += 1
                self._nacc_df += 1
                self._nacc_df_update += 1

    # def _mcmove_diffusion_radial(self,model):
    #     """This function do the MC move for the radial diffusion profile and
    #     adjust the coefficents of the model. A MC move in the parameter space
    #     is defined by
    #
    #     .. math::
    #
    #         a_{k,\\mathrm{new}}= a_{k} + \\Delta_\\text{MC} \\cdot (R - 0.5)
    #
    #     with :math:`\\Delta_\\text{MC}` as the MC step movewidth and :math:`R`
    #     as a random number between :math:`0` and :math:`1`. The choice of the
    #     coefficient is also made by determining a random number.
    #
    #     Parameters
    #     ----------
    #     model : Model
    #         Model object which set before with the model class
    #     """
    #     # Caclulate a random number to choose a random coefficient
    #     ## The first coefficent of the diffusion profile is fixed 0 all the time
    #     idx = np.random.randint(0,model._n_diff_radial)
    #
    #     # Calculate one new coefficient
    #     diff_radial_coeff_temp = copy.deepcopy(model._diff_radial_coeff)
    #     diff_radial_coeff_temp[idx] += self._delta_diff_radial * (np.random.random() - 0.5)
    #
    #     # Use the new coeff to calculate a new diffusion profile
    #     diff_radial_bin_temp = model._calc_profile(diff_radial_coeff_temp, model._diff_radial_basis)
    #
    #     # Calculate a new likelihood to check acceptance of the step
    #     log_like_try = self.log_likelihood_radial(model,diff_radial_bin_temp)
    #
    #     # Propagtor behavior (propagator is well behaved - :TODO: implement)
    #     if log_like_try is not None and not np.isnan(log_like_try):
    #         # Calculate different between new and old likelihood
    #         dlog = log_like_try - self._log_like_radial
    #
    #         # Calculate a random number for the acceptance criterion
    #         r = np.random.random()  #in [0,1[
    #
    #         # acceptance criterion
    #         if r < np.exp(dlog / self._temp):
    #             # Save new diffusion profile (after MC step)
    #             model._diff_radial_bin[:] = diff_radial_bin_temp[:]
    #
    #             # Save new coefficent vector
    #             model._diff_radial_coeff[:] = diff_radial_coeff_temp[:]
    #
    #             #Save new likelihood
    #             self._log_like_radial = log_like_try
    #
    #             # Update MC statistics
    #             self._nacc_diff_radial_coeff[idx] += 1
    #             self._nacc_diff_radial += 1
    #             self._nacc_diff_radial_update += 1

    def _update_movewidth_mc(self, imc, radial=False):
        """This function sets a new MC move width after a define number of MC
        steps :math:`n_\\text{MC,update}`. The new step width is estimate with

        .. math::

            \\Delta_\\text{MC} = \\exp \\left ( 0.1 \\cdot \\frac{n_\\text{accept}}{n_\\text{MC,update}} \\right)

        Parameters
        ----------
        imc : integer
            Current number of MC steps
        do_radial : bool
            True to calculate the radial diffusion
        """

        # If _num_mc_update>0 the move width update is on
        if self._num_mc_update > 0:

            # Update the move width
            if ((imc+1) % self._num_mc_update == 0) and not radial:
                self._delta_df *= np.exp(0.1 * (float(self._nacc_df_update) / self._num_mc_update - 0.3))
                self._delta_diff *= np.exp(0.1 * (float(self._nacc_diff_update) / self._num_mc_update - 0.3))

                self._nacc_df_update = 0
                self._nacc_diff_update = 0

            # if ((imc+1) % self._num_mc_update == 0) and radial:
            #     self._delta_diff_radial *= np.exp(0.1 * (float(self._nacc_diff_radial_update) / self._num_mc_update - 0.3))
            #     self._nacc_diff_radial_update = 0


    ###############
    # Rate matrix #
    ###############
    def _init_rate_matrix_pbc(self, bin_num, diff_bin, df_bin):
        """This function estimates the rate Matrix R for the current free energy
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
        bin_num : integer
            Number of bins
        diff_bin : list
            Ln diffusion profile over the bins
        df_bin : list
            Free energy profile over the bins

        Returns
        -------
        rate matrix : array
            Rate matrix for the current free energy and ln diffusion profile
        """

        # Initialize rate matrix
        rate = np.float64(np.zeros((bin_num, bin_num)))

        # Off-diagonal elements
        delta_df_bin = df_bin[1:]-df_bin[:-1]
        exp1 = diff_bin[:bin_num-1] - 0.5 * delta_df_bin
        exp2 = diff_bin[:bin_num-1] + 0.5 * delta_df_bin
        rate.ravel()[bin_num::bin_num+1] = np.exp(exp1)[:bin_num-1]
        rate.ravel()[1::bin_num+1] = np.exp(exp2)[:bin_num-1]

        # Corners and periodic boundary conditions
        rate[0, -1] = np.exp(diff_bin[-1]-0.5*(df_bin[0]-df_bin[-1]))
        rate[-1, 0] = np.exp(diff_bin[-1]-0.5*(df_bin[-1]-df_bin[0]))
        rate[0, 0] = - rate[1, 0] - rate[-1, 0]
        rate[-1, -1] = - rate[-2, -1] - rate[0, -1]

        # Diagonal elements
        for i in range(1, bin_num-1):
            rate[i, i] = - rate[i-1, i] - rate[i+1, i]

        return rate


    ##############
    # Likelihood #
    ##############
    def _log_likelihood_z(self, model,  temp=None):
        """This function estimate the likelihood of the current free energy or
        diffusion profile over the bins in a simulation box. It is used to
        calculated the diffusion in z-direction over z-coordinate. This
        likelihood is necessary to decide whether the MC step will be accepted.

        .. math::

            \\ln \ L = \\sum_{j \\rightarrow i} \\ln \\left ( \\left[ \\left( e^{\\mathbf{R}\\Delta_{ij}t_{\\alpha}} \\right)_{ij}\\right]^{N_{ij}(\\Delta_{ij}t_{\\alpha})} \\right)

        with :math:`R` as the rate matrix from :func:`_init_rate_matrix_pbc`,
        :math:`\\Delta_{ij}t_{\\alpha}` as the current lag time and
        :math:`N_{ij}(\\Delta_{ij}t_{\\alpha})` as the transition matrix.
        The transition matrix contains the numbers of all observed transition
        :math:`j \\rightarrow i` at a given lag time in a simulated trajectory.
        This matrix is sampled with the function :func:`poreana.sample.Sample.init_diffusion_mc` from
        a simulated trajectory at the beginning and depends on the lag time.

        Parameters
        ----------
        model : integer
            Model class set with the model function
        temp : list
            Profile which was adapted in the current MC (free energy or diffusion)

        Returns
        -------
        log_like : float
            Likelihood for the current profiles in a simulation box
        """

        # Calculate the current rate matrix for a trajectory with periodic boundary condition
        if temp is None:
            rate = self._init_rate_matrix_pbc(model._bin_num, model._diff_bin, model._df_bin)
        else:
            if self._choice > 0.5:
                rate = self._init_rate_matrix_pbc(model._bin_num,  temp, model._df_bin)
            else:
                rate = self._init_rate_matrix_pbc(model._bin_num, model._diff_bin, temp)

        # Calculate the current rate matrix for a trajectory with a reflected wall in z direction
        # elif not model._pbc:
        #     if temp is None:
        #         rate = self.init_rate_matrix_nopbc(model._bin_num, model._diff_bin, model._df_bin)
        #     else:
        #         if self._choice > 0.5:
        #             rate = self.init_rate_matrix_nopbc(model._bin_num,  temp, model._df_bin)
        #         else:
        #             rate = self.init_rate_matrix_nopbc(model._bin_num, model._diff_bin, temp)

        # Calculate the propagator
        propagator = sc.linalg.expm(self._len_step * model._dt * rate)

        # Calculate likelihood
        tiny = 1e-10
        mat = model._trans_mat[self._len_step] * np.log(propagator.clip(tiny))
        log_like = np.sum(mat)

        return log_like

    # def log_likelihood_radial(self, model, wrad):
    #     """This function estimate the likelihood of the current radial diffusion
    #     profile over the bins in a simulation box. This likelihood is necessary
    #     to decide whether the MC step will be accepted.
    #
    #     .. math::
    #          \\ln \\ L(M) = \\sum_{m,i \\leftarrow j} \\ln \\left( \\left[ \\Delta r \\sum_{\\alpha_k} 2 r_m \\frac{J_0(\\alpha_kr_m)}{s^2J_1^2(x_k)} [e^{(\\mathbf{R}-\\alpha_k^2D)\\Delta_{ij}t_{\\alpha}}]_{ij} \\right] ^{N_{m,ij}(\\Delta_{ij}t_{\\alpha})} \\right)
    #
    #
    #     with :math:`t_{i}` as the current lag time and :math:`N_{m,ij}(t_i)` as
    #     the radial transition matrix, :math:`J_0` as the roots of the Bessel
    #     function and :math:`J_1` as the 1st order Bessel first type in those
    #     zeros. The variable :math:`s` is the length where the bessel function
    #     going to zero and :math:`r_m` is the middle of the radial bin. The lag
    #     time is :math:`t_{i}` and :math:`D` is the current radial diffusion
    #     profile. The rate matrix :math:`R` from :func:`_init_rate_matrix_pbc`
    #     which constant over the radial diffuison determination and is calculated
    #     with the normal diffusion and the free energy profile is need. This the
    #     reason why the normal diffusion calculation is necessary for the radial
    #     diffusion calculation.
    #
    #     The transition matrix :math:`N_{m,ij}(t_i)` contains all observed number
    #     of transition of a particle being in z-bin i and radial bin m after a
    #     lag time t, given that it was in z-bin j and r = 0 at the start. This
    #     matrix is sampled with the function :func:`init_diffusion_mc` from a
    #     simulated trajectory at the beginning and depends on the lag time.
    #
    #     The bessel part
    #
    #     .. math::
    #         \\mathrm{bessel} = \\sum_{\\alpha_k} 2r_m \\frac{J_0(\\alpha_kr_m)}{s^2J_1^2(x_k)}
    #
    #     of the likelihood does not depend on the radial diffusion profile and is determined with the function :func:`setup_bessel_box`
    #
    #     The function :func:`log_likelihood_radial` determine the rate matrix part of the Likelihood
    #
    #     .. math::
    #         \\mathrm{rate}_{\\mathrm{radial}} = \\left [e^{\\left(\\mathbf{R}-\\alpha_k^2D\\right)\\Delta_{ij}t_{\\alpha}}\\right]_{ij}
    #
    #     multiple it with the results from the :func:`setup_bessel_box` and
    #     logarithmize the result. Afterwards the results are multipe with the
    #     transition matrix :math:`N_{m,ij}(\\Delta_{ij}t_{\\alpha})`. The
    #     likelihood thus obtained provides the acceptance criteria for the MC
    #     alogirthm and is returned by the function .
    #
    #     Parameters
    #     ----------
    #     model : obj
    #         Model class set with the model function
    #     wrad : list
    #         Radial diffusion profile which was adapted in the current MC
    #
    #     Returns
    #     -------
    #     log_like : float
    #         Likelihood for the current radial diffusion profile in a simulation box
    #     """
    #     # Initialize log_like
    #     log_like = np.float64(0.0)
    #     tiny = np.empty((model._bin_num_rad,model._bin_num,model._bin_num))
    #     tiny.fill(1.e-32) # lower bound of propagator (to avoid NaN's)
    #
    #     # Calculate the current rate matrix
    #     if model._pbc:
    #         rate = self._init_rate_matrix_pbc(model._bin_num, model._diff_bin, model._df_bin)
    #     else:
    #         rate = self.init_rate_matrix_nopbc(model._bin_num, model._diff_bin, model._df_bin)
    #
    #     # Calculate propagtor
    #     rmax = 5 #max(model._bins_radial) # in units [dr]
    #
    #     # initialize Arrays
    #     rate_l = np.zeros((model._bin_num,model._bin_num),dtype=np.float64)
    #     propagator = np.zeros((model._bin_num_rad,model._bin_num,model._bin_num),dtype=np.float64)
    #     trans = np.zeros((model._bin_num_rad,model._bin_num,model._bin_num),dtype=np.float64)
    #
    #     # set up sink term
    #     sink = np.zeros((model._bin_num),dtype=np.float64)
    #
    #     # loop over l (index of Bessel function)
    #     for l in range(self._lmax):
    #         # Calculate sink term for on bessel index (dimension r=z)
    #         sink = np.exp(wrad) * self._bessel0_zeros[l]**2 / rmax**2
    #
    #         # Calculate the rate matrix with sink term
    #         # Sink term substract from the diagonal of the rate matrix
    #         rate_l[:,:] = rate[:,:]
    #         rate_l.ravel()[::model._bin_num+1] -= sink
    #
    #         # Calculate the exp of the rate - sink term
    #         mat_exp = sc.linalg.expm(self._len_step * model._dt * rate_l)
    #
    #         # Calculate the propagator - likelihood per r bin and per z bin (rxzxz)
    #         for k in range(model._bin_num_rad):
    #             propagator[k,:,:] += self._bessels[l,k] * mat_exp[:,:]
    #
    #     # Log of the propagator
    #     ln_prop = np.log(model._bin_radial_width * np.maximum(propagator,tiny))
    #
    #     # Set trans matrix
    #     for i in range(model._bin_num_rad):
    #         trans[i,:,:] = model._trans_mat_radial[self._len_step][i][:,:]
    #
    #     # Calculate likelihood - every raidal Transition matrix (zxz) are
    #     # multiplying the every radial propergator (zxz)
    #     log_like = np.sum(trans[:,:,:] * ln_prop[:,:,:])
    #
    #     return log_like
    #
    # def setup_bessel_box(self,model):
    #     """This function set the zeros of the 0th order Bessel first type and
    #     the bessel function for the radial likelihood.
    #
    #     .. math::
    #         \\mathrm{bessel} = \\sum_{\\alpha_k} 2r_m \\frac{J_0(\\alpha_kr_m)}{s^2J_1^2(x_k)}
    #
    #     The bessel matrix has the dimension of lxr and contains the constant
    #     part of the radial likelihood.
    #
    #     Parameters
    #     ----------
    #     model : Model
    #         Model object which set with the model class
    #     """
    #
    #     # Zeros of the bessel funktions of the 0th bessel function (x-values)
    #     self._bessel0_zeros = sc.special.jn_zeros(0,self._lmax)
    #
    #     # 1st order Bessel function zeropoints - y value of the 1th bessel
    #     # function at zeropoints of 0th bessel function (x-values above)
    #     bessel1_inzeros = sc.special.j1(self._bessel0_zeros)
    #
    #     # Set max r and r_m vector
    #     rmax = max(model._bins_radial) # maximal radius -> propagtor becomes zero
    #     r = np.arange(model._bin_radial_width/2,max(model._bins_radial),model._bin_radial_width,dtype=np.float64) # Vektors in the middle of the radial bins
    #
    #     # set up Bessel functions
    #     self._bessels = np.zeros((self._lmax,model._bin_num_rad),dtype=np.float64)
    #
    #     # Determine the bessel part of the propogator
    #     for l in range(self._lmax):
    #         # Set bessel part of propagtor
    #         self._bessels[l,:] = 2 * r * sc.special.j0(r / rmax * self._bessel0_zeros[l]) / bessel1_inzeros[l]**2 / rmax**2
