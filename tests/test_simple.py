import os
import sys

import copy
import shutil
import unittest

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import porems as pms
import poreana as pa


class UserModelCase(unittest.TestCase):
    #################
    # Remove Output #
    #################
    @classmethod
    def setUpClass(self):
        if os.path.isdir("tests"):
            os.chdir("tests")

        folder = 'output'
        pa.utils.mkdirp(folder)
        pa.utils.mkdirp(folder+"/temp")
        open(folder+"/temp.txt", 'a').close()

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

        # Load molecule
        mol_B = pms.Molecule(inp="data/benzene.gro")
        mol_W = pms.Molecule(inp="data/spc216.gro")
        mol_H = pms.Molecule(inp="data/heptane.gro")

        # Sample
        ## Single core
        sample = pa.Sample("data/pore_system_cylinder.obj", "data/traj_cylinder.xtc", mol_B)
        sample.init_density("output/dens_cyl_s.obj")
        sample.init_gyration("output/gyr_cyl_s.obj")
        sample.init_diffusion_bin("output/diff_cyl_s.obj")
        sample.sample(is_parallel=False)

        sample = pa.Sample("data/pore_system_slit.obj", "data/traj_slit.xtc", mol_W)
        sample.init_density("output/dens_slit.obj")
        sample.sample(is_parallel=False, is_pbc=False)

        sample = pa.Sample([6.00035, 6.00035, 19.09191], "data/traj_box.xtc", mol_H)
        sample.init_density("output/dens_box.obj")
        sample.init_gyration("output/gyr_box.obj")
        sample.sample(shift=[0, 0, 3.3], is_parallel=False)

        ## Parallel
        sample = pa.Sample("data/pore_system_cylinder.obj", "data/traj_cylinder.xtc", mol_B)
        sample.init_density("output/dens_cyl_p.obj")
        sample.init_gyration("output/gyr_cyl_p.obj")
        sample.init_diffusion_bin("output/diff_cyl_p.obj")
        sample.sample(is_parallel=True, is_pbc=False)


        # Sample MC Diffusion
        ## Single core (Pore System)
        sample = pa.Sample("data/pore_system_cylinder.obj", "data/traj_cylinder.xtc", mol_B)
        sample.init_diffusion_mc("output/diff_mc_cyl_s.obj", len_step=[1,2,5,10,20,30,40,50])
        sample.sample(is_parallel=False)

        ## Parallel (Box System)
        sample = pa.Sample("data/pore_system_cylinder.obj", "data/traj_cylinder.xtc", mol_B)
        sample.init_diffusion_mc("output/diff_mc_cyl_p.obj", len_step=[1,2,5,10,20,30,40,50])
        sample.sample(is_parallel=True, is_pbc=True)

        ## Test box system
        sample = pa.Sample([6.00035, 6.00035, 19.09191], "data/traj_box.xtc", mol_H)
        sample.init_diffusion_mc("output/diff_mc_box.obj", len_step=[1,2,5,10,20,30,40,50])
        sample.sample(shift=[0, 0, 3.3], is_parallel=False, is_pbc=True)




    #########
    # Utils #
    #########
    def test_utils(self):
        file_link = "output/test/test.txt"

        pa.utils.mkdirp("output/test")

        self.assertEqual(pa.utils.column([[1, 1, 1], [2, 2, 2]]), [[1, 2], [1, 2], [1, 2]])

        pa.utils.save([1, 1, 1], file_link)
        self.assertEqual(pa.utils.load(file_link), [1, 1, 1])

        self.assertEqual(round(pa.utils.mumol_m2_to_mols(3, 100), 4), 180.66)
        self.assertEqual(round(pa.utils.mols_to_mumol_m2(180, 100), 4), 2.989)
        self.assertEqual(round(pa.utils.mmol_g_to_mumol_m2(0.072, 512), 2), 0.14)
        self.assertEqual(round(pa.utils.mmol_l_to_mols(30, 1000), 4), 18.066)
        self.assertEqual(round(pa.utils.mols_to_mmol_l(18, 1000), 4), 29.8904)

        print()
        pa.utils.toc(pa.utils.tic(), message="Test", is_print=True)
        self.assertEqual(round(pa.utils.toc(pa.utils.tic(), is_print=True)), 0)


    ############
    # Geometry #
    ############
    def test_geometry(self):
        vec_a = [1, 1, 2]
        vec_b = [0, 3, 2]

        print()

        self.assertEqual(round(pa.geom.dot_product(vec_a, vec_b), 4), 7)
        self.assertEqual(round(pa.geom.length(vec_a), 4), 2.4495)
        self.assertEqual([round(x, 4) for x in pa.geom.vector(vec_a, vec_b)], [-1, 2, 0])
        self.assertIsNone(pa.geom.vector([0, 1], [0, 0, 0]))
        self.assertEqual([round(x, 4) for x in pa.geom.unit(vec_a)], [0.4082, 0.4082, 0.8165])
        self.assertEqual([round(x, 4) for x in pa.geom.cross_product(vec_a, vec_b)], [-4, -2, 3])
        self.assertEqual(round(pa.geom.angle(vec_a, vec_b), 4), 37.5714)


    ##########
    # Sample #
    ##########
    def test_sample(self):
        # self.skipTest("Temporary")
        print()

        # Define molecules
        mol = pms.Molecule("benzene", "BEN", inp="data/benzene.gro")
        mol2 = pms.Molecule("benzene", "BEN", inp="data/benzene.gro")
        mol2.add("H", [0, 0, 0])

        # Sanity checks
        pa.Sample("data/pore_system_cylinder.obj", "data/traj_cylinder.xtc", mol2)
        pa.Sample("data/pore_system_cylinder.obj", "data/traj_cylinder.xtc", mol, atoms=["C1"], masses=[1, 1])
        pa.Sample("data/pore_system_cylinder.obj", "data/traj_cylinder.xtc", mol).sample(shift=[1])

        # Diffusion
        sample = pa.Sample("data/pore_system_cylinder.obj", "data/traj_cylinder.xtc", mol, atoms=["C1"])
        sample.init_diffusion_bin("output/diff_np_s.obj", len_obs=3e-12)

        sample = pa.Sample([0, 0, 1], "data/traj_cylinder.xtc", mol, atoms=["C1"])
        sample.init_diffusion_bin("output/diff_box_test.obj", len_obs=3e-12)


    ##############
    # Adsorption #
    ##############
    def test_adsorption(self):
        # Calculate adsorption
        ads_s = pa.adsorption.calculate("output/dens_cyl_s.obj")
        ads_p = pa.adsorption.calculate("output/dens_cyl_p.obj")

        self.assertEqual(round(ads_s["conc"]["mumol_m2"], 2), 0.15)
        self.assertEqual(round(ads_s["num"]["in"], 2), 10.68)
        self.assertEqual(round(ads_p["conc"]["mumol_m2"], 2), 0.15)
        self.assertEqual(round(ads_p["num"]["in"], 2), 10.68)


    ###########
    # Density #
    ###########
    def test_density(self):
        # Calculate density
        dens_s = pa.density.calculate("output/dens_cyl_s.obj", target_dens=16)
        dens_p = pa.density.calculate("output/dens_cyl_p.obj", target_dens=16)

        self.assertEqual(round(dens_s["dens"]["in"], 3), 12.941)
        self.assertEqual(round(dens_s["dens"]["ex"], 3), 16.446)
        self.assertEqual(round(dens_p["dens"]["in"], 3), 12.941)
        self.assertEqual(round(dens_p["dens"]["ex"], 3), 16.446)

        dens_slit = pa.density.calculate("output/dens_slit.obj", target_dens=997)
        dens_box = pa.density.calculate("output/dens_box.obj")

        # Plot density
        plt.figure()
        pa.density.plot(dens_s, target_dens=0.146)
        pa.density.plot(dens_s, target_dens=0.146, is_mean=True)
        plt.savefig("output/density.pdf", format="pdf", dpi=1000)
        # plt.show()

        plt.figure()
        pa.density.plot(dens_slit, is_mean=True)
        plt.savefig("output/density_slit.pdf", format="pdf", dpi=1000)
        # plt.show()

        plt.figure()
        pa.density.plot(dens_s, intent="in")
        pa.density.plot(dens_p, intent="in")
        plt.legend(["Single core", "Parallel"])
        plt.savefig("output/density_in.pdf", format="pdf", dpi=1000)
        # plt.show()

        plt.figure()
        pa.density.plot(dens_s, intent="ex")
        pa.density.plot(dens_p, intent="ex")
        plt.legend(["Single core", "Parallel"])
        plt.savefig("output/density_ex.pdf", format="pdf", dpi=1000)
        # plt.show()

        plt.figure()
        pa.density.plot(dens_box, intent="ex")
        plt.savefig("output/density_box.pdf", format="pdf", dpi=1000)
        # plt.show()

        print()
        pa.density.plot(dens_s, intent="DOTA")


    #################
    # Bin Diffusion #
    #################
    def test_diffusion(self):
        # Bin diffusion
        plt.figure()
        pa.diffusion.bins("output/diff_cyl_s.obj")
        pa.diffusion.bins("output/diff_cyl_s.obj", is_norm=True)
        plt.savefig("output/diffusion_bins.pdf", format="pdf", dpi=1000)
        # plt.show()

        # CUI diffusion
        plt.figure()
        pa.diffusion.cui("output/diff_cyl_s.obj", is_fit=True)
        plt.savefig("output/diffusion_cui.pdf", format="pdf", dpi=1000)
        # plt.show()

        # Mean diffusion based on bins
        plt.figure()
        mean_s = pa.diffusion.mean("output/diff_cyl_s.obj", "output/dens_cyl_s.obj", is_check=True)
        plt.savefig("output/diff_mean_check.pdf", format="pdf", dpi=1000)
        mean_p = pa.diffusion.mean("output/diff_cyl_p.obj", "output/dens_cyl_p.obj")

        self.assertEqual(round(mean_s, 2), 1.13)
        self.assertEqual(round(mean_p, 2), 1.13)

    ################
    # MC Diffusion #
    ################

    # Test the entire mc diffusion method
    def test_mc_pore(self):
        # self.skipTest("Temporary")

        # Set Cosine Model for diffusion and energy profile
        model = pa.CosineModel("output/diff_mc_cyl_s.obj", 6, 10)

        # Set the MC class and options
        model._len_step = [10,20,30,40,50]
        MC = pa.MC(model,8000,1500,print_output=False)

        # Do the MC alogirthm
        MC.do_mc_cycles(model,"output/diff_test_mc.obj")

        # Plot diffusion coefficient over inverse lagtime
        plt.figure()
        diff, diff_mean, diff_table = pa.diffusion.diffusion_fit("output/diff_test_mc.obj")
        plt.savefig("output/diffusion_fit.svg", format="svg", dpi=1000)

        # Plot pore diffusion coefficient over inverse lagtime
        plt.figure()
        diff_pore, diff_mean_pore, diff_table = pa.diffusion.diffusion_pore_fit("data/pore_system_cylinder.obj","output/diff_test_mc.obj")
        plt.savefig("output/diffusion_pore_fit.svg", format="svg", dpi=1000)

        # Plot diffusion profile over box length
        plt.figure()
        pa.diffusion.diff_profile("output/diff_test_mc.obj", infty_profile = True)
        plt.savefig("output/diffusion_profile.svg", format="svg", dpi=1000)

        # Plot diffusion profile in the pore area
        plt.figure()
        pa.diffusion.diff_pore_profile("data/pore_system_cylinder.obj","output/diff_test_mc.obj", infty_profile = True)
        plt.savefig("output/diffusion_pore_profile.svg", format="svg", dpi=1000)

        # Plot free energy profile over box length
        plt.figure()
        pa.diffusion.df_profile("output/diff_test_mc.obj",[10])
        plt.savefig("output/energy_profile.svg", format="svg", dpi=1000)

        # Plot transition matrix as a heat map
        plt.figure()
        pa.diffusion.plot_trans_mat("output/diff_test_mc.obj",10)
        plt.savefig("output/transition_heatmap.svg", format="svg", dpi=1000)

        # Check if diffusion coefficient is in the range
        self.assertEqual(abs(diff - (1.6 * 10**-9) ) < 0.3 * 10**-9, True)
        self.assertEqual(abs(diff_pore - (1.2 * 10**-9) ) < 0.3 * 10**-9, True)

    # Test entire code for a box system
    def test_mc_box(self):
        # self.skipTest("Temporary")

        # Set Cosine Model for diffusion and energy profile
        model = pa.CosineModel("output/diff_mc_box.obj", 6, 10)

        # Set the MC class and options
        model._len_step = [10,20,30,40,50]
        MC = pa.MC(model,5000,2500,print_output=False)

        # Do the MC alogirthm
        MC.do_mc_cycles(model,"output/diff_test_mc_box.obj")

        # Plot diffusion coefficient over inverse lagtime
        plt.figure()
        diff, diff_mean, diff_table = pa.diffusion.diffusion_fit("output/diff_test_mc_box.obj")
        plt.savefig("output/diffusion_fit_box.svg", format="svg", dpi=1000)


        # Plot diffusion profile over box length
        plt.figure()
        pa.diffusion.diff_profile("output/diff_test_mc_box.obj", infty_profile = True)
        plt.savefig("output/diffusion_profile_box.svg", format="svg", dpi=1000)

        # Plot free energy profile over box length
        plt.figure()
        pa.diffusion.df_profile("output/diff_test_mc_box.obj",[10])
        plt.savefig("output/energy_profile_box.svg", format="svg", dpi=1000)

        # Plot transition matrix as a heat map
        plt.figure()
        pa.diffusion.plot_trans_mat("output/diff_test_mc_box.obj",10)
        plt.savefig("output/transition_heatmap_box.svg", format="svg", dpi=1000)

        # Check if diffusion coefficient is in the range
        self.assertEqual(abs(diff - (1.0 * 10**-8) ) < 0.3 * 10**-8, True)

    def test_print_out(self):
        # self.skipTest("Temporary")

        # Set Cosine Model for diffusion and energy profile
        model = pa.CosineModel("output/diff_mc_cyl_s.obj", 6, 10, print_output=True)

        # Set the MC class and options
        model._len_step = [10]
        MC = pa.MC(model,100,2000,print_output=True)

        # Do the MC alogirthm
        MC.do_mc_cycles(model,"output/diff_test_mc.obj")




    # Test parallelisation of transition matrix
    def test_sample_p_s(self):
        # self.skipTest("Temporary")

        # Load Transition matrix for single
        trans = pa.utils.load("output/diff_mc_cyl_s.obj")
        trans_s = trans["data"]

        # Load Transition matrix for parallel
        trans_2 = pa.utils.load("output/diff_mc_cyl_p.obj")
        trans_p = trans_2["data"]

        list = []
        for i in [1,2,5,10,20,30,40,50]:
            list.append(np.array_equal(trans_s[i],trans_p[i]))

        # Check is parallelisation correct
        self.assertEqual(list, [True]*8)

    # Test sampling of the transition matrix
    def test_sample_sample_trans(self):
        # self.skipTest("Temporary")

        # Load molecule
        mol_B = pms.Molecule(inp="data/benzene.gro")

        # Test sampling
        sample = pa.Sample("data/pore_system_cylinder.obj", "data/traj_cylinder.xtc", mol_B)
        sample.init_diffusion_mc("output/sampling_test.obj", len_step=[1,2,5,10,20,30,40,50])
        sample.sample(is_parallel=False)

        # Load Transition matrix for single
        trans = pa.utils.load("data/trans_check.obj")
        trans_s = trans["data"]

        # Load Transition matrix for parallel
        trans_2 = pa.utils.load("output/sampling_test.obj")
        trans_p = trans_2["data"]

        list = []
        for i in [1,2,5,10,20,30,40,50]:
            list.append(np.array_equal(trans_s[i],trans_p[i]))

        # Check is parallelisation correct
        self.assertEqual(list, [True]*8)

    # Test initalize likelihood
    def test_init_likelihood(self):
        # self.skipTest("Temporary")

        # Set the cosinus model
        model = pa.CosineModel("output/diff_mc_cyl_s.obj", 6, 10)

        # Set the MC class
        MC = pa.MC(model)

        # Set the variable because this happen in the do_mc_cycles function -> not necessary to call to check the likelihood and Check if the initalize likelihood is correct
        MC._len_step = 1
        self.assertEqual(abs(MC.log_likelihood_z(model)-(-128852.33005868513) < 10**-5),True)

        # Set the variable because this happen in the do_mc_cycles function -> not necessary to call to check the likelihood and Check if the initalize likelihood is correct
        MC._len_step = 2
        self.assertEqual(abs(MC.log_likelihood_z(model)-(-165354.76731180004) < 10**-5),True)

        # Set the variable because this happen in the do_mc_cycles function -> not necessary to call to check the likelihood and Check if the initalize likelihood is correct
        MC._len_step = 10
        self.assertEqual(abs(MC.log_likelihood_z(model)-(-258946.70553844847) < 10**-5),True)


    # Check initial profiles
    def test_init_profiles(self):
        # self.skipTest("Temporary")

        # Check cosine model
        # Set the cosine model
        model = pa.CosineModel("output/diff_mc_cyl_s.obj", 6, 10)

        # Check if the initialized profiles are corret
        self.assertEqual(np.array_equal(np.round(model._diff_bin,3), np.array([-1.394] * model._bin_num)), True)
        self.assertEqual(np.array_equal(model._df_bin, np.array([0] * model._bin_num)), True)

        # Set the step model
        model = pa.StepModel("output/diff_mc_cyl_s.obj", 6, 10)

        # Check if the initialized profiles are corret
        self.assertEqual(np.array_equal(np.round(model._diff_bin,3), np.array([-1.394] * model._bin_num)), True)
        self.assertEqual(np.array_equal(model._df_bin, np.array([0] * model._bin_num)), True)

    # Test input/output tables
    def test_outputs(self):
        # self.skipTest("Temporary")

        # Check tables
        pa.diffusion.print_model_inputs("data/check_tables.obj",print_con=False)
        pa.diffusion.print_mc_inputs("data/check_tables.obj",print_con=False)
        pa.diffusion.print_statistics_mc("data/check_tables.obj",print_con=False)
        pa.diffusion.print_coeff("data/check_tables.obj",print_con=False)

        pa.diffusion.print_model_inputs("data/check_tables.obj",print_con=True)
        pa.diffusion.print_mc_inputs("data/check_tables.obj",print_con=True)
        pa.diffusion.print_statistics_mc("data/check_tables.obj",print_con=True)
        pa.diffusion.print_coeff("data/check_tables.obj",print_con=True)

        # Check output which is not coveraged by the entire MC test
        pa.diffusion.diff_pore_profile("data/pore_system_cylinder.obj","data/check_tables.obj", infty_profile = False)
        pa.diffusion.diff_profile("data/check_tables.obj", infty_profile = False)
        pa.diffusion.df_profile("data/check_tables.obj")

        # Check kwargs of transition heatmap
        kwargs = {"vmin":0,"vmax":0.5, "xticklabels":30, "yticklabels":30,"cbar":True,"square":False}
        pa.diffusion.plot_trans_mat("output/diff_mc_cyl_s.obj",10,kwargs)

    # Test currently only one can be initialize
    def test_init_bin_mc(self):
        # self.skipTest("Temporary")

        # Load molecule
        mol_B = pms.Molecule(inp="data/benzene.gro")

        # Test bin -> MC
        sample = pa.Sample("data/pore_system_cylinder.obj", "data/traj_cylinder.xtc", mol_B)
        sample.init_diffusion_bin("output/test.obj")
        sample.init_diffusion_mc("output/test.obj")

        # Test MC -> Bin
        sample = pa.Sample("data/pore_system_cylinder.obj", "data/traj_cylinder.xtc", mol_B)
        sample.init_diffusion_mc("output/test.obj")
        sample.init_diffusion_bin("output/test.obj")


    ############
    # Gyration #
    ############
    def test_gyration(self):
        # self.skipTest("Temporary")

        # Plot gyration radius
        plt.figure()
        mean_s = pa.gyration.plot("output/gyr_cyl_s.obj", "output/dens_cyl_s.obj", is_mean=True)
        plt.savefig("output/gyration_s.pdf", format="pdf", dpi=1000)
        plt.figure()
        mean_p = pa.gyration.plot("output/gyr_cyl_p.obj", "output/dens_cyl_p.obj")
        plt.savefig("output/gyration_p.pdf", format="pdf", dpi=1000)

        self.assertEqual(round(mean_s["in"], 2), 0.12)
        self.assertEqual(round(mean_s["ex"], 2), 0.15)
        self.assertEqual(round(mean_p["in"], 2), 0.12)
        self.assertEqual(round(mean_p["ex"], 2), 0.15)

        plt.figure()
        pa.gyration.plot("output/gyr_cyl_s.obj", "output/dens_cyl_s.obj", intent="in")
        pa.gyration.plot("output/gyr_cyl_p.obj", "output/dens_cyl_p.obj", intent="in")
        plt.legend(["Single core", "Parallel"])
        plt.savefig("output/gyration_in.pdf", format="pdf", dpi=1000)

        plt.figure()
        pa.gyration.plot("output/gyr_cyl_s.obj", "output/dens_cyl_s.obj", intent="ex")
        pa.gyration.plot("output/gyr_cyl_p.obj", "output/dens_cyl_p.obj", intent="ex")
        plt.legend(["Single core", "Parallel"])
        plt.savefig("output/gyration_ex.pdf", format="pdf", dpi=1000)

        plt.figure()
        pa.gyration.plot("output/gyr_box.obj", "output/dens_box.obj", intent="ex")
        plt.savefig("output/gyration_box.pdf", format="pdf", dpi=1000)

        print()
        pa.gyration.plot("output/gyr_cyl_s.obj", "output/dens_cyl_s.obj", intent="DOTA")


if __name__ == '__main__':
    unittest.main(verbosity=2)
