import os
import sys

import copy
import shutil
import unittest

import numpy as np
import pandas as pd
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

        sample = pa.Sample("data/pore_system_cylinder.obj", "data/traj_cylinder.xtc", mol_B)
        sample.init_diffusion_mc("output/diff_mc_cyl_s.obj", len_step=[1,2,5,10,20,30,40,50])
        sample.sample(is_parallel=False)

        sample = pa.Sample([6.00035, 6.00035, 19.09191], "data/traj_box.xtc", mol_H)
        sample.init_diffusion_mc("output/diff_mc_box.obj", len_step=[1,2,5,10,20,30,40,50])
        sample.sample(shift=[0, 0, 3.3], is_parallel=False, is_pbc=True)

        ## Parallel
        sample = pa.Sample("data/pore_system_cylinder.obj", "data/traj_cylinder.xtc", mol_B)
        sample.init_density("output/dens_cyl_p.obj")
        sample.init_gyration("output/gyr_cyl_p.obj")
        sample.init_diffusion_bin("output/diff_cyl_p.obj")
        sample.sample(is_parallel=True, is_pbc=False)

        sample = pa.Sample("data/pore_system_cylinder.obj", "data/traj_cylinder.xtc", mol_B)
        sample.init_diffusion_mc("output/diff_mc_cyl_p.obj", len_step=[1,2,5,10,20,30,40,50])
        sample.sample(is_parallel=True, is_pbc=True)


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


        # Test the error output if bin and MC diffusion calculation is initialized
        # Test bin -> MC
        sample = pa.Sample("data/pore_system_cylinder.obj", "data/traj_cylinder.xtc", mol)
        sample.init_diffusion_bin("output/test.obj")
        sample.init_diffusion_mc("output/test.obj")

        # Test MC -> Bin
        sample = pa.Sample("data/pore_system_cylinder.obj", "data/traj_cylinder.xtc", mol)
        sample.init_diffusion_mc("output/test.obj")
        sample.init_diffusion_bin("output/test.obj")


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
        dens_s = pa.density.bins("output/dens_cyl_s.obj", target_dens=16)
        dens_p = pa.density.bins("output/dens_cyl_p.obj", target_dens=16)

        self.assertEqual(round(dens_s["dens"]["in"], 3), 12.941)
        self.assertEqual(round(dens_s["dens"]["ex"], 3), 16.446)
        self.assertEqual(round(dens_p["dens"]["in"], 3), 12.941)
        self.assertEqual(round(dens_p["dens"]["ex"], 3), 16.446)

        dens_slit = pa.density.bins("output/dens_slit.obj", target_dens=997)
        dens_box = pa.density.bins("output/dens_box.obj")

        # Plot density
        plt.figure()
        pa.density.bins_plot(dens_s, target_dens=0.146)
        pa.density.bins_plot(dens_s, target_dens=0.146, is_mean=True)
        plt.savefig("output/density.pdf", format="pdf", dpi=1000)
        # plt.show()

        plt.figure()
        pa.density.bins_plot(dens_slit, is_mean=True)
        plt.savefig("output/density_slit.pdf", format="pdf", dpi=1000)
        # plt.show()

        plt.figure()
        pa.density.bins_plot(dens_s, intent="in")
        pa.density.bins_plot(dens_p, intent="in")
        plt.legend(["Single core", "Parallel"])
        plt.savefig("output/density_in.pdf", format="pdf", dpi=1000)
        # plt.show()

        plt.figure()
        pa.density.bins_plot(dens_s, intent="ex")
        pa.density.bins_plot(dens_p, intent="ex")
        plt.legend(["Single core", "Parallel"])
        plt.savefig("output/density_ex.pdf", format="pdf", dpi=1000)
        # plt.show()

        plt.figure()
        pa.density.bins_plot(dens_box, intent="ex")
        plt.savefig("output/density_box.pdf", format="pdf", dpi=1000)
        # plt.show()

        print()
        pa.density.bins_plot(dens_s, intent="DOTA")


    ############
    # Gyration #
    ############
    def test_gyration(self):
        # Plot gyration radius
        plt.figure()
        mean_s = pa.gyration.bins_plot("output/gyr_cyl_s.obj", "output/dens_cyl_s.obj", is_mean=True)
        plt.savefig("output/gyration_s.pdf", format="pdf", dpi=1000)
        plt.figure()
        mean_p = pa.gyration.bins_plot("output/gyr_cyl_p.obj", "output/dens_cyl_p.obj")
        plt.savefig("output/gyration_p.pdf", format="pdf", dpi=1000)

        self.assertEqual(round(mean_s["in"], 2), 0.12)
        self.assertEqual(round(mean_s["ex"], 2), 0.15)
        self.assertEqual(round(mean_p["in"], 2), 0.12)
        self.assertEqual(round(mean_p["ex"], 2), 0.15)

        plt.figure()
        pa.gyration.bins_plot("output/gyr_cyl_s.obj", "output/dens_cyl_s.obj", intent="in")
        pa.gyration.bins_plot("output/gyr_cyl_p.obj", "output/dens_cyl_p.obj", intent="in")
        plt.legend(["Single core", "Parallel"])
        plt.savefig("output/gyration_in.pdf", format="pdf", dpi=1000)

        plt.figure()
        pa.gyration.bins_plot("output/gyr_cyl_s.obj", "output/dens_cyl_s.obj", intent="ex")
        pa.gyration.bins_plot("output/gyr_cyl_p.obj", "output/dens_cyl_p.obj", intent="ex")
        plt.legend(["Single core", "Parallel"])
        plt.savefig("output/gyration_ex.pdf", format="pdf", dpi=1000)

        plt.figure()
        pa.gyration.bins_plot("output/gyr_cyl_s.obj", "output/dens_cyl_s.obj", intent="in", is_norm=True)
        pa.gyration.bins_plot("output/gyr_cyl_s.obj", "output/dens_cyl_s.obj", intent="ex", is_norm=True)
        plt.legend(["Interior", "Exterior"])
        plt.savefig("output/gyration_norm.pdf", format="pdf", dpi=1000)

        plt.figure()
        pa.gyration.bins_plot("output/gyr_box.obj", "output/dens_box.obj", intent="ex")
        plt.savefig("output/gyration_box.pdf", format="pdf", dpi=1000)

        print()
        pa.gyration.bins_plot("output/gyr_cyl_s.obj", "output/dens_cyl_s.obj", intent="DOTA")


    #################
    # Bin Diffusion #
    #################
    def test_diffusion_bin(self):
        # Bin diffusion
        plt.figure()
        pa.diffusion.bins_plot(pa.diffusion.bins("output/diff_cyl_s.obj"))
        pa.diffusion.bins_plot(pa.diffusion.bins("output/diff_cyl_s.obj", is_norm=True))
        plt.savefig("output/diffusion_bins.pdf", format="pdf", dpi=1000)
        # plt.show()

        # CUI diffusion
        plt.figure()
        pa.diffusion.cui("output/diff_cyl_s.obj", is_fit=True)
        plt.savefig("output/diffusion_cui.pdf", format="pdf", dpi=1000)
        # plt.show()

        # Mean diffusion based on bins
        plt.figure()
        mean_s = pa.diffusion.mean(pa.diffusion.bins("output/diff_cyl_s.obj"), pa.density.bins("output/dens_cyl_s.obj"), is_check=True)
        plt.savefig("output/diff_mean_check.pdf", format="pdf", dpi=1000)
        mean_p = pa.diffusion.mean(pa.diffusion.bins("output/diff_cyl_p.obj"), pa.density.bins("output/dens_cyl_p.obj"))

        self.assertEqual(round(mean_s, 2), 1.13)
        self.assertEqual(round(mean_p, 2), 1.13)


    ################
    # MC Diffusion #
    ################

    # Test model class
    def test_diffusion_mc_model(self):
        # self.skipTest("Temporary")

        # Check cosine model
        model = pa.CosineModel("output/diff_mc_cyl_s.obj", 6, 10)

        # Check if the initialized profiles are corret
        self.assertEqual(np.array_equal(np.round(model._diff_bin,3), np.array([-1.394] * model._bin_num)), True)
        self.assertEqual(np.array_equal(model._df_bin, np.array([0] * model._bin_num)), True)

        # Check step model
        model = pa.StepModel("output/diff_mc_cyl_s.obj", 6, 10)

        # Check if the initialized profiles are corret
        self.assertEqual(np.array_equal(np.round(model._diff_bin,3), np.array([-1.394] * model._bin_num)), True)
        self.assertEqual(np.array_equal(model._df_bin, np.array([0] * model._bin_num)), True)


    # Test MC class
    def test_diffusion_mc_mc(self):
        # self.skipTest("Temporary")

        # Pore diffusion
        # Set Cosine Model for diffusion and energy profile
        model = pa.CosineModel("output/diff_mc_cyl_s.obj", 6, 10)

        # Set the variable because this happen in the do_mc_cycles function -> not necessary to call to check the likelihood and Check if the initalize likelihood is correct
        pa.MC._len_step = 1
        self.assertEqual(round(pa.MC()._log_likelihood_z(model),2),-128852.33)

        # Set the variable because this happen in the do_mc_cycles function -> not necessary to call to check the likelihood and Check if the initalize likelihood is correct
        pa.MC._len_step = 2
        self.assertEqual(round(pa.MC()._log_likelihood_z(model),2),-165354.77)

        # Set the variable because this happen in the do_mc_cycles function -> not necessary to call to check the likelihood and Check if the initalize likelihood is correct
        pa.MC._len_step = 10
        self.assertEqual(round(pa.MC()._log_likelihood_z(model),2),-258946.71)

        # Set len_step for MC run test
        model._len_step = [10,20,30,40]

        #### Test Single ####
        # Do the MC alogirthm
        pa.MC().run(model,"output/diff_test_mc.obj", nmc_eq=8000, nmc=2000, print_output=False, is_parallel=False)

        # Plot diffusion coefficient over inverse lagtime
        plt.figure()
        diff, diff_mean, diff_table = pa.diffusion.mc_fit("output/diff_test_mc.obj")
        plt.savefig("output/mc_fit.pdf", format="pdf", dpi=1000)

        # Plot pore diffusion coefficient over inverse lagtime
        plt.figure()
        diff_pore, diff_mean_pore, diff_table = pa.diffusion.mc_fit("output/diff_test_mc.obj", section="is_pore")
        plt.savefig("output/mc_fit_pore.pdf", format="pdf", dpi=1000)

        # Check if diffusion coefficient is in the range
        self.assertEqual(abs(diff - (1.6 * 10**-9) ) < 0.3 * 10**-9, True)
        self.assertEqual(abs(diff_pore - (1.2 * 10**-9) ) < 0.3 * 10**-9, True)

        #### Test Parallel ####
        # Do the MC alogirthm
        pa.MC().run(model,"output/diff_test_mc.obj", nmc_eq=8000, nmc=2000, print_output=False, is_parallel=True)

        # Plot diffusion coefficient over inverse lagtime
        plt.figure()
        diff, diff_mean, diff_table = pa.diffusion.mc_fit("output/diff_test_mc.obj")
        plt.savefig("output/mc_fit.pdf", format="pdf", dpi=1000)

        # Plot pore diffusion coefficient over inverse lagtime
        plt.figure()
        diff_pore, diff_mean_pore, diff_table = pa.diffusion.mc_fit("output/diff_test_mc.obj", section="is_pore")
        plt.savefig("output/mc_fit_pore.pdf", format="pdf", dpi=1000)

        # Check if diffusion coefficient is in the range
        self.assertEqual(abs(diff - (1.6 * 10**-9) ) < 0.3 * 10**-9, True)
        self.assertEqual(abs(diff_pore - (1.2 * 10**-9) ) < 0.3 * 10**-9, True)

        # Test MC output
        # Set Step Model for diffusion and energy profile
        model = pa.StepModel("output/diff_mc_cyl_s.obj", 6, 10, print_output=True)

        # Set Cosine Model for diffusion and energy profile
        model = pa.CosineModel("output/diff_mc_cyl_s.obj", 6, 10, print_output=True)

        # Set the MC class and options
        model._len_step = [10]

        # Do the MC alogirthm
        pa.MC().run(model,"output/diff_test_mc.obj", nmc_eq=8000, nmc=2000, print_output=True, is_parallel=False)


    def test_diffusion_mc_box(self):
        # Box diffusion
        # self.skipTest("Temporary")

        # Set Cosine Model for diffusion and energy profile
        model = pa.CosineModel("output/diff_mc_box.obj", 6, 10)

        # Set the MC class and options
        model._len_step = [10,20,30,40,50]

        # Do the MC alogirthm
        pa.MC().run(model,"output/diff_test_mc_box.obj", nmc_eq=1000, nmc=2000, print_output=False, is_parallel=False)

        # Plot diffusion coefficient over inverse lagtime
        plt.figure()
        diff, diff_mean, diff_table = pa.diffusion.mc_fit("output/diff_test_mc_box.obj")
        plt.savefig("output/diffusion_fit_box.pdf", format="pdf", dpi=1000)

        # Check if diffusion coefficient is in the range
        self.assertEqual(abs(diff - (1.3 * 10**-8) ) < 0.3 * 10**-8, True)


    ##########
    # :TODO: # - Titel ist nichts aussagend und sowas wie drüber, auch woanders testen und wenn, dann auch ergebnisse vergleichen und nicht ob es durchgelaufen ist
    ##########
    def test_parallel_sample(self):
        # self.skipTest("Temporary")

        # Load Transition matrix for single
        trans = pa.utils.load("output/diff_mc_cyl_s.obj")
        trans_s = trans["data"]

        # Load Transition matrix for parallel
        trans_2 = pa.utils.load("output/diff_mc_cyl_p.obj")
        trans_p = trans_2["data"]

        # Test if transition matrix of single and parallel calculation is the same
        list = []
        for i in [1,2,5,10,20,30,40,50]:
            list.append(np.array_equal(trans_s[i],trans_p[i]))

        # Check is parallelisation correct
        self.assertEqual(list, [True]*8)


    ###############
    # Free Energy #
    ###############
    def test_freeenergy_mc(self):
        # Plot free energy profile over box length
        plt.figure()
        pa.freeenergy.mc_profile("output/diff_test_mc.obj")
        plt.savefig("output/energy_profile.pdf", format="pdf", dpi=1000)


    ##########
    # Tables #
    ##########
    def test_tables(self):
        # Check tables
        pa.tables.mc_model("data/check_output.obj", print_con=False)
        pa.tables.mc_model("data/box_output.obj", print_con=False)
        pa.tables.mc_inputs("data/check_output.obj", print_con=False)
        pa.tables.mc_statistics("data/check_output.obj", print_con=False)
        pa.tables.mc_lag_time("data/check_output.obj", print_con=False)

        pa.tables.mc_model("data/check_output.obj", print_con=True)
        pa.tables.mc_inputs("data/check_output.obj", print_con=True)
        pa.tables.mc_statistics("data/check_output.obj", print_con=True)
        pa.tables.mc_lag_time("data/check_output.obj", print_con=True)

    def test_diffusion_output(self):

        # Check output which is not coveraged by the entire MC test
        # Check diffusion profile function
        pa.diffusion.mc_profile("data/check_output.obj", len_step=[10,20,30,40], infty_profile = False)
        pa.diffusion.mc_profile("data/check_output.obj", len_step=[10,20,30,40], section = "is_pore", infty_profile = True)
        pa.diffusion.mc_profile("data/check_output.obj", len_step=[10,20,30,40], section = "is_res", infty_profile = True)
        pa.diffusion.mc_profile("data/check_output.obj", section = [1,10], infty_profile = True)

        # Check diffusion fitting function
        pa.diffusion.mc_fit("data/check_output.obj", section = "is_pore")
        pa.diffusion.mc_fit("data/check_output.obj", section = "is_res")
        pa.diffusion.mc_fit("data/check_output.obj", section=[0,10])



        # Check transition matrix heatmap
        pa.diffusion.mc_trans_mat("data/check_output.obj",10)
        pa.diffusion.mc_trans_mat("data/check_output_sample.obj",10)


        # Check if box not pore system
        pa.diffusion.mc_fit("data/box_output.obj", section = "is_pore")
        pa.diffusion.mc_fit("data/box_output.obj", section = "is_res")
        pa.diffusion.mc_profile("data/box_output.obj", section = "is_pore")
        pa.diffusion.mc_profile("data/box_output.obj", section = "is_res")

if __name__ == '__main__':
    unittest.main(verbosity=2)
