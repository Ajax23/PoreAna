import os
import sys

import copy
import shutil
import unittest

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import porems as pms
import poreana as pa

import numpy as np

class UserModelCase(unittest.TestCase):
    #################
    # Remove Output #
    #################
    @classmethod
    def setUpClass(self):
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
        mol = pms.Molecule("benzene", "BEN", inp="data/benzene.gro")
        mol_W = pms.Molecule("water", "SOL", inp="data/spc216.gro")

        # Sample
        ## Single core
        sample = pa.Sample("data/pore_system_cylinder.obj", "data/traj_cylinder.xtc", mol)
        sample.init_density("output/dens_cyl_s.obj")
        sample.init_gyration("output/gyr_cyl_s.obj")
        # sample.init_diffusion_bin("output/diff_cyl_s.obj")
        sample.init_diffusion_mc("output/diff_mc_cyl_s.obj", len_step=[1,2,5,10,20,30,40,50])
        sample.sample(is_parallel=False, is_pbc=True)

        # sample = pa.Sample("data/pore_system_slit.obj", "data/traj_slit.xtc", mol_W)
        # sample.init_density("output/dens_slit.obj")
        # sample.sample(is_parallel=False, is_pbc=False)

        ## Parallel
        sample = pa.Sample("data/pore_system_cylinder.obj", "data/traj_cylinder.xtc", mol)
        sample.init_density("output/dens_cyl_p.obj")
        sample.init_gyration("output/gyr_cyl_p.obj")
        # sample.init_diffusion_bin("output/diff_cyl_p.obj")
        sample.init_diffusion_mc("output/diff_mc_cyl_p.obj", len_step=[1,2,5,10,20,30,40,50])
        sample.sample(is_parallel=True, is_pbc=True)


    #########
    # Utils #
    #########
    # Test the entire mc diffusion method
    def test_mc(self):
        #self.skipTest("Temporary")

        # Set Cosine Model for diffusion and energy profile
        model = pa.CosineModel("output/diff_mc_cyl_s.obj", 6, 10)

        # Set the MC class and options
        model._len_step = [10,20,30,40,50]
        MC = pa.MC(model,20000,20000)

        # Do the MC alogirthm
        MC.do_mc_cycles(model,"output/diff_test_mc.obj")

        # Plot diffusion coefficient over inverse lagtime
        plt.figure()
        diff, diff_table = pa.diffusion.diffusion_fit("output/diff_test_mc.obj")
        plt.savefig("output/diffusion_fit.svg", format="svg", dpi=1000)

        # Plot diffusion profile over box length
        plt.figure()
        pa.diffusion.diff_profile("output/diff_test_mc.obj", infty_profile = True)
        plt.savefig("output/diffusion_profile.svg", format="svg", dpi=1000)

        # Plot free energy profile over box length
        plt.figure()
        pa.diffusion.df_profile("output/diff_test_mc.obj",[10])
        plt.savefig("output/energy_profile.svg", format="svg", dpi=1000)

        # Plot transition matrix as a heat map
        plt.figure()
        pa.diffusion.plot_trans_mat("output/diff_test_mc.obj",10)
        plt.savefig("output/transition_heatmap.svg", format="svg", dpi=1000)

        # Check if diffusion coefficient is in the range
        self.assertEqual(abs(diff - (1.4 * 10**-9) ) < 0.3 * 10**-9, True)

    # Test parallelisation of transition matrix
    def test_sample_p_s(self):
        #self.skipTest("Temporary")

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
    def test_sample(self):
        #self.skipTest("Temporary")

        # Load Transition matrix for single
        trans = pa.utils.load("data/trans_check.obj")
        trans_s = trans["data"]

        # Load Transition matrix for parallel
        trans_2 = pa.utils.load("output/diff_mc_cyl_s.obj")
        trans_p = trans_2["data"]

        list = []
        for i in [1,2,5,10,20,30,40,50]:
            list.append(np.array_equal(trans_s[i],trans_p[i]))

        # Check is parallelisation correct
        self.assertEqual(list, [True]*8)

    # Test initalize likelihood
    def test_init_likelihood(self):
        #self.skipTest("Temporary")

        # Set the cosinus model
        model = pa.CosineModel("output/diff_mc_cyl_s.obj", 6, 10)

        # Set the MC class
        MC = pa.MC(model)

        # Set the variable because this happen in the do_mc_cycles function -> not necessary to call to check the likelihood and Check if the initalize likelihood is correct
        MC._len_step = 1
        self.assertEqual(MC.log_likelihood_box(model),  -128852.33005868513)

        # Set the variable because this happen in the do_mc_cycles function -> not necessary to call to check the likelihood and Check if the initalize likelihood is correct
        MC._len_step = 2
        self.assertEqual(MC.log_likelihood_box(model),  -165354.76731180004)

        # Set the variable because this happen in the do_mc_cycles function -> not necessary to call to check the likelihood and Check if the initalize likelihood is correct
        MC._len_step = 10
        self.assertEqual(MC.log_likelihood_box(model),  -258946.70553844847)


    # Check initial profiles
    def test_init_profiles(self):
        #self.skipTest("Temporary")

        # Set the cosinus model
        model = pa.CosineModel("output/diff_mc_cyl_s.obj", 6, 10)

        # Check if the initialized profiles are corret
        self.assertEqual(np.array_equal(model._diff_bin, np.array([-1.3937803336775594] * model._bin_num)), True)
        self.assertEqual(np.array_equal(model._df_bin, np.array([0] * model._bin_num)), True)

if __name__ == '__main__':
    unittest.main(verbosity=2)
