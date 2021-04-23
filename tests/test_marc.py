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
    #def setUpClass(self):
        # folder = 'output'
        # pa.utils.mkdirp(folder)
        # pa.utils.mkdirp(folder+"/temp")
        # open(folder+"/temp.txt", 'a').close()
        #
        # for filename in os.listdir(folder):
        #     file_path = os.path.join(folder, filename)
        #     if os.path.isfile(file_path) or os.path.islink(file_path):
        #         os.unlink(file_path)
        #     elif os.path.isdir(file_path):
        #         shutil.rmtree(file_path)
        #
        # # Load molecule
        # mol = pms.Molecule("benzene", "BEN", inp="data/benzene.gro")
        #
        # # Sample
        # sample = pa.Sample("data/pore_system.obj", "data/traj.xtc", mol, is_nojump=False)
        # sample.init_diffusion_bin("output/diff.obj")
        # sample.sample(is_parallel=False)
        #
        # sample = pa.Sample("data/pore_system.obj", "data/traj_nojump.xtc", mol, is_nojump=True)
        # sample.init_diffusion_bin("output/diff.obj")
        # sample.sample(is_parallel=False)
        #
        # sample = pa.Sample("data/pore_system.obj", "data/traj.xtc", mol, is_nojump=False)
        # sample.init_diffusion_bin("output/diff_p.obj")
        # sample.sample(is_parallel=True)
        #
        # sample = pa.Sample("data/pore_system.obj", "data/traj_nojump.xtc", mol, is_nojump=True)
        # sample.init_diffusion_bin("output/diff_nj.obj")
        # sample.sample(is_parallel=True)


    #########
    # Utils #
    #########
    def test_sample(self):
        # self.skipTest("Temporary")
        #print()

        # Define molecules
        mol = pms.Molecule("benzene", "BEN", inp="data/benzene.gro")

        # Sanity checks
        sample = pa.Sample("data/pore_system_cylinder.obj", "data/traj_cylinder.xtc", mol)

        # Diffusion
        #sample.init_diffusion_bin("output/diff_2.obj")
        sample.init_diffusion_mc("output/diff.obj",bin_num=10, len_step=[10,20])
        sample.sample(is_parallel=False)
        a = pa.utils.load("output/diff.obj")
        c = a["data"]
        print(c)
        sample.init_diffusion_mc("output/diff_2.obj", bin_num=10,len_step=[10,20])
        sample.sample(is_parallel=True)
        b = pa.utils.load("output/diff_2.obj")
        d = b["data"]
        print(d[10])
        print(np.array_equal(c[10],d[10]))
        print(np.array_equal(c[20],d[20]))

    # def test_mc(self):
    #     # self.skipTest("Temporary")
    #     #print()
    #
    #     # Define molecules
    #     mol = pms.Molecule("benzene", "BEN", inp="data/benzene.gro")
    #
    #     # Sanity checks
    #     sample = pa.Sample("data/pore_system.obj", "data/traj_nojump.xtc", mol, is_nojump=True)
    #
    #     # Diffusion
    #     #sample.init_diffusion_bin("output/diff_2.obj")
    #     sample.init_diffusion_mc("output/diff.obj", len_step=[10,20,30,40])
    #     sample.sample(is_parallel=True)
    #
    #     model = pa.CosinusModel("output/diff.obj", 6, 10)
    #
    #     MC = pa.MC(model,10000,10000)
    #
    #     MC.do_mc_cycles(model,"output/diff_test_mc.obj")
    #
    #     diff = pa.post_process.diffusion("output/diff_test_mc.obj")
    #     plt.show()
    #     pa.post_process.diff_profile("output/diff_test_mc.obj")
    #     plt.show()
    #     pa.post_process.diffusion_pore("data/pore_system.obj","output/diff_test_mc.obj")
    #     plt.show()
    #     pa.post_process.plot_trans_mat("output/diff_test_mc.obj",10)
    #     plt.show()


    # def test_sample_trans(self):
    #     self.skipTest("Temporary")
    #
    #     # Set the desired step length and the box length
    #     len_step = [1,2,3,5,10,15,20,30,40]
    #     box_length = 5.0849
    #
    #     # Load molecule
    #     mol = pms.Molecule("oxygen", "O", inp="data/mcdiff/oxygen.gro")
    #
    #     # Calculate transition matrix for different step length
    #     md.pore.diffusion_mcdiff.sample_box_sim("data/mcdiff/traj_test.pdb", "data/mcdiff/traj_test.trr", "output/pores/diff_test_trans.obj", mol,len_step,box_length,2e-12,pbc=True)
    #
    #     # Load the output obj files and load the transition matrices
    #     obj = md.utils.load( "output/pores/diff_test_trans.obj")
    #     trans_list = obj["trans_mat"]
    #
    #     # Check if the calculated matrices are the same as in the files
    #     vec = []
    #     for i in len_step:
    #         vec.append(np.array_equal(trans_list[i],md.pore.diffusion_mcdiff.read_trans_mat_ref("data/mcdiff/trans_mat/transitions.nbins100.{}.pbc.dat".format(i))))
    #
    #     # Check if the code is correct
    #     self.assertEqual(vec, [True]*9)
    #
    # def test_init_profiles(self):
    #     self.skipTest("Temporary")
    #     len_step = [10]
    #     box_length = 5.0849
    #
    #     mol = pms.Molecule("oxygen", "O", inp="data/mcdiff/oxygen.gro")
    #     #self.skipTest("Temporary")
    #     md.pore.diffusion_mcdiff.sample_box_sim("data/mcdiff/traj_test.pdb", "data/mcdiff/traj_test.trr", "output/pores/diff_test.obj", mol,len_step, box_length)
    #
    #     # Set the cosinus model
    #     model = md.pore.mcdiff.CosinusModel("output/pores/diff_test.obj", 6, 10)
    #
    #     print(len(model._diff_bin),len(np.array([1.35261946] * model._bin_num)))
    #     print(np.array_equal(model._diff_bin, np.array([1.35261946] * model._bin_num)))
    #     print(np.array_equal(model._df_bin, np.array([0] * model._bin_num)))
    #
    #     self.assertEqual(np.array_equal(model._diff_bin, np.array([1.35261946] * model._bin_num)), True)
    #     self.assertEqual(np.array_equal(model._df_bin, np.array([0] * model._bin_num)), True)
    #
    #
    # def test_init_likelihood(self):
    #     self.skipTest("Temporary")
    #
    #     # Set the desired step length and the box length
    #     len_step = [10]
    #     box_length = 5.0849
    #
    #     # Load molecule
    #     mol = pms.Molecule("oxygen", "O", inp="data/mcdiff/oxygen.gro")
    #
    #     # Calculate transition matrix
    #     md.pore.diffusion_mcdiff.sample_box_sim("data/mcdiff/traj_test.pdb", "data/mcdiff/traj_test.trr", "output/pores/diff_test.obj", mol,len_step, box_length)
    #
    #     # Set the cosinus model
    #     model = md.pore.mcdiff.CosinusModel("output/pores/diff_test.obj", 6, 10)
    #
    #     # Set the MC class
    #     MC = md.pore.mcdiff.MC(model,1,1000,5000)
    #
    #     # Set the variable because this happen in the do_mc_cycles function -> not necessary to call to check the likelihood
    #     MC._len_step = 10
    #
    #     # Check if the initalize likelihood is correct
    #     self.assertEqual(MC.log_likelihood_box(model), -193331.16050932498)








if __name__ == '__main__':
    unittest.main(verbosity=2)
