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
        sample = pa.Sample("data/pore_system.obj", "data/traj_nojump.xtc", mol, is_nojump=True)

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

if __name__ == '__main__':
    unittest.main(verbosity=2)
