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




    #########
    # Utils #
    #########
    def test_dems(self):
        mol_W = pms.Molecule(inp="data/tip4p2005.gro")
        mol_W.set_masses([15.9994,1.0079,1.0079,0])


        # Sample
        ## Single core
        sample = pa.Sample("data/pore.yml", "data/traj_test.xtc", mol_W)
        sample.init_density("output/dens_cyl_s.h5", remove_pore_from_res=False)
        sample.sample(is_parallel=False)
        dens_s = pa.density.bins("output/dens_cyl_s.h5", target_dens=16)
        pa.density.bins_plot(dens_s, target_dens=0.146, is_mean=True)

if __name__ == '__main__':
    unittest.main(verbosity=2)
