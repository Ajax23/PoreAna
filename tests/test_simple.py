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
        mol = pms.Molecule("spc216", "SOL", inp="data/spc216.gro")

        # Sample density
        pa.sample.density("data/pore_system.obj", "data/traj.xtc", "output/dens.obj", mol)

        # Sample diffusion
        pa.sample.diffusion_bin("data/pore_system.obj", "data/traj.xtc", "output/diff.obj", mol, len_obs=4e-12)


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

        # Load molecule
        mol = pms.Molecule("spc216", "SOL", inp="data/spc216.gro")
        mol2 = copy.deepcopy(mol)
        mol2.add("H", [1, 1, 1], name="H3")
        mol2.add("H", [1, 1, 1], name="H4")

        # Sample density
        pa.sample.density("data/pore_system.obj", "data/traj.xtc", "output/dens.obj", mol, atoms=["O1"])
        pa.sample.density("data/pore_system.obj", "data/traj.xtc", "output/dens.obj", mol, atoms=["O1"], masses=[1, 2])
        pa.sample.density("data/pore_system.obj", "data/traj.xtc", "output/dens.obj", mol2, is_force=True)


        # Sample diffusion
        pa.sample.diffusion_bin("data/pore_system.obj", "data/traj.xtc", "output/diff.obj", mol, atoms=["O1"])
        pa.sample.diffusion_bin("data/pore_system.obj", "data/traj.xtc", "output/diff.obj", mol, atoms=["O1"], masses=[1, 2])
        pa.sample.diffusion_bin("data/pore_system.obj", "data/traj.xtc", "output/diff.obj",  mol2, is_force=True)
        pa.sample.diffusion_bin("data/pore_system.obj", "data/traj.xtc", "output/diff.obj",  mol, len_obs=3e-12, is_force=True)


    ###########
    # Density #
    ###########
    def test_adsorption(self):
        # Calculate adsorption
        ads = pa.adsorption.calculate("data/pore_system.obj", "output/dens.obj")
        self.assertEqual(round(ads["c"][1], 2), 71.65)
        self.assertEqual(round(ads["n"][1], 2), 7402.67)


    ###########
    # Density #
    ###########
    def test_density(self):
        # Calculate density
        dens = pa.density.calculate("output/dens.obj", target_dens=1000)
        self.assertEqual(round(dens["in"] [3], 2), 992.77)
        self.assertEqual(round(dens["out"][3], 2), 975.54)

        # Plot density
        plt.figure()
        pa.density.plot(dens, target_dens=33)
        pa.density.plot(dens, target_dens=33, is_mean=True)
        plt.savefig("output/density.pdf", format="pdf", dpi=1000)
        # plt.show()

        plt.figure()
        pa.density.plot(dens, intent="in")
        pa.density.plot(dens, intent="out")
        plt.savefig("output/density_intent.pdf", format="pdf", dpi=1000)
        # plt.show()

        print()
        pa.density.plot(dens, intent="DOTA")


    #################
    # Bin Diffusion #
    #################
    def test_diffusion(self):
        # Bin diffusion
        plt.figure()
        pa.diffusion.bins("output/diff.obj")
        pa.diffusion.bins("output/diff.obj", is_norm=True)
        plt.savefig("output/diffusion_bins.pdf", format="pdf", dpi=1000)
        # plt.show()

        # CUI diffusion
        plt.figure()
        pa.diffusion.cui("output/diff.obj", is_fit=True)
        plt.savefig("output/diffusion_cui.pdf", format="pdf", dpi=1000)
        # plt.show()

        # Mean diffusion based on bins
        pa.diffusion.mean("output/diff.obj", "output/dens.obj")


if __name__ == '__main__':
    unittest.main(verbosity=2)
