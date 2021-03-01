import os
import sys

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


    #########
    # Utils #
    #########
    def test_utils(self):
        file_link = "output/test/test.txt"

        pms.utils.mkdirp("output/test")

        with open(file_link, "w") as file_out:
            file_out.write("TEST")
        pms.utils.copy(file_link, file_link+"t")
        pms.utils.replace(file_link+"t", "TEST", "DOTA")
        with open(file_link+"t", "r") as file_in:
            for line in file_in:
                self.assertEqual(line, "DOTA\n")

        self.assertEqual(pms.utils.column([[1, 1, 1], [2, 2, 2]]), [[1, 2], [1, 2], [1, 2]])

        pms.utils.save([1, 1, 1], file_link)
        self.assertEqual(pms.utils.load(file_link), [1, 1, 1])

        self.assertEqual(round(pms.utils.mumol_m2_to_mols(3, 100), 4), 180.66)
        self.assertEqual(round(pms.utils.mols_to_mumol_m2(180, 100), 4), 2.989)
        self.assertEqual(round(pms.utils.mmol_g_to_mumol_m2(0.072, 512), 2), 0.14)
        self.assertEqual(round(pms.utils.mmol_l_to_mols(30, 1000), 4), 18.066)
        self.assertEqual(round(pms.utils.mols_to_mmol_l(18, 1000), 4), 29.8904)

        print()
        pms.utils.toc(pms.utils.tic(), message="Test", is_print=True)
        self.assertEqual(round(pms.utils.toc(pms.utils.tic(), is_print=True)), 0)


    ############
    # Geometry #
    ############
    def test_geometry(self):
        vec_a = [1, 1, 2]
        vec_b = [0, 3, 2]

        print()

        self.assertEqual(round(pms.geom.dot_product(vec_a, vec_b), 4), 7)
        self.assertEqual(round(pms.geom.length(vec_a), 4), 2.4495)
        self.assertEqual([round(x, 4) for x in pms.geom.vector(vec_a, vec_b)], [-1, 2, 0])
        self.assertIsNone(pms.geom.vector([0, 1], [0, 0, 0]))
        self.assertEqual([round(x, 4) for x in pms.geom.unit(vec_a)], [0.4082, 0.4082, 0.8165])
        self.assertEqual([round(x, 4) for x in pms.geom.cross_product(vec_a, vec_b)], [-4, -2, 3])
        self.assertEqual(round(pms.geom.angle(vec_a, vec_b), 4), 37.5714)
        self.assertEqual(round(pms.geom.angle_polar(vec_a), 4), 0.7854)
        self.assertEqual(round(pms.geom.angle_azi(vec_b), 4), 0.9828)
        self.assertEqual(round(pms.geom.angle_azi([0, 0, 0]), 4), 1.5708)
        self.assertEqual([round(x, 4) for x in pms.geom.main_axis(1)], [1, 0, 0])
        self.assertEqual([round(x, 4) for x in pms.geom.main_axis(2)], [0, 1, 0])
        self.assertEqual([round(x, 4) for x in pms.geom.main_axis(3)], [0, 0, 1])
        self.assertEqual([round(x, 4) for x in pms.geom.main_axis("x")], [1, 0, 0])
        self.assertEqual([round(x, 4) for x in pms.geom.main_axis("y")], [0, 1, 0])
        self.assertEqual([round(x, 4) for x in pms.geom.main_axis("z")], [0, 0, 1])
        self.assertEqual(pms.geom.main_axis("h"), "Wrong axis definition...")
        self.assertEqual(pms.geom.main_axis(100), "Wrong axis definition...")
        self.assertEqual(pms.geom.main_axis(0.1), "Wrong axis definition...")
        self.assertEqual([round(x, 4) for x in pms.geom.rotate(vec_a, "x", 90, True)], [1.0, -2.0, 1.0])
        self.assertIsNone(pms.geom.rotate(vec_a, [0, 1, 2, 3], 90, True))
        self.assertIsNone(pms.geom.rotate(vec_a, "h", 90, True))


    ###########
    # Density #
    ###########
    def test_density(self):
        # self.skipTest("Temporary")

        # Load molecule
        mol = pms.Molecule("spc216", "SOL", inp="data/spc216.gro")

        # Sample trajectory
        pa.density.sample("data/pore_system.obj", "data/traj.pdb", "data/traj.trr", "output/dens.obj", mol, is_force=True)

        # Calculate density
        dens = pa.density.calculate("output/dens.obj")

        # Plot density
        plt.figure()
        pa.density.plot(dens)
        plt.savefig("output/density.pdf", format="pdf", dpi=1000)
        # plt.show()


    #############
    # Diffusion #
    #############
    def test_diffusion(self):
        # self.skipTest("Temporary")

        # Load molecule
        mol = pms.Molecule("spc216", "SOL", inp="data/spc216.gro")

        # Sample trajectory
        pa.diffusion.sample("data/pore_system.obj", "data/traj.pdb", "data/traj.trr", "output/diff.obj", mol, len_obs=4e-12, is_force=True)

        # Bin diffusion
        plt.figure()
        pa.diffusion.bins("output/diff.obj")
        plt.savefig("output/diffusion_bins.pdf", format="pdf", dpi=1000)
        # plt.show()

        # CUI diffusion
        plt.figure()
        pa.diffusion.cui("output/diff.obj")
        plt.savefig("output/diffusion_cui.pdf", format="pdf", dpi=1000)
        # plt.show()

        # Mean diffusion based on bins
        pa.diffusion.mean("output/diff.obj", "output/dens.obj")


if __name__ == '__main__':
    unittest.main(verbosity=2)
