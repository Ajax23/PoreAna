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
        mol = pms.Molecule("benzene", "BEN", inp="data/benzene.gro")

        # Sample
        sample = pa.Sample("data/pore_system.obj", "data/traj.xtc", mol, is_nojump=False)
        sample.init_density("output/dens.obj")
        sample.init_gyration("output/gyr.obj")
        sample.init_diffusion_bin("output/diff.obj")
        sample.sample(is_parallel=False)

        sample = pa.Sample("data/pore_system.obj", "data/traj_nojump.xtc", mol, is_nojump=True)
        sample.init_density("output/dens_snj.obj")
        sample.sample(is_parallel=False)

        sample = pa.Sample("data/pore_system.obj", "data/traj.xtc", mol, is_nojump=False)
        sample.init_density("output/dens_p.obj")
        sample.init_gyration("output/gyr_p.obj")
        sample.init_diffusion_bin("output/diff_p.obj")
        sample.sample(is_parallel=True)

        sample = pa.Sample("data/pore_system.obj", "data/traj_nojump.xtc", mol, is_nojump=True)
        sample.init_density("output/dens_nj.obj")
        sample.init_gyration("output/gyr_nj.obj")
        sample.init_diffusion_bin("output/diff_nj.obj")
        sample.sample(is_parallel=True)


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
        pa.Sample("data/pore_system.obj", "data/traj.xtc", mol2)
        pa.Sample("data/pore_system.obj", "data/traj.xtc", mol, atoms=["C1"], masses=[1, 1])

        # Diffusion
        sample = pa.Sample("data/pore_system.obj", "data/traj.xtc", mol, atoms=["C1"])
        sample.init_diffusion_bin("output/diff.obj", len_obs=3e-12)


    ##############
    # Adsorption #
    ##############
    def test_adsorption(self):
        # Calculate adsorption
        ads = pa.adsorption.calculate("data/pore_system.obj", "output/dens.obj")
        ads_p = pa.adsorption.calculate("data/pore_system.obj", "output/dens_p.obj")
        ads_nj = pa.adsorption.calculate("data/pore_system.obj", "output/dens_nj.obj")

        self.assertEqual(round(ads["conc"]["mumol_m2"], 2), 0.15)
        self.assertEqual(round(ads["num"]["in"], 2), 10.68)
        self.assertEqual(round(ads_p["conc"]["mumol_m2"], 2), 0.15)
        self.assertEqual(round(ads_p["num"]["in"], 2), 10.68)
        self.assertEqual(round(ads_nj["conc"]["mumol_m2"], 2), 0.15)
        self.assertEqual(round(ads_nj["num"]["in"], 2), 10.68)


    ###########
    # Density #
    ###########
    def test_density(self):
        # Calculate density
        dens = pa.density.calculate("output/dens.obj", target_dens=16)
        dens_p = pa.density.calculate("output/dens_p.obj", target_dens=16)
        dens_nj = pa.density.calculate("output/dens_nj.obj", target_dens=16)

        self.assertEqual(round(dens["dens"]["in"], 3), 12.941)
        self.assertEqual(round(dens["dens"]["ex"], 3), 15.977)
        self.assertEqual(round(dens_p["dens"]["in"], 3), 12.941)
        self.assertEqual(round(dens_p["dens"]["ex"], 3), 15.977)
        self.assertEqual(round(dens_nj["dens"]["in"], 3), 12.946)
        self.assertEqual(round(dens_nj["dens"]["ex"], 3), 16.178)

        # Plot density
        plt.figure()
        pa.density.plot(dens, target_dens=0.146)
        pa.density.plot(dens, target_dens=0.146, is_mean=True)
        plt.savefig("output/density.pdf", format="pdf", dpi=1000)
        # plt.show()

        plt.figure()
        pa.density.plot(dens, intent="in")
        pa.density.plot(dens, intent="ex")
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
        diff = pa.diffusion.bins("output/diff.obj")
        pa.diffusion.bins("output/diff.obj", is_norm=True)
        plt.savefig("output/diffusion_bins.pdf", format="pdf", dpi=1000)
        # plt.show()

        # CUI diffusion
        plt.figure()
        pa.diffusion.cui("output/diff.obj", is_fit=True)
        plt.savefig("output/diffusion_cui.pdf", format="pdf", dpi=1000)
        # plt.show()

        # Mean diffusion based on bins
        plt.figure()
        mean = pa.diffusion.mean("output/diff.obj", "output/dens.obj", is_check=True)
        plt.savefig("output/diff_mean_check.pdf", format="pdf", dpi=1000)
        mean_p = pa.diffusion.mean("output/diff_p.obj", "output/dens_p.obj")
        mean_nj = pa.diffusion.mean("output/diff_nj.obj", "output/dens_nj.obj")

        self.assertEqual(round(mean, 2), 1.13)
        self.assertEqual(round(mean_p, 2), 1.13)
        self.assertEqual(round(mean_nj, 2), 1.13)


    ############
    # Gyration #
    ############
    def test_gyration(self):
        # Plot gyration radius
        plt.figure()
        mean = pa.gyration.plot("output/gyr.obj", "output/dens.obj", is_mean=True)
        plt.savefig("output/gyration.pdf", format="pdf", dpi=1000)
        plt.figure()
        mean_p = pa.gyration.plot("output/gyr_p.obj", "output/dens_p.obj")
        plt.savefig("output/gyration_p.pdf", format="pdf", dpi=1000)
        plt.figure()
        mean_nj = pa.gyration.plot("output/gyr_nj.obj", "output/dens_nj.obj")
        plt.savefig("output/gyration_nj.pdf", format="pdf", dpi=1000)

        self.assertEqual(round(mean["in"], 2), 0.12)
        self.assertEqual(round(mean["ex"], 2), 0.38)
        self.assertEqual(round(mean_p["in"], 2), 0.12)
        self.assertEqual(round(mean_p["ex"], 2), 0.38)
        self.assertEqual(round(mean_nj["in"], 2), 0.12)
        self.assertEqual(round(mean_nj["ex"], 2), 0.39)

        plt.figure()
        pa.gyration.plot("output/gyr.obj", "output/dens.obj", intent="in")
        pa.gyration.plot("output/gyr.obj", "output/dens.obj", intent="ex")
        plt.savefig("output/gyration_intent.pdf", format="pdf", dpi=1000)

        print()
        pa.gyration.plot("output/gyr.obj", "output/dens.obj", intent="DOTA")


if __name__ == '__main__':
    unittest.main(verbosity=2)
