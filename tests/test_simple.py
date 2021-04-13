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
        mol_W = pms.Molecule("water", "SOL", inp="data/spc216.gro")

        # Sample
        ## Single core
        sample = pa.Sample("data/pore_system_cylinder.obj", "data/traj_cylinder.xtc", mol)
        sample.init_density("output/dens_cyl_s.obj")
        sample.init_gyration("output/gyr_cyl_s.obj")
        sample.init_diffusion_bin("output/diff_cyl_s.obj")
        sample.sample(is_parallel=False)

        sample = pa.Sample("data/pore_system_slit.obj", "data/traj_slit.xtc", mol_W)
        sample.init_density("output/dens_slit.obj")
        sample.sample(is_parallel=False, is_pbc=False)

        ## Parallel
        sample = pa.Sample("data/pore_system_cylinder.obj", "data/traj_cylinder.xtc", mol)
        sample.init_density("output/dens_cyl_p.obj")
        sample.init_gyration("output/gyr_cyl_p.obj")
        sample.init_diffusion_bin("output/diff_cyl_p.obj")
        sample.sample(is_parallel=True, is_pbc=False)


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

        # Diffusion
        sample = pa.Sample("data/pore_system_cylinder.obj", "data/traj_cylinder.xtc", mol, atoms=["C1"])
        sample.init_diffusion_bin("output/diff_np_s.obj", len_obs=3e-12)


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


    ############
    # Gyration #
    ############
    def test_gyration(self):
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

        print()
        pa.gyration.plot("output/gyr_cyl_s.obj", "output/dens_cyl_s.obj", intent="DOTA")


if __name__ == '__main__':
    unittest.main(verbosity=2)
