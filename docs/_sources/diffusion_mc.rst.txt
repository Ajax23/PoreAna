:orphan:

.. raw:: html

  <div class="container-fluid">
    <div class="row">
      <div class="col-md-10">
        <div style="text-align: justify; text-justify: inter-word;">

Diffusion analysis in a pore (MC Method)
========================================
The MC diffusion analysis needs the sampled object file using the mc diffusion routine

.. code-block:: python

  import porems as pms
  import poreana as pa


  # Load molecule
  mol = pms.Molecule("benzene", "BEN", inp="data/benzene.gro")

  # Set inputs
  len_steps = [1,2,5,10,20,30,40]

  # Sample transition matrix
  sample = pa.Sample("data/pore_system_cylinder.obj", "data/traj_cylinder.xtc", mol)
  sample.init_diffusion_mc("output/diff_mc_cyl_s.obj", len_step)

``Finished frame 2001/2001...``

After sampling, a model has to set and the MC Alogirthm started

.. code-block:: python

  # Set Cosine Model for diffusion and energy profile
  model = pa.CosineModel("output/diff_mc_cyl_s.obj", 6, 10)

  # Set the MC class and options
  MC = pa.MC(model,5000,5000)

  # Do the MC alogirthm
  MC.do_mc_cycles(model,"output/diff_test_mc.obj")

The results of the MC Alogrithm the diffusion can be calculated

.. code-block:: python

  # Print the results for the normal diffusion
  diff, diff_table = pa.diffusion.diffusion_fit("output/diff_test_mc.obj")

``Diffusion: 1.8418e-09 m^2/s``

``Mean Diffusion: 1.8306e-09 m^2/s``

``Standard deviation: 7.2669e-11 m^2/s``

.. figure::  /pics/fit.svg
  :align: center
  :width: 50%
  :name: fig1

or the diffusion and free energy profile over the entire system can be displayed

.. code-block:: python

  # Plot diffusion profile over the simulation box
  pa.diffusion.diff_profile("diff_mc.obj",print_df=True)

  # Plot free energy profile over the simulation box
  pa.diffusion.df_profile("diff_mc.obj",[10])

.. figure::  /pics/diff_profile.svg
  :align: center
  :width: 49%
  :name: fig2

.. figure::  /pics/df_profile.svg
  :align: center
  :width: 49%
  :name: fig3


Additionally, the pore area can be considered more closely

.. code-block:: python

  # Plot the lag time extrapolation for the pore ares
  pa.diffusion.diff_pore_profile("data/pore_system_cylinder.obj","diff_mc.obj",print_df=True)

  # Plot diffusion profile in a pore
  pa.diffusion.diff_pore_profile("data/pore_system_cylinder.obj","diff_mc.obj")


``Diffusion: 1.4061e-09 m^2/s``

``Mean Diffusion: 1.3905e-09 m^2/s``

``Standard deviation: 8.6283e-11 m^2/s``

.. figure::  /pics/fit_pore.svg
  :align: center
  :width: 50%
  :name: fig4

.. figure::  /pics/diff_profile_pore.svg
  :align: center
  :width: 50%
  :name: fig5

.. raw:: html

        </div>
      </div>
    </div>
  </div>
