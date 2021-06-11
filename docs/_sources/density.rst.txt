:orphan:

.. raw:: html

  <div class="container-fluid">
    <div class="row">
      <div class="col-md-10">
        <div style="text-align: justify; text-justify: inter-word;">

Density Analysis in a Pore
==========================

The density analysis needs the sampled object file using the density
routine

.. code-block:: python

    import porems as pms
    import poreana as pa

    mol = pms.Molecule(inp="data/benzene.gro")

    sample = pa.Sample("data/pore_system_cylinder.obj", "data/traj_cylinder.xtc", mol, [])
    sample.init_density("output/dens.obj")
    sample.sample()


``Finished frame 2001/2001...``


The calculation of the density profile is done using the bins function

.. code-block:: python

    dens = pa.density.bins("output/dens.obj")

``Density inside  Pore = 0.100 #/nm^3 ;  12.941 kg/m^3``

``Density outside Pore = 0.127 #/nm^3 ;  16.446 kg/m^3``


and viewed using the plot function

.. code-block:: python

    pa.density.bins_plot(dens)


.. figure::  /pics/density_01.svg
  :align: center
  :width: 70%
  :name: fig1


.. raw:: html

        </div>
      </div>
    </div>
  </div>
