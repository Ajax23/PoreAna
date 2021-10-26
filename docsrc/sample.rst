:orphan:

.. raw:: html

  <div class="container-fluid">
    <div class="row">
      <div class="col-md-10">
        <div style="text-align: justify; text-justify: inter-word;">

Sampling Trajectory Files
=========================

The used GROMACS trajectory is sampled using the ChemFiles package. In
order to avoid artefacts due to periodic boundary conditions, molecules
are joined using the *-pbc mol* command in GROMACS which places the center
of mass of the molecules into the box. Furthermore, PoreAna only analyses
one molecule type per routine, that’s why the desired molecule type has
to be extracted first

.. code:: bash

   gmx_mpi trjconv -f traj.xtc -s topol.tpr -o traj_mol.xtc -pbc mol

The sampling routine uses the molecule structure defined by the PoreMS
package

.. code-block:: python

    import porems as pms

    mol = pms.Molecule(inp="data/benzene.gro")
    mol


.. parsed-literal::

        Residue Name Type      x      y      z
    0         0   C1    C  0.196  0.108  0.109
    1         0   C2    C  0.300  0.048  0.181
    2         0   C3    C  0.295  0.046  0.320
    3         0   C4    C  0.188  0.104  0.387
    4         0   C5    C  0.084  0.163  0.315
    5         0   C6    C  0.088  0.165  0.176
    6         0   H1    H  0.199  0.110  0.000
    7         0   H2    H  0.384  0.004  0.128
    8         0   H3    H  0.376  0.000  0.376
    9         0   H4    H  0.184  0.102  0.496
    10        0   H5    H  0.000  0.208  0.368
    11        0   H6    H  0.008  0.211  0.120

Before running the sampling routine, first a Sample object has to be
created with the pore system (*pore_syste.obj* file provided by PoreMS),
trajectory file, the corresponding molecule object and a list of atoms
names to be considered (empty list for all)

.. code-block:: python

    import poreana as pa

    sample = pa.Sample("data/pore_system_cylinder.obj", "data/traj_cylinder.xtc", mol, [])

The next step is activating the analysis routines to be used during the
sampling process using the initialize functions with their corresponding
inputs

.. code-block:: python

    sample.init_density("output/dens.h5")
    sample.init_gyration("output/gyr.h5")
    sample.init_diffusion_bin("output/diff.h5")

If you would like to calculated the diffusion and the free energy profile
you have to set the MC Diffusion initalize function instead of the Bin Diffusion.
Attention currently only the initialization of one diffusion calculation method is possible.

.. code-block:: python

    sample.init_density("output/dens.h5")
    sample.init_gyration("output/gyr.h5")
    sample.init_diffusion_mc("output/diff.h5")

Finally, the sample function is initiated with the
option to run in parallel and deactivating the periodic boundary
conditions

.. code-block:: python

    sample.sample(is_parallel=True, is_pbc=False)

``Finished frame 2001/2001...``


.. raw:: html

        </div>
      </div>
    </div>
  </div>
