:orphan:

.. raw:: html

  <div class="container-fluid">
    <div class="row">
      <div class="col-md-10">
        <div style="text-align: justify; text-justify: inter-word;">

Diffusion analysis in a pore
============================

.. code-block:: python

  import porems as pms
  import moldyn as md

.. code-block:: python

  mol = Molecule("spc216", "SOL")
  mol.add("O", [5.1100, 7.8950, 2.4240])
  mol.add("H", [5.1599, 7.9070, 2.3432])
  mol.add("H", [5.1080, 7.7996, 2.4310])
  mol.zero()

  print(mol)

.. raw:: html

  <div class="nboutput nblast">
    <div class="output_area rendered_html">
      <table border="1" class="dataframe">
        <thead>
          <tr style="text-align: right;">
            <th></th>
            <th>Axis 1</th>
            <th>Axis 2</th>
            <th>Axis 3</th>
            <th>Type</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th>0</th>
            <td>0.0020</td>
            <td>0.0954</td>
            <td>0.0808</td>
            <td>O</td>
          </tr>
          <tr>
            <th>1</th>
            <td>0.0519</td>
            <td>0.1074</td>
            <td>0.0000</td>
            <td>H</td>
          </tr>
          <tr>
            <th>2</th>
            <td>0.0000</td>
            <td>0.0000</td>
            <td>0.0878</td>
            <td>H</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>


.. code-block:: python

  md.pore.diffusion.sample("pore.obj", "traj.pdb", "traj.trr", "diff.obj", mol, len_obs=4e-12, is_force=True)


.. code-block:: python

  md.pore.diffusion.bins("diff.obj")

.. figure::  /pics/diffusion_bins.svg
  :align: center
  :width: 50%
  :name: fig1


.. code-block:: python

  md.pore.diffusion.cui("diff.obj")

``Diffusion axial:  8.123 10^-9 m^2s^-1``

``Diffusion radial: 630.953 10^-9 m^2 s^-1; Number of zeros: 21; Radius:  0.27``

.. figure::  /pics/diffusion_cui.svg
  :align: center
  :width: 50%
  :name: fig2


.. code-block:: python

  md.pore.diffusion.mean("diff.obj", "dens.obj")

``Mean Diffusion axial: 4.594 10^-9 m^2s^-1``


.. raw:: html

        </div>
      </div>
    </div>
  </div>
