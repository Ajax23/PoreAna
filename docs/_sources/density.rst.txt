:orphan:

.. raw:: html

  <div class="container-fluid">
    <div class="row">
      <div class="col-md-10">
        <div style="text-align: justify; text-justify: inter-word;">

Density analysis in a Pore
==========================

.. code-block:: python

  import porems as pms
  import moldyn as md

.. code-block:: python

  mol = pms.Molecule("spc216", "SOL")
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

  md.pore.density.sample("pore.obj", "traj.pdb", "traj.trr", "dens.obj", mol, is_force=True)


.. code-block:: python

  dens = md.pore.density.calculate("dens.obj")

``Density inside  Pore = 32.877 #/nm^3 ; 983.542 kg/m^3``

``Density outside Pore = 32.895 #/nm^3 ; 984.083 kg/m^3``


.. code-block:: python

  md.pore.density.plot(dens)

.. figure::  /pics/density.svg
  :align: center
  :width: 70%
  :name: fig1


.. raw:: html

        </div>
      </div>
    </div>
  </div>
