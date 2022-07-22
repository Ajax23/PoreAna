.. raw:: html

    </div>
    <div class=col-md-9 content>

Sample
======

.. currentmodule:: poreana.sample

.. autoclass:: Sample


  .. rubric:: Sampling

  .. autosummary::

     ~Sample._sample_helper
     ~Sample.sample


  .. rubric:: Density

  .. autosummary::

    ~Sample.init_density
    ~Sample._density
    ~Sample._density_data


  .. rubric:: Gyration Radius

  .. autosummary::

    ~Sample.init_gyration
    ~Sample._gyration
    ~Sample._gyration_data


  .. rubric:: Bin Diffusion

  .. autosummary::

    ~Sample.init_diffusion_bin
    ~Sample._diffusion_bin
    ~Sample._diffusion_bin_data
    ~Sample._diffusion_bin_step


  .. rubric:: MC Diffusion

  .. autosummary::

    ~Sample.init_diffusion_mc
    ~Sample._diffusion_mc_data
    ~Sample._diffusion_mc


  .. rubric:: Bin Structure

  .. autosummary::

    ~Sample._bin_ex
    ~Sample._bin_in
    ~Sample._bin_in_const_A
    ~Sample._bin_window
    ~Sample._bin_mc
