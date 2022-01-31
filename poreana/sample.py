################################################################################
# Sample                                                                       #
#                                                                              #
"""Sample functions to be run on clusters."""
################################################################################


import sys
import math
import numpy as np
import chemfiles as cf
import multiprocessing as mp

import poreana.utils as utils
import poreana.geometry as geometry


class Sample:
    """This class samples a trajectory to determine different properties.
    Different properties can be initialized to be run at the same time during
    the sampling run. The output is stored in form of pickle files for later
    calculation using methods provided in the package.

    It is advisable to run the sampling on a cluster due to a high time and
    resource consumption.

    The system can either be a pore system - variable **system** is a file link
    to the *pore_system* object file - or a simple simulation box - variable
    **system** is a list containing the dimensions in nano meter.

    Parameters
    ----------
    system : string, list
        Link to poresystem object file or a list of dimensions for a simple box
        analysis
    link_traj : string
        Link to trajectory file (trr or xtc)
    mol : Molecule
        Molecule to calculate the density for
    atoms : list, optional
        List of atom names, leave empty for whole molecule
    masses : list, optional
        List of atom masses, leave empty to read molecule object masses
    entry : float, optional
        Remove pore entrance from calculation
    """
    def __init__(self, system, link_traj, mol, atoms=[], masses=[], entry=0.5):
        # Initialize
        self._pore = utils.load(system, file_type="yml") if isinstance(system, str) else None
        self._box = system if isinstance(system, list) else []
        self._traj = link_traj
        self._mol = mol
        self._atoms = atoms
        self._masses = masses
        self._entry = entry

        # Set analysis routines
        self._is_density = False
        self._is_gyration = False
        self._is_diffusion_bin = False
        self._is_diffusion_mc = False

        # Get molecule ids
        self._atoms = [atom.get_name() for atom in mol.get_atom_list()] if not self._atoms else self._atoms
        self._atoms = [atom_id for atom_id in range(mol.get_num()) if mol.get_atom_list()[atom_id].get_name() in self._atoms]

        # Check masses
        if not self._masses:
            if len(self._atoms)==mol.get_num():
                self._masses = mol.get_masses()
            elif len(self._atoms) == 1:
                self._masses = [1]
        self._sum_masses = sum(self._masses)

        # Check atom mass consistency
        if self._atoms and not len(self._masses) == len(self._atoms):
            print("Length of variables *atoms* and *masses* do not match!")
            return

        # Get number of frames
        traj = cf.Trajectory(self._traj)
        self._num_frame = traj.nsteps

        # Get numer of residues
        frame = traj.read()
        num_res = len(frame.topology.atoms)/mol.get_num()

        # Check number of residues
        if abs(int(num_res)-num_res) >= 1e-5:
            print("Number of atoms is inconsistent with number of residues.")
            return

        # Create residue list with all relevant atom ids
        self._res_list = {}
        for res_id in range(int(num_res)):
            self._res_list[res_id] = [res_id*mol.get_num()+atom for atom in range(mol.get_num()) if atom in self._atoms]

        # Get pore properties
        self._pore_props = {}
        if self._pore:
            self._pore_props["type"] = self._pore["shape"]
            self._pore_props["res"] = self._pore["reservoir"]
            self._pore_props["focal"] = self._pore["centroid"]
            self._pore_props["box"] = self._pore["dimensions"]

            # Get pore diameter
            if self._pore_props["type"] == "CYLINDER":
                self._pore_props["diam"] = self._pore["diameter"]
            elif self._pore_props["type"] == "SLIT":
                self._pore_props["diam"] = self._pore["height"]


    ########
    # Bins #
    ########
    def _bin_in(self, bin_num):
        """This function creates a simple bin structure for the interior of the
        pore based on the pore diameter.

        Parameters
        ----------
        bin_num : integer
            Number of bins to be used

        Returns
        -------
        data : dictionary
            Dictionary containing a list of the bin width and a data list
        """
        # Define bins
        width = [self._pore_props["diam"]/2/bin_num*x for x in range(bin_num+2)]
        bins = [0 for x in range(bin_num+1)]

        return {"width": width, "bins": bins}

    def _bin_ex(self, bin_num):
        """This function creates a simple bin structure for the exterior of the
        pore based on the reservoir length.

        Parameters
        ----------
        bin_num : integer
            Number of bins to be used

        Returns
        -------
        data : dictionary
            Dictionary containing a list of the bin width and a data list
        """
        # Process system
        z_length = self._pore_props["res"] if self._pore_props else self._box[2]

        # Define bins
        width = [z_length/bin_num*x for x in range(bin_num+1)]
        bins = [0 for x in range(bin_num+1)]

        return {"width": width, "bins": bins}

    def _bin_window(self, bin_num, len_window):
        """This function creates window list for each bin for the interior of
        the pore based on the pore diameter.

        Parameters
        ----------
        bin_num : integer
            Number of bins to be used
        len_window : integer
            Window length

        Returns
        -------
        data : dictionary
            Dictionary containing a list of the bin width and a data list
        """
        # Define bins
        width = [self._pore_props["diam"]/2/bin_num*x for x in range(bin_num+2)]
        bins = [[0 for y in range(len_window)] for x in range(bin_num+1)]

        return {"width": width, "bins": bins}

    def _bin_mc(self, bin_num, direction):
        """This function creates a simple bin structure for the pore and
        resevoir.

        Parameters
        ----------
        bin_num : integer
            Number of bins to be used
        direction : integer, optional
            Direction of descretization of the simulation box (x = 0; y = 1; z = 2)

        Returns
        -------
        data : dictionary
            Dictionary containing a list of the bin width and a data list
        """
        # Ask for system type (box or pore system)
        if self._pore:
            z_length = self._pore_props["box"][direction]
        else:
            z_length = self._box[direction]

        # Define bins
        bins = [z_length/bin_num*x for x in range(bin_num+1)]
        return {"bins": bins}


    ###########
    # Density #
    ###########
    def init_density(self, link_out, bin_num=150, remove_pore_from_res=True):
        """Enable density sampling routine.

        Parameters
        ----------
        link_out : string
            Link to output hdf5, obj or yml data file
        bin_num : integer, optional
            Number of bins to be used
        remove_pore_from_res : bool, optional
            True to remove an extended pore volume from the reservoirs to only
            consider the reservoir space intersecting the crystal grid
        """
        # Initialize
        self._is_density = True
        self._dens_inp = {"output": link_out, "bin_num": bin_num,
                          "remove_pore_from_res": remove_pore_from_res}

    def _density_data(self):
        """Create density data structure.

        Returns
        -------
        data : dictionary
            Density data structure
        """
        # Initialize
        bin_num = self._dens_inp["bin_num"]
        data = {}

        # Fill dictionary
        data["ex_width"] = self._bin_ex(bin_num)["width"]
        data["ex"] = self._bin_ex(bin_num)["bins"]

        if self._pore:
            data["in_width"] = self._bin_in(bin_num)["width"]
            data["in"] = self._bin_in(bin_num)["bins"]

        return data

    def _density(self, data, region, dist, com):
        """This function samples the density inside and outside of the pore.

        All atoms are sampled each frame if they are inside or outside the
        bounds of the pore minus an optional entry length on both sides.

        Inside the pore the atom instances will be added to radial cylindric
        slices :math:`r_i-r_{i-1}` and outside to rectangular slices
        :math:`z_j-z_{j-1}` with pore radius :math:`r_i` of radial slice
        :math:`i` and length :math:`z_j` of slice :math:`j`.

        Parameters
        ----------
        data : dictionary
            Data dictionary containing bins for the pore interior and exterior
        region : string
            Indicator wether molecule is inside or outside pore
        dist : float
            Distance of center of mass to pore surface area
        com : list
            Center of mass of current molecule
        """
        # Initialize
        bin_num = self._dens_inp["bin_num"]

        # Molecule is inside pore
        if region=="in":
            index = int(dist/data["in_width"][1])
            if index <= bin_num:
                data["in"][index] += 1

        # Molecule is in the reservoir
        elif region=="ex":
            # Calculate distance to crystobalit and apply perodicity
            lentgh = abs(com[2]-self._pore_props["box"][2]) if self._pore and com[2] > self._pore_props["box"][2]/2 else com[2]
            index = int(lentgh/data["ex_width"][1])

            # Only consider reservoir space in vicinity of crystobalit
            # Remove an extended pore volume from the reservoir
            if self._pore and self._dens_inp["remove_pore_from_res"]:
                is_add = index <= bin_num and dist > self._pore_props["diam"]/2 and com[2]<=self._pore_props["box"][2]
            else:
                is_add = index <= bin_num

            if is_add:
                data["ex"][index] += 1


    ############
    # Gyration #
    ############
    def init_gyration(self, link_out, bin_num=150):
        """Enable gyration sampling routine.

        Parameters
        ----------
        link_out : string
            Link to output hdf5, obj or yml data file
        bin_num : integer, optional
            Number of bins to be used
        """
        # Initialize
        self._is_gyration = True
        self._gyr_inp = {"output": link_out, "bin_num": bin_num}

    def _gyration_data(self):
        """Create gyration data structure.

        Returns
        -------
        data : dictionary
            Gyration data structure
        """
        # Initialize
        bin_num = self._gyr_inp["bin_num"]
        data = {}

        # Fill dictionary
        data["ex_width"] = self._bin_ex(bin_num)["width"]
        data["ex"] = self._bin_ex(bin_num)["bins"]

        if self._pore:
            data["in_width"] = self._bin_in(bin_num)["width"]
            data["in"] = self._bin_in(bin_num)["bins"]

        return data

    def _gyration(self, data, region, dist, com, pos):
        """This function calculates the gyration radius of molecules inside the
        pore.

        All atoms are sampled each frame if they are inside the bounds of the
        pore minus an optional entry length on both sides.

        Inside the pore the atom instances will be added to radial cylindric
        slices :math:`s_{r,j}-s_{r,j-1}` and outside to rectangular slices
        :math:`s_{z,k}-s_{z,k-1}` with pore radius :math:`s_{r,j}` of radial
        slice :math:`j` and length :math:`s_{z,k}` of slice :math:`k`.

        The gyration radius is calculated using

        .. math::

            R_g=\\left(\\frac{\\sum_i\\|\\boldsymbol{r}_i\\|^2m_i}{\\sum_im_i})\\right)^{\\frac{1}{2}}

        with mass :math:`m_i` and position :math:`\\boldsymbol{r}_i` of atom
        :math:`i` with respect to the center of mass of the molecule.

        Parameters
        ----------
        data : dictionary
            Data dictionary containing bins for the pore interior and exterior
        region : string
            Indicator wether molecule is inside or outside pore
        dist : float
            Distance of center of mass to pore surface area
        com : list
            Center of mass of current molecule
        pos : list
            List of atom positions of current molecule
        """
        # Initialize
        bin_num = self._gyr_inp["bin_num"]

        # Calculate gyration radius
        r_g = (sum([geometry.length(geometry.vector(pos[atom_id], com))**2*self._masses[atom_id] for atom_id in range(len(self._atoms))])/self._sum_masses)**0.5

        # Add molecule to bin
        if region=="in":
            index = int(dist/data["in_width"][1])
            if index <= bin_num:
                data["in"][index] += r_g

        elif region=="ex":
            # Calculate distance to crystobalit and apply perodicity
            lentgh = abs(com[2]-self._pore_props["box"][2]) if self._pore and com[2] > self._pore_props["box"][2]/2 else com[2]
            index = int(lentgh/data["ex_width"][1])

            # Only consider reservoir space in vicinity of crystobalit
            # Remove an extended pore volume from the reservoir
            if self._pore:
                is_add = index <= bin_num and dist > self._pore_props["diam"]/2 and com[2]<=self._pore_props["box"][2]
            else:
                is_add = index <= bin_num

            if is_add:
                data["ex"][index] += r_g


    #############
    # Diffusion #
    #############
    def init_diffusion_bin(self, link_out, bin_num=50, len_obs=16e-12, len_frame=2e-12, len_step=2, bin_step_size=1):
        """Enable diffusion sampling routine.

        Parameters
        ----------
        link_out : string
            Link to hdf5, obj or yml data file
        bin_num : integer, optional
            Number of bins to be used
        len_obs : float, optional
            Observation length of a window in seconds
        len_frame : float, optional
            Length of a frame in seconds
        len_step : integer, optional
            Length of the step size between frames
        bin_step_size : integer, optional
            Number of allowed bins for the molecule to leave
        """
        # Initialize
        if self._is_diffusion_mc:
            print("Binning and MC-approaches cannot be run in parallel.")
            return
        if self._pore:
            self._is_diffusion_bin = True
        else:
            print("Bin diffusion currently only usable for pore system.")
            return

        # Define window length
        len_window = len_obs/len_step/len_frame+1
        if not len_window == int(len_window):
            obs_u = (math.ceil(len_window)-1)*len_step*len_frame
            obs_d = (math.floor(len_window)-1)*len_step*len_frame
            print("Observation length not possible with current inputs. Alternatively use len_obs="+"%.1e" % obs_u+" or len_obs="+"%.1e" % obs_d+".")
            return
        else:
            len_window = int(len_window)

        # Create input dictionalry
        self._diff_bin_inp = {"output": link_out, "bin_step_size": bin_step_size,
                              "bin_num": bin_num, "len_step": len_step,
                              "len_frame": len_frame, "len_window": len_window}

    def _diffusion_bin_data(self):
        """Create bin diffusion data structure.

        Returns
        -------
        data : dictionary
            Bin diffusion data structure
        """
        # Initialize
        bin_num = self._diff_bin_inp["bin_num"]
        len_window = self._diff_bin_inp["len_window"]

        # Create dictionary
        data = {}
        data["width"] = self._bin_window(bin_num, len_window)["width"]
        bins = ["z", "r", "n", "z_tot", "r_tot", "n_tot"]
        for bin in bins:
            data[bin] = self._bin_window(bin_num, len_window)["bins"]

        return data

    def _diffusion_bin_step(self, idx):
        """Helper function to define allowed bin step list.

        Parameters
        ----------
        idx : integer
            Bin index
        """
        out_list = [idx+x for x in range(self._diff_bin_inp["bin_step_size"], 0, -1)]
        out_list += [idx]
        out_list += [idx-x for x in range(1, self._diff_bin_inp["bin_step_size"]+1)]
        return out_list

    def _diffusion_bin(self, data, region, dist, com_list, idx_list, res_id, com):
        """This function samples the mean square displacement of a molecule
        group in a pore in both axial and radial direction separated in radial
        bins.

        First a centre of mass-list is filled with :math:`w\\cdot s` frames with
        window length :math:`w` and stepsize :math:`s`. Each following frame
        removes the first com of the list and a new added to the end of it. This
        way only one loop over the frames is needed, since each frame is only
        needed for :math:`w\\cdot s` frames in total.

        All molecule com's are sampled each window if they are inside the bounds
        of the pore minus an entry length on both sides. Once the com leaves the
        boundary, it is no longer sampled for this specific window.
        Additionally, the radial bin index is checked for each frame. If the
        molecule com stays in the pore for the whole window length and in the
        same starting bin plus an allowed offset, the msd is added to it is
        added to a the corresponding window starting radial bin. The sub
        volumes, or rather bins, of the radial distance are calculated by

        .. math::

            V_i^\\text{radial}=\\pi z_\\text{pore}(r_i^2-r_{i-1}^2)

        with :math:`r` of bin :math:`i` and pore length :math:`z_\\text{pore}`.

        Once the first com-list is filled, the mean square displacement (msd)
        is sampled each frame. The axial diffusion only considers the
        deviation in :math:`z` direction

        .. math::

            \\text{msd}_\\text{axial}=\\langle\\left[z(0)-z(t_i)\\right]^2\\rangle
            =\\frac{\\left[z(0)-z(t_i)\\right]^2}{M_i}

        with time :math:`t` and normalization :math:`M` at bin :math:`i`.
        The radial diffusion only considers the radial components :math:`x` and
        :math:`y`

        .. math::

            \\text{msd}_\\text{radial}=\\langle\\left[r(0)-r(t_i)\\right]^2\\rangle
            =\\frac{\\left[\\sqrt{x^2(0)+y^2(0)}-\\sqrt{x^2(t_i)+y^2(t_i)}\\right]^2}{M_i}.

        Parameters
        ----------
        data : dictionary
            Data dictionary containing bins for axial and radial diffusion
        region : string
            Indicator wether molecule is inside or outside pore
        dist : float
            Distance of center of mass to pore surface area
        com_list : list
            List of dictionaries containing coms of all molecules for each frame
        index_list : list
            List of dictionaries containing bin id of all molecules for each frame
        res_id : integer
            Current residue id
        com : list
            Center of mass of current molecule
        """
        # Initialize
        bin_num = self._diff_bin_inp["bin_num"]
        len_step = self._diff_bin_inp["len_step"]
        len_window = self._diff_bin_inp["len_window"]

        # Only sample diffusion inside the pore
        if region == "in":
            # Calculate bin index
            index = math.floor(dist/data["width"][1])

            # Add com and bin index to global lists
            com_list[-1][res_id] = com
            idx_list[-1][res_id] = index

            # Start sampling when initial window is filled
            if len(com_list) == len_window*len_step and res_id in com_list[0]:
                # Set reference position
                pos_ref = com_list[0][res_id]
                idx_ref = idx_list[0][res_id]

                # Create temporary msd lists
                msd_z = [0 for x in range(len_window)]
                msd_r = [0 for x in range(len_window)]
                norm = [0 for x in range(len_window)]
                len_msd = 0

                # Run through position list to sample msd
                for step in range(0, len_window*len_step, len_step):
                    # Check if com is inside pore
                    if res_id in com_list[step]:
                        # Initialize step information
                        pos_step = com_list[step][res_id]
                        idx_step = idx_list[step][res_id]

                        # Get window index
                        win_idx = int(step/len_step)

                        # Add to msd
                        msd_z[win_idx] += (pos_ref[2]-pos_step[2])**2
                        msd_r[win_idx] += geometry.length(geometry.vector(pos_ref, [pos_step[0], pos_step[1], pos_ref[2]]))**2

                        # Add to normalize
                        norm[win_idx] += 1

                        # Check if com is within range of reference bin
                        if idx_step in self._diffusion_bin_step(idx_ref):
                            len_msd += 1

                        # COM left radial bin
                        else: break
                    # COM left the boundary
                    else: break

                # Save msd
                if idx_ref <= bin_num:
                    for i in range(len_window):
                        # Add to total list
                        data["z_tot"][idx_ref][i] += msd_z[i]
                        data["r_tot"][idx_ref][i] += msd_r[i]
                        data["n_tot"][idx_ref][i] += norm[i]

                        # Add to bin calculation list if msd is permissible
                        if len_msd == len_window:
                            data["z"][idx_ref][i] += msd_z[i]
                            data["r"][idx_ref][i] += msd_r[i]
                            data["n"][idx_ref][i] += norm[i]


    ################
    # MC Diffusion #
    ################
    def init_diffusion_mc(self, link_out, len_step, bin_num=100, len_frame=2e-12, direction = 2):
        """Enable diffusion sampling routine with the MC Alogrithm.

        This function sample the transition matrix for the diffusion
        calculation with the Monte Carlo diffusion methode for a cubic
        simulation box. The sample of the transition matrix is to be run on
        the cluster due to a high time and resource consumption. The output,
        a data object, is then used to calculate the self-diffusion using
        further calculation functions for the MC Diffusion methode.

        It is necessary to caculate the transition matrix for different step
        length and so for different lag times. A lagtime
        :math:`\\Delta t_{\\alpha}` is defined by

        .. math::

            \\Delta t_{\\alpha} = t_{i,\\alpha} - t_{j,\\alpha}

        with :math:`i` and :math:`j` as the current state of the system at two
        different times and :math:`\\alpha` as past time between the two states.
        To sample the transition matrix the frame length :math:`t` has to
        be specifiy and this frame length is for all lag times the same.
        The variation of the lag time is happend by adapting the step length
        :math:`s` which indicates the interval between the frames. The lag time
        is than calculated by

        .. math::

            \\Delta t_{\\alpha} = t \cdot s

        After the sampling a model class has to set and then the MC calculation
        can run. Subsequently the final mean diffusion coefficient can be
        determined with a extrapolation to
        :math:`\\Delta t_{\\alpha} \\rightarrow \infty`.
        For the etxrapolation we need the mean diffusion over the bins for
        different chosen lag times. That's why we have to calculate the results
        and the transition matrix for several lag times. More information about
        post processing and the extrapolation that you can find in
        :func:`poreana.diffusion.mc_fit`.

        The direction of descretization can be also choosen using the input variable
        :math:`\\mathrm{direction}`. The simulation box
        can be divided in every spatial direction and so the transition matrix
        is sampled in the chosen direction and the diffusion is calculated in
        this direction.

        Parameters
        ----------
        link_out : string
            Link to hdf5, obj or yml data file
        len_step : integer
            Length of the step size between frames
        bin_num : integer, optional
            Number of bins to be used
        len_frame : float, optional
            Length of a frame in seconds
        direction : integer, optional
            Direction of descretization of the simulation box (**0** (x-axis);
            **1** (y-axis); **2** (z-axis))
        """
        # Initialize
        if self._is_diffusion_bin:
            print("Binning and MC-approaches cannot be run in parallel.")
            return

        if direction not in [0,1,2]:
            print("Wrong directional input. Possible inputs are 0 (x-axis), 1 (y-axis), and 2 (z-axis)...")
            return

        # Enable routine
        self._is_diffusion_mc = True

        # Calculate bins
        bins = self._bin_mc(bin_num, direction)["bins"]

        # Sort len_step list
        len_step.sort()

        # Create input dictionalry
        self._diff_mc_inp = {"output": link_out, "bins": bins,
                              "bin_num": bin_num, "len_step": len_step,
                              "len_frame": len_frame, "is_pbc": True, "direction" : int(direction)}

    def _diffusion_mc_data(self):
        """Create mc diffusion data structure.

        Returns
        -------
        data : dictionary
            Bin diffusion data structure
        """
        # Initialize
        bin_num = self._diff_mc_inp["bin_num"]
        len_step = self._diff_mc_inp["len_step"]

        # Create dictionary
        data = {}

        # Initialize transition matrix
        for step in len_step:
            data[step] = np.zeros((bin_num+2, bin_num+2), int)

        return data

    def _diffusion_mc(self, data, idx_list, com, res_id, frame_list, frame_id):
        """This function sample the transition matrix for the diffusion
        calculation with the Monte Carlo diffusion methode for a cubic
        simulation box. The sample of the transition matrix is to be run on
        the cluster due to a high time and resource consumption. The output,
        a h5 data file, is then used to calculate the self-diffusion using
        further calculation functions for the MC Diffusion methode.

        It is necessary to caculate the transition matrix for different step
        length and so for different lag times. A lagtime
        :math:`\\Delta t_{\\alpha}` is defined by

        .. math::

            \\Delta t_{\\alpha} = t_{i,\\alpha} - t_{j,\\alpha}

        with :math:`i` and :math:`j` as the current state of the system at two
        different times and :math:`\\alpha` as past time between the two states.
        To sample the transition matrix the frame length :math:`t` has to
        be specifiy and this frame length is for all lag times the same.
        The variation of the lag time is happend by adapting the step length
        :math:`s` which indicates the interval between the frames. The lag time
        is than calculated by

        .. math::

            \\Delta t_{\\alpha} = t \cdot s

        After the sampling a model class has to set and then the MC calculation
        can run. Subsequently the final mean diffusion coefficient can be
        determined with a extrapolation to
        :math:`\\Delta t_{\\alpha} \\rightarrow \infty`.
        For the etxrapolation we need the mean diffusion over the bins for
        different chosen lag times. That's why we have to calculate the results
        and the transition matrix for several lag times. More information about
        post processing and the extrapolation that you can find in
        :func:`diffusion.diffusion_fit`

        Parameters
        ----------
        data : dictionary
            Data dictionary containing bins for axial and radial diffusion
        index_list : list
            List of dictionaries containing bin id of all molecules for each frame
        com : list
            Center of mass of current molecule
        res_id : integer
            Current residue id
        frame_list : list
            List of frame ids to process
        frame_id : integer
            Current frame_id
        """
        # Initialize
        len_step = self._diff_mc_inp["len_step"]
        bins = self._diff_mc_inp["bins"]
        direction = self._diff_mc_inp["direction"]

        # Calculate bin index
        idx_list[-1][res_id] = np.digitize(com[direction], bins)

        # Sample the transition matrix for the len_step
        if frame_list[0]==0:
            for step in len_step:
                if len(idx_list) >= (step+1):
                    # Calculate transition matrix in z direction
                    start = idx_list[-(step+1)][res_id]
                    end = idx_list[-1][res_id]
                    data[step][end, start] += 1

        # For parallel calculation
        if frame_list[0]!=0 and frame_id>=(frame_list[0] + self._diff_mc_inp["len_step"][-1]):
            for step in len_step:
                if len(idx_list) >= (step+1):
                    # Calculate transition matrix in z direction
                    start = idx_list[-(step+1)][res_id]
                    end = idx_list[-1][res_id]
                    data[step][end, start] += 1

    ############
    # Sampling #
    ############
    def sample(self, shift=[0, 0, 0], np=0, is_pbc=True, is_broken=False, is_parallel=True):
        """This function runs all enabled sampling routines. The output is
        stored in form of pickle files for later calculation using methods
        provided in the package.

        Note that the sampling should be run on a cluster due to a high time and
        resource consumption.

        Parameters
        ----------
        shift : list, optional
            Vector for translating atoms in nm
        np : integer, optional
            Number of cores to use
        is_pbc : bool, optional
            True to apply periodic boundary conditions
        is_broken : bool, optional
            True to check for broken molecules during sampling
        is_parallel : bool, optional
            True to run parallelized sampling
        """
        # Process input
        if not len(shift)==3:
            print("Sample - Wrong shift dimension.")
            return

        # Get number of cores
        np = np if np and np<=mp.cpu_count() else mp.cpu_count()

        # Run sampling helper
        if is_parallel:
            # Divide number of frames on processors
            frame_num = math.floor(self._num_frame/np)

            # Define bounds
            frame_start = [frame_num*i for i in range(np)]
            frame_end = [frame_num*(i+1) if i<np-1 else self._num_frame for i in range(np)]

            # Substract window filling for bin diffusion
            if self._is_diffusion_bin:
                frame_start = [x-self._diff_bin_inp["len_window"]*self._diff_bin_inp["len_step"]+1 if i>0 else x for i, x in enumerate(frame_start)]

            if self._is_diffusion_mc:
                frame_end = [x+max(self._diff_mc_inp["len_step"]) for i, x in enumerate(frame_end)]
                for i in range(len(frame_end)):
                    if frame_end[i] >= self._num_frame:
                        frame_end[i] = frame_end[-1]-max(self._diff_mc_inp["len_step"])

            # Create working lists for processors
            frame_np = [list(range(frame_start[i], frame_end[i])) for i in range(np)]

            # Run parallel search
            pool = mp.Pool(processes=np)
            results = [pool.apply_async(self._sample_helper, args=(frame_list, shift, is_pbc, is_broken,)) for frame_list in frame_np]
            pool.close()
            pool.join()
            output = [x.get() for x in results]

            # Destroy object
            del results
        else:
            # Run sampling
            output = [self._sample_helper(list(range(self._num_frame)), shift, is_pbc, is_broken)]

        # Concatenate output and create pickle object files
        system = {"sys": "pore", "props": self._pore_props} if self._pore else {"sys": "box", "props": {"length" :self._box}}
        inp = {"num_frame": self._num_frame, "mass": self._mol.get_mass(), "entry": self._entry}

        if self._is_density:
            inp_dens = inp.copy()
            inp_dens.update(self._dens_inp)
            inp_dens.pop("output")
            data_dens = output[0]["density"]
            for out in output[1:]:
                if self._pore:
                    data_dens["in"] = [x+y for x, y in zip(data_dens["in"], out["density"]["in"])]
                data_dens["ex"] = [x+y for x, y in zip(data_dens["ex"], out["density"]["ex"])]

            # Pickle
            results = {system["sys"]: system["props"], "inp": inp_dens, "data": data_dens, "type": "dens_bin"}
            utils.save(results, self._dens_inp["output"])


        if self._is_gyration:
            inp_gyr = inp.copy()
            inp_gyr.update(self._gyr_inp)
            inp_gyr.pop("output")
            data_gyr = output[0]["gyration"]
            for out in output[1:]:
                if self._pore:
                    data_gyr["in"] = [x+y for x, y in zip(data_gyr["in"], out["gyration"]["in"])]
                data_gyr["ex"] = [x+y for x, y in zip(data_gyr["ex"], out["gyration"]["ex"])]

            # Pickle
            results = {system["sys"]: system["props"], "inp": inp_gyr, "data": data_gyr, "type": "gyr_bin"}
            utils.save(results, self._gyr_inp["output"])

        if self._is_diffusion_bin:
            inp_diff = inp.copy()
            inp_diff.update(self._diff_bin_inp)
            inp_diff.pop("output")
            data_diff = output[0]["diffusion_bin"]
            for out in output[1:]:
                for i in range(self._diff_bin_inp["bin_num"]):
                    for j in range(self._diff_bin_inp["len_window"]):
                        data_diff["z"][i][j] += out["diffusion_bin"]["z"][i][j]
                        data_diff["r"][i][j] += out["diffusion_bin"]["r"][i][j]
                        data_diff["n"][i][j] += out["diffusion_bin"]["n"][i][j]
                        data_diff["z_tot"][i][j] += out["diffusion_bin"]["z_tot"][i][j]
                        data_diff["r_tot"][i][j] += out["diffusion_bin"]["r_tot"][i][j]
                        data_diff["n_tot"][i][j] += out["diffusion_bin"]["n_tot"][i][j]

            # Pickle
            results = {system["sys"]: system["props"], "inp": inp_diff, "data": data_diff, "type": "diff_bin"}
            utils.save(results, self._diff_bin_inp["output"])

        if self._is_diffusion_mc:
            inp_diff = inp.copy()
            inp_diff.update(self._diff_mc_inp)
            inp_diff.pop("output")
            data_diff = output[0]["diffusion_mc"]
            for step in self._diff_mc_inp["len_step"]:
                data_diff[step] = data_diff[step]
                for out in output[1:]:
                    data_diff[step] += out["diffusion_mc"][step]

            for step in self._diff_mc_inp["len_step"]:
                data_diff[step] = data_diff[step][1:-1,1:-1]

            # Save results in dictionary
            results = {system["sys"]: system["props"], "inp": inp_diff, "data": data_diff, "type": "diff_mc"}

            # Save dictionary to h5-file
            utils.save(results, self._diff_mc_inp["output"])


    def _sample_helper(self, frame_list, shift, is_pbc, is_broken):
        """Helper function for sampling run.

        Parameters
        ----------
        frame_list :
            List of frame ids to process
        shift : list
            Vector for translating atoms in nm
        is_pbc : bool
            True to apply periodic boundary conditions
        is_broken : bool
            True to check for broken molecules during sampling

        Returns
        -------
        output : dictionary
            Dictionary containing all sampled data
        """
        # Initialize
        box = self._pore_props["box"] if self._pore else self._box
        res = self._pore_props["res"] if self._pore else 0
        com_list = []
        idx_list = []

        # Create local data structures
        output = {}
        if self._is_density:
            output["density"] = self._density_data()
        if self._is_gyration:
            output["gyration"] = self._gyration_data()
        if self._is_diffusion_bin:
            output["diffusion_bin"] = self._diffusion_bin_data()
        if self._is_diffusion_mc:
            output["diffusion_mc"] = self._diffusion_mc_data()

        # Calculate length index and com lists
        if self._is_diffusion_bin:
            len_fill = self._diff_bin_inp["len_window"]*self._diff_bin_inp["len_step"]
        elif self._is_diffusion_mc:
            len_fill = self._diff_mc_inp["len_step"][-1]+1
        else:
            len_fill = 1

        # Load trajectory
        traj = cf.Trajectory(self._traj)
        frame_form = "%"+str(len(str(self._num_frame)))+"i"

        # Run through frames
        for frame_id in frame_list:
            # Read frame
            frame = traj.read_step(frame_id)
            positions = frame.positions

            # Add new dictionaries and remove unneeded references
            if len(com_list) >= len_fill:
                idx_list.pop(0)
                com_list.pop(0)
            idx_list.append({})
            com_list.append({})

            # Run through residues
            for res_id in self._res_list:
                # Get position vectors
                pos = [[positions[self._res_list[res_id][atom_id]][i]/10+shift[i] for i in range(3)] for atom_id in range(len(self._atoms))]

                # Calculate centre of mass
                com_no_pbc = [sum([pos[atom_id][i]*self._masses[atom_id] for atom_id in range(len(self._atoms))])/self._sum_masses for i in range(3)]

                # Check if molecule is broken
                if is_broken:
                    for i in range(3):
                        if abs(com_no_pbc[i]-pos[0][i])>box[i]/3:
                            print("Sample - Broken molecule found - ResID: "+"%5i"%res_id+", AtomID: "+"%5i"%atom_id)

                # Apply periodic boundary conditions
                if is_pbc:
                    com = [com_no_pbc[i]-math.floor(com_no_pbc[i]/box[i])*box[i] for i in range(3)]
                else:
                    com = com_no_pbc

                # Calculate distance towards center axis
                if self._pore:
                    if self._pore_props["type"]=="CYLINDER":
                        dist = geometry.length(geometry.vector([self._pore_props["focal"][0], self._pore_props["focal"][1], com[2]], com))
                    elif self._pore_props["type"]=="SLIT":
                        dist = abs(self._pore_props["focal"][1]-com[1])
                else:
                    dist = 0

                # Set region - in-interior, ex-exterior
                region = ""
                if self._pore and com[2] > res+self._entry and com[2] < box[2]-res-self._entry:
                    region = "in"
                elif not self._pore or com[2] < res or com[2] > box[2]-res:
                    region = "ex"

                # Remove window filling instances except from first processor
                if self._is_diffusion_bin:
                    is_sample = len(com_list)==len_fill or frame_id<=len_fill
                else:
                    is_sample = True

                # Sampling routines
                if is_sample:
                    if self._is_density:
                        self._density(output["density"], region, dist, com)
                    if self._is_gyration:
                        self._gyration(output["gyration"], region, dist, com_no_pbc, pos)
                if self._is_diffusion_bin:
                    self._diffusion_bin(output["diffusion_bin"], region, dist, com_list, idx_list, res_id, com)
                if self._is_diffusion_mc:
                    self._diffusion_mc(output["diffusion_mc"], idx_list, com, res_id, frame_list, frame_id)

            # Progress
            if (frame_id+1)%10==0 or frame_id==0 or frame_id==self._num_frame-1:
                sys.stdout.write("Finished frame "+frame_form%(frame_id+1)+"/"+frame_form%self._num_frame+"...\r")
                sys.stdout.flush()
        print()
        return output
