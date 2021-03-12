################################################################################
# Sample                                                                       #
#                                                                              #
"""Sample functions to be run on clusters."""
################################################################################


import os
import sys
import math
import chemfiles as cf

import porems as pms
import poreana.utils as utils
import poreana.geometry as geometry


class Sample:
    """This class samples a trajectory to determine different properties.
    Different properties can be initialized to be run at the same time during
    the sampling run. The output is stored in form of pickle files for later
    calculation using methods provided in the package.

    Note that the sampling should be run on a cluster due to a high time and
    resource consumption.

    Parameters
    ----------
    link_pore : string
        Link to poresystem object file
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
    def __init__(self, link_pore, link_traj, mol, atoms=[], masses=[], entry=0.5):
        # Initialize
        self._pore_props = {}
        self._traj = link_traj
        self._mol = mol
        self._atoms = atoms
        self._masses = masses
        self._entry = entry
        self._pore = utils.load(link_pore)

        # Set analysis routines
        self._is_density = False
        self._is_diffusion_bin = False
        self._is_gyration = False

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

        # Check consistency
        if self._atoms and not len(self._masses) == len(self._atoms):
            print("Length of variables *atoms* and *masses* do not match!")
            return

        # Get pore properties
        self._pore_props["res"] = self._pore.reservoir()
        self._pore_props["focal"] = self._pore.centroid()
        self._pore_props["box"] = self._pore.box()
        self._pore_props["box"][2] += 2*self._pore_props["res"]

        # Get pore diameter
        if isinstance(self._pore, pms.PoreCylinder):
            self._pore_props["diam"] = self._pore.diameter()
        # elif isinstance(self._pore, pms.PoreSlit):
        #     self._pore_props["diam"] = self._pore.height()


    ###########
    # Density #
    ###########
    def init_density(self, link_out, bin_num=150):
        """Enable density sampling routine.

        Parameters
        ----------
        link_out : string
            Link to output object file
        bin_num : integer, optional
            Number of bins to be used
        """
        # Initialize
        res = self._pore_props["res"]
        diam = self._pore_props["diam"]
        self._is_density = True
        self._dens_out = link_out

        # Define bins
        self._dens_in = [[diam/2/bin_num*x for x in range(bin_num+2)], [0 for x in range(bin_num+1)]]
        self._dens_ex = [[res/bin_num*x for x in range(bin_num+1)], [0 for x in range(bin_num+1)]]

    def _density(self, region, dist, com):
        """This function samples the density inside and outside of the pore.

        All atoms are sampled each frame if they are inside or outside the
        bounds of the pore minus an optional entry length on both sides.

        Inside the pore the atom instances will be added to radial cylindric
        slices :math:`r_i-r_{i-1}` and outside to rectangular slices
        :math:`z_j-z_{j-1}` with pore radius :math:`r_i` of radial slice
        :math:`i` and length :math:`z_j` of slice :math:`j`.

        Parameters
        ----------
        region : string
            Indicator wether molecule is inside or outside pore
        dist : float
            Distance of center of mass to pore surface area
        com : list
            Center of mass of current molecule
        """
        bin_num = len(self._dens_in[1])-1

        # Add molecule to bin
        if region=="in":
            index = math.floor(dist/self._dens_in[0][1])
            if index <= bin_num:
                self._dens_in[1][index] += 1

        elif region=="ex":
            # Calculate distance to crystobalit and apply perodicity
            lentgh = com[2] if com[2] < self._pore_props["focal"][2] else abs(com[2]-self._pore_props["box"][2])
            index = math.floor(lentgh/self._dens_ex[0][1])

            # Only consider reservoir space in vicinity of crystobalit - remove pore
            if dist > self._pore_props["diam"]/2 and index <= bin_num:
                self._dens_ex[1][index] += 1


    ############
    # Gyration #
    ############
    def init_gyration(self, link_out, bin_num=150):
        """Enable gyration radius sampling routine.

        Parameters
        ----------
        link_out : string
            Link to output object file
        bin_num : integer, optional
            Number of bins to be used
        """
        # Initialize
        res = self._pore_props["res"]
        diam = self._pore_props["diam"]
        self._is_gyration = True
        self._gyr_out = link_out

        # Define bins
        self._gyr_in = [[diam/2/bin_num*x for x in range(bin_num+2)], [0 for x in range(bin_num+1)]]
        self._gyr_ex = [[res/bin_num*x for x in range(bin_num+1)], [0 for x in range(bin_num+1)]]

    def _gyration(self, region, dist, com, pos):
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
        bin_num = len(self._gyr_in[1])-1

        r_g = (sum([geometry.length(geometry.vector(pos[atom_id], com))**2*self._masses[atom_id] for atom_id in range(len(self._atoms))])/self._sum_masses)**0.5

        # Add molecule to bin
        if region=="in":
            index = math.floor(dist/self._gyr_in[0][1])
            if index <= bin_num:
                self._gyr_in[1][index] += r_g

        elif region=="ex":
            # Calculate distance to crystobalit and apply perodicity
            lentgh = com[2] if com[2] < self._pore_props["focal"][2] else abs(com[2]-self._pore_props["box"][2])
            index = math.floor(lentgh/self._gyr_ex[0][1])

            # Only consider reservoir space in vicinity of crystobalit - remove pore
            if dist > self._pore_props["diam"]/2 and index <= bin_num:
                self._gyr_ex[1][index] += r_g


    #############
    # Diffusion #
    #############
    def init_diffusion_bin(self, link_out, bin_num=50, len_obs=16e-12, len_frame=2e-12, len_step=2, bin_step_size=1):
        """Enable diffusion sampling routine.

        Parameters
        ----------
        link_out : string
            Link to output object file
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
        self._is_diffusion_bin = True
        self._diff_out = link_out
        self._diff_bin_step_size = bin_step_size
        self._diff_len_step = len_step
        self._diff_len_frame = len_frame

        # Define window length
        len_window = len_obs/len_step/len_frame+1
        if not len_window == int(len_window):
            obs_u = (math.ceil(len_window)-1)*len_step*len_frame
            obs_d = (math.floor(len_window)-1)*len_step*len_frame
            print("Observation length not possible with current inputs. Alternatively use len_obs="+"%.1e" % obs_u+" or len_obs="+"%.1e" % obs_d+".")
            return
        else:
            len_window = int(len_window)
        self._diff_len_window = len_window

        # Define bins
        self._diff_in = [self._pore_props["diam"]/2/bin_num*x for x in range(bin_num+2)]

        self._diff_bin_z = [[0 for y in range(len_window)] for x in range(bin_num+1)]
        self._diff_bin_r = [[0 for y in range(len_window)] for x in range(bin_num+1)]
        self._diff_bin_n = [[0 for y in range(len_window)] for x in range(bin_num+1)]

        self._diff_bin_tot_z = [[0 for y in range(len_window)] for x in range(bin_num+1)]
        self._diff_bin_tot_r = [[0 for y in range(len_window)] for x in range(bin_num+1)]
        self._diff_bin_tot_n = [[0 for y in range(len_window)] for x in range(bin_num+1)]

    def _diffusion_bin_step(self, idx):
        """Helper function to define allowed bin step list.

        Parameters
        ----------
        idx : integer
            Current bin index
        """
        out_list = [idx+x for x in range(self._diff_bin_step_size, 0, -1)]
        out_list += [idx]
        out_list += [idx-x for x in range(1, self._diff_bin_step_size+1)]
        return out_list

    def _diffusion_bin(self, region, dist, com_list, idx_list, res_id, com):
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
        link_out : string
            Link to output object file
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
        bin_num = len(self._diff_in)-2
        len_step = self._diff_len_step
        len_window = self._diff_len_window

        # Only sample diffusion inside the pore
        if region == "in":
            # Calculate bin index
            index = math.floor(dist/self._diff_in[1])

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
                        self._diff_bin_tot_z[idx_ref][i] += msd_z[i]
                        self._diff_bin_tot_r[idx_ref][i] += msd_r[i]
                        self._diff_bin_tot_n[idx_ref][i] += norm[i]

                        # Add to bin calculation list if msd is permissible
                        if len_msd == len_window:
                            self._diff_bin_z[idx_ref][i] += msd_z[i]
                            self._diff_bin_r[idx_ref][i] += msd_r[i]
                            self._diff_bin_n[idx_ref][i] += norm[i]


    ############
    # Sampling #
    ############
    def sample(self, is_force=False):
        """This function runs all enabled sampling routines. The output is
        stored in form of pickle files for later calculation using methods
        provided in the package.

        Note that the sampling should be run on a cluster due to a high time and
        resource consumption.

        Parameters
        ----------
        is_force : bool, optional
            True to overwrite existing object files
        """
        # Load trajectory
        mol = self._mol
        box = self._pore_props["box"]
        res = self._pore_props["res"]
        traj = cf.Trajectory(self._traj)
        num_frame = traj.nsteps
        res_list = {}
        com_list = []
        idx_list = []

        # Export inputs
        self._inp = {"frame": num_frame, "mass": mol.get_mass(),
                     "entry": self._entry, "res": self._pore_props["res"],
                     "diam": self._pore_props["diam"], "box": self._pore_props["box"]}

        # Run through frames
        for frame_id in range(num_frame):
            # Read frame
            frame = traj.read()
            positions = frame.positions

            # Add new dictionaries and remove unneeded references
            if self._is_diffusion_bin and frame_id >= (self._diff_len_window*self._diff_len_step):
                com_list.pop(0)
                idx_list.pop(0)
            com_list.append({})
            idx_list.append({})

            # Create list of relevant atom ids
            if not res_list:
                # Get number of residues in system
                num_res = len(frame.topology.atoms)/mol.get_num()

                # Check number of residues
                if abs(int(num_res)-num_res) >= 1e-5:
                    print("Number of atoms is inconsistent with number of residues.")
                    return

                # Check relevant atoms
                for res_id in range(int(num_res)):
                    res_list[res_id] = [res_id*mol.get_num()+atom for atom in range(mol.get_num()) if atom in self._atoms]

            # Run through residues
            for res_id in res_list:
                # Get position vectors
                pos = [[positions[res_list[res_id][atom_id]][i]/10 for i in range(3)] for atom_id in range(len(self._atoms))]

                # Calculate centre of mass
                com = [sum([pos[atom_id][i]*self._masses[atom_id] for atom_id in range(len(self._atoms))])/self._sum_masses for i in range(3)]

                # Remove edge molecules
                is_edge = False
                for i in range(3):
                    is_edge = True if abs(com[i]-pos[0][i])>self._pore_props["res"] else is_edge

                # Check if com was calculated on edges
                if not is_edge:
                    # Calculate distance towards center axis
                    if isinstance(self._pore, pms.PoreCylinder):
                        dist = geometry.length(geometry.vector([self._pore_props["focal"][0], self._pore_props["focal"][1], com[2]], com))
                    # elif isinstance(self._pore, pms.PoreSlit):
                    #     dist = abs(self._pore_props["focal"][1]-com[1])

                    # Set region - in-inside, ex-outside
                    region = ""
                    if com[2] > res+self._entry and com[2] < box[2]-res-self._entry:
                        region = "in"
                    elif com[2] <= res or com[2] > box[2]-res:
                        region = "ex"

                    # Sample density
                    if self._is_density:
                        self._density(region, dist, com)
                    if self._is_gyration:
                        self._gyration(region, dist, com, pos)
                    if self._is_diffusion_bin:
                        self._diffusion_bin(region, dist, com_list, idx_list, res_id, com)

            # Progress
            sys.stdout.write("Finished frame "+"%4i" % (frame_id+1)+"/"+"%4i" % num_frame+"...\r")
            sys.stdout.flush()
        print()

        # Save pickle object files
        if self._is_density:
            utils.save({"inp": self._inp, "in": self._dens_in, "ex": self._dens_ex}, self._dens_out)
        if self._is_gyration:
            utils.save({"inp": self._inp, "in": self._gyr_in, "ex": self._gyr_ex}, self._gyr_out)
        if self._is_diffusion_bin:
            inp = {key: val for key, val in self._inp.items()}
            inp["bins"] = len(self._diff_in)-2
            inp["step"] = self._diff_len_step
            inp["window"] = self._diff_len_window
            inp["frame"] = self._diff_len_frame
            utils.save({"inp": inp, "bins": self._diff_in,
                        "axial":  self._diff_bin_z, "axial_tot":  self._diff_bin_tot_z,
                        "radial": self._diff_bin_r, "radial_tot": self._diff_bin_tot_r,
                        "norm":   self._diff_bin_n, "norm_tot":   self._diff_bin_tot_n}, self._diff_out)
