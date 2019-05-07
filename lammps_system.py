import numpy as np


class Particle(object):
    """
    Contains a single LAMMPS particle.
    """
    _id = 1

    def __init__(self, atom_type, coordinates, mol_type=0, mass=1):
        """

        Initialise a single particle.

        :param atom_type: LAMMPS atom type
        :type atom_type: int
        :param coordinates: x, y, and z coordinates of the particle
        :type coordinates: length-3 numpy.ndarray (most containers ought to work)
        :param mol_type: if the particle belongs to a molecule, this is that molecule's index
        :type mol_type: int
        :param mass: particle mass
        :type mass: float
        """
        self.id = Particle._id
        Particle._id += 1
        self.atom_type = atom_type
        self.mol_type = mol_type
        self.coordinates = coordinates.copy()
        self.mass = mass

    def __repr__(self):
        return "{} {} {} {:.16e} {:.16e} {:.16e} 0 0 0".format(self.id, self.mol_type, self.atom_type,
                                                               *self.coordinates)
        # if self.atom_type == 1 and self.mol_type == 1:
        #     wall = 1
        # elif self.atom_type == 1 and self.mol_type == 2:
        #     wall = 2
        # elif self.coordinates[2] == 1:
        #     wall = -1
        # elif self.coordinates[2] == 119:
        #     wall = -2
        # else:
        #     wall = 0
        # gp = self.coordinates.copy()
        # gp[2] = 0
        # data = [self.id, self.atom_type, self.coordinates[0], self.coordinates[1], self.coordinates[2], self.mol_type,
        #         wall, gp[0], gp[1], gp[2]]
        # fmt = "{:>5n}{:>5n}{:20.10f}{:20.10f}{:20.10f}{:>5n}{:>5n}{:20.10f}{:20.10f}{:20.10f}"
        # return fmt.format(*data)


class Bonds(object):
    """
    Contains the bonds in a LAMMPS system
    """

    def __init__(self):
        self.b_list = []
        self.bonds = 0
        self.bond_types = 0

    def add_bond(self, b_type, p_i, p_j):
        """

        Add a bond to the bond list.

        :param b_type: LAMMPS bond type
        :type b_type: int
        :param p_i: the first particle in the bond
        :type p_i: Particle
        :param p_j: the second particle in the bond
        :type p_j: Particle

        """
        b_i = p_i.id
        b_j = p_j.id
        self.b_list.append([b_type, b_i, b_j])
        self.bonds += 1
        self.bond_types = max(self.bond_types, b_type)

    def get_header(self):
        return "{:n} bonds\n{:n} bond types".format(self.bonds, self.bond_types)

    def __len__(self):
        return len(self.b_list)

    def __repr__(self):
        string = "\nBonds\n\n"
        fmt = "{:10n}{:10n}{:10n}{:10n}\n"
        for n, triple in enumerate(self.b_list):
            string += fmt.format(n + 1, *triple)
        return string.rstrip()


class Angles(object):
    """
    Contains the angles in a LAMMPS system
    """

    def __init__(self):
        self.a_list = []
        self.angles = 0
        self.angle_types = 0

    def add_angle(self, a_type, p_i, p_j, p_k):
        """

        Add an angle to the angle list.

        :param a_type: LAMMPS angle type
        :type a_type: int
        :param p_i: the first particle in the angle
        :type p_i: Particle
        :param p_j: the second particle in the angle
        :type p_j: Particle
        :param p_k: the third particle in the angle
        :type p_k: Particle

        """
        a_i = p_i.id
        a_j = p_j.id
        a_k = p_k.id
        self.a_list.append([a_type, a_i, a_j, a_k])
        self.angles += 1
        self.angle_types = max(self.angle_types, a_type)

    def get_header(self):
        return "{:<10n}angles\n{:<10n}angle types\n".format(self.angles, self.angle_types)

    def __len__(self):
        return len(self.a_list)

    def __repr__(self):
        string = "\nAngles\n\n"
        fmt = "{:10n}{:10n}{:10n}{:10n}{:10n}\n"
        for n, quad in enumerate(self.a_list):
            string += fmt.format(n + 1, *quad)
        return string.rstrip()


class Particles(object):
    """
    Contains particles, bonds, and angles and calculates other properties
    necessary to construct a LAMMPS datafile
    """

    def __init__(self):
        self.p_list = []
        self.limits = np.zeros(6)
        self.atoms = 0
        self.atom_types = 0
        self.mol_types = 0
        self.masses = {}
        self.bonds = Bonds()
        self.angles = Angles()
        self.elements = ["C", "N", "O", "F", "Si", "P", "S", "Cl"]

    def add_particle(self, p):
        """

        Add a particle to the system, updated the total atom types and the box
        limits accordingly.

        :param p: the particle to be added
        :type p: Particle

        """
        self.limits[0::2] = np.minimum(self.limits[0::2], p.coordinates)
        self.limits[1::2] = np.maximum(self.limits[1::2], p.coordinates)
        self.p_list.append(p)
        self.atoms += 1
        self.atom_types = max(self.atom_types, p.atom_type)
        self.mol_types = max(self.mol_types, p.mol_type)
        self.masses[p.atom_type] = p.mass

    def get_atom_mol_types(self):
        return self.atom_types, self.mol_types

    def add_bond(self, b_type, p_i, p_j):
        self.bonds.add_bond(b_type, p_i, p_j)

    def add_angle(self, a_type, p_i, p_j, p_k):
        self.angles.add_angle(a_type, p_i, p_j, p_k)

    def add_list(self, pl):
        """

        Add a list of particles to the system.

        :param pl: a list of particles.
        :type pl: list of Particles

        """
        [self.add_particle(p) for p in pl]

    def get_limits(self):
        lim_string = "{:.16e} {:.16e} xlo xhi\n{:.16e} {:.16e} ylo yhi\n{:.16e} {:.16e} zlo zhi\n\n"
        return lim_string.format(*self.limits)

    def get_masses(self):
        return "Masses\n\n" + "\n".join("{:n} {:n}".format(k, v) for (k, v) in self.masses.items()) + "\n\n"

    def get_header(self):
        string = "{:n} atoms\n{:n} atom types".format(self.atoms, self.atom_types)
        if len(self.bonds) > 0:
            string += "\n" + self.bonds.get_header()
        if len(self.angles) > 0:
            string += "\n" + self.angles.get_header()
        return string + "\n\n"

    def get_atoms_and_bonds(self):
        string = "Atoms\n\n" + "\n".join(p.__repr__() for p in self.p_list)
        if len(self.bonds) > 0:
            string += "\n" + self.bonds.__repr__()
        if len(self.angles) > 0:
            string += "\n" + self.angles.__repr__()
        return string

    def get_xyz_string(self):
        fmt = "{:<5s}{:20.10f}{:20.10f}{:20.10f}\n"
        xyz_string = "{}\n\n".format(self.atoms)
        for p in self.p_list:
            p_type = p.atom_type
            p_element = self.elements[p_type - 1]
            p_coords = p.coordinates
            xyz_string += fmt.format(p_element, *p_coords)
        return xyz_string

    def __repr__(self):
        string = "comment\n\n"
        string += self.get_header()
        string += self.get_limits()
        string += self.get_masses()
        string += self.get_atoms_and_bonds()
        return string
