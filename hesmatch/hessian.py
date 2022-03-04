import numpy as np
from scipy.linalg import eigh
from ase.units import Hartree, mol, kJ, Bohr, nm


class Hessian():
    def __init__(self, hessian, hes_format, hes_unit, masses):
        self.masses = masses

        if hes_format == 'upper':
            self.n_atoms = int((np.sqrt(8*hessian.size+1)-1)/6)
            self.matrix = self.from_upper_to_matrix(hessian)
        elif hes_format == 'lower':
            self.n_atoms = int((np.sqrt(8*hessian.size+1)-1)/6)
            self.matrix = self.from_lower_to_matrix(hessian)
        else:
            self.n_atoms = int(np.sqrt(hessian.size)/3)
            self.matrix = hessian

        if hes_unit == 2:  # kJ mol-1 nm-2 to kJ mol-1 A-2
            self.matrix /= nm**2
        elif hes_unit == 3:  # Hartree Bohr-2 to kJ mol-1 A-2
            self.matrix *= Hartree * mol / kJ / Bohr**2

        if np.any(masses):
            self.matrix = self.mass_weigh()

        self.eigval, self.eigvec = self.diagonalize()

    def from_upper_to_matrix(self, upper):
        """
        Parameters
        ----------
        upper : 1D Numpy array
            Hessian matrix in upper triangle format

        Returns
        -------
        matrix : 2D Numpy array
            Hessian matrix as a (3n, 3n) 2D matrix
        """
        matrix = np.zeros((3*self.n_atoms, 3*self.n_atoms))
        count = 0
        for i in range(3*self.n_atoms):
            for j in range(i, 3*self.n_atoms):
                matrix[i, j] = upper[count]
                matrix[j, i] = matrix[i, j]
                count += 1
        return matrix

    def from_lower_to_matrix(self, lower):
        """
        Parameters
        ----------
        lower : 1D Numpy array
            Hessian matrix in lower triangle format

        Returns
        -------
        matrix : 2D Numpy array
            Hessian matrix as a (3n, 3n) 2D matrix

        """
        matrix = np.zeros((3*self.n_atoms, 3*self.n_atoms))
        count = 0
        for i in range(3*self.n_atoms):
            for j in range(i+1):
                matrix[i, j] = lower[count]
                matrix[j, i] = matrix[i, j]
                count += 1
        return matrix

    def mass_weigh(self):
        """
        Returns
        -------
        2D Numpy array (3n, 3n)
            Mass-weighted 2D Hessian matrix
        """
        mass_sq = self.masses * self.masses[:, np.newaxis]
        mass_sq = np.repeat(mass_sq, 3, axis=0)
        mass_sq = np.repeat(mass_sq, 3, axis=1)
        return self.matrix/np.sqrt(mass_sq)

    def diagonalize(self):
        """
        Returns
        -------
        freq : 1D Numpy array
            Eigenvalues, which are the vibrational frequencies (cm-1)
        vec : 3D Numpy array (3n-6, n, 3)
            Eigenvectors, which are the vibrational modes
        """

        # Need to switch these units to ase.units
        const_amu = 1.6605389210e-27
        const_avogadro = 6.0221412900e+23
        const_speedoflight = 299792.458
        kj2j = 1e3
        ang2meter = 1e-10
        to_omega2 = kj2j/ang2meter**2/(const_avogadro*const_amu)  # 1/s**2
        to_waveno = 1e-5/(2.0*np.pi*const_speedoflight)  # cm-1

        val, vec = eigh(self.matrix)
        vec = np.reshape(np.transpose(vec), (3*self.n_atoms, self.n_atoms, 3))[6:]

        if np.any(self.masses):
            for i in range(self.n_atoms):
                vec[:, i, :] = vec[:, i, :] / np.sqrt(self.masses[i])

        # For linear molecules 5: ? Is the clipping necessary?
        freq = np.sqrt(val.clip(min=0)[6:] * to_omega2) * to_waveno
        return freq, vec
