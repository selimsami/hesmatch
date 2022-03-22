from typing import Sequence, Union
from numbers import Real
import numpy as np
from ase.vibrations import VibrationsData
from ase import Atoms


class VibrationsData(VibrationsData):

    @classmethod
    def from_lower_triangle(cls, atoms: Atoms,
                hessian_lower_triangle: Union[Sequence[Sequence[Real]], np.ndarray],
                indices: Sequence[int] = None) -> 'VibrationsData':
        """Instantiate VibrationsData when the Hessian is given as the
        lower triangle of the matrix in ((3N)**2+3N)/2 format

        Args:
            atoms: Equilibrium geometry of vibrating system

            hessian: Second-derivative in energy with respect to
                Cartesian nuclear movements as a ((3N)**2+3N)/2 array.

            indices: Indices of (non-frozen) atoms included in Hessian

        """
        if indices is None:
            indices = range(len(atoms))
        assert indices is not None  # Show Mypy that indices is now a sequence

        hessian_lower_triangle_array = np.asarray(hessian_lower_triangle)
        n_atoms = cls._check_dimensions(atoms, hessian_lower_triangle_array,
                                        indices=indices, triangle=True)

        hessian_2d_array = np.zeros((3*n_atoms, 3*n_atoms))

        count = 0
        for i in range(3*n_atoms):
            for j in range(i+1):
                hessian_2d_array[i, j] = hessian_lower_triangle_array[count]
                hessian_2d_array[j, i] = hessian_2d_array[i, j]
                count += 1

        return cls(atoms, hessian_2d_array.reshape(n_atoms, 3, n_atoms, 3),
                   indices=indices)

    @classmethod
    def from_upper_triangle(cls, atoms: Atoms,
                hessian_upper_triangle: Union[Sequence[Sequence[Real]], np.ndarray],
                indices: Sequence[int] = None) -> 'VibrationsData':
        """Instantiate VibrationsData when the Hessian is given as the
        upper triangle of the matrix in ((3N)**2+3N)/2 format

        Args:
            atoms: Equilibrium geometry of vibrating system

            hessian: Second-derivative in energy with respect to
                Cartesian nuclear movements as a ((3N)**2+3N)/2 array.

            indices: Indices of (non-frozen) atoms included in Hessian

        """
        if indices is None:
            indices = range(len(atoms))
        assert indices is not None  # Show Mypy that indices is now a sequence

        hessian_upper_triangle_array = np.asarray(hessian_upper_triangle)
        n_atoms = cls._check_dimensions(atoms, hessian_upper_triangle_array,
                                        indices=indices, triangle=True)

        hessian_2d_array = np.zeros((3*n_atoms, 3*n_atoms))

        count = 0
        for i in range(3*n_atoms):
            for j in range(i, 3*n_atoms):
                hessian_2d_array[i, j] = hessian_upper_triangle_array[count]
                hessian_2d_array[j, i] = hessian_2d_array[i, j]
                count += 1

        return cls(atoms, hessian_2d_array.reshape(n_atoms, 3, n_atoms, 3),
                   indices=indices)

    @staticmethod
    def _check_dimensions(atoms: Atoms,
                          hessian: np.ndarray,
                          indices: Sequence[int],
                          two_d: bool = False,
                          triangle: bool = False) -> int:
        """Sanity check on array shapes from input data

        Args:
            atoms: Structure
            indices: Indices of atoms used in Hessian
            hessian: Proposed Hessian array
            two_d: Whether the Hessian is in 2D format
            triangle: Whether the Hessian is in 1D triangle format

        Returns:
            Number of atoms contributing to Hessian

        Raises:
            ValueError if Hessian dimensions does not match the reference shape

        """

        n_atoms = len(atoms[indices])

        if two_d:
            ref_shape = [n_atoms * 3, n_atoms * 3]
            ref_shape_txt = '{n:d}x{n:d}'.format(n=(n_atoms * 3))

        elif triangle:
            ref_shape = [((n_atoms*3)**2+n_atoms*3)/2]
            ref_shape_txt = '{n:d}'.format(n=int(((n_atoms*3)**2+n_atoms*3)/2))

        else:
            ref_shape = [n_atoms, 3, n_atoms, 3]
            ref_shape_txt = '{n:d}x3x{n:d}x3'.format(n=n_atoms)

        if (isinstance(hessian, np.ndarray)
            and hessian.shape == tuple(ref_shape)):
            return n_atoms
        else:
            raise ValueError("Hessian for these atoms should be a "
                             "{} numpy array.".format(ref_shape_txt))
