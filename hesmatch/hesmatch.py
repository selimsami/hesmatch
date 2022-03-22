from ase import Atoms
import numpy as np
from .hessian import VibrationsData
from ase.units import Hartree, mol, kJ, Bohr, nm


def hesmatch(ref_hessian, match_hessians, masses=None, ref_format='2d', match_format='2d',
             ref_unit=1, match_unit=1):
    """

    Parameters
    ----------
    ref_hessian : TYPE
        DESCRIPTION.
    match_hessians : TYPE
        DESCRIPTION.
    masses : TYPE, optional
        DESCRIPTION. The default is None.
    ref_format : TYPE, optional
        DESCRIPTION. The default is '2d'.
    match_format : TYPE, optional
        DESCRIPTION. The default is '2d'.
    ref_unit : TYPE, optional
        DESCRIPTION. The default is 1.
    match_unit : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    None.


    """

    ref = do_vibrational_analysis(ref_hessian, ref_format, ref_unit, masses)
    matches = [do_vibrational_analysis(match_hessian, match_format, match_unit, masses)
               for match_hessian in match_hessians]

    for i, hes in enumerate([ref]+matches):
        print(f'\n\n\n\nHESSIAN NUMBER: {i+1}\n')
        print('\neigval', hes.get_frequencies())
        print()
        print(hes.get_modes())


def do_vibrational_analysis(hessian, hes_format, unit, masses):
    n_atoms = len(masses)
    molecule = Atoms(numbers=np.ones(n_atoms), masses=masses)

    if unit == 1:  # kJ mol-1 A-2 to eV A-2
        hessian *= kJ / mol
    if unit == 2:  # kJ mol-1 nm-2 to eV A-2
        hessian *= kJ / mol / nm**2
    elif unit == 3:  # Hartree Bohr-2 to eV A-2
        hessian *= Hartree / Bohr**2

    if hes_format == '2d':
        vib_data = VibrationsData.from_2d(molecule, hessian)
    elif hes_format == 'upper':
        vib_data = VibrationsData.from_upper_triangle(molecule, hessian)
    elif hes_format == 'lower':
        vib_data = VibrationsData.from_lower_triangle(molecule, hessian)

    return vib_data
