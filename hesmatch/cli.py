from colt import from_commandline
import numpy as np
from .hesmatch import hesmatch


@from_commandline("""
    # Path of the reference Hessian file
    ref_file = :: existing_file

    # Path of the matched Hessian file(s)
    match_file = :: list(existing_file)

    # File containing the mass of the atoms for mass-weighed analysis
    mass_file = :: existing_file, optional

    # Format of the provided Hessian matrix for the reference
    ref_format = 2d :: str :: [2d, upper, lower]

    # Format of the matched Hessian matrices
    match_format = 2d :: str :: [2d, upper, lower]

    # Units of the reference Hessian matrix (1: kJ mol-1 A-2, 2: kJ mol-1 nm-2, 3: Hartree Bohr-2)
    ref_unit = 1 :: int :: [1, 2, 3]

    # Units of the matched Hessian matrices (1: kJ mol-1 A-2, 2: kJ mol-1 nm-2, 3: Hartree Bohr-2)
    match_unit = 1 :: int :: [1, 2, 3]

    """, description={'alias': 'hesmatch'})
def cli(ref_file, match_file, mass_file, ref_format, match_format, ref_unit, match_unit):

    ref = read_hessian([ref_file], ref_format)[0]
    match = read_hessian(match_file, match_format)

    if mass_file:
        masses = read_1d_file(mass_file)
    else:
        masses = None

    hesmatch(ref, match, masses, ref_format, match_format, ref_unit, match_unit)


def read_hessian(hes_files, hes_format):
    hessians = []
    for hes_file in hes_files:
        if hes_format == '2d':
            hessians.append(read_2d_file(hes_file))
        else:
            hessians.append(read_1d_file(hes_file))
    return hessians


def read_2d_file(file):
    return np.loadtxt(file)


def read_1d_file(file):
    data = []
    with open(file, 'r') as f:
        for line in f:
            if line.strip != '':
                data.extend(line.split())
    return np.array(data, dtype=float)


if __name__ == '__main__':
    cli()
