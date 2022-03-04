from .hessian import Hessian


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

    ref = Hessian(ref_hessian, ref_format, ref_unit, masses)
    matches = [Hessian(match_hessian, match_format, match_unit, masses) for match_hessian in
               match_hessians]

    for i, hes in enumerate([ref]+matches):
        print(f'\n\n\n\nHESSIAN NUMBER: {i+1}\n')
        print('natoms', hes.n_atoms)
        print('\neigval', hes.eigval)
        print('\neigvec', hes.eigvec)
