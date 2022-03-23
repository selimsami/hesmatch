import numpy as np
from pulp import LpVariable, LpProblem, LpMaximize, lpSum


def calc_freq_diff(ref_freqs, match_freqs):
    diff = np.abs(ref_freqs - match_freqs)
    error = np.abs(diff / ref_freqs * 100)
    return diff, error


def do_matching(overlap_matrix, ref_freqs, match_freqs, match_modes, weight=100):
    """
    Match to the reference vibrational modes/frequencies by minimizing:
        Sum_ij[ ModeRef_i * ModeMatch_j + |FreqRef_i - FreqMatch_j| / weight ]

    A higher weight increases importance of matching the overlap matrix.

    Parameters
    ----------
    overlap_matrix : (n_mode, n_mode) Numpy array
    ref_freqs : (n_mode) Numpy array
    match_freqs : (n_mode) Numpy array
    match_modes : (n_mode, n_atom, 3) Numpy array
    weight : float, optional

    Returns
    -------
    chosen_overlaps : (n_mode) Numpy array
    match_freqs : (n_mode) Numpy array
    match_modes : (n_mode, n_atom, 3) Numpy array

    """
    chosen_overlaps, unordered_ref_freqs, unordered_match_freqs = [], [], []
    n_freqs = len(overlap_matrix)

    choices = LpVariable.dicts("choice", (range(n_freqs), range(n_freqs)), cat="Binary")
    prob = LpProblem("freq macher", LpMaximize)

    for i in range(n_freqs):
        prob += lpSum([choices[j][i] for j in range(n_freqs)]) == 1
        prob += lpSum([choices[i][j] for j in range(n_freqs)]) == 1

    prob += lpSum([choices[i][j]*(overlap_matrix[j][i]-abs(ref_freqs[i] - match_freqs[j])/weight)
                   for j in range(n_freqs) for i in range(n_freqs)])
    prob.solve()

    for var in prob.variables():
        if var.varValue == 1.0:
            i, j = [int(n) for n in var.name.split('_')[1:]]
            unordered_ref_freqs.append(ref_freqs[i])
            unordered_match_freqs.append(match_freqs[j])
            chosen_overlaps.append(overlap_matrix[j][i])

    sort = np.argsort(unordered_ref_freqs)
    match_freqs = np.array(unordered_match_freqs)[sort]
    chosen_overlaps = np.array(chosen_overlaps)[sort]
    match_modes = match_modes[sort]

    return chosen_overlaps, match_freqs, match_modes


def normalize_modes(modes):
    """
    Normalize the vibrational modes so that root mean square is equal to 1.
    """

    normalization = (modes * modes).sum(axis=1).sum(axis=1)**0.5
    return modes / normalization[:, np.newaxis, np.newaxis]


def calc_overlap_matrix(ref_modes, match_modes):
    """
    Calculate all combination of overlaps between ref and match vibrational modes.
    """
    return np.abs((match_modes[:, np.newaxis] * ref_modes[np.newaxis]).sum(axis=2).sum(axis=2))


def matcher(ref, matches):

    ref_freqs = ref.get_frequencies().real[6:]
    ref_modes = normalize_modes(ref.get_modes()[6:])

    for match in matches:
        match_freqs = match.get_frequencies().real[6:]
        match_modes = normalize_modes(match.get_modes()[6:])
        overlap_matrix = calc_overlap_matrix(ref_modes, match_modes)

        chosen_overlaps, match_freqs, match_modes = do_matching(overlap_matrix, ref_freqs,
                                                                match_freqs, match_modes)

        diff, error = calc_freq_diff(ref_freqs, match_freqs)
        print(chosen_overlaps, diff, error)
