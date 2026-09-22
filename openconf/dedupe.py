"""Conservative Cartesian RMSD conformer deduplication."""

from collections.abc import Sequence
from itertools import permutations

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.intp]

# Rotation and permutation assignment each depend on the other, so they are refined in
# alternating sweeps. Convergence is detected, so this is only an upper bound.
_PERMUTATION_SWEEPS = 4


def rmsd_dedupe(
    mol: Chem.Mol,
    conf_ids: list[int],
    energies: Sequence[float] | None = None,
    *,
    use_heavy_atoms_only: bool = True,
    rmsd_threshold: float = 0.25,
    max_atom_deviation: float | None = None,
) -> list[int]:
    """Deduplicate conformers using fixed-correspondence Cartesian RMSD.

    Conformers are aligned with proper-rotation Kabsch fits. Atom
    correspondence remains fixed except that, in all-atom mode, two to four
    hydrogens bonded to the same heavy atom may permute. No moment-of-inertia
    or independent rotor-symmetry pruning is applied.

    Args:
        mol: molecule containing conformers
        conf_ids: conformer identifiers to process
        energies: optional energies aligned with `conf_ids`; lowest-energy
            duplicate representative is retained
        use_heavy_atoms_only: use only heavy atoms for comparison
        rmsd_threshold: RMSD below which conformers are duplicates
        max_atom_deviation: maximum aligned displacement allowed for every
            compared atom; defaults to twice `rmsd_threshold`

    Returns:
        Retained conformer identifiers

    Raises:
        ValueError: inputs or thresholds are invalid
    """
    if not np.isfinite(rmsd_threshold) or rmsd_threshold <= 0.0:
        raise ValueError("rmsd_threshold must be finite and positive")
    if max_atom_deviation is None:
        max_atom_deviation = 2.0 * rmsd_threshold
    if not np.isfinite(max_atom_deviation) or max_atom_deviation <= 0.0:
        raise ValueError("max_atom_deviation must be finite and positive")
    if energies is not None and len(energies) != len(conf_ids):
        raise ValueError("energies must align with conf_ids")
    if len(set(conf_ids)) != len(conf_ids):
        raise ValueError("conf_ids must be unique")
    if len(conf_ids) <= 1:
        return conf_ids.copy()

    if energies is None:
        order = np.arange(len(conf_ids))
    else:
        energy_array = np.asarray(energies, dtype=float)
        if np.any(np.isnan(energy_array)):
            raise ValueError("energies must not contain NaN")
        order = np.argsort(energy_array, kind="stable")

    ordered_ids = [conf_ids[int(index)] for index in order]
    atom_indices = _selected_atom_indices(mol, use_heavy_atoms_only)
    coordinates = _extract_coordinates(mol, ordered_ids, atom_indices)
    permutation_groups = () if use_heavy_atoms_only else _same_parent_hydrogen_groups(mol)
    mask = _prune_coordinates(
        coordinates,
        rmsd_threshold=rmsd_threshold,
        max_atom_deviation=max_atom_deviation,
        permutation_groups=permutation_groups,
    )
    return [conf_id for conf_id, keep in zip(ordered_ids, mask, strict=True) if keep]


def _selected_atom_indices(mol: Chem.Mol, use_heavy_atoms_only: bool) -> IntArray:
    if use_heavy_atoms_only:
        indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
        if indices:
            return np.asarray(indices, dtype=np.intp)
    return np.arange(mol.GetNumAtoms(), dtype=np.intp)


def _extract_coordinates(
    mol: Chem.Mol,
    conf_ids: Sequence[int],
    atom_indices: IntArray,
) -> FloatArray:
    coordinates = np.empty((len(conf_ids), len(atom_indices), 3), dtype=float)
    for index, conf_id in enumerate(conf_ids):
        coordinates[index] = mol.GetConformer(int(conf_id)).GetPositions()[atom_indices]
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("conformer coordinates must be finite")
    return coordinates


def _same_parent_hydrogen_groups(mol: Chem.Mol) -> tuple[IntArray, ...]:
    groups: list[IntArray] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        hydrogen_indices = sorted(
            neighbor.GetIdx()
            for neighbor in atom.GetNeighbors()
            if neighbor.GetAtomicNum() == 1 and neighbor.GetDegree() == 1
        )
        if 2 <= len(hydrogen_indices) <= 4:
            groups.append(np.asarray(hydrogen_indices, dtype=np.intp))
    return tuple(groups)


def _prune_coordinates(
    coordinates: FloatArray,
    *,
    rmsd_threshold: float,
    max_atom_deviation: float,
    permutation_groups: Sequence[IntArray],
) -> NDArray[np.bool_]:
    centered = coordinates - coordinates.mean(axis=1, keepdims=True)
    radial_distances = np.linalg.norm(centered, axis=2)
    for group in permutation_groups:
        radial_distances[:, group] = np.sort(radial_distances[:, group], axis=1)

    group_permutations = tuple(
        np.asarray(list(permutations(range(len(group)))), dtype=np.intp) for group in permutation_groups
    )

    keep = np.ones(len(coordinates), dtype=bool)
    kept_indices: list[int] = []
    for candidate_index in range(len(coordinates)):
        if not kept_indices:
            kept_indices.append(candidate_index)
            continue

        reference_indices = np.asarray(kept_indices, dtype=np.intp)
        radial_differences = radial_distances[reference_indices] - radial_distances[candidate_index]
        squared_radial_differences = radial_differences**2
        viable = (np.mean(squared_radial_differences, axis=1) < rmsd_threshold**2) & (
            np.max(squared_radial_differences, axis=1) < max_atom_deviation**2
        )
        viable_reference_indices = reference_indices[viable]

        if not len(viable_reference_indices):
            is_duplicate = False
        else:
            if permutation_groups:
                rmsds, max_deviations = _aligned_metrics_with_permutations(
                    centered[viable_reference_indices],
                    centered[candidate_index],
                    permutation_groups,
                    group_permutations,
                )
            else:
                rmsds, max_deviations = _aligned_metrics_to_centered_references(
                    centered[viable_reference_indices],
                    centered[candidate_index],
                )
            is_duplicate = bool(np.any((rmsds < rmsd_threshold) & (max_deviations < max_atom_deviation)))

        if is_duplicate:
            keep[candidate_index] = False
        else:
            kept_indices.append(candidate_index)

    return keep


def _superpose_to_references(
    centered_references: FloatArray,
    centered_candidates: FloatArray,
) -> FloatArray:
    """Rotate every candidate onto its paired reference with a proper-rotation Kabsch fit.

    Args:
        centered_references: origin-centred reference coordinates, one per reference
        centered_candidates: origin-centred candidate coordinates, one per reference

    Returns:
        Candidate coordinates rotated onto their references
    """
    covariance = np.einsum("kma,kmb->kab", centered_candidates, centered_references)
    left, _, right_transpose = np.linalg.svd(covariance)
    reflections = np.linalg.det(left @ right_transpose) < 0
    left[reflections, :, -1] *= -1
    return np.einsum("kma,kab->kmb", centered_candidates, left @ right_transpose)


def _aligned_metrics_with_permutations(
    centered_references: FloatArray,
    centered_candidate: FloatArray,
    permutation_groups: Sequence[IntArray],
    group_permutations: Sequence[IntArray],
) -> tuple[FloatArray, FloatArray]:
    """Align one candidate to every reference, permuting equivalent atoms within groups.

    Refines one group at a time, keeping an arrangement only when it strictly improves
    (RMSD, maximum deviation), and stops once a sweep changes nothing. Each arrangement is
    scored under its own superposition, because a rotation fitted to the wrong atom
    correspondence can make the correct arrangement look worse than the identity and strand
    the search at a fixed point.

    All references and all arrangements of a group are superposed in one batch, which is
    what makes this affordable: the work is unchanged but it lands in a handful of stacked
    SVD calls instead of one call per reference per arrangement.

    Args:
        centered_references: origin-centred reference coordinates
        centered_candidate: origin-centred candidate coordinates
        permutation_groups: atom indices whose assignment may permute
        group_permutations: arrangements to consider per group, ordered as permutation_groups

    Returns:
        Best per-reference RMSD and maximum aligned atom displacement found

    Note:
        Returns the best metrics seen during refinement, which need not come from the final
        arrangement, matching the behaviour of the scalar search this replaces.
    """
    reference_count, atom_count = centered_references.shape[:2]
    rows = np.arange(reference_count)
    mapping = np.broadcast_to(np.arange(atom_count, dtype=np.intp), (reference_count, atom_count)).copy()
    best_rmsds, best_maxima = _metrics_for_mappings(centered_references, centered_candidate, mapping)

    for _ in range(_PERMUTATION_SWEEPS):
        changed = False
        for group, arrangements in zip(permutation_groups, group_permutations, strict=True):
            trial_count = len(arrangements)
            trials = np.repeat(mapping[:, None, :], trial_count, axis=1)
            trials[:, :, group] = group[arrangements]
            trial_rmsds, trial_maxima = _metrics_for_mappings(
                np.repeat(centered_references[:, None], trial_count, axis=1).reshape(-1, atom_count, 3),
                centered_candidate,
                trials.reshape(-1, atom_count),
            )
            trial_rmsds = trial_rmsds.reshape(reference_count, trial_count)
            trial_maxima = trial_maxima.reshape(reference_count, trial_count)

            # The current arrangement is always among the trials, so it supplies the
            # baseline that a replacement has to beat outright.
            current = np.all(group[arrangements][None] == mapping[:, group][:, None], axis=2).argmax(axis=1)
            tie_break = np.broadcast_to(np.arange(trial_count), (reference_count, trial_count))
            proposed = np.lexsort((tie_break, trial_maxima, trial_rmsds), axis=1)[:, 0]
            chosen = np.where(
                _strictly_better(
                    trial_rmsds[rows, proposed],
                    trial_maxima[rows, proposed],
                    trial_rmsds[rows, current],
                    trial_maxima[rows, current],
                ),
                proposed,
                current,
            )
            if not np.array_equal(chosen, current):
                mapping[:, group] = group[arrangements[chosen]]
                changed = True

            improved = _strictly_better(trial_rmsds[rows, chosen], trial_maxima[rows, chosen], best_rmsds, best_maxima)
            best_rmsds = np.where(improved, trial_rmsds[rows, chosen], best_rmsds)
            best_maxima = np.where(improved, trial_maxima[rows, chosen], best_maxima)
        if not changed:
            break

    return best_rmsds, best_maxima


def _strictly_better(
    rmsds: FloatArray,
    maxima: FloatArray,
    reference_rmsds: FloatArray,
    reference_maxima: FloatArray,
) -> NDArray[np.bool_]:
    """Compare (RMSD, maximum deviation) pairs lexicographically.

    Args:
        rmsds: candidate RMSD values
        maxima: candidate maximum deviations
        reference_rmsds: RMSD values to beat
        reference_maxima: maximum deviations to beat

    Returns:
        Whether each candidate pair sorts strictly before its reference pair
    """
    return (rmsds < reference_rmsds) | ((rmsds == reference_rmsds) & (maxima < reference_maxima))


def _metrics_for_mappings(
    centered_references: FloatArray,
    centered_candidate: FloatArray,
    mappings: IntArray,
) -> tuple[FloatArray, FloatArray]:
    """Superpose a candidate onto references under per-reference atom mappings.

    Args:
        centered_references: origin-centred reference coordinates
        centered_candidate: origin-centred candidate coordinates
        mappings: candidate atom index per reference and position

    Returns:
        Per-reference RMSD and maximum aligned atom displacement
    """
    candidates = np.broadcast_to(centered_candidate, (len(mappings), *centered_candidate.shape))
    permuted = np.take_along_axis(candidates, mappings[..., None], 1)
    displacements = np.linalg.norm(
        _superpose_to_references(centered_references, permuted) - centered_references, axis=2
    )
    return (
        np.sqrt(np.mean(displacements**2, axis=1)),
        np.max(displacements, axis=1),
    )


def _aligned_metrics_to_centered_references(
    centered_references: FloatArray,
    centered_candidate: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    covariance = np.einsum(
        "ma,kmb->kab",
        centered_candidate,
        centered_references,
    )
    left, _, right_transpose = np.linalg.svd(covariance)
    reflections = np.linalg.det(left @ right_transpose) < 0
    left[reflections, :, -1] *= -1
    rotations = left @ right_transpose
    aligned = np.einsum("ma,kab->kmb", centered_candidate, rotations)
    displacements = np.linalg.norm(aligned - centered_references, axis=2)
    return (
        np.sqrt(np.mean(displacements**2, axis=1)),
        np.max(displacements, axis=1),
    )
