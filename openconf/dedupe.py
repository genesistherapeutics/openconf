"""Conservative Cartesian RMSD conformer deduplication."""

from collections.abc import Sequence
from itertools import permutations

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.intp]


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

    keep = np.ones(len(coordinates), dtype=bool)
    kept_indices: list[int] = []
    for candidate_index, candidate in enumerate(coordinates):
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
        elif permutation_groups:
            is_duplicate = any(
                _is_similar(
                    coordinates[reference_index],
                    candidate,
                    rmsd_threshold,
                    max_atom_deviation,
                    permutation_groups,
                )
                for reference_index in viable_reference_indices
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


def _is_similar(
    reference: FloatArray,
    candidate: FloatArray,
    rmsd_threshold: float,
    max_atom_deviation: float,
    permutation_groups: Sequence[IntArray],
) -> bool:
    rmsd, maximum = _aligned_rmsd_and_max(
        reference,
        candidate,
        permutation_groups,
    )
    return rmsd < rmsd_threshold and maximum < max_atom_deviation


def _aligned_rmsd_and_max(
    reference: FloatArray,
    candidate: FloatArray,
    permutation_groups: Sequence[IntArray],
) -> tuple[float, float]:
    mapping = np.arange(len(reference), dtype=np.intp)
    best_rmsd, best_maximum, *_ = _aligned_metrics(reference, candidate)

    for _ in range(4):
        changed = False
        for group in permutation_groups:
            group_mapping = mapping
            group_rmsd, group_maximum, *_ = _aligned_metrics(
                reference,
                candidate[mapping],
            )
            for permutation in permutations(group.tolist()):
                trial_mapping = mapping.copy()
                trial_mapping[group] = permutation
                rmsd, maximum, *_ = _aligned_metrics(
                    reference,
                    candidate[trial_mapping],
                )
                if (rmsd, maximum) < (group_rmsd, group_maximum):
                    group_mapping = trial_mapping
                    group_rmsd = rmsd
                    group_maximum = maximum

            if not np.array_equal(group_mapping, mapping):
                mapping = group_mapping
                changed = True
            if (group_rmsd, group_maximum) < (best_rmsd, best_maximum):
                best_rmsd = group_rmsd
                best_maximum = group_maximum
        if not changed:
            break

    return best_rmsd, best_maximum


def _aligned_metrics(
    reference: FloatArray,
    candidate: FloatArray,
) -> tuple[float, float, FloatArray]:
    reference_centered = reference - reference.mean(axis=0)
    candidate_centered = candidate - candidate.mean(axis=0)
    covariance = candidate_centered.T @ reference_centered
    left, _, right_transpose = np.linalg.svd(covariance)
    if np.linalg.det(left @ right_transpose) < 0:
        left[:, -1] *= -1
    rotation = left @ right_transpose
    displacements = np.linalg.norm(
        candidate_centered @ rotation - reference_centered,
        axis=1,
    )
    return (
        float(np.sqrt(np.mean(displacements**2))),
        float(np.max(displacements)),
        rotation,
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
