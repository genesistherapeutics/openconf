"""Tests for pose-constrained (FEP-style analogue) conformer generation."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem


def _make_butylbenzene_pose() -> tuple[Chem.Mol, list[int]]:
    """Return an H-added butylbenzene with one embedded conformer and ring atom indices."""
    mol = Chem.MolFromSmiles("CCCCc1ccccc1")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    ring_atoms = [4, 5, 6, 7, 8, 9]  # benzene ring heavy-atom indices
    return mol, ring_atoms


def test_analogue_preset_values():
    """'analogue' preset returns the expected config."""
    from openconf import preset_config

    cfg = preset_config("analogue")
    assert cfg.max_out == 50
    assert cfg.n_steps == 150
    assert cfg.do_final_refine is True
    assert cfg.parent_strategy == "softmax"
    assert cfg.final_select == "diverse"


# ---------------------------------------------------------------------------
# Core correctness
# ---------------------------------------------------------------------------


def test_constrained_atoms_exactly_fixed():
    """Constrained atom positions must be bit-for-bit identical to the input pose."""
    from openconf import generate_conformers_from_pose

    mol, ring_atoms = _make_butylbenzene_pose()

    ref_pos = {i: np.array(mol.GetConformer(0).GetAtomPosition(i)) for i in ring_atoms}
    ensemble = generate_conformers_from_pose(mol, constrained_atoms=ring_atoms)

    for rec in ensemble.records:
        conf = ensemble.mol.GetConformer(rec.conf_id)
        for i in ring_atoms:
            pos = np.array(conf.GetAtomPosition(i))
            assert np.allclose(pos, ref_pos[i], atol=1e-9), (
                f"Atom {i} drifted in conformer {rec.conf_id}: ref={ref_pos[i]}, got={pos}"
            )


def test_generate_conformers_from_pose_returns_conformers():
    """Basic smoke test: function runs and produces at least one conformer."""
    from openconf import generate_conformers_from_pose

    mol, ring_atoms = _make_butylbenzene_pose()
    ensemble = generate_conformers_from_pose(mol, constrained_atoms=ring_atoms)

    assert ensemble.n_conformers > 0
    assert ensemble.n_conformers <= 50  # analogue preset max_out
    assert all(np.isfinite(e) for e in ensemble.energies)


def test_free_atoms_do_move():
    """At least one free (non-constrained) atom should differ across conformers."""
    from openconf import generate_conformers_from_pose

    mol, ring_atoms = _make_butylbenzene_pose()
    ensemble = generate_conformers_from_pose(mol, constrained_atoms=ring_atoms)

    # Butyl chain atom indices (heavy atoms 0-3)
    free_atoms = [0, 1, 2, 3]

    # Collect positions of free atoms across all conformers
    all_positions = []
    for rec in ensemble.records:
        conf = ensemble.mol.GetConformer(rec.conf_id)
        pos = np.array([conf.GetAtomPosition(i) for i in free_atoms])
        all_positions.append(pos)

    # At least two conformers should differ in at least one free atom
    assert len(all_positions) >= 2
    assert not all(np.allclose(all_positions[0], p, atol=0.01) for p in all_positions[1:]), (
        "All conformers have identical free-atom coordinates — no exploration occurred"
    )


# ---------------------------------------------------------------------------
# Input variants
# ---------------------------------------------------------------------------


def test_mol_without_hs_input():
    """Works when the input mol has only heavy atoms (AddHs applied internally)."""
    from openconf import generate_conformers_from_pose

    mol = Chem.MolFromSmiles("CCCCc1ccccc1")
    # Embed without adding Hs — only heavy-atom conformer
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    ring_atoms = [4, 5, 6, 7, 8, 9]

    ensemble = generate_conformers_from_pose(mol, constrained_atoms=ring_atoms)
    assert ensemble.n_conformers > 0


def test_first_conformer_used_as_seed():
    """When mol has multiple conformers, only the first is used as the seed."""
    from openconf import ConformerConfig, generate_conformers_from_pose

    mol = Chem.MolFromSmiles("CCCCc1ccccc1")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=3, randomSeed=0)
    ring_atoms = [4, 5, 6, 7, 8, 9]

    ref_pos = {i: np.array(mol.GetConformer(0).GetAtomPosition(i)) for i in ring_atoms}
    config = ConformerConfig(max_out=5, n_steps=20, pool_max=50)
    ensemble = generate_conformers_from_pose(mol, constrained_atoms=ring_atoms, config=config)

    for rec in ensemble.records:
        conf = ensemble.mol.GetConformer(rec.conf_id)
        for i in ring_atoms:
            pos = np.array(conf.GetAtomPosition(i))
            assert np.allclose(pos, ref_pos[i], atol=1e-9)


def test_no_conformer_raises():
    """Passing a mol with no conformer raises ValueError."""
    from openconf import generate_conformers_from_pose

    mol = Chem.MolFromSmiles("CCCCc1ccccc1")
    mol = Chem.AddHs(mol)
    # deliberately do NOT embed

    with pytest.raises(ValueError, match="conformer"):
        generate_conformers_from_pose(mol, constrained_atoms=[4, 5, 6, 7, 8, 9])


def test_config_and_preset_both_raises():
    """Passing both config and preset raises ValueError."""
    from openconf import ConformerConfig, generate_conformers_from_pose

    mol, ring_atoms = _make_butylbenzene_pose()
    with pytest.raises(ValueError):
        generate_conformers_from_pose(
            mol,
            constrained_atoms=ring_atoms,
            config=ConformerConfig(),
            preset="analogue",
        )


# ---------------------------------------------------------------------------
# Rotor filtering
# ---------------------------------------------------------------------------


def test_filter_constrained_rotors_eliminates_double_sided():
    """A rotor whose distal fragments on both sides contain constrained heavy atoms is excluded.

    Biphenyl with all 12 carbons constrained: the single biaryl C–C bond has
    constrained ring carbons beyond the axis atom on both sides, so it cannot
    be rotated without displacing a constrained atom and must be eliminated.
    """
    from openconf import build_rotor_model, filter_constrained_rotors, prepare_molecule

    mol = prepare_molecule(Chem.MolFromSmiles("c1ccccc1-c1ccccc1"))
    rm = build_rotor_model(mol)
    full_count = rm.n_rotatable  # 1 bond: the biaryl C-C

    # Constrain every carbon; the biaryl bond has constrained distal heavy
    # atoms on both sides and should be removed.
    all_carbons = frozenset(i for i, a in enumerate(mol.GetAtoms()) if a.GetAtomicNum() == 6)
    filtered = filter_constrained_rotors(rm, all_carbons)

    assert filtered.n_rotatable < full_count
    assert filtered.n_rotatable == 0


def test_filter_constrained_rotors_boundary_attachment_kept():
    """Scaffold-edge bond is kept when both axis atoms are constrained but distal side is free.

    Butylbenzene with the benzene ring AND the adjacent chain carbon (C3, index 3)
    all constrained: the C3-ring bond has two constrained axis atoms, but the free
    butyl chain (C0–C2) lies entirely beyond C3. The boundary-attachment rule keeps
    the bond so the chain can still be explored.
    """
    from openconf import build_rotor_model, filter_constrained_rotors, prepare_molecule
    from openconf.perceive import _atoms_on_side

    mol = prepare_molecule(Chem.MolFromSmiles("CCCCc1ccccc1"))
    rm = build_rotor_model(mol)

    # Ring atoms (4-9) plus the first chain carbon attached to the ring (3).
    # Both atoms of the ring-chain bond are now constrained.
    constrained = frozenset([3, 4, 5, 6, 7, 8, 9])
    filtered = filter_constrained_rotors(rm, constrained)

    # The C3-ring bond must survive: free chain atoms are the distal fragment.
    assert filtered.n_rotatable > 0

    # Every surviving rotor's distal moving fragment must be constraint-free.
    for rotor in filtered.rotors:
        atom_i, atom_j = rotor.atom_idxs
        moving = _atoms_on_side(mol, atom_j, atom_i)
        distal = moving - {atom_j}
        assert not constrained & distal, (
            f"Rotor {rotor.atom_idxs} has constrained atoms in distal fragment: {constrained & distal}"
        )


def test_filter_constrained_rotors_free_side_reoriented():
    """When the free fragment is on the 'wrong' side of a rotor it is flipped.

    Butylbenzene with only ring atoms constrained: the ring-to-chain bond's
    free side (butyl chain) becomes the moving side. All 4 rotors are kept.
    """
    from openconf import build_rotor_model, filter_constrained_rotors, prepare_molecule
    from openconf.perceive import _atoms_on_side

    mol = prepare_molecule(Chem.MolFromSmiles("CCCCc1ccccc1"))
    rm = build_rotor_model(mol)

    constrained = frozenset(range(4, 10))  # ring atoms
    filtered = filter_constrained_rotors(rm, constrained)

    # All 4 chain rotors should be preserved (just possibly reoriented).
    assert filtered.n_rotatable == rm.n_rotatable

    # For every remaining rotor the distal moving fragment must be entirely free.
    # atom_j sits on the rotation axis and never physically translates, so it is
    # excluded from the check — only the atoms beyond it (distal) must be free.
    for rotor in filtered.rotors:
        atom_i, atom_j = rotor.atom_idxs
        moving = _atoms_on_side(mol, atom_j, atom_i)
        distal = moving - {atom_j}
        assert not constrained & distal, (
            f"Rotor {rotor.atom_idxs} has constrained atoms in distal fragment: {constrained & distal}"
        )


def test_filtered_constrained_torsions_do_not_translate_core():
    """Filtered torsion moves leave constrained atom coordinates unchanged."""
    from rdkit.Chem import rdMolTransforms

    from openconf import build_rotor_model, filter_constrained_rotors, prepare_molecule

    mol = prepare_molecule(Chem.MolFromSmiles("CCCCc1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=0)
    constrained = frozenset(range(4, 10))  # benzene ring atoms
    filtered = filter_constrained_rotors(build_rotor_model(mol), constrained)

    assert filtered.rotors
    constrained_indices = sorted(constrained)
    for rotor in filtered.rotors:
        trial = Chem.Mol(mol)
        conf = trial.GetConformer(0)
        before = conf.GetPositions().copy()
        angle = rdMolTransforms.GetDihedralDeg(conf, *rotor.dihedral_atoms)

        rdMolTransforms.SetDihedralDeg(conf, *rotor.dihedral_atoms, angle + 73.0)

        after = conf.GetPositions()
        assert np.allclose(after[constrained_indices], before[constrained_indices], atol=1e-8), (
            f"Rotor {rotor.atom_idxs} translated constrained atoms"
        )


def test_boundary_constrained_torsion_moves_only_free_fragment():
    """Boundary torsion with constrained axis atoms moves free distal fragment."""
    from rdkit.Chem import rdMolTransforms

    from openconf import build_rotor_model, filter_constrained_rotors, prepare_molecule

    mol = prepare_molecule(Chem.MolFromSmiles("CCCCc1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=0)
    constrained = frozenset([3, 4, 5, 6, 7, 8, 9])
    filtered = filter_constrained_rotors(build_rotor_model(mol), constrained)
    boundary_rotor = next((rotor for rotor in filtered.rotors if set(rotor.atom_idxs) == {3, 4}), None)

    assert boundary_rotor is not None
    conf = mol.GetConformer(0)
    before = conf.GetPositions().copy()
    angle = rdMolTransforms.GetDihedralDeg(conf, *boundary_rotor.dihedral_atoms)

    rdMolTransforms.SetDihedralDeg(conf, *boundary_rotor.dihedral_atoms, angle + 90.0)

    after = conf.GetPositions()
    constrained_indices = sorted(constrained)
    free_chain_indices = [0, 1, 2]
    assert np.allclose(after[constrained_indices], before[constrained_indices], atol=1e-8)
    assert max(float(np.linalg.norm(after[idx] - before[idx])) for idx in free_chain_indices) > 0.05


def test_filter_constrained_rotors_no_constrained_atoms():
    """Empty constraint set leaves the rotor model unchanged."""
    from openconf import build_rotor_model, filter_constrained_rotors, prepare_molecule

    mol = prepare_molecule(Chem.MolFromSmiles("CCCCc1ccccc1"))
    rm = build_rotor_model(mol)
    filtered = filter_constrained_rotors(rm, frozenset())

    assert filtered.n_rotatable == rm.n_rotatable


def test_filter_constrained_ring_flips():
    """Ring flips involving constrained ring atoms are removed."""
    from openconf import build_rotor_model, filter_constrained_rotors, prepare_molecule

    # Cyclohexyl-methyl: the ring should flip; constrain the ring
    mol = prepare_molecule(Chem.MolFromSmiles("CC1CCCCC1"))
    rm = build_rotor_model(mol)
    assert len(rm.ring_flips) == 1

    # Constrain all ring atoms (indices vary with H-addition; use atom symbols)
    ring_indices = frozenset(i for i, a in enumerate(mol.GetAtoms()) if a.IsInRing() and a.GetAtomicNum() == 6)
    filtered = filter_constrained_rotors(rm, ring_indices)
    assert len(filtered.ring_flips) == 0, "Ring flip should be removed when ring is constrained"


def test_filter_ring_flips_with_constrained_subtrees_removed():
    """Ring flips with constrained attached subtrees are removed after filtering."""
    from openconf import build_rotor_model, filter_constrained_rotors, prepare_molecule

    # Butyl-cyclohexane: constraining attached chain makes subtree-aware flip unsafe.
    mol = prepare_molecule(Chem.MolFromSmiles("CCCCC1CCCCC1"))
    rm = build_rotor_model(mol)
    assert len(rm.ring_flips) == 1

    constrained = frozenset([0, 1, 2, 3])
    filtered = filter_constrained_rotors(rm, constrained)
    assert len(filtered.ring_flips) == 0, "Ring flip should be removed when attached subtree is constrained"


# ---------------------------------------------------------------------------
# Move suppression
# ---------------------------------------------------------------------------


def test_global_shake_suppressed_in_constrained_mode():
    """_select_move_type never returns 'global_shake' when constraint_spec is set."""
    from openconf.config import ConformerConfig, ConstraintSpec
    from openconf.perceive import build_rotor_model, prepare_molecule
    from openconf.propose.hybrid import HybridProposer
    from openconf.torsionlib import TorsionLibrary

    mol = prepare_molecule(Chem.MolFromSmiles("CCCCc1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=0)
    rm = build_rotor_model(mol)

    spec = ConstraintSpec(constrained_atoms=frozenset(range(4, 10)))
    config = ConformerConfig(random_seed=0, shake_period=1)  # shake_period=1 forces periodic shakes
    proposer = HybridProposer(mol, rm, TorsionLibrary(), config, constraint_spec=spec)

    # Run enough steps to trigger shake_period multiple times
    move_types = {proposer._select_move_type(step) for step in range(200)}
    assert "global_shake" not in move_types, f"global_shake appeared in constrained mode: {move_types}"


# ---------------------------------------------------------------------------
# Internal-coordinate constraints
# ---------------------------------------------------------------------------


def test_internal_coordinate_constraints_freeze_reference_geometry():
    """Bond, angle, and torsion constraints can freeze input conformer values."""
    from rdkit.Chem import rdMolTransforms

    from openconf import (
        AngleConstraintSpec,
        BondConstraintSpec,
        ConformerConfig,
        ConstraintSpec,
        TorsionConstraintSpec,
        generate_conformers,
    )

    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    AllChem.EmbedMolecule(mol, randomSeed=0)
    conf = mol.GetConformer(0)
    ref_distance = conf.GetAtomPosition(1).Distance(conf.GetAtomPosition(2))
    ref_angle = rdMolTransforms.GetAngleDeg(conf, 0, 1, 2)
    ref_torsion = rdMolTransforms.GetDihedralDeg(conf, 0, 1, 2, 3)

    config = ConformerConfig(
        max_out=3,
        pool_max=10,
        n_steps=8,
        minimize_batch_size=1,
        random_seed=0,
        constraint_spec=ConstraintSpec(
            bond_constraints=(BondConstraintSpec(1, 2),),
            angle_constraints=(AngleConstraintSpec(0, 1, 2),),
            torsion_constraints=(TorsionConstraintSpec(0, 1, 2, 3),),
        ),
    )
    ensemble = generate_conformers(mol, config=config, add_hs=False)

    assert ensemble.n_conformers > 0
    for record in ensemble.records:
        trial_conf = ensemble.mol.GetConformer(record.conf_id)
        distance = trial_conf.GetAtomPosition(1).Distance(trial_conf.GetAtomPosition(2))
        angle = rdMolTransforms.GetAngleDeg(trial_conf, 0, 1, 2)
        torsion = rdMolTransforms.GetDihedralDeg(trial_conf, 0, 1, 2, 3)

        assert abs(distance - ref_distance) <= 0.02
        assert abs(angle - ref_angle) <= 1.0
        assert abs(((torsion - ref_torsion + 180.0) % 360.0) - 180.0) <= 1.0


def test_explicit_internal_coordinate_constraints_do_not_require_reference_conformer():
    """Explicit bond, angle, and torsion targets work without input coordinates."""
    from rdkit.Chem import rdMolTransforms

    from openconf import (
        AngleConstraintSpec,
        BondConstraintSpec,
        ConformerConfig,
        ConstraintSpec,
        TorsionConstraintSpec,
        generate_conformers,
    )

    config = ConformerConfig(
        max_out=2,
        pool_max=10,
        n_seeds=3,
        n_steps=0,
        minimize_batch_size=1,
        random_seed=0,
        constraint_spec=ConstraintSpec(
            bond_constraints=(BondConstraintSpec(1, 2, distance=1.54, tolerance=0.02, force_constant=100000.0),),
            angle_constraints=(
                AngleConstraintSpec(0, 1, 2, angle_deg=112.0, tolerance_deg=2.0, force_constant=100000.0),
            ),
            torsion_constraints=(
                TorsionConstraintSpec(
                    0,
                    1,
                    2,
                    3,
                    dihedral_deg=180.0,
                    tolerance_deg=2.0,
                    force_constant=100000.0,
                ),
            ),
        ),
    )
    ensemble = generate_conformers("CCCC", config=config)

    assert ensemble.n_conformers > 0
    for record in ensemble.records:
        conf = ensemble.mol.GetConformer(record.conf_id)
        distance = conf.GetAtomPosition(1).Distance(conf.GetAtomPosition(2))
        angle = rdMolTransforms.GetAngleDeg(conf, 0, 1, 2)
        torsion = rdMolTransforms.GetDihedralDeg(conf, 0, 1, 2, 3)

        assert abs(distance - 1.54) <= 0.04
        assert abs(angle - 112.0) <= 3.0
        assert abs(((torsion - 180.0 + 180.0) % 360.0) - 180.0) <= 3.0


def test_staged_constraint_minimization_preserves_position_anchors():
    """Angle restraint must not move position-constrained downstream atoms."""
    from rdkit.Chem import rdMolTransforms

    from openconf.constraints import (
        AngleConstraint,
        ConstraintModel,
        PositionConstraint,
        minimize_with_constraints,
    )

    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer(0)
    reference = conf.GetAtomPosition(3)
    reference_position = (float(reference.x), float(reference.y), float(reference.z))
    reference_bond = conf.GetAtomPosition(2).Distance(reference)
    initial_angle = rdMolTransforms.GetAngleDeg(conf, 0, 1, 2)
    constraints = ConstraintModel(
        position_constraints=(PositionConstraint(3, reference_position),),
        angle_constraints=(AngleConstraint(0, 1, 2, initial_angle + 28.0, initial_angle + 32.0, 1000.0),),
    )
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")

    result = minimize_with_constraints(
        mol,
        0,
        constraints,
        200,
        mmff_props=props,
        require_convergence=True,
    )

    assert result is not None
    assert result.converged
    assert result.constraints_satisfied
    assert conf.GetAtomPosition(3).Distance(reference) <= 1e-9
    assert abs(conf.GetAtomPosition(2).Distance(conf.GetAtomPosition(3)) - reference_bond) <= 0.01


def test_staged_constraint_minimization_does_not_rewrite_ring_coordinates():
    """Zero-iteration constrained minimization must leave ring geometry unchanged."""
    from openconf.constraints import ConstraintModel, DistanceConstraint, minimize_with_constraints

    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    conf = mol.GetConformer(0)
    initial_positions = conf.GetPositions().copy()
    initial_distance = conf.GetAtomPosition(0).Distance(conf.GetAtomPosition(1))
    constraints = ConstraintModel(
        distance_constraints=(DistanceConstraint(0, 1, initial_distance + 0.5, initial_distance + 0.5, 1000.0),)
    )
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")

    result = minimize_with_constraints(mol, 0, constraints, 0, mmff_props=props)

    assert result is not None
    assert np.allclose(conf.GetPositions(), initial_positions, atol=1e-12)


# ---------------------------------------------------------------------------
# SDF output
# ---------------------------------------------------------------------------


def test_constrained_ensemble_to_sdf(tmp_path):
    """Constrained ensemble can be written to and read back from SDF."""
    from openconf import ConformerConfig, generate_conformers_from_pose

    mol, ring_atoms = _make_butylbenzene_pose()
    config = ConformerConfig(max_out=5, n_steps=20, pool_max=50)
    ensemble = generate_conformers_from_pose(mol, constrained_atoms=ring_atoms, config=config)

    out = tmp_path / "analogue.sdf"
    ensemble.to_sdf(str(out))
    assert out.exists()

    supplier = Chem.SDMolSupplier(str(out), removeHs=False)
    mols_read = [m for m in supplier if m is not None]
    assert len(mols_read) == ensemble.n_conformers
