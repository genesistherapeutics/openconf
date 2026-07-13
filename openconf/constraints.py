"""Reusable geometry constraints for conformer relaxation."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from .exceptions import OpenConfValueError

_DEFAULT_POSITION_FORCE_CONSTANT = 1000.0
_METAL_POSITION_FORCE_CONSTANT = 1e4
_METAL_LIGAND_DISTANCE_TOLERANCE = 0.05
_METAL_LIGAND_DISTANCE_FORCE_CONSTANT = 100000.0


class _BondConstraintSpecLike(Protocol):
    atom_i: int
    atom_j: int
    distance: float | None
    tolerance: float
    force_constant: float


class _AngleConstraintSpecLike(Protocol):
    atom_i: int
    atom_j: int
    atom_k: int
    angle_deg: float | None
    tolerance_deg: float
    force_constant: float


class _TorsionConstraintSpecLike(Protocol):
    atom_i: int
    atom_j: int
    atom_k: int
    atom_l: int
    dihedral_deg: float | None
    tolerance_deg: float
    force_constant: float


@dataclass(frozen=True)
class PositionConstraint:
    """Harmonic position constraint on an atom.

    Attributes:
        atom_idx: constrained atom index
        position: reference Cartesian position
        force_constant: force constant in kcal/mol/A^2
    """

    atom_idx: int
    position: tuple[float, float, float]
    force_constant: float = _DEFAULT_POSITION_FORCE_CONSTANT


@dataclass(frozen=True)
class DistanceConstraint:
    """Harmonic distance-window constraint between two atoms.

    Attributes:
        atom_i: first atom index
        atom_j: second atom index
        min_distance: lower distance bound in Angstrom
        max_distance: upper distance bound in Angstrom
        force_constant: force constant in kcal/mol/A^2
    """

    atom_i: int
    atom_j: int
    min_distance: float
    max_distance: float
    force_constant: float


@dataclass(frozen=True)
class AngleConstraint:
    """Harmonic angle-window constraint between three atoms.

    Attributes:
        atom_i: first atom index
        atom_j: central atom index
        atom_k: third atom index
        min_angle_deg: lower angle bound in degrees
        max_angle_deg: upper angle bound in degrees
        force_constant: force constant in kcal/mol/rad^2
    """

    atom_i: int
    atom_j: int
    atom_k: int
    min_angle_deg: float
    max_angle_deg: float
    force_constant: float


@dataclass(frozen=True)
class TorsionConstraint:
    """Harmonic torsion-window constraint between four atoms.

    Attributes:
        atom_i: first atom index
        atom_j: second atom index
        atom_k: third atom index
        atom_l: fourth atom index
        min_dihedral_deg: lower dihedral bound in degrees
        max_dihedral_deg: upper dihedral bound in degrees
        force_constant: force constant in kcal/mol/rad^2
    """

    atom_i: int
    atom_j: int
    atom_k: int
    atom_l: int
    min_dihedral_deg: float
    max_dihedral_deg: float
    force_constant: float


@dataclass(frozen=True)
class ConstraintModel:
    """Geometry constraints shared by all minimization paths.

    Attributes:
        position_constraints: atom position constraints
        distance_constraints: atom-pair distance constraints
        angle_constraints: atom-angle constraints
        torsion_constraints: atom-torsion constraints
    """

    position_constraints: tuple[PositionConstraint, ...] = ()
    distance_constraints: tuple[DistanceConstraint, ...] = ()
    angle_constraints: tuple[AngleConstraint, ...] = ()
    torsion_constraints: tuple[TorsionConstraint, ...] = ()

    @property
    def has_constraints(self) -> bool:
        """Return whether model contains any force-field constraints."""
        return bool(
            self.position_constraints or self.distance_constraints or self.angle_constraints or self.torsion_constraints
        )

    @property
    def constrained_atoms(self) -> frozenset[int]:
        """Atom indices with position constraints."""
        return frozenset(constraint.atom_idx for constraint in self.position_constraints)

    @classmethod
    def empty(cls) -> "ConstraintModel":
        """Return model with no constraints."""
        return cls()

    @classmethod
    def from_atom_positions(
        cls,
        mol: Chem.Mol,
        atom_indices: frozenset[int],
        force_constant: float = _DEFAULT_POSITION_FORCE_CONSTANT,
    ) -> "ConstraintModel":
        """Build position constraints from first conformer.

        Args:
            mol: molecule containing reference conformer
            atom_indices: atoms to constrain
            force_constant: force constant in kcal/mol/A^2

        Returns:
            Constraint model with atom position constraints
        """
        if not atom_indices or mol.GetNumConformers() == 0:
            return cls.empty()

        conf = mol.GetConformer(mol.GetConformers()[0].GetId())
        constraints = []
        for atom_idx in sorted(atom_indices):
            pos = conf.GetAtomPosition(int(atom_idx))
            constraints.append(
                PositionConstraint(
                    atom_idx=int(atom_idx),
                    position=(float(pos.x), float(pos.y), float(pos.z)),
                    force_constant=force_constant,
                )
            )
        return cls(position_constraints=tuple(constraints))

    @classmethod
    def from_internal_coordinates(
        cls,
        mol: Chem.Mol,
        bond_constraints: tuple[_BondConstraintSpecLike, ...] = (),
        angle_constraints: tuple[_AngleConstraintSpecLike, ...] = (),
        torsion_constraints: tuple[_TorsionConstraintSpecLike, ...] = (),
    ) -> "ConstraintModel":
        """Build internal-coordinate constraints, using first conformer as reference when needed.

        Args:
            mol: molecule containing optional reference conformer
            bond_constraints: bond constraint specifications
            angle_constraints: angle constraint specifications
            torsion_constraints: torsion constraint specifications

        Returns:
            Constraint model with distance, angle, and torsion constraints
        """
        if not bond_constraints and not angle_constraints and not torsion_constraints:
            return cls.empty()

        conf = mol.GetConformer(mol.GetConformers()[0].GetId()) if mol.GetNumConformers() else None
        distances: list[DistanceConstraint] = []
        angles: list[AngleConstraint] = []
        torsions: list[TorsionConstraint] = []

        for constraint in bond_constraints:
            distance = constraint.distance
            if distance is None:
                if conf is None:
                    msg = "Bond constraint without explicit distance requires molecule with reference conformer."
                    raise OpenConfValueError(msg)
                distance = float(
                    conf.GetAtomPosition(int(constraint.atom_i)).Distance(conf.GetAtomPosition(int(constraint.atom_j)))
                )
            distances.append(
                DistanceConstraint(
                    atom_i=int(constraint.atom_i),
                    atom_j=int(constraint.atom_j),
                    min_distance=max(0.0, float(distance) - float(constraint.tolerance)),
                    max_distance=float(distance) + float(constraint.tolerance),
                    force_constant=float(constraint.force_constant),
                )
            )

        for constraint in angle_constraints:
            angle = constraint.angle_deg
            if angle is None:
                if conf is None:
                    msg = "Angle constraint without explicit angle requires molecule with reference conformer."
                    raise OpenConfValueError(msg)
                angle = float(
                    rdMolTransforms.GetAngleDeg(
                        conf,
                        int(constraint.atom_i),
                        int(constraint.atom_j),
                        int(constraint.atom_k),
                    )
                )
            angles.append(
                AngleConstraint(
                    atom_i=int(constraint.atom_i),
                    atom_j=int(constraint.atom_j),
                    atom_k=int(constraint.atom_k),
                    min_angle_deg=float(angle) - float(constraint.tolerance_deg),
                    max_angle_deg=float(angle) + float(constraint.tolerance_deg),
                    force_constant=float(constraint.force_constant),
                )
            )

        for constraint in torsion_constraints:
            dihedral = constraint.dihedral_deg
            if dihedral is None:
                if conf is None:
                    msg = "Torsion constraint without explicit dihedral requires molecule with reference conformer."
                    raise OpenConfValueError(msg)
                dihedral = float(
                    rdMolTransforms.GetDihedralDeg(
                        conf,
                        int(constraint.atom_i),
                        int(constraint.atom_j),
                        int(constraint.atom_k),
                        int(constraint.atom_l),
                    )
                )
            torsions.append(
                TorsionConstraint(
                    atom_i=int(constraint.atom_i),
                    atom_j=int(constraint.atom_j),
                    atom_k=int(constraint.atom_k),
                    atom_l=int(constraint.atom_l),
                    min_dihedral_deg=float(dihedral) - float(constraint.tolerance_deg),
                    max_dihedral_deg=float(dihedral) + float(constraint.tolerance_deg),
                    force_constant=float(constraint.force_constant),
                )
            )

        return cls(
            distance_constraints=tuple(distances),
            angle_constraints=tuple(angles),
            torsion_constraints=tuple(torsions),
        )

    @classmethod
    def from_metal_shell(
        cls,
        mol: Chem.Mol,
        metal_atom_indices: frozenset[int],
        position_force_constant: float = _METAL_POSITION_FORCE_CONSTANT,
        distance_tolerance: float = _METAL_LIGAND_DISTANCE_TOLERANCE,
        distance_force_constant: float = _METAL_LIGAND_DISTANCE_FORCE_CONSTANT,
    ) -> "ConstraintModel":
        """Build metal position and first-shell distance constraints.

        Args:
            mol: molecule containing reference conformer
            metal_atom_indices: metal atom indices
            position_force_constant: metal position force constant
            distance_tolerance: allowed metal-ligand distance tolerance
            distance_force_constant: metal-ligand distance force constant

        Returns:
            Constraint model for metal centers and direct ligands
        """
        if not metal_atom_indices or mol.GetNumConformers() == 0:
            return cls.empty()

        conf = mol.GetConformer(mol.GetConformers()[0].GetId())
        positions = cls.from_atom_positions(mol, metal_atom_indices, position_force_constant).position_constraints
        distances: list[DistanceConstraint] = []
        for metal_idx in sorted(metal_atom_indices):
            metal_pos = conf.GetAtomPosition(int(metal_idx))
            for neighbor in mol.GetAtomWithIdx(int(metal_idx)).GetNeighbors():
                ligand_idx = int(neighbor.GetIdx())
                ligand_pos = conf.GetAtomPosition(ligand_idx)
                distance = float(metal_pos.Distance(ligand_pos))
                if distance <= 0.0:
                    continue
                distances.append(
                    DistanceConstraint(
                        atom_i=int(metal_idx),
                        atom_j=ligand_idx,
                        min_distance=max(0.0, distance - distance_tolerance),
                        max_distance=distance + distance_tolerance,
                        force_constant=distance_force_constant,
                    )
                )
        return cls(position_constraints=positions, distance_constraints=tuple(distances))

    def combine(self, other: "ConstraintModel") -> "ConstraintModel":
        """Return merged constraints, with later position constraints winning.

        Args:
            other: constraints to merge after self

        Returns:
            Combined constraint model
        """
        positions = {constraint.atom_idx: constraint for constraint in self.position_constraints}
        positions.update({constraint.atom_idx: constraint for constraint in other.position_constraints})
        distances = {
            (min(c.atom_i, c.atom_j), max(c.atom_i, c.atom_j), c.min_distance, c.max_distance): c
            for c in self.distance_constraints
        }
        distances.update(
            {
                (min(c.atom_i, c.atom_j), max(c.atom_i, c.atom_j), c.min_distance, c.max_distance): c
                for c in other.distance_constraints
            }
        )
        angles = {(c.atom_i, c.atom_j, c.atom_k, c.min_angle_deg, c.max_angle_deg): c for c in self.angle_constraints}
        angles.update(
            {(c.atom_i, c.atom_j, c.atom_k, c.min_angle_deg, c.max_angle_deg): c for c in other.angle_constraints}
        )
        torsions = {
            (c.atom_i, c.atom_j, c.atom_k, c.atom_l, c.min_dihedral_deg, c.max_dihedral_deg): c
            for c in self.torsion_constraints
        }
        torsions.update(
            {
                (c.atom_i, c.atom_j, c.atom_k, c.atom_l, c.min_dihedral_deg, c.max_dihedral_deg): c
                for c in other.torsion_constraints
            }
        )
        return ConstraintModel(
            position_constraints=tuple(positions[idx] for idx in sorted(positions)),
            distance_constraints=tuple(distances[key] for key in sorted(distances)),
            angle_constraints=tuple(angles[key] for key in sorted(angles)),
            torsion_constraints=tuple(torsions[key] for key in sorted(torsions)),
        )

    def reset_positions(self, mol: Chem.Mol, conf_id: int) -> None:
        """Snap constrained atoms and distances back to reference geometry.

        Args:
            mol: molecule containing conformer
            conf_id: conformer ID to update
        """
        if not self.position_constraints and not self.distance_constraints:
            return
        conf = mol.GetConformer(int(conf_id))
        position_atoms = {constraint.atom_idx for constraint in self.position_constraints}
        for constraint in self.position_constraints:
            conf.SetAtomPosition(constraint.atom_idx, constraint.position)
        for constraint in self.distance_constraints:
            atom_i = int(constraint.atom_i)
            atom_j = int(constraint.atom_j)
            if atom_i in position_atoms and atom_j in position_atoms:
                continue
            pos_i = conf.GetAtomPosition(atom_i)
            pos_j = conf.GetAtomPosition(atom_j)
            vec = np.array([pos_j.x - pos_i.x, pos_j.y - pos_i.y, pos_j.z - pos_i.z])
            dist = float(np.linalg.norm(vec))
            if constraint.min_distance <= dist <= constraint.max_distance or dist <= 1e-12:
                continue
            target = 0.5 * (constraint.min_distance + constraint.max_distance)
            scaled = vec * (target / dist)
            if atom_i in position_atoms:
                conf.SetAtomPosition(atom_j, (pos_i.x + scaled[0], pos_i.y + scaled[1], pos_i.z + scaled[2]))
            elif atom_j in position_atoms:
                conf.SetAtomPosition(atom_i, (pos_j.x - scaled[0], pos_j.y - scaled[1], pos_j.z - scaled[2]))

    def max_position_drift(self, mol: Chem.Mol, conf_id: int) -> float:
        """Return maximum drift of position-constrained atoms.

        Args:
            mol: molecule containing conformer
            conf_id: conformer ID to inspect

        Returns:
            Maximum distance from reference position
        """
        if not self.position_constraints:
            return 0.0
        conf = mol.GetConformer(int(conf_id))
        max_drift = 0.0
        for constraint in self.position_constraints:
            pos = conf.GetAtomPosition(constraint.atom_idx)
            ref = np.array(constraint.position)
            cur = np.array([pos.x, pos.y, pos.z])
            max_drift = max(max_drift, float(np.linalg.norm(cur - ref)))
        return max_drift


def add_constraints_to_force_field(ff: object, constraints: ConstraintModel, family: str) -> None:
    """Add constraints to RDKit force field object when supported.

    Args:
        ff: RDKit force field object
        constraints: constraints to apply
        family: force-field family, either `"MMFF"` or `"UFF"`
    """
    if not constraints.has_constraints:
        return

    position_method = getattr(ff, f"{family}AddPositionConstraint", None)
    if position_method is not None:
        for constraint in constraints.position_constraints:
            position_method(int(constraint.atom_idx), 0.0, float(constraint.force_constant))

    distance_method = getattr(ff, f"{family}AddDistanceConstraint", None)
    if distance_method is not None:
        for constraint in constraints.distance_constraints:
            distance_method(
                int(constraint.atom_i),
                int(constraint.atom_j),
                False,
                float(constraint.min_distance),
                float(constraint.max_distance),
                float(constraint.force_constant),
            )

    angle_method = getattr(ff, f"{family}AddAngleConstraint", None)
    if angle_method is not None:
        for constraint in constraints.angle_constraints:
            angle_method(
                int(constraint.atom_i),
                int(constraint.atom_j),
                int(constraint.atom_k),
                False,
                float(constraint.min_angle_deg),
                float(constraint.max_angle_deg),
                float(constraint.force_constant),
            )

    torsion_method = getattr(ff, f"{family}AddTorsionConstraint", None)
    if torsion_method is not None:
        for constraint in constraints.torsion_constraints:
            torsion_method(
                int(constraint.atom_i),
                int(constraint.atom_j),
                int(constraint.atom_k),
                int(constraint.atom_l),
                False,
                float(constraint.min_dihedral_deg),
                float(constraint.max_dihedral_deg),
                float(constraint.force_constant),
            )
