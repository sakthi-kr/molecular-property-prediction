"""Purely structure-derived, stateless RDKit representations.

All features are computed molecule-by-molecule without looking at logD labels
or other compounds. Any feature scaling is fitted on training data only later.
"""
from __future__ import annotations

import math

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors


# Fixed feature definitions and order, versioned by configs/preparation.json.
DESCRIPTOR_FUNCTIONS = (
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Descriptors.MolLogP),  # logP estimate is not the logD label.
    ("TPSA", rdMolDescriptors.CalcTPSA),
    ("HBA", Lipinski.NumHAcceptors),
    ("HBD", Lipinski.NumHDonors),
    ("RotatableBonds", Lipinski.NumRotatableBonds),
    ("FractionCSP3", rdMolDescriptors.CalcFractionCSP3),
    ("RingCount", Lipinski.RingCount),
    ("AromaticRings", Lipinski.NumAromaticRings),
    ("AliphaticRings", Lipinski.NumAliphaticRings),
    ("HeavyAtomCount", Lipinski.HeavyAtomCount),
    ("Heteroatoms", Lipinski.NumHeteroatoms),
    ("FormalCharge", lambda mol: sum(a.GetFormalCharge() for a in mol.GetAtoms())),
    ("LabuteASA", rdMolDescriptors.CalcLabuteASA),
    ("BertzCT", Descriptors.BertzCT),
)
DESCRIPTOR_NAMES = tuple(name for name, _ in DESCRIPTOR_FUNCTIONS)


def calculate_features(smiles: list[str], *, radius: int, bits: int, include_chirality: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return (float64 RDKit 2D descriptors, uint8 Morgan bit fingerprints)."""
    if not smiles:
        raise ValueError("No molecular structures to featurize")
    if radius < 0 or bits < 64 or bits > 32768:
        raise ValueError("Invalid fingerprint radius or number of bits")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=bits, includeChirality=include_chirality,
    )
    x_desc = np.empty((len(smiles), len(DESCRIPTOR_NAMES)), dtype=np.float64)
    x_morgan = np.empty((len(smiles), bits), dtype=np.uint8)
    for idx, item in enumerate(smiles):
        mol = Chem.MolFromSmiles(item)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"Invalid SMILES at feature row {idx}")
        values = [float(fn(mol)) for _, fn in DESCRIPTOR_FUNCTIONS]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"Non-finite descriptors at feature row {idx}")
        x_desc[idx] = values
        x_morgan[idx] = generator.GetFingerprintAsNumPy(mol).astype(np.uint8, copy=False)
    return x_desc, x_morgan
