"""
Turn a SMILES string into a fixed-size "adjacency matrix image": off-diagonal
entries encode bond order between atoms i,j (0/single/aromatic/double/triple
-> distinct grayscale levels), diagonal entries encode atomic number (scaled)
so atom identity isn't thrown away, not just connectivity. Padded/cropped
into a 32x32 canvas so it can go through the same 2D block-DCT JPEG pipeline
used for MNIST/CIFAR unchanged.
"""
import numpy as np
from rdkit import Chem

BOND_LEVEL = {
    Chem.BondType.SINGLE: 80,
    Chem.BondType.AROMATIC: 120,
    Chem.BondType.DOUBLE: 160,
    Chem.BondType.TRIPLE: 200,
}

MAX_ATOMS = 32

def smiles_to_adjacency(smiles, max_atoms=MAX_ATOMS):
    """Returns a max_atoms x max_atoms uint8 array, or None if the SMILES is
    invalid or has more than max_atoms heavy atoms."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n = mol.GetNumAtoms()
    if n > max_atoms:
        return None
    mat = np.zeros((max_atoms, max_atoms), dtype=np.float64)
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        mat[i, i] = min(255, atom.GetAtomicNum() * 6)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        level = BOND_LEVEL.get(bond.GetBondType(), 80)
        mat[i, j] = level
        mat[j, i] = level
    return mat.astype(np.uint8)


if __name__ == '__main__':
    for smiles in ['CCO', 'c1ccccc1', 'CC(=O)O']:
        mat = smiles_to_adjacency(smiles)
        n_atoms = Chem.MolFromSmiles(smiles).GetNumAtoms()
        print(f"{smiles!r}: {n_atoms} atoms, nonzero entries={np.count_nonzero(mat)}")
