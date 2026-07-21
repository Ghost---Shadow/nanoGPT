"""
Encode the same adjacency matrix (mol_graph.smiles_to_adjacency) as a plain
sparse-tensor triple list instead of JPEG-DCT compressing it: just the
nonzero (row, col, value) entries, upper triangle only (matrix is
symmetric), as exact integers with no quantization/lossy compression and no
imposed 2D block structure.
"""
import numpy as np

def adjacency_to_sparse_string(mat):
    """mat: NxN array. Returns 'row-col-value' triples (upper triangle,
    row<=col, nonzero only) joined by commas, row-major order."""
    n = mat.shape[0]
    triples = []
    for i in range(n):
        for j in range(i, n):
            v = int(mat[i, j])
            if v != 0:
                triples.append(f"{i}-{j}-{v}")
    return ','.join(triples)

def sparse_string_to_ranges(s):
    """Returns list of (i, j, char_start, char_end) for each triple's
    character span within s -- used to map attention back to matrix cells."""
    ranges = []
    pos = 0
    for part in s.split(','):
        i, j, v = part.split('-')
        start = pos
        pos += len(part)
        ranges.append((int(i), int(j), start, pos))
        pos += 1  # comma
    return ranges


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'chem_jpeg'))
    from mol_graph import smiles_to_adjacency
    for smiles in ['CCO', 'c1ccccc1', 'CC(=O)O']:
        mat = smiles_to_adjacency(smiles)
        s = adjacency_to_sparse_string(mat)
        print(f"{smiles!r}: {len(s)} chars -> {s}")
