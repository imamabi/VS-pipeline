from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_scaffolds(mols):
    scaffolds = []
    for mol in mols:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        scaffolds.append(scaf)
    return scaffolds

def scaffold_statistics(mols):
    scaf_smiles = [Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) for m in mols]
    unique_scaffolds = set(scaf_smiles)
    freq = {s: scaf_smiles.count(s) for s in unique_scaffolds}
    return freq
