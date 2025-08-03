from rdkit import Chem
from rdkit.Chem import AllChem

def generate_3d(mol):
    """
    Generate 3D conformer and energy minimize using MMFF or UFF.
    
    Args:
        mol (RDKit Mol): input molecule
    
    Returns:
        mol (RDKit Mol): molecule with 3D coordinates
    """
    mol = Chem.AddHs(mol)  # Add hydrogens
    params = AllChem.ETKDGv3()  # Use advanced ETKDG algorithm
    params.randomSeed = 0xf00d
    
    if AllChem.EmbedMolecule(mol, params) == -1:
        raise ValueError("3D embedding failed.")
    
    # Optimize geometry using MMFF or UFF
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    
    return mol
