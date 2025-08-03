from rdkit import Chem
from rdkit.Chem import rdMolStandardize

def load_molecules(input_data, input_type="smiles"):
    """
    Load molecules from SMILES list or SDF file.
    
    Args:
        input_data (list or str): SMILES list or SDF file path
        input_type (str): 'smiles' or 'sdf'
    
    Returns:
        list of RDKit Mol objects
    """
    mols = []
    if input_type == "smiles":
        for smi in input_data:
            mol = Chem.MolFromSmiles(smi)
            mols.append(mol)
    elif input_type == "sdf":
        suppl = Chem.SDMolSupplier(input_data)
        mols = [mol for mol in suppl if mol is not None]
    else:
        raise ValueError("Unsupported input type: 'smiles' or 'sdf'")
    return mols

def standardize_molecules(mols):
    """
    Standardize molecules (neutralize charges, sanitize, remove salts).
    
    Args:
        mols (list): list of RDKit Mol objects
    
    Returns:
        clean_mols (list): standardized molecules
        problematic (list): list of (index, error, SMILES) for failed molecules
    """
    clean_mols = []
    problematic = []
    normalizer = rdMolStandardize.Normalizer()
    reionizer = rdMolStandardize.Reionizer()
    
    for i, mol in enumerate(mols):
        if mol is None:
            problematic.append((i, "Invalid molecule", None))
            continue
        try:
            mol = Chem.RemoveHs(mol)
            mol = normalizer.normalize(mol)
            mol = reionizer.reionize(mol)
            Chem.SanitizeMol(mol)
            clean_mols.append(mol)
        except Exception as e:
            problematic.append((i, str(e), Chem.MolToSmiles(mol) if mol else None))
    
    return clean_mols, problematic
