# Importing packages
import os
import sys
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from tqdm import tqdm
from io import StringIO
from rdkit import rdBase
from rdkit.Chem.Draw import IPythonConsole
import numpy as np
import argparse
import time

# Import Open Drug Discovery Toolkit (ODDT)
import oddt
from oddt.toolkits import extras

# Get the input file name from the command line
if len(sys.argv) < 2:
    print("Usage: python 2D_similarity_search.py <file_name>")
    sys.exit(1)

file_name = sys.argv[1]

# Function to separate valid and invalid molecules
def readmols(suppl):
    ok = []
    failures = []
    sio = sys.stderr = StringIO()
    for i, m in enumerate(suppl):
        if m is None:
            failures.append((i, sio.getvalue()))
            sio = sys.stderr = StringIO()  # Reset the error logger
        else:
            ok.append((i, m))
    return ok, failures

# Load molecules from the file
suppl = Chem.ForwardSDMolSupplier(f"./{file_name}")
ok, failures = readmols(suppl)

# Store failed molecules in bad_mol
bad_mol = [i for i, _ in failures]

# Store valid molecules in mol_ready
mol_ready = [m for _, m in ok]

# Compute molecular fingerprints
def compute_fingerprint(molecule, fp_type="morgan"):
    """Compute molecular fingerprint, defaulting to Morgan fingerprint."""
    if fp_type == "morgan":
        return AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048)
    elif fp_type == "rdkit":
        return Chem.RDKFingerprint(molecule)
    else:
        raise ValueError("Unsupported fingerprint type")

# Calculate Tanimoto similarity between two fingerprints
def calculate_similarity(fp1, fp2):
    """Calculate Tanimoto similarity between two fingerprints."""
    return DataStructs.TanimotoSimilarity(fp1, fp2)

# Perform similarity search with cutoff
def perform_similarity_search(molecules, reference_molecules, fp_type="morgan", similarity_cutoff=0.85):
    """Perform 2D similarity search and return molecules with similarity above the cutoff."""
    reference_fps = [compute_fingerprint(ref_mol, fp_type) for ref_mol in reference_molecules]
    
    similar_molecules = []
    for mol in molecules:
        mol_fp = compute_fingerprint(mol, fp_type)
        if any(calculate_similarity(mol_fp, ref_fp) >= similarity_cutoff for ref_fp in reference_fps):
            similar_molecules.append(mol)
    
    return similar_molecules

# Save the similar molecules to an output file
def save_similar_molecules(molecules, output_file):
    writer = Chem.SDWriter(output_file)
    for mol in molecules:
        writer.write(mol)
    writer.close()

# Main function
def main():
    parser = argparse.ArgumentParser(description="2D Similarity Search Script")
    parser.add_argument("-r", "--reference", required=True, help="Reference file (SMILES or SDF format)")
    parser.add_argument("-o", "--output", required=True, help="Output file (SDF format)")
    parser.add_argument("-c", "--cutoff", type=float, default=0.85, help="Similarity cutoff (default: 0.85)")
    parser.add_argument("-f", "--fingerprint", type=str, default="morgan", choices=["morgan", "rdkit"],
                        help="Fingerprint type (default: Morgan)")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("Loading reference molecules...")
    reference_suppl = Chem.SDMolSupplier(args.reference)
    reference_ok, _ = readmols(reference_suppl)
    reference_molecules = [m for _, m in reference_ok]
    
    print(f"Performing similarity search with cutoff {args.cutoff} and fingerprint type '{args.fingerprint}'...")
    similar_molecules = perform_similarity_search(mol_ready, reference_molecules,
                                                  fp_type=args.fingerprint, similarity_cutoff=args.cutoff)
    
    print(f"Found {len(similar_molecules)} similar molecules.")
    
    print("Saving similar molecules to output file...")
    save_similar_molecules(similar_molecules, args.output)
    
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
