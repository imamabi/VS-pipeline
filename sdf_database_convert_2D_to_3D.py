#Importing packages
import os
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
from io import StringIO
import sys
from rdkit import rdBase
from tqdm import tqdm
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
Chem.WrapLogs()

#Import open drug discovery toolkit 
import oddt
from oddt.toolkits import extras

# Get the percentage of sorted results from the command line argument
if len(sys.argv) < 2:
    print("Usage: python 2D_3D_convert.py <file_name>")
    sys.exit(1)

file_name = sys.argv[1]
#A function to get ok mols and failed mols
def readmols(suppl):
    ok=[]
    failures=[]
    sio = sys.stderr = StringIO()
    for i,m in enumerate(suppl):
        if m is None:
            failures.append((i,sio.getvalue()))
            sio = sys.stderr = StringIO() # reset the error logger
        else:
            ok.append((i,m))
    return ok,failures

suppl = Chem.ForwardSDMolSupplier(f"./{file_name}")
ok,failures = readmols(suppl)

#This part will store the failed molecules in bad_mol
bad_mol = []
for i,fail in failures:
    bad_mol.append(i)
    #print(i,fail)

#This part will store the ok molecules in mol_ready
mol_ready = []
for i,m in ok:
    mol_ready.append(m)
    #print (i,m)

#This part will separate the 2D from the 3D molecules
# Lists to store 2D and 3D molecules
molecules_2d = []
molecules_3d = []

# Iterate over the molecules in the input file
for mol in mol_ready:
    if mol is not None:
        if mol.GetNumConformers() == 0 or mol.GetConformer().Is3D():
            # If the molecule has no conformers or the conformer is 3D, it is considered 3D
            molecules_3d.append(mol)
        else:
            # If the molecule has at least one conformer and the conformer is 2D, it is considered 2D
            molecules_2d.append(mol)


#converting the 2D molecules in the datset to 3D
if molecules_2d is not None:       
    
    mol_2D = molecules_2d

    # List to store the converted molecules
    converted_molecules = []

    # List to store unrecognized molecules
    unrecognized_molecules = []

    # Iterate over the molecules in the input file
    for mol in mol_2D:
        if mol is not None:
            try:
                # Add hydrogen atoms to the molecule
                '''Note: Structure might not contain H atoms'''
                mol = Chem.AddHs(mol)
                # Perform 3D conversion using the ETKDG algorithm
                Chem.SanitizeMol(mol)

                # Check if the molecule contains unrecognized atoms
                unrecognized_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
                if unrecognized_atoms:
                    unrecognized_molecules.append(mol)
                else:
                    AllChem.EmbedMolecule(mol, AllChem.ETKDG())

                    # Optimize the 3D structure using a force field
                    AllChem.MMFFOptimizeMolecule(mol)

                    # Append the converted molecule to the list
                    converted_molecules.append(mol)
            except Exception as e:
                # If an error occurs, append the molecule to the list of unrecognized molecules
                unrecognized_molecules.append((mol, str(e)))

#Optimizing the 3D molecules in original dataset
if molecules_2d is not None:
    
    mol_3D = molecules_3d

    optimized_molecules = converted_molecules #Define optimized_molecule as progression from converted_molecules
    unoptimized_mol = []
    for mol in mol_3D:
        try:
        #optimize the 3D molecules
            AllChem.MMFFOptimizeMolecule(mol)

        #Append the optimized molecule
            optimized_molecules.append(mol)

        except Exception as e:
            print ('The molecule ' f'{mol}'+'returns error: ', str(e)) #Show faulty molecule and error and error statement
            unoptimized_mol.append(mol) #Append faulty 3D molecule in unoptimized
    
optimized_molecules=optimized_molecules #update optimized molecules

# Write the optimized molecules to a new SDF file
output_file_opt = f"{file_name}_output_optimized.sdf"
writer = Chem.SDWriter(output_file_opt)
for mol in optimized_molecules:
    writer.write(mol)
writer.close()

#Auditing the molecule converstion
print ('Total mols in database:  ', (len(ok)+len(failures)), '\nTotal 2D mols in database:  ', len(molecules_2d), 
       '\nTotal 3D mols in database:  ', len(molecules_3d), '\nOptimized mols from database:  ', len(optimized_molecules),
       '\nUnrecognized 2D mols in database:  ', len(unrecognized_molecules), '\nUnoptimized 3D mols in database:  ', len (unoptimized_mol),
       '\nFailed mols in database:  ', len(bad_mol))

