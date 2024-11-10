# VS-pipeline
This scripts are for;
2D similiarity search for search for molecules that are identical targets.
To run 2D similiary search

```markdown
python 2D_similarity_search.py -i molecules.sdf -r reference_molecules.sdf -o similar_molecules.sdf -c 0.90 -f rdkit

The sdf_database_convert_2D_to_3D.py preprocessed ligand molecules in multi-sdf file.
The molecules are filtered, converted to 3D and minimized to make ready for docking.
