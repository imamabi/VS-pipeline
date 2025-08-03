import argparse
import sys
import os
from rdkit import Chem
from rdkit.Chem import SDWriter

from data_processing import load_molecules, standardize_molecules
from conformer_generation import generate_3d
from similarity_search import calculate_similarities
from visualization import compute_features, reduce_dimensions, butina_clustering, plot_clusters
from scaffold_analysis import scaffold_statistics

def save_molecules(mols, file_path):
    """Helper function to save molecules to SDF."""
    writer = SDWriter(file_path)
    for mol in mols:
        writer.write(mol)
    writer.close()

def main():
    parser = argparse.ArgumentParser(description="Virtual Screening Pipeline using RDKit")
    parser.add_argument("-i", "--input", required=True, help="Input dataset file (SDF)")
    parser.add_argument("-r", "--reference", required=True, help="Reference molecules file (SDF)")
    parser.add_argument("-o", "--output", required=True, help="Output file for docking-ready molecules (SDF)")
    parser.add_argument("-c", "--cutoff", type=float, default=0.9, help="Similarity cutoff (default: 0.90)")
    parser.add_argument("-f", "--features", default="both", choices=["ecfp", "fcfp", "both"], help="Feature type for visualization")
    args = parser.parse_args()

    print("=== Virtual Screening Pipeline ===")
    print(f"Input: {args.input}")
    print(f"Reference: {args.reference}")
    print(f"Cutoff: {args.cutoff}")
    print(f"Features: {args.features}")

    # Create output directory for intermediate files
    out_dir = os.path.dirname(args.output) or "."
    
    # Step 1: Load molecules
    print("\n[Step 1] Loading molecules...")
    dataset_mols = load_molecules(args.input, "sdf")
    reference_mols = load_molecules(args.reference, "sdf")
    print(f"Dataset size: {len(dataset_mols)}, Reference size: {len(reference_mols)}")

    # Step 2: Standardize molecules
    print("[Step 2] Standardizing molecules...")
    dataset_mols, errors = standardize_molecules(dataset_mols)
    if errors:
        print(f"Warning: {len(errors)} molecules failed standardization and were skipped.")
    save_molecules(dataset_mols, os.path.join(out_dir, "standardized.sdf"))
    print("Standardized molecules saved.")

    # Step 3: Similarity search
    print("[Step 3] Calculating similarities (ECFP + FCFP)...")
    similarities = calculate_similarities(reference_mols, dataset_mols, combine_method="average")
    
    # Save raw similarity scores
    sim_file = os.path.join(out_dir, "similarity_scores.csv")
    with open(sim_file, "w") as f:
        f.write("Reference,Dataset,Combined,ECFP,FCFP\n")
        for ref_idx, sim_list in similarities.items():
            for idx, combined_score, ecfp_score, fcfp_score in sim_list:
                f.write(f"{ref_idx},{idx},{combined_score},{ecfp_score},{fcfp_score}\n")
    print(f"Similarity scores saved to {sim_file}")

    # Step 4: Filter molecules by cutoff
    print(f"[Step 4] Filtering molecules with similarity >= {args.cutoff}...")
    selected_mols = []
    for ref_idx, sim_list in similarities.items():
        for idx, combined_score, _, _ in sim_list:
            if combined_score >= args.cutoff:
                selected_mols.append(dataset_mols[idx])
    
    selected_mols = list(set(selected_mols))
    if not selected_mols:
        print("No molecules passed the similarity cutoff. Exiting.")
        sys.exit(0)
    filtered_file = os.path.join(out_dir, "filtered_molecules.sdf")
    save_molecules(selected_mols, filtered_file)
    print(f"Filtered molecules saved to {filtered_file}")

    # Step 5: Generate 3D for filtered molecules
    print("[Step 5] Generating 3D and minimizing for docking...")
    docking_ready = []
    for mol in selected_mols:
        try:
            mol_3d = generate_3d(mol)
            docking_ready.append(mol_3d)
        except Exception as e:
            print(f"3D generation failed for {Chem.MolToSmiles(mol)}: {e}")
    save_molecules(docking_ready, args.output)
    print(f"Docking-ready molecules saved to {args.output}")

    # Step 6: Visualization & clustering
    print("[Step 6] Dimensionality reduction (UMAP) and clustering (Butina)...")
    features, _ = compute_features(docking_ready, method=args.features)
    coords = reduce_dimensions(features, method="umap")
    clusters = butina_clustering(docking_ready, cutoff=0.6)
    cluster_labels = [next(idx for idx, cluster in enumerate(clusters) if i in cluster) for i in range(len(docking_ready))]
    plot_clusters(coords, cluster_labels, title="Docking-Ready Molecule Clusters")

    # Step 7: Scaffold analysis
    print("[Step 7] Scaffold analysis...")
    scaffold_freq = scaffold_statistics(docking_ready)
    print("\nTop scaffolds:")
    for scaf, count in sorted(scaffold_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{scaf}: {count}")

    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
