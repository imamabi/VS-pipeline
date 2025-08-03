import argparse
import sys
from rdkit import Chem
from rdkit.Chem import SDWriter

from data_processing import load_molecules, standardize_molecules
from conformer_generation import generate_3d
from similarity_search import calculate_similarities, get_top_n
from visualization import compute_features, reduce_dimensions, butina_clustering, plot_clusters
from scaffold_analysis import scaffold_statistics

def main():
    # Argument Parser
    parser = argparse.ArgumentParser(description="Virtual Screening Pipeline using RDKit")
    parser.add_argument("-i", "--input", required=True, help="Input dataset file (SDF)")
    parser.add_argument("-r", "--reference", required=True, help="Reference molecules file (SDF)")
    parser.add_argument("-o", "--output", required=True, help="Output file for similar molecules (SDF)")
    parser.add_argument("-c", "--cutoff", type=float, default=0.9, help="Similarity cutoff (default: 0.90)")
    parser.add_argument("-f", "--features", default="both", choices=["ecfp", "fcfp", "both"], help="Feature type for visualization")
    args = parser.parse_args()

    print("=== Virtual Screening Pipeline ===")
    print(f"Input dataset: {args.input}")
    print(f"Reference molecules: {args.reference}")
    print(f"Output file: {args.output}")
    print(f"Similarity cutoff: {args.cutoff}")
    print(f"Feature method: {args.features}")

    # Load molecules
    print("\n[Step 1] Loading molecules...")
    dataset_mols = load_molecules(args.input, "sdf")
    reference_mols = load_molecules(args.reference, "sdf")
    print(f"Dataset size: {len(dataset_mols)}, Reference size: {len(reference_mols)}")

    # Standardize molecules
    print("[Step 2] Standardizing molecules...")
    dataset_mols, errors = standardize_molecules(dataset_mols)
    if errors:
        print(f"Warning: {len(errors)} molecules failed standardization and were skipped.")

    # Generate 3D conformers
    print("[Step 3] Generating 3D conformers...")
    for mol in dataset_mols:
        try:
            generate_3d(mol)
        except Exception as e:
            print(f"3D generation failed for {Chem.MolToSmiles(mol)}: {e}")

    # Similarity search
    print("[Step 4] Calculating similarities (ECFP + FCFP)...")
    similarities = calculate_similarities(reference_mols, dataset_mols, combine_method="average")
    
    # Filter molecules based on cutoff
    print(f"[Step 5] Filtering molecules with similarity >= {args.cutoff}...")
    selected_mols = []
    for ref_idx, sim_list in similarities.items():
        for idx, combined_score, ecfp_score, fcfp_score in sim_list:
            if combined_score >= args.cutoff:
                selected_mols.append(dataset_mols[idx])
    
    selected_mols = list(set(selected_mols))  # Remove duplicates
    print(f"Selected {len(selected_mols)} molecules above cutoff.")

    # Save output to SDF
    if selected_mols:
        print(f"[Step 6] Writing selected molecules to {args.output}...")
        writer = SDWriter(args.output)
        for mol in selected_mols:
            writer.write(mol)
        writer.close()
    else:
        print("No molecules passed the similarity cutoff. Exiting.")
        sys.exit(0)

    # Visualization
    print("[Step 7] Dimensionality reduction (UMAP) and clustering (Butina)...")
    features, _ = compute_features(selected_mols, method=args.features)
    coords = reduce_dimensions(features, method="umap")
    clusters = butina_clustering(selected_mols, cutoff=0.6)
    
    cluster_labels = []
    for i in range(len(selected_mols)):
        label = [idx for idx, cluster in enumerate(clusters) if i in cluster][0]
        cluster_labels.append(label)
    
    plot_clusters(coords, cluster_labels, title="Selected Molecules Clusters")

    # Scaffold analysis
    print("[Step 8] Scaffold analysis...")
    scaffold_freq = scaffold_statistics(selected_mols)
    print("\nTop scaffolds:")
    for scaf, count in sorted(scaffold_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{scaf}: {count}")

    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
