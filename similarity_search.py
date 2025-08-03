from rdkit.Chem import AllChem
from rdkit import DataStructs

def compute_fingerprint(mol, fp_type="ecfp", radius=2, nBits=2048):
    """
    Compute molecular fingerprint.
    
    Args:
        mol (RDKit Mol): input molecule
        fp_type (str): 'ecfp' (atom environment) or 'fcfp' (functional class)
    
    Returns:
        RDKit ExplicitBitVect fingerprint
    """
    if fp_type == "ecfp":
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits, useFeatures=False)
    elif fp_type == "fcfp":
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits, useFeatures=True)
    else:
        raise ValueError("fp_type must be 'ecfp' or 'fcfp'")

def calculate_similarities(reference_mols, dataset_mols, combine_method="average", weight=(0.5, 0.5)):
    """
    Calculate similarity using ECFP and FCFP and combine scores.
    
    Args:
        reference_mols (list): list of reference molecules
        dataset_mols (list): list of dataset molecules
        combine_method (str): 'average' or 'weighted'
        weight (tuple): weights for weighted method (ECFP, FCFP)
    
    Returns:
        dict: ref_idx -> [(dataset_idx, combined_score, ecfp_score, fcfp_score)]
    """
    similarities = {}
    
    # Compute fingerprints for all molecules
    ref_ecfp = [compute_fingerprint(m, "ecfp") for m in reference_mols]
    ref_fcfp = [compute_fingerprint(m, "fcfp") for m in reference_mols]
    data_ecfp = [compute_fingerprint(m, "ecfp") for m in dataset_mols]
    data_fcfp = [compute_fingerprint(m, "fcfp") for m in dataset_mols]
    
    for i in range(len(reference_mols)):
        sim_list = []
        for j in range(len(dataset_mols)):
            sim_ecfp = DataStructs.TanimotoSimilarity(ref_ecfp[i], data_ecfp[j])
            sim_fcfp = DataStructs.TanimotoSimilarity(ref_fcfp[i], data_fcfp[j])
            
            # Combine scores
            if combine_method == "average":
                final_score = (sim_ecfp + sim_fcfp) / 2
            elif combine_method == "weighted":
                final_score = (sim_ecfp * weight[0]) + (sim_fcfp * weight[1])
            else:
                raise ValueError("combine_method must be 'average' or 'weighted'")
            
            sim_list.append((j, final_score, sim_ecfp, sim_fcfp))
        
        # Sort by combined similarity
        sim_list.sort(key=lambda x: x[1], reverse=True)
        similarities[i] = sim_list
    
    return similarities

def get_top_n(similarities, dataset_mols, n=5):
    """
    Get top N similar molecules for each reference.
    
    Args:
        similarities (dict): similarity results
        dataset_mols (list): molecules
        n (int): number of top results
    
    Returns:
        dict: ref_idx -> [(mol, combined_score, ecfp_score, fcfp_score)]
    """
    top_n_dict = {}
    for ref_idx, sim_list in similarities.items():
        top_n = sim_list[:n]
        top_n_mols = [(dataset_mols[idx], score, ecfp, fcfp) for idx, score, ecfp, fcfp in top_n]
        top_n_dict[ref_idx] = top_n_mols
    return top_n_dict
