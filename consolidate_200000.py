"""Consolidate 20 chunks (200,000 words) of SD2Vec into a single unified dictionary."""

import json
from pathlib import Path
import numpy as np

ROOT_DIR = Path("/Users/ishikawahiroki/sd2vec")
VOCAB_DIR = ROOT_DIR / "large_vocab_200000"
VECTORS_DIR = VOCAB_DIR / "vectors"
CONCEPTS_DIR = VOCAB_DIR / "concepts"
DIMENSIONS_FILE = ROOT_DIR / "dimensions_phase3.json"

TOTAL_CHUNKS = 20
CHUNK_SIZE = 10000
EXPECTED_CONCEPTS = 200000

def main():
    print("=== Consolidating 200,000-Word SD2Vec Dictionary ===")
    
    with open(DIMENSIONS_FILE, "r", encoding="utf-8") as f:
        dimensions = json.load(f)

    all_vectors = []
    all_stds = []
    all_concepts = []
    
    for i in range(TOTAL_CHUNKS):
        cid = f"chunk_{i:04d}"
        cdir = VECTORS_DIR / cid
        print(f"Loading {cid}...")
        
        vec = np.load(cdir / "vectors.npy")
        std = np.load(cdir / "repeat_std.npy")
        with open(cdir / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        assert vec.shape == (CHUNK_SIZE, 50), f"Bad shape for {cid}: {vec.shape}"
        assert std.shape == (CHUNK_SIZE, 50), f"Bad shape for {cid}: {std.shape}"
        assert len(meta["concepts"]) == CHUNK_SIZE, f"Bad concepts len for {cid}: {len(meta['concepts'])}"
        
        # Verify alignment with original concept file
        orig_concepts_file = CONCEPTS_DIR / f"{cid}.txt"
        with open(orig_concepts_file, "r", encoding="utf-8") as f:
            orig_concepts = [line.strip() for line in f if line.strip()]
        assert orig_concepts == meta["concepts"], f"Concept mismatch in {cid}!"
        
        all_vectors.append(vec)
        all_stds.append(std)
        all_concepts.extend(meta["concepts"])
        
    print(f"Concatenating arrays...")
    final_vectors = np.vstack(all_vectors)
    final_stds = np.vstack(all_stds)
    
    assert final_vectors.shape == (EXPECTED_CONCEPTS, 50)
    assert final_stds.shape == (EXPECTED_CONCEPTS, 50)
    assert len(all_concepts) == EXPECTED_CONCEPTS
    
    # Save unified files in large_vocab_200000/
    print("Saving unified vectors.npy...")
    np.save(VOCAB_DIR / "vectors.npy", final_vectors)
    
    print("Saving unified repeat_std.npy...")
    np.save(VOCAB_DIR / "repeat_std.npy", final_stds)
    
    print("Computing quality metrics...")
    quality = {
        "concept_count": EXPECTED_CONCEPTS,
        "dimension_count": 50,
        "vector_shape": [EXPECTED_CONCEPTS, 50],
        "mean_repeat_std": float(np.mean(final_stds)),
        "max_repeat_std": float(np.max(final_stds)),
        "nonzero_fraction": float(np.count_nonzero(final_vectors) / final_vectors.size),
    }
    with open(VOCAB_DIR / "quality.json", "w", encoding="utf-8") as f:
        json.dump(quality, f, ensure_ascii=False, indent=2)
        
    print("Saving metadata.json...")
    metadata = {
        "model": "LiquidAI/d1-3B",
        "concept_count": EXPECTED_CONCEPTS,
        "concepts": all_concepts,
        "dimensions": dimensions,
        "repeats": 1,
        "scale": [-3, 3],
        "method": "d1 system_one score decisions",
    }
    with open(VOCAB_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    print("\n=== Consolidation Complete! ===")
    print(f"Vectors Shape: {final_vectors.shape} (dtype: {final_vectors.dtype})")
    print(f"File Size: {(VOCAB_DIR / 'vectors.npy').stat().st_size / (1024 * 1024):.2f} MB")
    print(f"Quality: {quality}")

if __name__ == "__main__":
    main()
