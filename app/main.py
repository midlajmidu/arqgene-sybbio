import os
import pandas as pd
from .parser import parse_faa
from .annotator import mock_kegg_annotation
from .model_builder import build_simple_model

def run_pipeline():
    # Get project root dynamically
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    input_path = os.path.join(base_dir, "data", "protein.faa")
    output_path = os.path.join(base_dir, "results", "annotations.csv")

    if not os.path.exists(input_path):
        print("❌ protein.faa not found inside data folder")
        print("Expected path:", input_path)
        return

    proteins = parse_faa(input_path)
    print(f"✅ Parsed {len(proteins)} proteins")

    annotations = mock_kegg_annotation(proteins)

    df = pd.DataFrame(annotations)
    df.to_csv(output_path, index=False)

    print("✅ Annotation complete")
    print("📁 Saved to:", output_path)

        # Build and optimize model
    model = build_simple_model()
    solution = model.optimize()

    print("🔬 FBA Objective Value:", solution.objective_value)

if __name__ == "__main__":
    run_pipeline()