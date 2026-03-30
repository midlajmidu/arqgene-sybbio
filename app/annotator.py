import random

def mock_kegg_annotation(proteins):
    fake_ko_list = ["K00001", "K00002", "K00003", "K00004"]
    annotated = []

    for protein in proteins:
        annotated.append({
            "protein_id": protein["id"],
            "ko_id": random.choice(fake_ko_list)
        })

    return annotated