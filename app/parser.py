from Bio import SeqIO

def parse_faa(file_path):
    proteins = []

    for record in SeqIO.parse(file_path, "fasta"):
        proteins.append({
            "id": record.id,
            "sequence": str(record.seq)
        })

    return proteins