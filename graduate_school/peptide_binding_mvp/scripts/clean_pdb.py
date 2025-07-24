import argparse

def clean_pdb_line(line):
    if not line.startswith("ATOM"):
        return line
    # Extract columns according to standard PDB format
    atom = line[0:6]
    atom_serial = line[6:11]
    atom_name = line[12:16]
    alt_loc = line[16:17]
    res_name = line[17:20]
    chain_id = line[21:22]
    res_seq = line[22:26]
    insertion = line[26:27]
    x = line[30:38]
    y = line[38:46]
    z = line[46:54]
    occupancy = "  1.00"
    temp_factor = " 20.00"
    element = atom_name.strip()[0].rjust(2)

    cleaned_line = f"{atom}{atom_serial} {atom_name}{alt_loc}{res_name} {chain_id}{res_seq}{insertion}   {x}{y}{z}{occupancy}{temp_factor}           {element}\n"
    return cleaned_line

def clean_pdb(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            outfile.write(clean_pdb_line(line))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean ZDOCK PDB file to standard format")
    parser.add_argument("--input", "-i", required=True, help="Input ZDOCK PDB file")
    parser.add_argument("--output", "-o", required=True, help="Output cleaned PDB file")
    args = parser.parse_args()
    clean_pdb(args.input, args.output)