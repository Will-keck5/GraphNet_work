import uproot
import pandas as pd
import sys

def extract_coordinates(input_file, output_file):
    with uproot.open(input_file) as file:
        tree = file["output"]
        
        # Extract branches directly into a Pandas DataFrame
        df = tree.arrays(["mcx", "mcy", "mcz"], library="pd")
        
        # Write the DataFrame to a new CSV file
        df.to_csv(output_file, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_coords.py <input.ntuple.root> <output.csv>")
        sys.exit(1)
        
    extract_coordinates(sys.argv[1], sys.argv[2])