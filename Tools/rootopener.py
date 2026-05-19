import argparse
import os
import pickle
import sys
from hitman.tools.ratextract import DataExtractor

def main():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(
        description="Extract Hitman data from ROOT files and split into chunked pickle lists."
    )
    parser.add_argument(
        'input_files', 
        nargs='+', 
        help="List of input ROOT files to process."
    )
    parser.add_argument(
        '-n', '--size', 
        type=int, 
        required=True, 
        help="Number of events per output file (chunk size)."
    )
    parser.add_argument(
        '-o', '--outdir', 
        type=str, 
        default='./chunks', 
        help="Output directory for the chunked files."
    )
    parser.add_argument(
        '--prefix', 
        type=str, 
        default='hitman_events', 
        help="Prefix for output filenames."
    )

    args = parser.parse_args()

    # 2. Extract Data
    print(f"[INFO] Loading {len(args.input_files)} input files...")
    
    # Initialize your extractor
    try:
        extractor = DataExtractor(args.input_files)
    except Exception as e:
        print(f"[ERROR] Failed to initialize DataExtractor: {e}")
        sys.exit(1)

    print("[INFO] Processing events (this may take some time for large files)...")
    
    # We use get_hitman_reco_data() because it returns the "giant list" of dicts
    # defined in your snippet.
    full_event_list = extractor.get_hitman_reco_data()
    
    total_events = len(full_event_list)
    if total_events == 0:
        print("[WARNING] No events found. Exiting.")
        sys.exit(0)
        
    print(f"[INFO] Extracted {total_events} total events.")

    # 3. Chunk and Save
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        print(f"[INFO] Created output directory: {args.outdir}")

    chunk_size = args.size
    num_chunks = (total_events + chunk_size - 1) // chunk_size  # Ceiling division

    print(f"[INFO] Splitting into {num_chunks} files of ~{chunk_size} events each.")

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_events)
        
        # Slice the giant list
        event_chunk = full_event_list[start_idx:end_idx]
        
        # Generate filename
        outfile = os.path.join(
            args.outdir, 
            f"{args.prefix}_{i:04d}.pkl"
        )
        
        # Save to pickle
        with open(outfile, 'wb') as f:
            pickle.dump(event_chunk, f)
            
        print(f"   -> Saved {outfile} ({len(event_chunk)} events)")

    print("[INFO] Done.")

if __name__ == "__main__":
    main()