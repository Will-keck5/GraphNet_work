"""Recostruction seperate reconstruction"""
# =============================================================================
# SCRIPT: reconstruct_sequential.py
#
# This script processes multiple input .db files sequentially, collects
# all reconstruction results, and saves them to a single output .db file.
# =============================================================================
import os
import argparse
import glob
import sqlite3
from typing import List
import torch
import pandas as pd
from graphnet.models import Model
from graphnet.data.datamodule import GraphNeTDataModule
from graphnet.data.dataset import SQLiteDataset
from graphnet.models.graphs import KNNGraph
from graphnet.models.detector.liquido import LiquidO_v2
from graphnet.training.labels import Direction
from graphnet.utilities.config import ModelConfig

def perform_sequential_reconstruction(
    model_dir: str,
    data_path_pattern: str, # Takes a wildcard pattern e.g., "path/to/*.db"
    output_db: str,
    pulsemap: str,
    features: List[str],
    truth_table: str,
    gpus: List[int],
    batch_size: int,
    num_workers: int,
) -> None:
    # --- Find all input files ---
    print(f"Searching for files with pattern: {data_path_pattern}")
    input_files = sorted(glob.glob(data_path_pattern))
    if not input_files:
        print("Error: No input files found matching the pattern. Exiting.")
        return
    print(f"Found {len(input_files)} files to process.")

    # --- Load Model (once) ---
    print(f"Loading model from: {model_dir}")
    state_dict_path = os.path.join(model_dir, "state_dict.pth")
    config_path = os.path.join(model_dir, "model_config.yml")
    try:
        model_config = ModelConfig.load(config_path)
        model = Model.from_config(model_config, trust_pickle=True)
        model.load_state_dict(torch.load(state_dict_path))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    # --- Process each file and collect results ---
    all_results = []
    for i, file_path in enumerate(input_files):
        print("-" * 70)
        print(f"Processing file {i+1}/{len(input_files)}: {file_path}")
        try:
            # Set up data loader for the current file
            graph_definition = KNNGraph(detector=LiquidO_v2())
            data_module = GraphNeTDataModule(
                dataset_reference=SQLiteDataset,
                dataset_args={
                    "path": file_path,
                    "pulsemaps": pulsemap,
                    "features": features,
                    "truth": [],
                    "truth_table": truth_table,
                    "graph_definition": graph_definition,
                    "labels": {'direction': Direction(zenith_key='zenith', azimuth_key='azimuth')}
                },
                train_selection=[],
                val_selection=[],
                test_dataloader_kwargs={"batch_size": batch_size, "num_workers": num_workers},
            )
            test_dataloader = data_module.test_dataloader

            # Perform prediction
            results_df = model.predict_as_dataframe(
                test_dataloader,
                gpus=gpus,
                additional_attributes=['event_no', 'zenith', 'azimuth', 'energy']
            )
            all_results.append(results_df)
            print(f"Finished processing. Found {len(results_df)} events.")

        except Exception as e:
            print(f"!! An error occurred while processing {file_path}: {e}")
            print("!! Skipping this file and continuing with the next one.")
            continue

    # --- Combine and save all results ---
    if not all_results:
        print("No results were collected. Nothing to save.")
        return

    print("-" * 70)
    print("All files processed. Concatenating results...")
    final_df = pd.concat(all_results, ignore_index=True)
    print(f"Total of {len(final_df)} events collected. Writing to final database...")

    try:
        os.makedirs(os.path.dirname(output_db), exist_ok=True)
        conn = sqlite3.connect(output_db)
        final_df.to_sql('reconstruction', conn, if_exists='replace', index=False)
        conn.close()
        print(f"Successfully saved all results to '{output_db}' in table 'reconstruction'.")
    except Exception as e:
        print(f"An error occurred while writing the final database: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process multiple .db files sequentially and save to a single output .db file.")
    parser.add_argument("-m", "--model_dir", type=str, required=True)
    parser.add_argument("-d", "--data_path_pattern", type=str, required=True, help="Wildcard pattern for input data files (e.g., 'data/*.db').")
    parser.add_argument("-o", "--output_db", type=str, required=True, help="Path for the final output DB file.")
    parser.add_argument("--pulsemap", type=str, default="DAQData")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", nargs='+', type=int, default=[0])
    args = parser.parse_args()

    FEATURES = ['sipm_x', 'sipm_y', 'sipm_z', 'time', 'charge']
    TRUTH_TABLE = 'TruthData'

    perform_sequential_reconstruction(
        model_dir=args.model_dir,
        data_path_pattern=args.data_path_pattern,
        output_db=args.output_db,
        pulsemap=args.pulsemap,
        features=FEATURES,
        truth_table=TRUTH_TABLE,
        gpus=args.gpus,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

# explain to me how this script works
"""
This script is designed to process multiple SQLite database files containing events data sequentially, reconstructing the events using a pre-trained model, and saving the results into a single output SQLite database file. Here's a breakdown of how it works:
1. The script takes several command-line arguments, including the model directory, input data path pattern, output database path, and various processing options (e.g., batch size, number of workers, GPU IDs).
2. It defines a function `perform_sequential_reconstruction` that encapsulates the main processing logic.
3. Inside this function, it uses the `glob` module to find all input database files matching the specified pattern.
4. For each input file, it sets up a data loader and performs event reconstruction using the pre-trained model.
5. The results are collected into a list and concatenated into a final DataFrame.
6. Finally, the script saves the concatenated results into the specified output SQLite database file.
"""