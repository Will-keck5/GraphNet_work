"""Direction Reconstruction with GraphNeT."""
# Train to find direction.

import os
from typing import Any, Dict, List, Optional

import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

from graphnet.data.constants import TRUTH
from graphnet.models import StandardModel
from graphnet.models.detector.liquido import LiquidO_v2
from graphnet.models.gnn import DynEdge
from graphnet.models.graphs import KNNGraph
from graphnet.data.datamodule import GraphNeTDataModule
from graphnet.data.dataset import SQLiteDataset
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa
from graphnet.training.loss_functions import VonMisesFisher3DLoss
from graphnet.training.labels import Direction

# For InferenceOnly
from graphnet.models import Model
from graphnet.utilities.config import ModelConfig

import pandas as pd
import sqlite3
import glob

from pytorch_lightning.loggers import WandbLogger # Use WandB to track loss, acc, etc.

# Added as part of effort to remove ProgressBar
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def main(
    path: str,
    pulsemap: str,
    target: str,
    truth_table: str,
    gpus: Optional[List[int]],
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    num_workers: int,
    outdir: str,
    features: List[str],
    truth: List[str],
    InferenceOnly: bool,
    selections, # Holds the event numbers for the Test dataset, one group of numbers per file
    FeaturesVersion, # v0 or v1 or v2
) -> None:
    """Run example."""

    # Configuration
    config: Dict[str, Any] = {
        "path": path,
        "pulsemap": pulsemap,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "target": target,
        "early_stopping_patience": early_stopping_patience,
        "fit": {
            "gpus": gpus,
            "max_epochs": max_epochs,
        },
    }

    # Define graph representation
    graph_definition = KNNGraph(detector=LiquidO_v2())

    # Get DataLoaders
    data_module = GraphNeTDataModule(dataset_reference=SQLiteDataset,
                                     dataset_args = {
                                    "truth_table": truth_table,
                                    "pulsemaps": pulsemap,
                                    "truth": truth,
                                    "features": features,
                                    "path": data_path,
                                    "graph_definition": graph_definition,
                                    "labels": {'direction': Direction(zenith_key='zenith', azimuth_key='azimuth')}},
                                    train_dataloader_kwargs={"batch_size": batch_size,
                                                             "num_workers": num_workers,
                                                             "shuffle": True
                                                             },
                                    test_selection = selections,
                                )
    training_dataloader = data_module.train_dataloader
    validation_dataloader = data_module.val_dataloader
    test_dataloader = data_module.test_dataloader


    # Building model
    backbone = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )
    task = DirectionReconstructionWithKappa(
    hidden_size=backbone.nb_outputs,
    target_labels=[config["target"]],
    loss_function=VonMisesFisher3DLoss(),
)

    if not InferenceOnly:
        model = StandardModel(
            graph_definition=graph_definition,
            backbone=backbone,
            tasks=[task],
            optimizer_class=Adam,
            optimizer_kwargs={"lr": 1e-03},
            scheduler_class=ReduceLROnPlateau,
            scheduler_kwargs={
                "patience": 3,
            },
            scheduler_config={
                "frequency": 1,
                "monitor": "val_loss",
            },
        )

        # Create the exact same callbacks that would be created by default, minus ProgressBar
        callbacks = []
        if validation_dataloader is not None:
            # Add Early Stopping
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=config["early_stopping_patience"],
                )
            )
            # Add Model Check Point
            callbacks.append(
                ModelCheckpoint(
		    dirpath=os.path.join(outdir, "checkpoints"),
                    save_top_k=1,
                    monitor="val_loss",
                    mode="min",
                    filename=f"{model.backbone.__class__.__name__}-" + 
                            "{epoch}-{val_loss:.2f}-{train_loss:.2f}",
                )
            )


        # Training model
        model.fit(
            training_dataloader,
            validation_dataloader,
            early_stopping_patience=config["early_stopping_patience"],
            callbacks=callbacks,
            logger = wandb_logger,
            enable_progress_bar=False,
            **config["fit"],
        )

    if InferenceOnly:
        print(f'Reading model from:\n   {outdir}')
        model_config = ModelConfig.load(f"{outdir}/model_config.yml")
        model = Model.from_config(model_config)
        model.load_state_dict(f"{outdir}/state_dict.pth")


    # Get predictions
    additional_attributes = ['zenith', 'azimuth', 'energy']
    assert isinstance(additional_attributes, list)

    results = model.predict_as_dataframe(
        test_dataloader,
        additional_attributes=additional_attributes + ["event_no"],
        gpus=[0],
    )

    # Save predictions and model to file
    os.makedirs(outdir, exist_ok=True)

    results.to_csv(f"{outdir}/results.csv")

    model.save(f"{outdir}/model.pth")
    model.save_state_dict(f"{outdir}/state_dict.pth")
    model.save_config(f"{outdir}/model_config.yml")    
    
    results.to_parquet(f"{outdir}/results.parquet")

def GetTestEventNumberLists(data_path, percent_to_keep):
    """
    Extracts the last X% of event numbers from the "TruthData" table in each SQLite database file.
    """
    selections = []
    for db_file in data_path:
        print(f"Processing {db_file}")
        
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM TruthData")
        total_events = cursor.fetchone()[0]
        
        num_to_keep = int(total_events * percent_to_keep)
        
        cursor.execute(f"SELECT event_no FROM TruthData ORDER BY event_no DESC LIMIT {num_to_keep}")
        last_events = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        selections.append(last_events)
    
    return selections


if __name__ == "__main__":
    # Constants
    FeaturesVersion = 'v2'
    if FeaturesVersion == 'v2':
        features = ['sipm_x', 'sipm_y', 'sipm_z', 'time', 'charge']
        pulsemap = 'DAQData'

    truth = TRUTH.LIQUIDO
    truth_table = 'TruthData'
    target = 'direction'
    gpus = [0]
    max_epochs = 30
    early_stopping_patience = 15
    batch_size = 256
    num_workers = 4

    parser = argparse.ArgumentParser(description='GraphNet training args')
    parser.add_argument('-l','--OutdirLabel', 
                        type=str, 
                        default='', 
                        help='Optional label for outdir name.')
    parser.add_argument("-infonly","--InferenceOnly",
                    action = "store_true",
                    default = False,
                    help="If True, load existing model and just do inference step")
    parser.add_argument("-dp","--DataPath",
                    type=str,
                    default = '',
                    help="wildcard filename for creating data_path list")

    args = parser.parse_args()
    outdir_label = args.OutdirLabel
    InferenceOnly = args.InferenceOnly
    DataPath = args.DataPath

    data_path = glob.glob(DataPath)
    data_path.sort()

    if DataPath != '':
        print(f'Running with db files from DataPath: {DataPath}')
    print(f'Running with {len(data_path)} files.')

    if 'subset_' in data_path[0]:
        base_path = data_path[0].split("subset_")[0]
    else:
        base_path = os.path.dirname(data_path[0])
        
    outdir = os.path.join(base_path, f"{target}-g{len(gpus)}-bs{batch_size}-me{max_epochs}-{outdir_label}".strip("-"))

    if "/scratch/" in outdir:
        # Handle path replacement for specific systems if necessary
        pass

    print(f'Output directory path: {outdir}')
    os.makedirs(outdir, exist_ok=True)

    wandb_logger = WandbLogger(
                project="GraphNet-Direction",
                name = outdir_label,
                save_dir="/path/to/your/wandb/logs", # IMPORTANT: Change this path
                log_model=True,
            )

    percent_for_training = 0.95
    selections = GetTestEventNumberLists(data_path, percent_for_training)

    for i, selection in enumerate(selections):
        min_event_no = min(selection)
        max_event_no = max(selection)
        print(f"For file {i}:")
        print(f"  Minimum event_no: {min_event_no}")
        print(f"  Maximum event_no: {max_event_no}")

    main(
        path = data_path,
        pulsemap = pulsemap,
        target = target,
        truth_table = truth_table,
        gpus = gpus,
        max_epochs = max_epochs,
        early_stopping_patience = early_stopping_patience,
        batch_size = batch_size,
        num_workers = num_workers,
        outdir = outdir,
        features = features,
        truth = truth,
        InferenceOnly = InferenceOnly,
        selections = selections,
        FeaturesVersion = FeaturesVersion,
    )