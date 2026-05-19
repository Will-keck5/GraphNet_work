"""Example of training Model."""
# Train to find vertex.

import os
from typing import Any, Dict, List, Optional

import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


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

class Vertex(Label):
    """Class for producing a single particle vertex label."""

    def __init__(
        self,
        key: str = "vertex",
        vertex_x_key: str = "vertex_x",
        vertex_y_key: str = "vertex_y",
        vertex_z_key: str = "vertex_z",
    ):
        """Construct `Direction`.

        Args:
            key: The name of the field in `Data` where the label will be
                stored. That is, `graph[key] = label`.
            vertex_x_key: The name of the pre-existing key in `graph` that will
                be used to access the vertex x-coordinate.
            vertex_y_key: The name of the pre-existing key in `graph` that will
                be used to access the vertex y-coordinate.
            vertex_z_key: The name of the pre-existing key in `graph` that will
                be used to access the vertex z-coordinate.
        """
        self._x = vertex_x_key
        self._y = vertex_y_key
        self._z = vertex_z_key
   
        # Base class constructor
        super().__init__(key=key)

    def __call__(self, graph: Data) -> torch.tensor:
        """Compute label for `graph`."""
        x = graph[self._x].reshape(-1, 1)
        y = graph[self._y].reshape(-1, 1)
        z = graph[self._z].reshape(-1, 1)
        return torch.cat((x,y,z), dim = 1).squeeze(1)
    
class MSELoss(LossFunction):
    """Mean squared error loss."""

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Implement loss calculation."""
        # Check(s)
        assert prediction.dim() == 2
        if len(target.shape) == 3:
            target = target.squeeze(1)
        assert prediction.size() == target.size()
        elements = torch.mean((prediction - target) ** 2, dim=-1)
        return elements


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
    # v0: base definition with SiPM (x,y,z,t)
    # v1: extended definition including stereo layer info with SiPM (x,y,z,t,theta,phi)
    # v2: DAQData: just one time and charge value per channel
    if FeaturesVersion == 'v0':
        graph_definition = KNNGraph(detector=LiquidO_v0())
    elif FeaturesVersion == 'v1':
        graph_definition = KNNGraph(detector=LiquidO_v1())
    elif FeaturesVersion == 'v2':
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
                                    "labels": {'direction': Direction(
                                        zenith_key='zenith',
                                        azimuth_key='azimuth',
                                    )},
                                    train_dataloader_kwargs={"batch_size": batch_size,
                                                             "num_workers": num_workers,
                                                             "shuffle": True
                                                             },
                                    test_selection = selections,
                                )
    training_dataloader = data_module.train_dataloader
    validation_dataloader = data_module.val_dataloader
    #T/V/T
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
            # Don't use learning rate scheduler:
            ###optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
            # Use learning rate scheduler
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
        # Use callbacks=callbacks in model.fit below
        callbacks = [] # here, standard_model.py uses callbacks = [ProgressBar()]
        if validation_dataloader is not None:
            # Add Early Stopping (exactly as in default)
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=config["early_stopping_patience"],
                )
            )
            # Add Model Check Point (exactly as in default)
            callbacks.append(
                ModelCheckpoint(
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
            callbacks=callbacks,  # Use my custom callbacks above, which doesn't include ProgressBar
            #logger= None,
            logger = wandb_logger,
            # does not work: progress_bar_refresh_rate=0,
            enable_progress_bar=False,
            **config["fit"],
        )

    # Not yet tested:
    if InferenceOnly:
        print(f'Reading model from:\n   {outdir}')
        model_config = ModelConfig.load(f"{outdir}/model_config.yml")
        model = Model.from_config(model_config)  # With randomly initialised weights.
        model.load_state_dict(f"{outdir}/state_dict.pth")  # Now with trained weight.


    # Get predictions
    additional_attributes = ['zenith', 'azimuth', 'energy']
    assert isinstance(additional_attributes, list)  # mypy

    # Get predictions
    # This had to be replaced by what's above: additional_attributes = model.target_labels

    results = model.predict_as_dataframe(
        #T/V/T validation_dataloader,
        test_dataloader,
        additional_attributes=additional_attributes + ["event_no"],
        gpus=[0], # Just use a single GPU for inference.  N GPUs overwrite the .csv file N times,
                  # leaving 1/N of the events after finishing training and inference.
        #gpus=config["fit"]["gpus"],
    )

    # Save predictions and model to file
    os.makedirs(outdir, exist_ok=True)

    # Save results as .csv and .parquet
    results.to_csv(f"{outdir}/results.csv")
    #results.to_parquet(f"{outdir}/results.parquet")

    # Save full model (including weights) to .pth file - not version safe
    # Note: Models saved as .pth files in one version of graphnet
    #       may not be compatible with a different version of graphnet.
    model.save(f"{outdir}/model.pth")

    # Save model config and state dict - Version safe save method.
    # This method of saving models is the safest way.
    model.save_state_dict(f"{outdir}/state_dict.pth")
    model.save_config(f"{outdir}/model_config.yml")    
    
    # Put this at the end in case it crashes the job...
    results.to_parquet(f"{outdir}/results.parquet")

def GetTestEventNumberLists(data_path, percent_to_keep):
    """
    Extracts the last 10% of event numbers from the "TruthData" table in each SQLite database file.

    Args:
    - data_path (list): List of file paths to SQLite database files.
    - percent_to_keep (float): Percentage of event numbers to keep from each database.

    Returns:
    - selections (list): List containing the last 10% of event numbers for each database file.
    """
    # List to store selections for each file
    selections = []

    # Iterate over each file path
    for db_file in data_path:
        print(f"Processing {db_file}")
        
        # Connect to the SQLite database
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Get the total number of events
        cursor.execute("SELECT COUNT(*) FROM TruthData")
        total_events = cursor.fetchone()[0]
        
        # Calculate the number of events to keep (last 10%)
        num_to_keep = int(total_events * percent_to_keep)
        # Get the last 10% of event numbers
        cursor.execute(f"SELECT event_no FROM TruthData ORDER BY event_no DESC LIMIT {num_to_keep}")
        last_events = [row[0] for row in cursor.fetchall()]
        
        # Close the connection to the database
        conn.close()
        
        # Append last events to selections list
        selections.append(last_events)
    
    return selections


if __name__ == "__main__":
    # Constants
    # Define FeaturesVersion
    FeaturesVersion = 'v2' # v0 or v1 or v2
    if FeaturesVersion == 'v0':
        features = ['sipm_x', 'sipm_y', 'sipm_z', 't'] # detector LiquidO_v0
        pulsemap = 'HitData'
    elif FeaturesVersion == 'v1':
        features = ['sipm_x', 'sipm_y', 'sipm_z', 't', 'sipm_zenith', 'sipm_azimuth'] # detector LiquidO_v1
        pulsemap = 'HitData'
    elif FeaturesVersion == 'v2':
        features = ['sipm_x', 'sipm_y', 'sipm_z', 'time', 'charge'] # detector LiquidO_v2    
        pulsemap = 'DAQData'
    truth = TRUTH.LIQUIDO
    truth_table = 'TruthData'
    # data_path is defined via command line arg and glob, see below
    target = 'direction'
    gpus = [0] # [0,1,2,3] # multiple GPUs only used for training (overridden in code for inference step)
    max_epochs = 1 # 30 # 35
    early_stopping_patience = 15 # 10 20 3
    batch_size = 256 # 24 # 20
    num_workers = 4 # 8 4  (Should this match --cpus-per-task= in SLURM script?)

    # Build the output directory path.  
    # Strip off everything past subset*/ from data_path.
    # Add 'target', len(gpus), batch_size: vertex_g{len(gpus)}_b{batch_size}
    # Add optional label using argparse

    import os
    import argparse

    # Set up command line argument parsing
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
    data_path.sort() # sort the data_path filenames

    if DataPath != '':
        print(f'Running with db files from DataPath: {DataPath}')
    print(f'Running with {len(data_path)} files.')

    if 'subset_' in data_path[0]:
        base_path = data_path[0].split("subset_")[0]
    elif 'DirectFromRoot' in data_path[0]:
        base_path = data_path[0].split("DirectFromRoot")[0] # should probably key on '*.db" instead...
    # NOT SURE IF ALL THESE ARE RELEVANT.  This script just reads the .db files
    # and the data_path should look something like 
    # electrons_uniform_IN2P3_v01/db_Mux1.6/subset_*/merged/merged.db
    # So the "if" above will work and all the "elif" below will never be true
    elif 'h5_Mux4' in data_path[0]:
        base_path = data_path[0].split("h5_Mux4")[0]
        print(f'Remember to set base_path for different Mux runs!')
    elif 'h5_Mux1.4' in data_path[0]:
        base_path = data_path[0].split("h5_Mux1.4")[0]
        print(f'Remember to set base_path for different Mux runs!')
    elif 'h5_Mux1.6' in data_path[0]:
        base_path = data_path[0].split("h5_Mux1.6")[0]
        print(f'Remember to set base_path for different Mux runs!')
    elif 'h5_Mux1.8' in data_path[0]:
        base_path = data_path[0].split("h5_Mux1.8")[0]
        print(f'Remember to set base_path for different Mux runs!')
    else:
        base_path = os.path.dirname(data_path[0]) # use the full base pathname with Garrett's directly processed files
        
    #/storage/group/dfc13/default/cowen/LiquidO/GNN/electrons_uniform_IN2P3_v01/h5_Mux4/merged/merged*.db

    # Build the output directory path
    outdir = os.path.join(base_path, f"{target}-g{len(gpus)}-bs{batch_size}-me{max_epochs}-{outdir_label}".strip("-"))

    # Write out the results.csv and model files to a more permanent location
    if "/scratch/dfc13/LiquidO/GNN" in outdir:
        outdir = outdir.replace("/scratch/dfc13/LiquidO/GNN", "/storage/group/dfc13/default/liquido/CLOUD")

    print(f'Output directory path: {outdir}')

    # Create the directory if it does not exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print(f"Created output directory:\n   {outdir}")
    else:
        print(f"Output directory already exists:\n   {outdir}")

    #outdir = "/storage/group/dfc13/default/cowen/LiquidO/CNN-RNN/ratv02/electrons_uniform_IN2P3/GraphNet/electrons/subset_01/merged"

    # Set up WandB logger
    wandb_logger = WandbLogger(
                project="GraphNet-Vertex",
                name = outdir_label,
                save_dir="/storage/group/dfc13/default/cowen/LiquidO/GNN/WandBLogs",
                log_model=True,
            )

    # Here we can try to copy all the files in the current data_path over to fast disk.
    # If we are successful, data_path will be set to the new value.

    #import ../Tools/FileHandlingTools as FHT

    #PathToFastDisk = "/nvmetmp"
    #data_path = FHT.CopyAllFilesToFastDisk(InputPathname, PathToFastDisk)

    # Get lists of events, one per file, of the last 10% in each file.
    # Use for Test data
    percent_to_keep = 0.95
    selections = GetTestEventNumberLists(data_path, percent_to_keep)

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