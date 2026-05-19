from graphnet.data.dataconverter import DataConverter
from graphnet.data.writers import SQLiteWriter
# Import the necessary extractor for your raw data format (e.g., H5Extractor)
from graphnet.data.extractors.liquido import H5HitExtractor, H5TruthExtractor

# 1. Set up the DataConverter
#    This example assumes you're converting from HDF5 files.
#    You need to provide the extractor(s) for your raw data format.
#data_converter = DataConverter(
   #extractors=[H5HitExtractor(), H5TruthExtractor()],
   # writer=SQLiteWriter(),
   # output_dir="/storage/work/wjk5361/my_graphnet_project/",
    #output_filename="merged_data.db",
#)

# In your case, you may already have the unconsolidated .db files,
# so you can skip the conversion and just merge.
data_converter = DataConverter(writer=SQLiteWriter())


# 2. Call the merge_files method
#    Point it to the .db files you want to combine.
data_converter.merge_files(
    files="/scratch/dfc13/LiquidO/GNN/CLOUD_4_10_25/gammas_fill_2M_FP_parallel_16mm_splitevdaq/SQLite_DAQ/CLOUD_gammas_fill_FP_parallel_16mm_1*.db",
)

print("Finished merging files!")