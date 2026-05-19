"""FileHandlingTools"""

import os
import shutil
import errno
import psutil

import socket

import glob

def CopyAllFilesToFastDisk(PathsToFastDisk, SourceFileList, TargetSubdirectory):

   # Create a new function that takes in the full input file path.
   # Use the path-basename to create the same directory structure on the fast disk.
   # Copy over all the files in the source directory to the newly created directory.
   # Return the new data_path for training etc.

    NFilesCopied = 0

    # Check if any paths in PathsToFastDisk exist
    PathToFastDisk = None
    for path in PathsToFastDisk:
        if os.path.exists(path):
            PathToFastDisk = path
            break

    if PathToFastDisk is None:
        print(f"\nFHT.CopyAllFilesToFastDisk: None of {PathsToFastDisk} exist. No files copied to fast disk.\n")
        return NFilesCopied, None
    else:
        print(f"\nFHT.CopyAllFilesToFastDisk: Will copy files to {PathToFastDisk} on node {socket.gethostname()}.\n")

    # Remove any files in PathToFastDisk directory that were accessed >= some time ago
    # For example, for one week ago, set FileAge = (7 * 24 * 60 * 60) = (days * hours * mins * secs)
    DryRun = False
    FileAge = (4 * 24 * 60 * 60) # days * hours * mins * secs
    FileAge = (15 * 60) # mins * secs
    NFilesDeleted, GBFreed = DeleteOldFiles(PathToFastDisk, DryRun, FileAge)
    if DryRun:
        print(f'FileHandlingTools: DryRun: Would have deleted {NFilesDeleted} old files from {PathToFastDisk}, freeing {GBFreed:.1f} GB of space.\n')
    else:
        print(f'FileHandlingTools: Deleted {NFilesDeleted} old files from {PathToFastDisk}, freeing {GBFreed:.2f} GB of space.\n')


    # Create target directory
    TargetSubdirectory = TargetSubdirectory.lstrip('/') # strip leading "/" so os.path.join works properly
    target_dir = os.path.join(PathToFastDisk, TargetSubdirectory)
    #print(f'FHT.CopyAllFilesToFastDisk: ',
    #      f'Merged fast disk pathname {PathToFastDisk} with target subdir {TargetSubdirectory} into target {target_dir}')
    os.makedirs(target_dir, exist_ok=True)
    print(f"FHT.CopyAllFilesToFastDisk: Created directory {target_dir}")

    # Copy files from source directory, but only if they are not already present in the target directory
    # Keep track of how many files were already present.
    # If file needs to be copied over, use 
    NFilesAlreadyPresent = 0
    NFilesCopied = 0
    for source_file in SourceFileList:
        target_file = os.path.join(target_dir, os.path.basename(source_file))
        if os.path.exists(target_file):
            NFilesAlreadyPresent += 1
        else:
            try:
                shutil.copy2(source_file, target_file, follow_symlinks=True)
            except OSError as e:
                if e.errno == errno.ENOSPC: # Error-handling: no space for copy
                    # Calculate total space needed
                    total_size_needed = sum(os.path.getsize(f) for f in SourceFileList)/1024/1024/1024 # in GB
                    # Get available space on fast disk
                    stats = os.statvfs(PathToFastDisk)
                    available_space = stats.f_frsize * stats.f_bavail /1024/1024/1024 # also in GB
                    print(f"FHT.CopyAllFilesToFastDisk: On node {socket.gethostname()}, "
                            f"space needed = {total_size_needed:.1f} GB, "
                            f"space available on {PathToFastDisk} = {available_space:.1f} GB.\n "
                            f"File copying to fast disk terminated after {NFilesCopied} files copied at:\n   cp {source_file} {target_file}."
                            f"Will use files from original location.")
                    return 0, None # First argument equalling zero will cause calling code to use original location
                else:
                    raise # Other error codes get passed up the chain

            NFilesCopied += 1
        #print(f'FHT.CopyAllFilesToFastDisk: Copied {source_file} to {target_file}')

    # [Might want to move this chunk of code to an earlier place in order to free up space from the get-go.]
    # Remove any files in the target dir that are not in the source file list.
    NUnmatchedFiles = RemoveUnmatchedFiles(target_dir, SourceFileList)

    # Count files
    NSourceFiles = len(SourceFileList)
    NTargetFiles = len([f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))])

    print(f"FHT.CopyAllFilesToFastDisk:\n   The number of source files was {NSourceFiles}. "
            f"\n   The number of target files was {NTargetFiles}. "
            f"\n   {NFilesCopied} files were copied to {target_dir}. "
            f"\n   {NUnmatchedFiles} unmatched files were deleted from {target_dir}.")

    if NFilesAlreadyPresent != 0:
        print(f"There were {NFilesAlreadyPresent} files already present.  These were not copied again.\n")
    else:
        print(f"There were no files already present.\n")
        

    if NTargetFiles != NSourceFiles:
        raise RuntimeError(f"FHT.CopyAllFilesToFastDisk:\n   Number of source files "
                        f"({NSourceFiles}) in {os.path.dirname(SourceFileList[0])}\n   does not equal the number of target files\n   "
                        f"({NTargetFiles}) in {target_dir}")

    NFilesCopied = NTargetFiles

    return NFilesCopied, target_dir

def RemoveUnmatchedFiles(target_dir, source_file_list):
    # Convert source_file_list to a set of filenames (without paths) for faster lookup
    source_filenames = set(os.path.basename(f) for f in source_file_list)
    
    NUnmatchedFiles = 0
    # Iterate through files in target directory
    for filename in os.listdir(target_dir):
        # If the filename is not in our source list
        if filename not in source_filenames:
            # Construct full path to file
            file_path = os.path.join(target_dir, filename)
            # Check if it's a file (not a directory)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    NUnmatchedFiles += 1
                except OSError as e:
                    print(f"Error removing {filename}: {e}")

    return NUnmatchedFiles



def CopyFileToFastDisk(InputFilename, PathToFastDisk):
    """Perform copy of file to fast disk unless same file has already been copied"""

    # Check if the input file exists
    if not os.path.isfile(InputFilename):
        raise ValueError(f'Input file {InputFilename} does not exist')

    # Check if the PathToFastDisk is valid
    if not os.path.isdir(PathToFastDisk):
        print(f'The specified path to fast disk "{PathToFastDisk}" does not exist.')
        return InputFilename
    else:
        # Extract the file name from the input filename
        filename = os.path.basename(InputFilename)
        # Construct the target file name on the fast disk
        InputFilenameFastDisk = os.path.join(PathToFastDisk, filename)

        # Check if the file has already been copied to the fast disk
        if os.path.isfile(InputFilenameFastDisk):
            # Check if the file on the fast disk and the input file are identical in size
            if os.path.getsize(InputFilename) == os.path.getsize(InputFilenameFastDisk):
                print(f'File {InputFilename} has already been copied to {InputFilenameFastDisk}')
                return InputFilenameFastDisk

        # Check if there is enough space on the fast disk to hold the input file
        # If not, print an information message and just return the input file name
        space = psutil.disk_usage(PathToFastDisk).free
        if space < os.path.getsize(InputFilename):
            print(f'Insufficient space to copy {InputFilename} to {PathToFastDisk}\n' 
                  f'   Available space: {space}\n'
                  f'   Required space: {os.path.getsize(InputFilename)}')
            return InputFilename

        # Perform the copy
        print(f'Starting copy of {InputFilename} to {InputFilenameFastDisk}')
        shutil.copy2(InputFilename, InputFilenameFastDisk)
        print(f'Finished copy of {InputFilename} to {InputFilenameFastDisk}')

        return InputFilenameFastDisk


import os
import time
import pwd
from datetime import datetime, timedelta

def DeleteOldFiles(Path, DryRun, FileAge):
    # Get current user's ID
    user_id = os.getuid()
    
    # Calculate timestamp for FileAge ago.  For example, for one week ago, use (7 * 24 * 60 * 60).
    time_ago = time.time() - FileAge
    
    files_deleted = 0
    bytes_freed = 0
    
    # Walk through directory and all subdirectories
    for root, dirs, files in os.walk(Path):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                # Get file stats
                stats = os.stat(filepath)
                
                # Check if file is owned by current user
                if stats.st_uid == user_id:
                    # Check if last access was more than time_ago ago
                    if stats.st_atime < time_ago:
                        # Get file size before deleting
                        file_size = stats.st_size
                        
                        # Delete the file
                        if not DryRun:
                            os.remove(filepath)
                            print(f"FileHandlingTools: Deleted file: {filepath}")
                        
                        files_deleted += 1
                        bytes_freed += file_size
                        if DryRun:
                            print(f"FileHandlingTools: Would have deleted file: {filepath}")
                        
            except (FileNotFoundError, PermissionError) as e:
                print(f"FHT DeleteOldFiles: Error processing {filepath}:\n   {e}")
    
    # Convert bytes freed to GB for printing
    gb_freed = bytes_freed / (1024**3)
    
    return files_deleted, gb_freed

