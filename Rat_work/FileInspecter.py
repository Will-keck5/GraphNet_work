import uproot
import sys

def print_structure(node, prefix=""):
    """
    Recursively prints the structure of an uproot node (File, Directory, Tree, or Branch).
    """
    # Attempt to get the keys of the current node
    if not hasattr(node, "keys") or not callable(node.keys):
        return
        
    keys = node.keys()
    
    for i, key in enumerate(keys):
        is_last = (i == len(keys) - 1)
        pointer = "└── " if is_last else "├── "
        
        # Uproot appends cycle numbers to keys (e.g., 'output;1'), so we strip them for readability
        display_name = key.split(';')[0] if isinstance(key, str) else str(key)
        
        try:
            item = node[key]
            class_name = item.classname
        except Exception:
            class_name = "Unknown"
            item = None

        print(f"{prefix}{pointer}{display_name} [{class_name}]")
        
        # Recurse into the item if it has its own keys (like a Tree or nested Branch)
        extension = "    " if is_last else "│   "
        if item is not None:
            print_structure(item, prefix + extension)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python print_ntuple_structure.py <path_to_ntuple.root>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    
    print(file_path)
    # Open the ROOT file
    with uproot.open(file_path) as root_file:
        print_structure(root_file)

        