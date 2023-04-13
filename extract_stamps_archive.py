import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

# Function to extract a file
def extract(index):
    file_path = f"/global/cfs/cdirs/dessn/www/autoscan/stamps/stamps_{index}.tar"
    dest_dir = f"/global/cfs/cdirs/m4287/cosmology/dessn/stamps/stamps_{index}/"
    
    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Extract the file using pigz and tar to the destination directory
    subprocess.run(["tar", "-xvf", file_path, "-C", dest_dir])
    print(f"Extracted: stamps_{index}.tar to {dest_dir}")

# Main function
def main():
    indices = list(range(0, 11))

    # Extract files in parallel
    with ThreadPoolExecutor() as executor:
        executor.map(extract, indices)

if __name__ == "__main__":
    main()
