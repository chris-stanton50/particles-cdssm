import os
import argparse

def delete_files_with_extension(directory, extension):
    """
    Delete all files in the specified directory (and its subdirectories) with the given extension.

    Args:
        directory (str): Path to the directory to search.
        extension (str): File extension to delete (e.g., '.txt').
    """
    if not os.path.isdir(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return

    deleted_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                deleted_files.append(file_path)

    if deleted_files:
        print(f"Deleted {len(deleted_files)} files with extension '{extension}':")
        for file in deleted_files:
            print(f"  - {file}")
    else:
        print(f"No files with extension '{extension}' found in '{directory}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete files with a specific extension in a directory.")
    parser.add_argument("directory", type=str, help="Path to the directory.")
    parser.add_argument("extension", type=str, help="File extension to delete (e.g., '.txt').")
    args = parser.parse_args()

    delete_files_with_extension(args.directory, args.extension)