import os
import shutil

def copy_matching_images(folder_a, folder_b, folder_c):
    files_a = os.listdir(folder_a)
    files_c = os.listdir(folder_c)

    for file_a in files_a:
        if file_a in files_c:
            shutil.copy(os.path.join(folder_c, file_a), os.path.join(folder_b, file_a))
            print(f"Copied {file_a} from folder C to folder B")


def sort_and_rename_images(folder_path):
    files = os.listdir(folder_path)
    sorted_files = sorted(files)

    for i, file_name in enumerate(sorted_files):
        new_file_name = f"{i+1:04}.jpg"
        old_file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(folder_path, new_file_name)
        os.rename(old_file_path, new_file_path)
        print(f"Renamed {file_name} to {new_file_name}")

def copy_matching_images(folder_a, folder_b, folder_c):
    matching_prefixes = set()
    files_b = os.listdir(folder_b)
    for file_b in files_b:
        prefix = file_b[:4]
        matching_prefixes.add(prefix)

    files_a = os.listdir(folder_a)

    for file_a in files_a:
        prefix = file_a[:4]
        if prefix in matching_prefixes:
            shutil.copy(os.path.join(folder_a, file_a), os.path.join(folder_c, file_a))
            print(f"Copied {file_a} from folder A to folder C")

def remove_suffixes(folder_path):
    files = os.listdir(folder_path)

    for file_name in files:
        name, ext = os.path.splitext(file_name)
        new_name = name[:4]
        new_file_path = os.path.join(folder_path, new_name + ext)
        os.rename(os.path.join(folder_path, file_name), new_file_path)
        print(f"Renamed {file_name} to {new_name + ext}")