import os
import shutil
import zipfile
def extract_zip(remove_zip = True):
    current_dir = os.getcwd()

    if os.path.basename(current_dir) == "Downloaded_models":
        source_dir = current_dir
    else:
        source_dir = os.path.join(current_dir, "Downloaded_models")
    # Move zip files into correct locations
    zip1_path = os.path.join(source_dir,"Only_sequence.zip")
    if os.path.exists(zip1_path):
        dest1 = os.path.join("Models","Exclude_Refined_TrueOT","GRU-EMB","Ensemble")

        os.makedirs(dest1, exist_ok=True)
        try:
            destination = shutil.move(zip1_path, dest1)
            print(f'moved Only_sequence.zip to: {destination}')
        except Exception as e:
            print(e)
            exit()
        print(f'Unziping {dest1}\n...')
        with zipfile.ZipFile(os.path.join(dest1, "Only_sequence.zip"), 'r') as zip_ref:
            zip_ref.extractall(dest1)
        if remove_zip:
            try:
                os.remove(dest1)
                print(f"Removed: {dest1}")
            except Exception as e:
                print(f"Error removing {dest1}: {e}")



    dest2 = os.path.join("Models","Exclude_Refined_TrueOT","GRU-EMB","Ensemble","With_features_by_columns","10_ensembels","50_models","Binary_epigenetics")
    os.makedirs(dest2, exist_ok=True)

    for fname in os.listdir(source_dir):
        if fname.endswith(".zip") and not fname == "Only_sequence.zip":
            src_path = os.path.join(source_dir, fname)
            dest_path = os.path.join(dest2, fname)
            try:
                destination = shutil.move(src_path, dest_path)
                print(f'moved {fname} to: {destination}')
            except Exception as e:
                print(e)
                exit()
            print(f'Unziping {dest_path}\n...')
            with zipfile.ZipFile(dest_path, 'r') as zip_ref:
                zip_ref.extractall(dest2)
            if remove_zip:
                try:
                    os.remove(destination)
                    print(f"Removed: {destination}")
                except Exception as e:
                    print(f"Error removing {destination}: {e}")

if __name__ == "__main__":
    extract_zip()
        