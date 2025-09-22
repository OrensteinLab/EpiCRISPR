import os
base_folder = os.path.join("Models", "Exclude_Refined_TrueOT", "GRU-EMB", "Ensemble")
folders_to_clean = [
    base_folder,
    os.path.join(base_folder, "With_features_by_columns", "10_ensembels", "50_models", "Binary_epigenetics")
]

for folder in folders_to_clean:
    if not os.path.exists(folder):
        print(f"Skipping: {folder} (does not exist)")
        continue

    for filename in os.listdir(folder):
        if filename.endswith(".zip"):
            file_path = os.path.join(folder, filename)
            try:
                os.remove(file_path)
                print(f"Removed: {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {e}")

