from src.samba import build_SAMBA_distance_matrix, plot_umap, apply_meta_analysis

if __name__ == '__main__':
    # Set a cutoff for the smoothing
    CUTOFF = 0.8
    # List of datasets names
    list_data_names = ["D1", "D2", "D3"]

    # Folder where the datasets are saved
    folder = "example_data_meta"

    apply_meta_analysis(folder, list_data_names, CUTOFF)
