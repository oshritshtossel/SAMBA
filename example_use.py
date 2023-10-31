import MIPMLP
import pandas as pd

try:
    from samba import micro2matrix,build_SAMBA_distance_matrix,plot_umap
except:
    from src.samba import micro2matrix
    from src.samba import build_SAMBA_distance_matrix

if __name__ == '__main__':
    # Load the raw data in the required format
    df = pd.read_csv("example_data/for_preprocess.csv")
    tag = pd.read_csv("example_data/tag.csv",index_col=0)

    # Apply the MIPMLP with the defaultive parameters
    processed = MIPMLP.preprocess(df)

    # micro2matrix and saving the images in a prepared folder
    folder = "example_data/2D_images"
    micro2matrix(processed, folder, save=True)

    # Calculate the distance matrix according to SAMBA
    DM = build_SAMBA_distance_matrix(folder)

    DM.to_csv("result.csv")

    # Plot UMAP according to the distance matrix and some tag
    plot_umap(DM,tag,"example_data")
