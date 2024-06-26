import MIPMLP
import pandas as pd

# try:
#     from samba import micro2matrix, build_SAMBA_distance_matrix, plot_umap
# except:
from src.samba import micro2matrix
from src.samba import build_SAMBA_distance_matrix, plot_umap

if __name__ == '__main__':
    CLASS = False
    # Load the raw data in the required format
    df = pd.read_csv("example_data/for_preprocess.csv")
    tag = pd.read_csv("example_data/tag.csv", index_col=0)

    # Apply the MIPMLP with the defaultive parameters
    processed = MIPMLP.preprocess(df)

    # micro2matrix there is an option to save the images in a prepared folder
    folder = "example_data/2D_images"
    array_of_imgs,bact_names, ordered_df = micro2matrix(processed, folder, save=False)

    # Calculate the distance matrix according to SAMBA
    DM = build_SAMBA_distance_matrix(folder,imgs=array_of_imgs,ordered_df=ordered_df,bact_names=bact_names,class_=CLASS)

    # Plot UMAP according to the distance matrix and some tag
    plot_umap(DM, tag, "example_data")


