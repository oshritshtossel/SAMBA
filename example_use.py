import MIPMLP
import pandas as pd
from micro2matrix import micro2matrix
from SAMBA_metric import *

if __name__ == '__main__':
    # Load the raw data in the required format
    df = pd.read_csv("example_data/for_preprocess.csv")

    # Apply the MIPMLP with the defaultive parameters
    processed = MIPMLP.preprocess(df)

    # micro2matrix and saving the images in a prepared folder
    folder = "example_data/2D_images"
    micro2matrix(processed, folder, save=True)

    # Calculate the distance matrix according to SAMBA
    DM = build_SAMBA_distance_matrix(folder)
    c=0
