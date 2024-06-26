# GIMIC (Smoothed Graph IMages of the MICrobiome)

 This code is attached to the paper "GIMIC - Smoothed Graph-Image representation of Microbiome Samples induce an optimal distance".
 We introduce GIMIC (Smoothed Graph IMages of the MICrobiome) as the smoothed tree-based images. GIMIC highlights consistent patterns across different cohorts, and as a metric (the difference between two GIMIC images), it outperforms existing metrics in a wide array of tasks over many 16S and WGS datasets, both within and between cohorts. Technically, GIMIC employs a fast Fourier transform (FFT) with adjustable thresholding to smooth the images,
reducing noise and accentuating meaningful information. Various distance metrics, such as SAM and MSE, can be applied to the processed images.

## How to apply GIMIC
GIMIC's code is available at this [GitHub](https://github.com/oshritshtossel/SAMBA/new/master?readme=1) as well as [PyPi](https://pypi.org/project/samba-metric/).

### GIMIC's GitHub

#### GIMIC as a metric
There is an example in example_use.py.
You should follow the following steps:
1. Load the raw ASVs table in the following format: the first column is named "ID",
   each row represents a sample and each column represents an ASV. The last row 
   contains the taxonomy information, named "taxonomy".
   
    ```
    df = pd.read_csv("example_data/for_preprocess.csv")
    ```

   
2. Apply the MIPMLP with the defaulting parameters (see [MIPMLP](https://pypi.org/project/MIPMLP/) for more explanations).

    ```
    processed = MIPMLP.preprocess(df)
    ```
    
3. micro2matrix (translate microbiome into matrix according to [iMic](https://doi.org/10.1080/19490976.2023.2224474), and save the images in a prepared folder
   
   ```
    folder = "example_data/2D_images"
    micro2matrix(processed, folder, save=True)
    ```
    
4. Calculate the distance matrix according to GIMIC
   One can choose the FFT cutoff (in the range of [0,1]), and the final metric (one of "sam","mse","d1","d2","d3").
   
   ```
    DM = build_SAMBA_distance_matrix(folder,cutoff=CUTOFF,metric=METRIC)
    ```
5. If a tag table is available. One can load the tag file and visualize the data according to the SAMBA metric by the plot_umap function.
   "example_data" is the folder path for saving.
      ```
     tag = pd.read_csv("example_data/tag.csv",index_col=0)
     plot_umap(DM,tag,"example_data")
    ```

#### GIMIC as a cross-cohort visualization tool
There is an example in example_use_meta_analysis.py.\
You should follow the following steps:
1. Set a cutoff for the smoothing. (A float between 0 to 1, when 1 is no smooting)
   ```
    CUTOFF = 0.8
    ```
2. Provide a list of datasets names
   
   **NOTE: in each data folder there should be the following csvs 'for_preprocess.csv' and 'tag.csv', in the format of the files in example_data_meta folder.**
    ```
    list_data_names = ["D1","D2","D3"]
    ```
4. Provide a folder where the datasets are saved
   ```
    folder = "example_data_meta"
   ```
5. Call the 'apply_meta_analysis' function
   ```
   apply_meta_analysis(folder,list_data_names,CUTOFF)
   ```
   This function plots a convenient meta-analysis visualization and saves it.
### GIMIC's PyPi
1. Install the GIMIC package

```
pip install samba-metric
```
2. Apply the MIPMLP with the default parameters
   
```
processed = MIPMLP.preprocess(df)
```
or -  load a MIPMLP processed data directly.

```
processed = pd.read_csv("example_data.csv",index_col=0)

```

4. Apply GIMIC metric on a MIPMLP processed data:
```
from samba import *
CLASS = False
    # Load the raw data in the required format
    df = pd.read_csv("example_data/for_preprocess.csv")
    # If tag is available
    tag = pd.read_csv("example_data/tag.csv", index_col=0)

    # Apply the MIPMLP with the defaultive parameters
    processed = MIPMLP.preprocess(df)

    # micro2matrix there is an option to save the images in a prepared folder, the default is no saving at all
    folder = "example_data/2D_images"
    array_of_imgs,bact_names, ordered_df = micro2matrix(processed, folder, save=False)

    # Calculate the distance matrix according to GIMIC
    DM = build_SAMBA_distance_matrix(folder,imgs=array_of_imgs,ordered_df=ordered_df,bact_names=bact_names,class_=CLASS)

    # Plot UMAP according to the distance matrix and some tag (NOTE: only when tag is available)
    plot_umap(DM, tag, "example_data")

```
   ![Output](umap_plot.png)
   
5. Apply GIMIC cross-cohort visualization on several cohorts from the same phenotype.
   **NOTE: in each data folder there should be the following csvs 'for_preprocess.csv' and 'tag.csv', in the format of the files in example_data_meta folder.**
   ```
    # Set a cutoff for the smoothing
    CUTOFF = 0.8
    # List of datasets names
    list_data_names = ["D1","D2","D3"]

    # Folder where the datasets are saved
    folder = "example_data_meta"

    apply_meta_analysis(folder,list_data_names,CUTOFF)
   ```
![Alt text](circle_Example meta analysis.png)
    
