# SAMBA (Smoothed imAge MicroBiome distAnce)

 This code is attached to the paper "".
 SAMBA is a novel microbial metric. SAMBA utilizes the iMic method to transform microbial data into images, incorporating phylogenetic structure and abundance similarity. 
This image-based representation enhances data visualization and analysis. Moreover, SAMBA employs a fast Fourier transform (FFT) with adjustable thresholding to smooth the images,
reducing noise and accentuating meaningful information. Various distance metrics, such as SAM and MSE, can be applied to the processed images.

## How to apply SAMBA
SAMBA's code is available at this [GitHub](https://github.com/oshritshtossel/SAMBA/new/master?readme=1) as well as [pypi]().

### SAMBA's GitHub
There is an example in example_use.py.
You should follow the following steps:
1. Load the raw ASVs table in the following format: the first column names "ID",
   each row represent a sample and each column represents an ASV. The last row 
   contains the taxonomy information, names "taxonomy".
   
    ```
    df = pd.read_csv("example_data/for_preprocess.csv")
    ```

   
2. Apply the MIPMLP with the defaultive parameters (see [MIPMLP](https://pypi.org/project/MIPMLP/) for more explanations).

    ```
    processed = MIPMLP.preprocess(df)
    ```
    
3. micro2matrix (translate microbiome into matrix according to [iMic](https://arxiv.org/abs/2205.06525), and save the images in a prepared folder
   
   ```
    folder = "example_data/2D_images"
    micro2matrix(processed, folder, save=True)
    ```
    
4. Calculate the distance matrix according to SAMBA
   One can choose the FFT cutoff (in range [0,1]), and the final metric (one of "sam","mse","d1","d2","d3").
   
   ```
    DM = build_SAMBA_distance_matrix(folder,cutoff=CUTOFF,metric=METRIC)
    ```
    
    