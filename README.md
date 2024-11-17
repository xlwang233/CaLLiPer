# CaLLiPer - Contrastive Language-Location Pre-training
Welcome! This is the repository for *CaLLiPer* model presented in our paper Multimodal Contrastive Learning of Urban Space Representations from POI Data.

## :star: Highlights

- Simple and effective representation learning for urban spaces using POI data.
- The first multimodal contrastive learning model to align spatial and semantic information.
- Improved conceptualisation of urban space representations through location encoding.
- Enhanced modelling of POI semantics by pre-trained text encoders.
- State-of-the-art performance and interpretability.


## Requirements
- pytorch >= 2.2.0
- transformer >= 4.43.4
- pytorch-lightning == 2.3.3
- tensorboard >= 1.14.0
- scikit-learn 
...

## Data

### POI data
The use of [OS POI data](https://www.ordnancesurvey.co.uk/products/points-of-interest) requires an Educational Licence. For demonstration purpose, we provide a sample of the POI data used in our study in `/data/london_poi_202203/sample_poi.csv`. You can easily construct your own coordinate-text pairs as training data similar to this.

### Land use data
The land use data used in our experiment was derived from [Verisk data](https://www.verisk.com/en-gb/solutions/land-buildings-data/), obtained through [Digimap](https://digimap.edina.ac.uk/roam/map/verisk). Please find it in `/data/landuse_classification/sampled_points_landuse.geojson`

### Socioeconomic status data
The NS-SeC dataset used in our experiment was obtained from [ONS](https://www.ons.gov.uk/). The original data is stored in `data/socioeconomic/lon_lsoa_ns-sec.csv`. The preprocessing code of this dataset can be found in the evaluation notebook - `sdm.ipynb`


## Training
Specifiy the hyperparameters in `configs/default.yaml` and run the following command for model training.

```bash
python main.py --cofing configs/default.yaml
```
The output will be saved in `logs/{exp_name}/`, including Tensorboard events and model checkpoints. 

## Testing
After finishing the pre-training, one can evaluate the resulting CaLLiPer model checkpoint on two downstream tasks - LUC (`luc.ipynb`) and SDM (`sdm.ipynb`). For ease of replicating the results presented in our paper, we provide the pre-trained checkpoint of *CaLLiPer-SenTrans* - shared through [Google Drive](https://drive.google.com/drive/folders/1MIou67A5rSIHaGVB5DN2Sce3iHO7t_2W?usp=drive_link). Please download it and put it in `checkpoints/`.

See the two Jupyter Notebooks, `luc.ipynb` and `sdm.ipynb`, for the complete downstream model training process. The resulting downstream models will be saved in `downstream_res/...`. If you do not feel like training downstream models yourself, we have also provided them - you probably have already found them in `downstream_res/...`



Note that due to the size of *CaLLiPer-Llama*, we have not provided its checkpoint in this repo at the moment but will consider sharing it if requested.

## TODOs
We plan to add more stuff in the future:

:black_square_button: Baseline model (implementation or checkpoints)
:black_square_button: Visualisation code

## Acknowledgements

The implementation of various location encoding methods is based on [Space2Vec](https://github.com/gengchenmai/space2vec) and [Spherical Harmonics and Sinusoidal Representation Networks](https://github.com/MarcCoru/locationencoder).

We appreciate their inspiring works.

## Citation

```bibtex
@article{wang2024multimodal,
  title={Multimodal Contrastive Learning of Urban Space Representations from POI Data},
  author={Wang, Xinglei and Cheng, Tao and Law, Stephen and Zeng, Zichao and Yin, Lu and Liu, Junyuan},
  journal={arXiv preprint arXiv:2411.06229},
  year={2024}
}
```
