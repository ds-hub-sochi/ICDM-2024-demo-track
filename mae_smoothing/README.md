# mae_smoothing

This repository is dedicated for (1) obtaining embeddings using MAE-model and (2) calculating label smoothing coefficients.

<br>

1. Embeddings will be saved to the folder containing annotations. For calculating and saving embeddings run

`python mae_model/fetch_embeds.py `

`--finetune` - path to finetune-MAE checkpoint

`--anno_path` - path to annotation in MAE format

`--anno_type` – `paths` if annotation contains full paths to videos; `names` if only names of videos are specified

`--data_path` – path to folder with videos. Only works when `--anno_type=names`

<br>

2. For calculating and saving label smoothing coefficients run

`python smoothing.py`

`--class_sim_method` -  technique from article (https://arxiv.org/pdf/2006.14028.pdf) (`random_sampling`) or class-averaged vectors (`mean_embeds`)

`--data_path` - path to folder containing `embeds.npy, labels.npy`

`--betta` - parameter from above-mentioned article

If `class_list.txt` exists in `data_path` then .txt file with closest classes will be saved there.
