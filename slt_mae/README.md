Generally repo works the same as original one: https://github.com/MCG-NJU/VideoMAE.  
Our goal is to get a good pre-trained (Foundation) model, which can be easily fine-tuned for 
various sign languages. Therefore, the experiment consists of two stages. At the first stage, 
we train a pre-trained model (see [scripts/Pretrain](scripts/Pretrain)). During the second stage, 
we perform fine-tuning on the [WLASL](scripts/WLASL), [AUTSL](scripts/AUTSL), [GSL](scripts/GSL), 
and [Slovo](scripts/SLOVO) datasets to evaluate the model’s 
effectiveness in processing downstream tasks. To train the pre-trained model, we utilized the YouTube-ASL dataset and trained a ViT 
Large model on it adhering to the VideoMAE approach. Given that many videos from the YouTube-ASL 
dataset are very long, we randomly sampled up to 100 fragments containing 100 frames 
from each.

Changes in the original repository affect sampling, augmentation, and fine-tuning procedures. 

To repeat experiments download necessary files (smoothing soft targets, sign languages datasets 
and original Kinetics-400 MAE pre-train model)


| Sign language | Smoothing files                   |
|---------------|-----------------------------------|
| WLASL         | [Download](https://sc.link/BPmnu) |
| AUTSL         | [Download](https://sc.link/vxiV9) |
| GLS           | [Download](https://sc.link/rwHk0) |
| SLOVO         | [Download](https://sc.link/UjohL) |

