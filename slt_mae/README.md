Generally repo works the same as original one: https://github.com/MCG-NJU/VideoMAE

Added changes in sampling, augmentation and created soft targeting for finetuning. To repeat expirements download nessesary files (smoothing soft targets, sl datasets and original Kinetics-400 MAE pretrain model)


| Smoothing files |                                    |
|-----------------|------------------------------------|
| WLASL           | [Download](https://sc.link/BPmnu)  |
| AUTSL           | [Download](https://sc.link/vxiV9)  |
| GLS             | [Download](https://sc.link/rwHk0)  |
| SLOVO           | [Download](https://sc.link/UjohL)  |





## 🔒 License

The majority of this project is released under the CC-BY-NC 4.0 license as found in the [LICENSE](https://github.com/MCG-NJU/VideoMAE/blob/main/LICENSE) file. Portions of the project are available under separate license terms: [SlowFast](https://github.com/facebookresearch/SlowFast) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) are licensed under the Apache 2.0 license. [BEiT](https://github.com/microsoft/unilm/tree/master/beit) is licensed under the MIT license.

## ✏️ Citation

If you think this project is helpful, please feel free to leave a star⭐️ and cite our paper:

```
@inproceedings{tong2022videomae,
  title={Video{MAE}: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
  author={Zhan Tong and Yibing Song and Jue Wang and Limin Wang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}

@article{videomae,
  title={VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
  author={Tong, Zhan and Song, Yibing and Wang, Jue and Wang, Limin},
  journal={arXiv preprint arXiv:2203.12602},
  year={2022}
}
```
