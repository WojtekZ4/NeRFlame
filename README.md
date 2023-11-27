# NeRFlame

Traditional 3D face models are based on mesh representations with texture. One of the most important models is FLAME (Faces Learned with an Articulated Model and Expressions), which produces meshes of human faces that are fully controllable. 
Unfortunately, such models have problems with capturing geometric and appearance details. 
In contrast to mesh representation, the neural radiance field  (NeRF) produces extremely sharp renders. However, implicit methods are hard to animate and do not generalize well to unseen expressions. It is not trivial to effectively control NeRF models to obtain face manipulation. 

<img src="https://github.com/WojtekZ4/NeRFlame/blob/main/gifs/turn1.gif" width="300" height="300">

The present paper proposes a novel approach, named NeRFlame, which combines the strengths of both NeRF and FLAME methods. Our method enables high-quality rendering capabilities of NeRF while also offering complete control over the visual appearance, similar to FLAME.

<img src="https://github.com/WojtekZ4/NeRFlame/blob/main/gifs/turn2.gif" width="300" height="300">


In contrast to traditional NeRF-based structures that use neural networks for RGB color and volume density modeling, our approach utilizes the FLAME mesh as a distinct density volume. Consequently, color values exist only in the vicinity of the FLAME mesh. This FLAME framework is seamlessly incorporated into the NeRF architecture for predicting RGB colors, enabling our model to explicitly represent volume density and implicitly capture RGB colors.

## Installation

Tested on Python 3.8.

```
git clone https://github.com/WojtekZ4/NeRFlame.git
cd NeRFlame
pip install -r requirements.txt
```

Download and install Pytorch3d as described [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

Download FLAME models and landmark embedings and place them inside `FLAME` folder, as shown [here](https://github.com/soubhiksanyal/FLAME_PyTorch).


## How To Run?

### Quick Start


To train a face:
```
python scripts/experiments/run_nerf.py --hparams_path scripts/configs/faces_M1000_N.yaml --model face_M1000N_flame_nerf
```

---

### More Datasets
To play with other faces presented in the paper, download the data [here](https://drive.google.com/drive/folders/1znso9vWtrkYqdMrZU1U0-X2pHJcpTXpe?usp=share_link).


---

To train NeRFlame on different datasets: 

```
python scripts/experiments/run_nerf.py --hparams_path scripts/configs/{DATASET_CONFIG}.yaml --model {DATASET_MODEL}
```

replace `{DATASET_CONFIG}` with `faces_M1000_N` | `faces_f1036_A` | `faces_m1011_D`| etc. 
and `{DATASET_MODEL}` with `face_M1000N_flame_nerf` | `face_f1036A_flame_nerf` | `face_m1011D_flame_nerf`| etc.

---

To test NeRFlame trained on different datasets: 

```
python scripts/experiments/run_nerf.py --hparams_path scripts/configs/{DATASET_CONFIG}.yaml --model {DATASET_MODEL} --render_only
```


## Citation

Thanks to the NeRF authors:
```
@misc{mildenhall2020nerf,
    title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
    author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
    year={2020},
    eprint={2003.08934},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

As well as authors of this pytorch implementation:
```
@misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
```

Also thanks to the FLAME authors:
```
@article{FLAME:SiggraphAsia2017,
  title = {Learning a model of facial shape and expression from {4D} scans},
  author = {Li, Tianye and Bolkart, Timo and Black, Michael. J. and Li, Hao and Romero, Javier},
  journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
  volume = {36},
  number = {6},
  year = {2017},
  url = {https://doi.org/10.1145/3130800.3130813}
}
```

And for the pose dependent dynamic landmarks:
```
@inproceedings{RingNet:CVPR:2019,
title = {Learning to Regress 3D Face Shape and Expression from an Image without 3D Supervision},
author = {Sanyal, Soubhik and Bolkart, Timo and Feng, Haiwen and Black, Michael},
booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
month = jun,
year = {2019},
month_numeric = {6}
}
```







