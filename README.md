# [CVPR 2026 Findings] GATE

Chaewon Lee,
JunHyeok Heo, 
and Chang-Su Kim

Official code for **"GATE: Gaussian-Attentive Transformer for Uncertainty-Aware Age Estimation"**[[paper]](https://openaccess.thecvf.com/content/CVPR2026F/papers/Lee_GATE_Gaussian-Attentive_Transformer_for_Uncertainty-Aware_Age_Estimation_CVPRF_2026_paper.pdf)

### Requirements
- PyTorch 2.3.0
- torchvision 0.18.0
- CUDA 11.8
- cuDNN 8.7
- python 3.9
  
### Installation
Create conda environment:
```bash
    $ conda env create -f environment.yaml -n GATE
    $ conda activate GATE
```
Download repository:
```bash
    $ git clone https://github.com/cwlee00/GATE.git
```
Download weights:

GATE model [Google Drive](https://drive.google.com/drive/folders/1k2u1rX6DbsD-oHcTk-un39AAZma6P9li?usp=drive_link)

### Evaluation
For evaluation, please download the datasets and models, and then configure the path in [config.yml](https://github.com/cwlee00/GATE/tree/main/config)

```
python test_GATE-s.py \
--checkpoint=./weights/GATE_models/GATE_s_clap.pth \
--dataset=clap
--fold=eval_on_test
```
### Train
For training, please download the datasets, and then configure the path in [config.yml](https://github.com/cwlee00/GATE/tree/main/config)
```
python train_GATE-s.py \
--dataset=clap \
--fold=eval_on_test \
```

### Citation
Please cite the following paper if you feel this repository useful.
```bibtex
    @InProceedings{Lee_2026_CVPR,
    author    = {Lee, Chaewon and Heo, JunHyeok and Kim, Chang-Su},
    title     = {GATE: Gaussian-Attentive Transformer for Uncertainty-Aware Age Estimation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
    month     = {June},
    year      = {2026},
    pages     = {8736-8745}
    }
```

