# THY-Twin

<p align="center">
  <img src="assets/overview.jpg" width="80%">
</p>

<p align="center">
  <strong>THY-Twin: A Digital Navigation and Auditing Twin for Ultrasound-Guided Thyroid Ablation</strong>
</p>

This repository contains the example code and data for the above paper.

## Overview

THY-Twin is a digital twin framework for ultrasound-guided thyroid ablation, consisting of three main components:

- **Static&Treatment Twin**: Preoperative Static Twin: a short preoperative ultrasound sweep is used to reconstruct a patient-specific 3D thyroid model for individualized planning. Postoperative Treatment Twin: post-ablation ultrasound data are reconstructed and co-registered with the preoperative model to quantify ablation coverage and treatment completeness. 
- **Shadow Twin**: Intraoperative Shadow Twin: real-time 2D ultrasound frames are continuously aligned with the preoperative 3D twin to maintain spatial correspondence during needle guidance.
- **Needle Segmentation (needle_seg)**: Needle segmentation in thyroid ultrasound images

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+

### Setup

```bash
# Clone the repository
git clone git@github.com:cshennju/THY-Twin.git
cd THY-Twin

# Install dependencies in requirements.txt
```

---

## 1. Static&Treatment Twin

Neural Reconstruction Field (NeurRecField)

### Data Preparation

1. Download `static` folder from the Dataset Link below.
2. Place it in `data_train/` directory:

```
Static&Treatment Twin/
├── data_train/         # Put downloaded 'static' data here
│   ├── patient_001/
│   ├── patient_002/
│   └── ...
```

### Training

```bash
cd "Static&Treatment Twin"
python train_thy_ssim.py
```

### Key Parameters

- `--root_dir`: Path to training data

### Inference

Run inference on a single checkpoint to generate 3D volume:

```bash
python vis_thy.py --ckpt /path/to/your/model.ckpt --out_dir ./out_3d
```

**Output files:**
- `*_3d.npy` - NumPy array format
- `*_3d.nrrd` - NRRD format for 3D visualization

**Visualization with 3D Slicer:**
1. Download and install [3D Slicer](https://www.slicer.org/)
2. Open 3D Slicer → `File` → `Add Data`
3. Select the generated `*_3d.nrrd` file
4. Adjust window/level and rendering settings in the Volume Rendering module

---

## 2. Shadow Twin

Neural Registration Field (NeurRegField)

### Data Preparation

1. Download `shadow` folder from the Dataset Link below
2. Place test data in `test_thyroid/` directory
3. Put model checkpoint in `experiments/` directory:

```
Shadow Twin/
├── test_thyroid/        # Put downloaded 'shadow' data here
└── experiments/
    └── trained_model.pth   # Put model here
```

### Testing

```bash
cd "Shadow Twin"
python test_ours_thyroid.py \

    --dataset_root /path/to/test/data \

    --model /path/to/trained_model.pth \
```
---

## 3. Needle Segmentation (needle_seg)

Needle Segmentation Network in thyroid ultrasound images.

### Data Preparation

1. Download `needle` folder from the Dataset Link below
2. Place NPZ files in the `data_fine/` directory:

```
needle_seg/
├── data_fine/
│   └── test.npz         # Downloaded test data
└── model/
    └── best_model.pth   # Put model here
```

**NPZ file format:**
```python
# Data format: npz file containing
# - arr_0: images (N, H, W, 3)
# - arr_1: labels (N, H, W, 1)
```

### Configuration

Edit `config_test.py` to set paths:

```python
TEST_FILENAME = 'data/test.npz'
TEST_DIR = './results'
BEST_MODEL = 'model/best_model.pth'
```

### Testing

```bash
cd needle_seg
python Test.py
```
---

## Datasets and Models

Download example data and pre-trained models: **[NJU Box](https://box.nju.edu.cn/seafhttp/f/f71608a8dc2d40268a34/?op=view)**

```
THY-Twin_Data/
├── needle/              # Needle segmentation data
├── shadow/              # Shadow Twin registration data
├── static/              # Static&Treatment Twin data
└── model/               # All pre-trained models
    ├── needle
    └── shadow
```
---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{thy_twin_2026,
  title={THY-Twin: A Digital Navigation and Auditing Twin for Ultrasound-Guided Thyroid Ablation},
  author={Shen, Chengkang and Zhou, You and Zhu, Hao and Wang, Longqiang and Zhang, XueJing and Cao, Xun and Jiang, Ming and Jin, Yunjie and Lin, Yi and Ma, Zhan},
  journal={...},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or issues, please open an issue on [GitHub Issues](https://github.com/cshennju/THY-Twin/issues).

---

## Acknowledgments

This work builds upon the following open-source projects:

**[1]** Shen, C. et al. CardiacField: Computational echocardiography for automated heart function estimation using two-dimensional echocardiography probes. *European Heart Journal - Digital Health* 6, 137–146 (2025).

**[2]** Hui, X. et al. Ultrasound-guided needle tracking with deep learning: A novel approach with photoacoustic ground truth. *Photoacoustics* 34, 100575 (2023).

**[3]** Lei, L. et al. Epicardium prompt-guided real-time cardiac ultrasound frame-to-volume registration. In *MICCAI* 618–628 (2024).

We thank the authors for releasing their code. Please also consider citing their work.


