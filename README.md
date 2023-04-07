# AICT: AN ADAPTIVE IMAGE COMPRESSION TRANSFORMER
Official TensorFlow implementation of [AICT: AN ADAPTIVE IMAGE COMPRESSION TRANSFORMER](https://arxiv.org/).

* [AICT](#aict)
  * [Tags](#tags)
  * [Overall AICT Framework](#overall-aict-framework)
  * [Disclaimer](#disclaimer)
  * [Documentation](#documentation)
  * [Requirements](#requirements)
  * [Folder Structure](#folder-structure)
  * [CLI-Usage](#cli-usage)
  * [Rate-Distortion Coding Performance](#rate-distortion-coding-performance)
  * [Citation](#citation)
  * [License](#license)
    
<!-- /code_chunk_output -->

## Tags
<code>Swin Transformer</code> <code>ConvNeXt</code> <code>Adaptive Resolution</code> <code>Neural Codecs</code> <code>Image Compression</code> <code>TensorFlow</code>

## Overall AICT Framework
![AICT framework](https://github.com/ahmedgh970/AICT/blob/main/figures/AICT.jpg)

## Disclaimer
Please do not hesitate to open an issue to inform of any problem you may find within this repository. Also, you can [email me](mailto:ahmed.ghorbel888@gmail.com?subject=[GitHub]) for questions or comments. 

## Documentation
* This repository is built upon the official TensorFlow implementation of [Channel-Wise Autoregressive Entropy Models for Learned Image Compression](https://ieeexplore.ieee.org/abstract/document/9190935). This baseline is referred to as [Conv-ChARM](https://github.com/ahmedgh970/AICT/blob/main/conv-charm.py).
* We provide lightweight versions of the models by removing the latent residual prediction (LRP) transform and slicing latent means and scales, as done in the [Tensorflow reimplementation of SwinT-ChARM](https://github.com/Nikolai10/SwinT-ChARM) from the original paper [TRANSFORMER-BASED TRANSFORM CODING](https://openreview.net/pdf?id=IDwN6xjHnK8).
* Refer to the [ResizeCompression github repo](https://github.com/treammm/ResizeCompression), as the official implementation of the paper [Estimating the Resize Parameter in End-to-end Learned Image Compression](https://arxiv.org/abs/2204.12022).
* Refer to the [TensorFlow Compression (TFC) library](https://github.com/tensorflow/compression) to build your own ML models with end-to-end optimized data compression built in.
* Refer to the [API documentation](https://www.tensorflow.org/api_docs/python/tfc) for a complete classes and functions description of the TensorFlow Compression (TFC) library.
 

## Requirements
<code>Python >= 3.6</code> <code>tensorflow_compression</code> <code>tensorflow_datasets</code> <code>tensorflow_addons</code> <code>einops</code> 

All packages used in this repository are listed in [requirements.txt](https://github.com/ahmedgh970/AICT/blob/main/requirements.txt).
To install those, run:
```
pip install -r requirements.txt
```

## Folder Structure
``` 
AICT-main
│
├── conv-charm.py                 # Conv-ChARM Model
├── swint-charm.py                # SwinT-ChARM Model
├── ict.py                        # ICT Model
├── aict.py                       # AICT Model
│
├── layers/
│   └── convNext.py/              # ConvNeXt block layers
│   └── swins/                    # Swin Transformer block layers
│   └── scaleAdaptation.py/       # Scale Adaptation module  layers
│
├── utils.py                      # Utility functions
├── config.py                     # Architecture configurations
├── requirements.txt              # Requirements
└── figures/                      # Documentation figures
```

## CLI Usage
Every model can be trained and evaluated individually using:
```
python aict.py train
```
```
python aict.py evaluate --test_dir [-I] --tfci_output_dir [-O] --png_output_dir [-P] --results_file [-R]
```

## Rate-Distortion coding performance
![Rate-Distortion coding performance on KODAK](https://github.com/ahmedgh970/AICT/blob/main/figures/rd_performance.png)

Table 1. BD-rate↓ (PSNR) performance of BPG (4:4:4), Conv-ChARM, SwinT-ChARM, ICT, and AICT compared to the VTM-18.0 for the four considered datasets.

| Image Codec | Kodak | Tecnick | JPEG-AI | CLIC21 | Average |
| --- | --- | --- | --- | --- | --- |
| BPG444 | 22.28% | 28.02% | 28.37% | 28.02% | 26.67% |
| Conv-ChARM | 2.58% | 3.72% | 9.66% | 2.14% | 4.53% |
| SwinT-ChARM | -1.92% | -2.50% | 2.91% | -3.22% | -1.18% |
| ICT (ours) | -5.10% | -5.91% | -1.14% | -6.44% | -4.65% |
| AICT (ours) | -5.09% | -5.99% | -2.03% | -7.33% | -5.11% |


## Citation
If you use this library for research purposes, please cite:
```
@inproceedings{ghorbel2023aict,
  title={AICT: AN ADAPTIVE IMAGE COMPRESSION TRANSFORMER},
  author={Ghorbel, Ahmed and Hamidouche, Wassim and Luce, Morin},
  booktitle={},
  year={2023}
}
```

## License
This project is licensed under the MIT License. See LICENSE for more details
