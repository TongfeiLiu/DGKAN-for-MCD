# [AAAI 2026] DGKAN: Dual-branch Graph Kolmogorov-Arnold Network for Unsupervised Multimodal Change Detection
![Paper](https://img.shields.io/badge/Paper-AAAI-blue)
![Python](https://img.shields.io/badge/Python-3.7%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red)

This repository provides the official implementation of the paper:

Tongfei Liu, Jianjian Xu, Tao Lei, Yingbo Wang, Xiaogang Du, Zhiyong Lv. [DGKAN: Dual-branch Graph Kolmogorov-Arnold Network for Unsupervised Multimodal Change Detection](https://ojs.aaai.org/index.php/AAAI/article/view/37665) [C]. **Proceedings of the AAAI Conference on Artificial Intelligence**, 2026, 40(9), 7278-7286. https://doi.org/10.1609/aaai.v40i9.37665

## 📖 Abstract
Multimodal change detection (MCD) has important applications in disaster assessment, but the nonlinear distortion of features and spatial misalignment caused by sensor imaging differences make it difficult to obtain changes through direct comparison. To overcome the above problems, this study aims to realize MCD by capturing the modality-independent structural commonality features between Multimodal Remote Sensing Images (MRSIs). To achieve this, we devise a basic Graph Kolmogorov-Arnold Network (GKAN) to excavate spatial structural relationships and cross-modal nonlinear mappings simultaneously. Based on this, we propose a Dual-branch GKAN (DGKAN) for unsupervised MCD, which can capture spatial-spectral structural commonality features and compare them directly to detect changes. Concretely, the GKAN is used within the DGKAN to build two autoencoders consisting of a Siamese encoder and two independent decoders to learn spatial-spectral structural commonality features through feature reconstruction. Besides, we introduce a Covariance Structural Commonality Loss (CSCL), which guides the network in extracting spatial-spectral structural commonality features between MRSIs by unsupervised constraints on the distributional consistency of cross-modal features. Experiments on several MCD datasets show that the proposed DGKAN can achieve convincing results, and ablation studies verify the effectiveness of the GKAN and CSCL.
The framework of the proposed DGKAN is presented as follows:
![Framework of our proposed DGKAN)](https://github.com/TongfeiLiu/DGKAN-for-MCD/blob/main/Figures/Freamwork.png)

## 📜 Citation
If you find our work useful for your research, please consider citing our paper:

```bibtex
@inproceedings{Liu2026DGKAN, 
    title={DGKAN: Dual-branch Graph Kolmogorov-Arnold Network for Unsupervised Multimodal Change Detection},
    author={Liu, Tongfei and Xu, Jianjian and Lei, Tao and Wang, Yingbo and Du, Xiaogang and Lv, Zhiyong},
    journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
    year={2026}, 
    volume={40}, 
    number={9}, 
    month={Mar.}, 
    pages={7278-7286}
}

@ARTICLE{TIP2025CFRL,
  author={Liu, Tongfei and Zhang, Mingyang and Gong, Maoguo and Zhang, Qingfu and Jiang, Fenlong and Zheng, Hanhong and Lu, Di},
  journal={IEEE Transactions on Image Processing}, 
  title={Commonality Feature Representation Learning for Unsupervised Multimodal Change Detection}, 
  year={2025},
  volume={34},
  number={},
  pages={1219-1233},
  doi={10.1109/TIP.2025.3539461}
}
```

## ❤️Acknowledgement
We are very grateful for the outstanding contributions of the publicly available MCD datasets and codes [1,2,3] and [4].

```
[1] https://sites.google.com/view/luppino/data.
[2] Professor Michele Volpi's webpage at https://sites.google.com/site/michelevolpiresearch/home.
[3] Professor Max Mignotte's webpage (http://www-labs.iro.umontreal.ca/~mignotte/).
[4] https://github.com/yulisun.
```

---

## 📮 Contact

If you have any problems (or cooperation) when running the code, please do not hesitate to contact us by:

- Jianjian Xu: xujianjian.Leo@sust.edu.cn
- Tongfei Liu: liutongfei_home@hotmail.com

---


