# UPT-Flow: Multi-Scale Transformer-Guided Normalizing Flow for Low-Light Image Enhancement

This is the official PyTorch code for the paper "[[UPT-Flow: A Multi-Scale Transformer-Guided Normalizing Flow for Low-Light Image Enhancement]](https://www.sciencedirect.com/science/article/abs/pii/S0031320324008276)". This paper was published in Pattern Recognition.

#### 🔥🔥🔥 

> **Abstract:** Low-light images often suffer from information loss and RGB value degradation due to extremely low or nonuniform lighting conditions. Many existing methods primarily focus on optimizing the appearance distance between the enhanced image and the ground truth, while neglecting the explicit modeling of information loss regions or incorrect information points in low-light images. To address this, this paper proposes an RGB value Unbalanced Points-guided multi-scale Transformer-based conditional normalizing Flow (UPT-Flow) for low-light image enhancement. We design an unbalanced point map prior based on the differences in the proportion of RGB values for each pixel in the image, which is used to modify traditional self-attention and mitigate the negative effects of areas with information distortion in the attention calculation. The Multi-Scale Transformer (MSFormer) is composed of several global-local transformer blocks, which encode rich global contextual information and local fine-grained details for conditional normalizing flow. In the invertible network of flow, we design cross-coupling conditional affine layers based on channel and spatial attention, enhancing the expressive power of a single flow step. Without bells and whistles, extensive experiments on low-light image enhancement, night traffic monitoring enhancement, low-light object detection, and nighttime image segmentation have demonstrated that our proposed method achieves state-of-the-art performance across a variety of real-world scenes. 


## 🔗 Contents

- [x] [Datasets](https://github.com/NJUPT-IPR-XuLintao/UPT-Flow/blob/main/README.md#-datasets)
- [ ] Training
- [x] [Testing](https://github.com/NJUPT-IPR-XuLintao/UPT-Flow/blob/main/README.md#-testing)
- [x] [Results](https://github.com/NJUPT-IPR-XuLintao/UPT-Flow/blob/main/README.md#-results)
- [x] [Acknowledgements](https://github.com/NJUPT-IPR-XuLintao/UPT-Flow/blob/main/README.md#-acknowledgements)

## 🔍 Datasets

1、LOLv2 (real & synthetic): Wenhan Yang, Haofeng Huang, Wenjing Wang, Shiqi Wang, and Jiaying Liu. "Sparse Gradient Regularized Deep Retinex Network for Robust Low-Light Image Enhancement", TIP, 2021. [[Baiduyun (extracted code: l9xm)]](https://pan.baidu.com/s/1U9ePTfeLlnEbr5dtI1tm5g) [Google Drive](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view?usp=sharing) 

2、 MIT-Adobe FiveK: Please refer to [MBPNet(TIP2023)](https://github.com/kbzhang0505/MBPNet)

3、SMID and SDSD (indoor & outdoor): Please refer to [SNRNet(CVPR2022)](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance)

## 🔍 Testing

**Pre-trained models for 6 datasets can be obtained from [Google Cloud Drive](https://drive.google.com/drive/folders/1kc1gYk3oTNkV-wZuqUjcZDNbZXqwq5Np?usp=sharing)**

1、Modify the paths to dataset and pre-trained mode. You need to modify the following path in the config files in `./confs`
```python
#### Test Settings
dataroot_unpaired: 
dataroot_GT: put high-light images  
dataroot_LR: put low-light images
model_path: put pre-trained model
```

2、Test the model

To test the model with paired data and obtain the evaluation results, e.g., PSNR, SSIM, and LPIPS. You need to specify the data path ```dataroot_LR```, ```dataroot_GT```, and model path ```model_path``` in the config file. Then run
```bash
test.py 
```

Note that for the LOL datasets, set the window_size to 5, and for the remaining datasets, set it to 8. Modify [Lowlight_Encoder.py](https://github.com/NJUPT-IPR-XuLintao/UPT-Flow/blob/77f391d6b5eb64b2d702c26a782fd70a71c75af4/UPT-Flow/models/modules/Lowlight_Encoder.py#L727)


## 🔍 Results

We achieved state-of-the-art performance on *low-light image enhancement*, *night traffic monitoring enhancement*, *low-light object detection* and *Nighttime semantic segmentation*. More results can be found in the paper.

<details>
<summary>Quantitative Comparison (click to expan)</summary>


  <p align="center">
  <img width="900" src="figs/table1.jpg">
	</p>

  <p align="center">
  <img width="500" src="figs/table2.jpg">

  </details>

<details>
<summary>Visual Comparison (click to expan)</summary>


  <p align="center">
  <img width="900" src="figs/fig1.jpg">
	</p>

  <p align="center">
  <img width="900" src="figs/fig2.jpg">
	</p>

  <p align="center">
  <img width="900" src="figs/fig3.jpg">
	</p>

   <p align="center">
  <img width="900" src="figs/fig4.jpg">
	</p>
 
  <p align="center">
  <img width="900" src="figs/fig5.jpg">
	</p>
 
  </details>

## Contact

If you have any questions, please feel free to contact me via email at lintao_xu@163.com.

## Citation
If you find our work useful for your research, please cite our paper
```
@article{xu2025upt,
  title={UPT-Flow: Multi-scale transformer-guided normalizing flow for low-light image enhancement},
  author={Xu, Lintao and Hu, Changhui and Hu, Yin and Jing, Xiaoyuan and Cai, Ziyun and Lu, Xiaobo},
  journal={Pattern Recognition},
  volume={158},
  pages={111076},
  year={2025},
  publisher={Elsevier}
}
```

## 💡 Acknowledgements
The codes are based on [LLFlow](https://github.com/wyf0912/LLFlow), [Restormer](https://github.com/swz30/Restormer), and [Uformer](https://github.com/ZhendongWang6/Uformer). Please also follow their licenses. Thanks for their awesome works.


