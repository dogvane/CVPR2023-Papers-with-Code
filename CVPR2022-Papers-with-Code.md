# CVPR 2022 论文和开源项目合集(Papers with Code)

[CVPR 2022](https://cvpr2022.thecvf.com/) 论文和开源项目合集(papers with code)！

CVPR 2022 收录列表ID：https://drive.google.com/file/d/15JFhfPboKdUcIH9LdbCMUFmGq_JhaxhC/view

> 注1：欢迎各位大佬提交issue，分享CVPR 2022论文和开源项目！
>
> 注2：关于往年CV顶会论文以及其他优质CV论文和大盘点，详见： https://github.com/amusi/daily-paper-computer-vision
>
> - [CVPR 2019](CVPR2019-Papers-with-Code.md)
> - [CVPR 2020](CVPR2020-Papers-with-Code.md)
> - [CVPR 2021](CVPR2021-Papers-with-Code.md)

如果你想了解最新最优质的的CV论文、开源项目和学习资料，欢迎扫码加入【CVer学术交流群】！互相学习，一起进步~ 

![](CVer学术交流群.png)

## 【CVPR 2022 论文开源目录】

- [Backbone](#Backbone)
- [CLIP](#CLIP)
- [GAN](#GAN)
- [GNN](#GNN)
- [MLP](#MLP)
- [NAS](#NAS)
- [OCR](#OCR)
- [NeRF](#NeRF)
- [3D Face](#3D Face)
- [长尾分布(Long-Tail)](#Long-Tail)
- [Visual Transformer](#Visual-Transformer)
- [视觉和语言(Vision-Language)](#VL)
- [自监督学习(Self-supervised Learning)](#SSL)
- [数据增强(Data Augmentation)](#DA)
- [知识蒸馏(Knowledge Distillation)](#KD)
- [目标检测(Object Detection)](#Object-Detection)
- [目标跟踪(Visual Tracking)](#VT)
- [语义分割(Semantic Segmentation)](#Semantic-Segmentation)
- [实例分割(Instance Segmentation)](#Instance-Segmentation)
- [全景分割(Panoptic Segmentation)](#Panoptic-Segmentation)
- [小样本分类(Few-Shot Classification)](#FFC)
- [小样本分割(Few-Shot Segmentation)](#FFS)
- [图像抠图(Image Matting)](#Matting)
- [视频理解(Video Understanding)](#VU)
- [图像编辑(Image Editing)](#Image-Editing)
- [Low-level Vision](#LLV)
- [超分辨率(Super-Resolution)](#Super-Resolution)
- [去模糊(Deblur)](#Deblur)
- [3D点云(3D Point Cloud)](#3D-Point-Cloud)
- [3D目标检测(3D Object Detection)](#3D-Object-Detection)
- [3D语义分割(3D Semantic Segmentation)](#3DSS)
- [3D目标跟踪(3D Object Tracking)](#3D-Object-Tracking)
- [3D人体姿态估计(3D Human Pose Estimation)](#3D-Human-Pose-Estimation)
- [3D语义场景补全(3D Semantic Scene Completion)](#3DSSC)
- [3D重建(3D Reconstruction)](#3D-R)
- [行人重识别(Person Re-identification)](#ReID)
- [伪装物体检测(Camouflaged Object Detection)](#COD)
- [深度估计(Depth Estimation)](#Depth-Estimation)
- [立体匹配(Stereo Matching)](#Stereo-Matching)
- [特征匹配(Feature Matching)](#FM)
- [车道线检测(Lane Detection)](#Lane-Detection)
- [光流估计(Optical Flow Estimation)](#Optical-Flow-Estimation)
- [图像修复(Image Inpainting)](#Image-Inpainting)
- [图像检索(Image Retrieval)](#Image-Retrieval)
- [人脸识别(Face Recognition)](#Face-Recognition)
- [人群计数(Crowd Counting)](#Crowd-Counting)
- [医学图像(Medical Image)](#Medical-Image)
- [视频生成(Video Generation)](#Video Generation)
- [场景图生成(Scene Graph Generation)](#Scene-Graph-Generation)
- [参考视频目标分割(Referring Video Object Segmentation)](#R-VOS)
- [步态识别(Gait Recognition)](#GR)
- [风格迁移(Style Transfer)](#ST)
- [异常检测(Anomaly Detection](#AD)
- [对抗样本(Adversarial Examples)](#AE)
- [弱监督物体检测(Weakly Supervised Object Localization)](#WSOL)
- [雷达目标检测(Radar Object Detection)](#ROD)
- [高光谱图像重建(Hyperspectral Image Reconstruction)](#HSI)
- [图像拼接(Image Stitching)](#Image-Stitching)
- [水印(Watermarking)](#Watermarking)
- [Action Counting](#AC)
- [Grounded Situation Recognition](#GSR)
- [Zero-shot Learning](#ZSL)
- [DeepFakes](#DeepFakes)
- [数据集(Datasets)](#Datasets)
- [新任务(New Tasks)](#New-Tasks)
- [其他(Others)](#Others)

<a name="Backbone"></a>

# Backbone

**A ConvNet for the 2020s**
**2020年代的一台卷积神经网络**

- Paper: https://arxiv.org/abs/2201.03545
- Code: https://github.com/facebookresearch/ConvNeXt
- 中文解读：https://mp.weixin.qq.com/s/Xg5wPYExnvTqRo6s5-2cAw

**Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs**
**扩大你的核尺寸到31x31：重新审视CNN中的大型核设计**

- Paper: https://arxiv.org/abs/2203.06717

- Code: https://github.com/megvii-research/RepLKNet
- Code2: https://github.com/DingXiaoH/RepLKNet-pytorch

- 中文解读：https://mp.weixin.qq.com/s/_qXyIQut-JRW6VvsjaQlFg

**MPViT : Multi-Path Vision Transformer for Dense Prediction**
**MPViT：多路径视觉变分器用于稠密预测**

- Paper: https://arxiv.org/abs/2112.11010
- Code: https://github.com/youngwanLEE/MPViT
- 中文解读: https://mp.weixin.qq.com/s/Q9-crEOz5IYzZaNoq8oXfg

**Mobile-Former: Bridging MobileNet and Transformer**
**移动-Former：将MobileNet和Transformer相结合**

- Paper: https://arxiv.org/abs/2108.05895
- Code: None
- 中文解读：https://mp.weixin.qq.com/s/yo5KmB2Y7t2R4jiOKI87HQ

**MetaFormer is Actually What You Need for Vision**
**元学习实际上就是你所需要的视觉解决方案**

- Paper: https://arxiv.org/abs/2111.11418
- Code: https://github.com/sail-sg/poolformer

**Shunted Self-Attention via Multi-Scale Token Aggregation**
**通过多尺度令牌聚合的Shifted Self-Attention**

-  Paper(Oral): https://arxiv.org/abs/2111.15193
- Code: https://github.com/OliverRensu/Shunted-Transformer

**TVConv: Efficient Translation Variant Convolution for Layout-aware Visual Processing**
**TVConv：用于布局感知视觉处理的efficient翻译变体卷积**

- Paper: http://arxiv.org/abs/2203.10489
- Code: https://github.com/JierunChen/TVConv

**Learned Queries for Efficient Local Attention**
**学习有效的局部注意力**

- Paper(Oral): https://arxiv.org/abs/2112.11435
- Code: https://github.com/moabarar/qna

**RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality**
**重参数化局部感知神经网络：层次视觉 MLP**

- Paper: https://arxiv.org/abs/2112.11081
- Code: https://github.com/DingXiaoH/RepMLP

<a name="CLIP"></a>

# CLIP

**HairCLIP: Design Your Hair by Text and Reference Image**
**发带：通过文本和参考图片设计你的发型**

- Paper: https://arxiv.org/abs/2112.05142

- Code: https://github.com/wty-ustc/HairCLIP

**PointCLIP: Point Cloud Understanding by CLIP**
**PointCLIP：通过CLIP点云理解的原理**

- Paper: https://arxiv.org/abs/2112.02413
- Code: https://github.com/ZrrSkywalker/PointCLIP

**Blended Diffusion for Text-driven Editing of Natural Images**
**混合扩散用于自然图像的文本驱动编辑**

- Paper: https://arxiv.org/abs/2111.14818

- Code: https://github.com/omriav/blended-diffusion

<a name="GAN"></a>

# GAN

**SemanticStyleGAN: Learning Compositional Generative Priors for Controllable Image Synthesis and Editing**
**语义风格GAN：控制图像合成和编辑的可控图像生成**

- Homepage: https://semanticstylegan.github.io/

- Paper: https://arxiv.org/abs/2112.02236
- Demo: https://semanticstylegan.github.io/videos/demo.mp4

**Style Transformer for Image Inversion and Editing**
**图像反转和编辑风格转换器**

- Paper: https://arxiv.org/abs/2203.07932
- Code: https://github.com/sapphire497/style-transformer

**Unsupervised Image-to-Image Translation with Generative Prior**
**无监督图像到图像翻译生成器优先**

- Homepage: https://www.mmlab-ntu.com/project/gpunit/
- Paper: https://arxiv.org/abs/2204.03641
- Code: https://github.com/williamyang1991/GP-UNIT

**StyleGAN-V: A Continuous Video Generator with the Price, Image Quality and Perks of StyleGAN2**
**StyleGAN-V：风格生成器2：具有风格2价格、图像质量和福利的连续视频生成器**

- Homepage: https://universome.github.io/stylegan-v
- Paper: https://arxiv.org/abs/2112.14683
- Code: https://github.com/universome/stylegan-v

**OSSGAN: Open-set Semi-supervised Image Generation**
**OSSGAN: 开放集半监督图像生成**

- Paper: https://arxiv.org/abs/2204.14249
- Code: https://github.com/raven38/OSSGAN

**Neural Texture Extraction and Distribution for Controllable Person Image Synthesis**
**神经纹理提取和分布对于控制人脸图像合成的可控性**

- Paper: https://arxiv.org/abs/2204.06160
- Code: https://github.com/RenYurui/Neural-Texture-Extraction-Distribution

<a name="GNN"></a>

# GNN

**OrphicX: A Causality-Inspired Latent Variable Model for Interpreting Graph Neural Networks**
**OrphicX：解释图神经网络的隐变量因果模型**

- Paper: https://wanyu-lin.github.io/assets/publications/wanyu-cvpr2022.pdf 
- Code: https://github.com/WanyuGroup/CVPR2022-OrphicX

<a name="MLP"></a>

# MLP

**RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality**
**RepMLPNet：具有重新参数化局部性的层次视觉 MLP**

- Paper: https://arxiv.org/abs/2112.11081
- Code: https://github.com/DingXiaoH/RepMLP

<a name="NAS"></a>

# NAS

**β-DARTS: Beta-Decay Regularization for Differentiable Architecture Search**
**β-Darts：为可微架构搜索的贝塔衰减正则化**

- Paper: https://arxiv.org/abs/2203.01665
- Code: https://github.com/Sunshine-Ye/Beta-DARTS

**ISNAS-DIP: Image-Specific Neural Architecture Search for Deep Image Prior**
**ISNAS-DIP：针对深度图像优先的图像特定神经架构搜索**

- Paper: https://arxiv.org/abs/2111.15362
- Code: None

<a name="OCR"></a>

# OCR

**SwinTextSpotter: Scene Text Spotting via Better Synergy between Text Detection and Text Recognition**
**SwinTextSpotter：通过文本检测和文本识别实现场景文本检测**

- Paper: https://arxiv.org/abs/2203.10209

- Code: https://github.com/mxin262/SwinTextSpotter

<a name="NeRF"></a>

# NeRF

**Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields**
**Mip-NeRF 360：无尽抗锯齿神经辐射场**

- Homepage: https://jonbarron.info/mipnerf360/
- Paper: https://arxiv.org/abs/2111.12077

- Demo: https://youtu.be/YStDS2-Ln1s

**Point-NeRF: Point-based Neural Radiance Fields**
**点NeRF: 基于点的神经辐射场**

- Homepage: https://xharlie.github.io/projects/project_sites/pointnerf/
- Paper: https://arxiv.org/abs/2201.08845
- Code: https://github.com/Xharlie/point-nerf

**NeRF in the Dark: High Dynamic Range View Synthesis from Noisy Raw Images**
**NeRF in the Dark: 从杂乱原始图像中进行高动态范围视图合成**

- Paper: https://arxiv.org/abs/2111.13679
- Homepage: https://bmild.github.io/rawnerf/
- Demo: https://www.youtube.com/watch?v=JtBS4KBcKVc

**Urban Radiance Fields**
**城市光晕场**

- Homepage: https://urban-radiance-fields.github.io/

- Paper: https://arxiv.org/abs/2111.14643
- Demo: https://youtu.be/qGlq5DZT6uc

**Pix2NeRF: Unsupervised Conditional π-GAN for Single Image to Neural Radiance Fields Translation**
**Pix2NeRF: 无监督条件π-GAN用于单张图像到神经辐射场**

- Paper: https://arxiv.org/abs/2202.13162
- Code: https://github.com/HexagonPrime/Pix2NeRF

**HumanNeRF: Free-viewpoint Rendering of Moving People from Monocular Video**
**人类视差：从单目视频中的移动人的自由视角渲染**

- Homepage: https://grail.cs.washington.edu/projects/humannerf/
- Paper: https://arxiv.org/abs/2201.04127

- Demo: https://youtu.be/GM-RoZEymmw

<a name="3D Face"></a>

# 3D Face

**ImFace: A Nonlinear 3D Morphable Face Model with Implicit Neural Representations**
**ImFace：具有隐式神经表示的非线性3D可塑面部模型**

- Paper: https://arxiv.org/abs/2203.14510
- Code: https://github.com/MingwuZheng/ImFace 

<a name="Long-Tail"></a>

# 长尾分布(Long-Tail)

**Retrieval Augmented Classification for Long-Tail Visual Recognition**
**检索增强分类 长尾视觉识别**

- Paper: https://arxiv.org/abs/2202.11233
- Code: None

<a name="Visual-Transformer"></a>

# Visual Transformer

## Backbone

**MPViT : Multi-Path Vision Transformer for Dense Prediction**
**MPViT：多路径视觉变分器用于稠密预测**

- Paper: https://arxiv.org/abs/2112.11010
- Code: https://github.com/youngwanLEE/MPViT

**MetaFormer is Actually What You Need for Vision**
**元学习其实是你实现视觉所需的最佳工具。**

- Paper: https://arxiv.org/abs/2111.11418
- Code: https://github.com/sail-sg/poolformer

**Mobile-Former: Bridging MobileNet and Transformer**
**移动-Former：连接MobileNet和Transformer**

- Paper: https://arxiv.org/abs/2108.05895
- Code: None
- 中文解读：https://mp.weixin.qq.com/s/yo5KmB2Y7t2R4jiOKI87HQ

**Shunted Self-Attention via Multi-Scale Token Aggregation**
**通过多尺度令牌聚合的Shifted Self-Attention**

-  Paper(Oral): https://arxiv.org/abs/2111.15193
- Code: https://github.com/OliverRensu/Shunted-Transformer

**Learned Queries for Efficient Local Attention**
**学习有效的局部注意力**

- Paper(Oral): https://arxiv.org/abs/2112.11435
- Code: https://github.com/moabarar/qna

## 应用(Application)

**Language-based Video Editing via Multi-Modal Multi-Level Transformer**
**基于语言的多模态多层次Transformer视频编辑**

- Paper: https://arxiv.org/abs/2104.01122
- Code: None

**MixSTE: Seq2seq Mixed Spatio-Temporal Encoder for 3D Human Pose Estimation in Video**
**MixSTE: 3D 人类姿态估计的序列到序列混合时空编码器**

- Paper: https://arxiv.org/abs/2203.00859
- Code: None

**Embracing Single Stride 3D Object Detector with Sparse Transformer**
**拥抱单步3D对象检测与稀疏Transformer**

- Paper: https://arxiv.org/abs/2112.06375
- Code: https://github.com/TuSimple/SST
- 中文解读：https://zhuanlan.zhihu.com/p/476056546

**Multi-class Token Transformer for Weakly Supervised Semantic Segmentation**
**多类令牌传递器用于弱监督语义分割**

- Paper: https://arxiv.org/abs/2203.02891
- Code: https://github.com/xulianuwa/MCTformer

**Spatio-temporal Relation Modeling for Few-shot Action Recognition**
**空间-时间关系建模对于少样本动作识别**

- Paper: https://arxiv.org/abs/2112.05132
- Code: https://github.com/Anirudh257/strm

**Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction**
**带空格提示的中文翻译：带空格提示的中文翻译**

- Paper: https://arxiv.org/abs/2111.07910
- Code: https://github.com/caiyuanhao1998/MST

**Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling**
**点BERT: 带有遮罩点模型的3D点云预训练**

- Homepage: https://point-bert.ivg-research.xyz/
- Paper: https://arxiv.org/abs/2111.14819
- Code: https://github.com/lulutang0608/Point-BERT

**GroupViT: Semantic Segmentation Emerges from Text Supervision**
**GroupViT：语义分割通过文本监督浮现出来**

- Homepage: https://jerryxu.net/GroupViT/

- Paper: https://arxiv.org/abs/2202.11094
- Demo: https://youtu.be/DtJsWIUTW-Y

**Restormer: Efficient Transformer for High-Resolution Image Restoration**
**重建者：用于高分辨率图像复原的高效Transformer**

- Paper: https://arxiv.org/abs/2111.09881
- Code: https://github.com/swz30/Restormer

**Splicing ViT Features for Semantic Appearance Transfer**
**拼接ViT特征以实现语义外观转移**

- Homepage: https://splice-vit.github.io/
- Paper: https://arxiv.org/abs/2201.00424
- Code: https://github.com/omerbt/Splice

**Self-supervised Video Transformer**
**自监督视频变压器**

- Homepage: https://kahnchana.github.io/svt/
- Paper: https://arxiv.org/abs/2112.01514

- Code: https://github.com/kahnchana/svt

**Learning Affinity from Attention: End-to-End Weakly-Supervised Semantic Segmentation with Transformers**
**学习注意力中的相似性：端到端弱监督语义分割与变压器**

- Paper: https://arxiv.org/abs/2203.02664
- Code: https://github.com/rulixiang/afa

**Accelerating DETR Convergence via Semantic-Aligned Matching**
**通过语义对齐匹配加速DETR收敛**

- Paper: https://arxiv.org/abs/2203.06883
- Code: https://github.com/ZhangGongjie/SAM-DETR

**DN-DETR: Accelerate DETR Training by Introducing Query DeNoising**
**DN-DETR：通过引入查询去噪来加速DETR训练**

- Paper: https://arxiv.org/abs/2203.01305
- Code: https://github.com/FengLi-ust/DN-DETR
- 中文解读: https://mp.weixin.qq.com/s/xdMfZ_L628Ru1d1iaMny0w

**Style Transformer for Image Inversion and Editing**
**风格转换器用于图像反转和编辑**

- Paper: https://arxiv.org/abs/2203.07932
- Code: https://github.com/sapphire497/style-transformer

**MonoDTR: Monocular 3D Object Detection with Depth-Aware Transformer**
**MonoDTR：单目3D物体检测与深度感知变压器**

- Paper: https://arxiv.org/abs/2203.10981

- Code: https://github.com/kuanchihhuang/MonoDTR

**Mask Transfiner for High-Quality Instance Segmentation**
**带高质实例分割的蒙克转移器**

- Paper: https://arxiv.org/abs/2111.13673
- Code: https://github.com/SysCV/transfiner

**Language as Queries for Referring Video Object Segmentation**
**语言作为视频对象分割的查询**

- Paper: https://arxiv.org/abs/2201.00487
- Code:  https://github.com/wjn922/ReferFormer
- 中文解读：https://mp.weixin.qq.com/s/MkQT8QWSYoYVhJ1RSF6oPQ

**X-Trans2Cap: Cross-Modal Knowledge Transfer using Transformer for 3D Dense Captioning**
**X-Trans2Cap: 使用Transformer进行3D稠密语义捕捉的跨模态知识传递**

- Paper: https://arxiv.org/abs/2203.00843
- Code: https://github.com/CurryYuan/X-Trans2Cap

**AdaMixer: A Fast-Converging Query-Based Object Detector**
**AdaMixer：一个快速收敛的基于查询的对象检测器**

- Paper(Oral): https://arxiv.org/abs/2203.16507
- Code: https://github.com/MCG-NJU/AdaMixer

**Omni-DETR: Omni-Supervised Object Detection with Transformers**
**Omni-DETR: Omnivoxious Object Detection with Transformers**

- Paper: https://arxiv.org/abs/2203.16089
- Code: https://github.com/amazon-research/omni-detr

**SwinTextSpotter: Scene Text Spotting via Better Synergy between Text Detection and Text Recognition**
**SwinTextSpotter：通过文本检测和文本识别实现场景文本检测**

- Paper: https://arxiv.org/abs/2203.10209

- Code: https://github.com/mxin262/SwinTextSpotter

**TransRAC: Encoding Multi-scale Temporal Correlation with Transformers for Repetitive Action Counting**
**TRAC：使用变分自注意力机制编码多尺度时序相关性以实现重复动作计数**

- Paper(Oral): https://arxiv.org/abs/2204.01018
- Code: https://github.com/SvipRepetitionCounting/TransRAC

**Collaborative Transformers for Grounded Situation Recognition**
**合作变压器 for  grounded situation recognition**

- Paper: https://arxiv.org/abs/2203.16518
- Code: https://github.com/jhcho99/CoFormer

**NFormer: Robust Person Re-identification with Neighbor Transformer**
**NFormer: 用于增强人体识别的邻接自注意力**

- Paper: https://arxiv.org/abs/2204.09331
- Code: https://github.com/haochenheheda/NFormer

**Boosting Robustness of Image Matting with Context Assembling and Strong Data Augmentation**
**利用上下文组装和强数据增强来增强图像去模糊的鲁棒性**

- Paper: https://arxiv.org/abs/2201.06889
- Code: None

**Not All Tokens Are Equal: Human-centric Visual Analysis via Token Clustering Transformer**
**不是所有的令牌都是相等的：通过令牌聚类和Transformer进行人类中心视觉分析**

- Paper(Oral): https://arxiv.org/abs/2204.08680
- Code: https://github.com/zengwang430521/TCFormer

**A New Dataset and Transformer for Stereoscopic Video Super-Resolution**
**新数据集和Transformer用于立体视频超分辨率**

- Paper: https://arxiv.org/abs/2204.10039
- Code: https://github.com/H-deep/Trans-SVSR/
- Dataset: http://shorturl.at/mpwGX

**Safe Self-Refinement for Transformer-based Domain Adaptation**
**安全的自回归调整**

- Paper: https://arxiv.org/abs/2204.07683
- Code: https://github.com/tsun/SSRT

**Fast Point Transformer**
**快速点 transformer**

- Homepage: http://cvlab.postech.ac.kr/research/FPT/
- Paper: https://arxiv.org/abs/2112.04702
- Code: https://github.com/POSTECH-CVLab/FastPointTransformer

**Transformer Decoders with MultiModal Regularization for Cross-Modal Food Retrieval**
**带多模态约束的Transformer解码器在跨模态食品检索**

- Paper: https://arxiv.org/abs/2204.09730
- Code: https://github.com/mshukor/TFood

**DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation**
**DAFormer: 改进适应性语义分割的网络架构和训练策略**

- Paper: https://arxiv.org/abs/2111.14887
- Code: https://github.com/lhoyer/DAFormer

**Stratified Transformer for 3D Point Cloud Segmentation**
**3D 点云分割的分层Transformer**

- Paper: https://arxiv.org/pdf/2203.14508.pdf
- Code: https://github.com/dvlab-research/Stratified-Transformer 

<a name="VL"></a>

# 视觉和语言(Vision-Language)

**Conditional Prompt Learning for Vision-Language Models**
**条件提示学习为视觉语言模型**

- Paper: https://arxiv.org/abs/2203.05557
- Code: https://github.com/KaiyangZhou/CoOp

**Bridging Video-text Retrieval with Multiple Choice Question**
**跨视频文本检索与多选题**

- Paper: https://arxiv.org/abs/2201.04850
- Code: https://github.com/TencentARC/MCQ

**Visual Abductive Reasoning**
**视觉演绎推理**

- Paper: https://arxiv.org/abs/2203.14040
- Code: https://github.com/leonnnop/VAR

<a name="SSL"></a>

# 自监督学习(Self-supervised Learning)

**UniVIP: A Unified Framework for Self-Supervised Visual Pre-training**
**UniVIP：统一框架，用于自监督视觉预训练**

- Paper: https://arxiv.org/abs/2203.06965
- Code: None

**Crafting Better Contrastive Views for Siamese Representation Learning**
**构建更好的对比性视图以用于新加坡的表示学习**

- Paper: https://arxiv.org/abs/2202.03278
- Code: https://github.com/xyupeng/ContrastiveCrop
- 中文解读：https://mp.weixin.qq.com/s/VTP9D5f7KG9vg30U9kVI2A

**HCSC: Hierarchical Contrastive Selective Coding**
**HCSC：层次对比选择编码**

- Homepage: https://github.com/gyfastas/HCSC
- Paper: https://arxiv.org/abs/2202.00455
- 中文解读: https://mp.weixin.qq.com/s/jkYR8mYp-e645qk8kfPNKQ

**DiRA: Discriminative, Restorative, and Adversarial Learning for Self-supervised Medical Image Analysis**
**DiRA：用于自监督医疗图像分析的判别、恢复和对抗性学习**

- Paper: https://arxiv.org/abs/2204.10437

- Code: https://github.com/JLiangLab/DiRA

<a name="DA"></a>

# 数据增强(Data Augmentation)

**TeachAugment: Data Augmentation Optimization Using Teacher Knowledge**
**教学增强：利用教师知识进行数据增强优化**

- Paper: https://arxiv.org/abs/2202.12513
- Code: https://github.com/DensoITLab/TeachAugment

**AlignMixup: Improving Representations By Interpolating Aligned Features**
**AlignMixup: 通过插值一致特征来改进表示**

- Paper: https://arxiv.org/abs/2103.15375
- Code: https://github.com/shashankvkt/AlignMixup_CVPR22 

<a name="KD"></a>

# 知识蒸馏(Knowledge Distillation)

**Decoupled Knowledge Distillation**
**分离知识蒸馏**

- Paper: https://arxiv.org/abs/2203.08679
- Code: https://github.com/megvii-research/mdistiller
- 中文解读：https://mp.weixin.qq.com/s/-4AA0zKIXh9Ei9-vc5jOhw

<a name="Object-Detection"></a>

# 目标检测(Object Detection)

**BoxeR: Box-Attention for 2D and 3D Transformers**
**BoxR: 二维和三维变压器中的盒注意力**
- Paper: https://arxiv.org/abs/2111.13087
- Code: https://github.com/kienduynguyen/BoxeR
- 中文解读：https://mp.weixin.qq.com/s/UnUJJBwcAsRgz6TnQf_b7w

**DN-DETR: Accelerate DETR Training by Introducing Query DeNoising**
**DN-DETR：通过引入查询去噪来加速DETR训练**

- Paper: https://arxiv.org/abs/2203.01305
- Code: https://github.com/FengLi-ust/DN-DETR
- 中文解读: https://mp.weixin.qq.com/s/xdMfZ_L628Ru1d1iaMny0w

**Accelerating DETR Convergence via Semantic-Aligned Matching**
**通过语义对齐匹配加速DETR收敛**

- Paper: https://arxiv.org/abs/2203.06883
- Code: https://github.com/ZhangGongjie/SAM-DETR

**Localization Distillation for Dense Object Detection**
**密贴目标检测的局部化蒸发**

- Paper: https://arxiv.org/abs/2102.12252
- Code: https://github.com/HikariTJU/LD
- Code2: https://github.com/HikariTJU/LD
- 中文解读：https://mp.weixin.qq.com/s/dxss8RjJH283h6IbPCT9vg

**Focal and Global Knowledge Distillation for Detectors**
**焦点和全局知识蒸馏对于检测器**

- Paper: https://arxiv.org/abs/2111.11837
- Code: https://github.com/yzd-v/FGD
- 中文解读：https://mp.weixin.qq.com/s/yDkreTudC8JL2V2ETsADwQ

**A Dual Weighting Label Assignment Scheme for Object Detection**
**一个用于目标检测的双重加权标签分配方案**

- Paper: https://arxiv.org/abs/2203.09730
- Code: https://github.com/strongwolf/DW

**AdaMixer: A Fast-Converging Query-Based Object Detector**
**AdaMixer：一个快速收敛的基于查询的对象检测器**

- Paper(Oral): https://arxiv.org/abs/2203.16507
- Code: https://github.com/MCG-NJU/AdaMixer

**Omni-DETR: Omni-Supervised Object Detection with Transformers**
**Omni-DETR: 全天候监督下的目标检测**

- Paper: https://arxiv.org/abs/2203.16089
- Code: https://github.com/amazon-research/omni-detr

**SIGMA: Semantic-complete Graph Matching for Domain Adaptive Object Detection**
**SIGMA：用于领域自适应对象检测的语义完整图匹配**

- Paper(Oral): https://arxiv.org/abs/2203.06398
- Code: https://github.com/CityU-AIM-Group/SIGMA

## 半监督目标检测

**Dense Learning based Semi-Supervised Object Detection**
**稠密学习基于半监督的目标检测**

- Paper: https://arxiv.org/abs/2204.07300

- Code: https://github.com/chenbinghui1/DSL

# 目标跟踪(Visual Tracking)

**Correlation-Aware Deep Tracking**
**相关感深的跟踪**

- Paper: https://arxiv.org/abs/2203.01666
- Code: None

**TCTrack: Temporal Contexts for Aerial Tracking**
**TCTrack：用于空中追踪的时空上下文**

- Paper: https://arxiv.org/abs/2203.01885
- Code: https://github.com/vision4robotics/TCTrack

## 多模态目标跟踪

**Visible-Thermal UAV Tracking: A Large-Scale Benchmark and New Baseline**
**可见-热 UAV 追踪：大规模基准和新基准**

- Homepage: https://zhang-pengyu.github.io/DUT-VTUAV/

- Paper: https://arxiv.org/abs/2204.04120

## 多目标跟踪(Multi-Object Tracking)

**Learning of Global Objective for Network Flow in Multi-Object Tracking**
**多对象跟踪中的网络流全局目标的学习**

- Paper: https://arxiv.org/abs/2203.16210
- Code: None

**DanceTrack: Multi-Object Tracking in Uniform Appearance and Diverse Motion**
**舞轨：在统一外观和多样化的动作中进行多对象跟踪**

- Homepage: https://dancetrack.github.io
- Paper: https://arxiv.org/abs/2111.14690
- Dataset: https://github.com/DanceTrack/DanceTrack

<a name="Semantic-Segmentation"></a>

# 语义分割(Semantic Segmentation)

**Novel Class Discovery in Semantic Segmentation**
**语义分割中的类别发现新类**

- Homepage: https://ncdss.github.io/
- Paper: https://arxiv.org/abs/2112.01900
- Code: https://github.com/HeliosZhao/NCDSS

**Deep Hierarchical Semantic Segmentation**
**深度层次语义分割**

- Paper: https://arxiv.org/abs/2203.14335
- Code: https://github.com/0liliulei/HieraSeg 

**Rethinking Semantic Segmentation: A Prototype View**
**重新思考语义分割：一个原型视图**

- Paper(Oral): https://arxiv.org/abs/2203.15102
- Code: https://github.com/tfzhou/ProtoSeg

## 弱监督语义分割

**Class Re-Activation Maps for Weakly-Supervised Semantic Segmentation**
**弱监督语义分割类别的重新激活图**

- Paper: https://arxiv.org/abs/2203.00962
- Code: https://github.com/zhaozhengChen/ReCAM

**Multi-class Token Transformer for Weakly Supervised Semantic Segmentation**
**多类令牌转换器用于弱监督语义分割**

- Paper: https://arxiv.org/abs/2203.02891
- Code: https://github.com/xulianuwa/MCTformer

**Learning Affinity from Attention: End-to-End Weakly-Supervised Semantic Segmentation with Transformers**
**学习注意力中的向心力：端到端的弱监督语义分割与变压器**

- Paper: https://arxiv.org/abs/2203.02664
- Code: https://github.com/rulixiang/afa

**CLIMS: Cross Language Image Matching for Weakly Supervised Semantic Segmentation**
**CLIMS：弱监督语义分割的跨语言图像匹配**

- Paper: https://arxiv.org/abs/2203.02668
- Code: https://github.com/CVI-SZU/CLIMS 

**CCAM: Contrastive learning of Class-agnostic Activation Map for Weakly Supervised Object Localization and Semantic Segmentation**
**CCAM：对比学习弱监督下类别无关激活图的类别无关对象定位和语义分割**

- Paper: https://arxiv.org/abs/2203.13505
- Code: https://github.com/CVI-SZU/CCAM 

**FIFO: Learning Fog-invariant Features for Foggy Scene Segmentation**
**FIFO：学习在雾中场景分割中不变的特征**

- Homeapage: http://cvlab.postech.ac.kr/research/FIFO/
- Paper(Oral): https://arxiv.org/abs/2204.01587
- Code: https://github.com/sohyun-l/FIFO 

**Regional Semantic Contrast and Aggregation for Weakly Supervised Semantic Segmentation**
**区域语义对比和聚合弱监督语义分割**

- Paper: https://arxiv.org/abs/2203.09653
- Code: https://github.com/maeve07/RCA.git

## 半监督语义分割

**ST++: Make Self-training Work Better for Semi-supervised Semantic Segmentation**
**ST++：为半监督语义分割使自训练更有效**

- Paper: https://arxiv.org/abs/2106.05095
- Code: https://github.com/LiheYoung/ST-PlusPlus
- 中文解读：https://mp.weixin.qq.com/s/knSnlebdtEnmrkChGM_0CA

**Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels**
**半监督语义分割：使用不可靠伪标签**

- Homepage: https://haochen-wang409.github.io/U2PL/
- Paper: https://arxiv.org/abs/2203.03884
- Code: https://github.com/Haochen-Wang409/U2PL
- 中文解读: https://mp.weixin.qq.com/s/-08olqE7np8A1XQzt6HAgQ

**Perturbed and Strict Mean Teachers for Semi-supervised Semantic Segmentation**
**半监督语义分割中的扰动和严格均值教师**

- Paper: https://arxiv.org/pdf/2111.12903.pdf
- Code: https://github.com/yyliu01/PS-MT

## 域自适应语义分割

**Towards Fewer Annotations: Active Learning via Region Impurity and Prediction Uncertainty for Domain Adaptive Semantic Segmentation**
**走向更少的注释：通过区域不纯度和预测不确定性进行领域自适应语义分割的主动学习**

- Paper: https://arxiv.org/abs/2111.12940
- Code: https://github.com/BIT-DA/RIPU

**DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation**
**DAFormer: 改进适应性语义分割的网络架构和训练策略**

- Paper: https://arxiv.org/abs/2111.14887
- Code: https://github.com/lhoyer/DAFormer

## 无监督语义分割

**GroupViT: Semantic Segmentation Emerges from Text Supervision**
**GroupViT: 语义分割通过文本监督浮现出来**

- Homepage: https://jerryxu.net/GroupViT/
- Paper: https://arxiv.org/abs/2202.11094
- Demo: https://youtu.be/DtJsWIUTW-Y

## 少样本语义分割

**Generalized Few-shot Semantic Segmentation**
**泛化少样本语义分割**

- Paper: https://jiaya.me/papers/cvpr22_zhuotao.pdf
- Code: https://github.com/dvlab-research/GFS-Seg 

<a name="Instance-Segmentation"></a>

# 实例分割(Instance Segmentation)

**BoxeR: Box-Attention for 2D and 3D Transformers**
**BoxeR: 2D 和 3D 变压器中的盒注意力**
- Paper: https://arxiv.org/abs/2111.13087
- Code: https://github.com/kienduynguyen/BoxeR
- 中文解读：https://mp.weixin.qq.com/s/UnUJJBwcAsRgz6TnQf_b7w

**E2EC: An End-to-End Contour-based Method for High-Quality High-Speed Instance Segmentation**
**E2EC：一种基于终边的高质量高速实例分割方法**

- Paper: https://arxiv.org/abs/2203.04074
- Code: https://github.com/zhang-tao-whu/e2ec

**Mask Transfiner for High-Quality Instance Segmentation**
**高质量实例分割的掩膜生成器**

- Paper: https://arxiv.org/abs/2111.13673
- Code: https://github.com/SysCV/transfiner

**Open-World Instance Segmentation: Exploiting Pseudo Ground Truth From Learned Pairwise Affinity**
**开放世界实例分割：利用学习到的成对关系度量**

- Homepage: https://sites.google.com/view/generic-grouping/

- Paper: https://arxiv.org/abs/2204.06107
- Code: https://github.com/facebookresearch/Generic-Grouping

## 自监督实例分割

**FreeSOLO: Learning to Segment Objects without Annotations**
**FreeSOLO：无注释地分割对象**

- Paper: https://arxiv.org/abs/2202.12181
- Code: https://github.com/NVlabs/FreeSOLO

## 视频实例分割

**Efficient Video Instance Segmentation via Tracklet Query and Proposal**
**高效的视频实例分割通过轨迹查询和提议**

- Homepage: https://jialianwu.com/projects/EfficientVIS.html
- Paper: https://arxiv.org/abs/2203.01853
- Demo: https://youtu.be/sSPMzgtMKCE

**Temporally Efficient Vision Transformer for Video Instance Segmentation**
**时间高效的视频实例分割视觉转换器**

- Paper: https://arxiv.org/abs/2204.08412
- Code: https://github.com/hustvl/TeViT

<a name="Panoptic-Segmentation"></a>

# 全景分割(Panoptic Segmentation)

**Panoptic SegFormer: Delving Deeper into Panoptic Segmentation with Transformers**
**Panoptic 分割器：通过 Transformer 更深入地进行 Panoptic 分割**

- Paper: https://arxiv.org/abs/2109.03814
- Code: https://github.com/zhiqi-li/Panoptic-SegFormer

**Large-scale Video Panoptic Segmentation in the Wild: A Benchmark**
**野外的大规模视频全景分割：一个基准**

- Paper: https://github.com/VIPSeg-Dataset/VIPSeg-Dataset/blob/main/VIPSeg2022.pdf
- Code: https://github.com/VIPSeg-Dataset/VIPSeg-Dataset
- Dataset: https://github.com/VIPSeg-Dataset/VIPSeg-Dataset 

<a name="FFC"></a>

# 小样本分类(Few-Shot Classification)

**Integrative Few-Shot Learning for Classification and Segmentation**
**集成式少量样本分类和分割学习**

- Paper: https://arxiv.org/abs/2203.15712
- Code: https://github.com/dahyun-kang/ifsl

**Learning to Affiliate: Mutual Centralized Learning for Few-shot Classification**
**学习联盟：少样本分类的互相关中心学习**

- Paper: https://arxiv.org/abs/2106.05517
- Code: https://github.com/LouieYang/MCL

<a name="FFS"></a>

# 小样本分割(Few-Shot Segmentation)

**Learning What Not to Segment: A New Perspective on Few-Shot Segmentation**
**学习何不分割：对少量样本分割的新视角**

- Paper: https://arxiv.org/abs/2203.07615
- Code: https://github.com/chunbolang/BAM

**Integrative Few-Shot Learning for Classification and Segmentation**
**整式少样本学习用于分类和分割**

- Paper: https://arxiv.org/abs/2203.15712
- Code: https://github.com/dahyun-kang/ifsl

**Dynamic Prototype Convolution Network for Few-Shot Semantic Segmentation**
**动态原型卷积神经网络 for 少样本语义分割**

- Paper: https://arxiv.org/abs/2204.10638
- Code: None

<a name="Matting"></a>

# 图像抠图(Image Matting)

**Boosting Robustness of Image Matting with Context Assembling and Strong Data Augmentation**
**利用上下文组装和强数据增强来提高图像校正的鲁棒性**

- Paper: https://arxiv.org/abs/2201.06889
- Code: None

<a name="VU"></a>

# 视频理解(Video Understanding)

**Self-supervised Video Transformer**
**自监督视频变压器**

- Homepage: https://kahnchana.github.io/svt/
- Paper: https://arxiv.org/abs/2112.01514
- Code: https://github.com/kahnchana/svt

**TransRAC: Encoding Multi-scale Temporal Correlation with Transformers for Repetitive Action Counting**
**TransRAC：使用变分自注意力机制编码多尺度时序相关性以实现重复动作计数**

- Paper(Oral): https://arxiv.org/abs/2204.01018
- Code: https://github.com/SvipRepetitionCounting/TransRAC

**FineDiving: A Fine-grained Dataset for Procedure-aware Action Quality Assessment**
**精细潜水：一个用于程序感知动作质量评估的细粒度数据集**

- Paper(Oral): https://arxiv.org/abs/2204.03646

- Dataset: https://github.com/xujinglin/FineDiving
- Code: https://github.com/xujinglin/FineDiving
- 中文解读：https://mp.weixin.qq.com/s/8t12Y34eMNwvJr8PeryWXg

**Dual-AI: Dual-path Actor Interaction Learning for Group Activity Recognition**
**双重AI：双重路径演员互动学习用于群活动识别**

- Paper(Oral): https://arxiv.org/abs/2204.02148
- Code: None

## 行为识别(Action Recognition)

**Spatio-temporal Relation Modeling for Few-shot Action Recognition**
**空间-时间关系建模对于少样本动作识别**

- Paper: https://arxiv.org/abs/2112.05132
- Code: https://github.com/Anirudh257/strm

## 动作检测(Action Detection)

**End-to-End Semi-Supervised Learning for Video Action Detection**
**端到端半监督学习视频动作检测**

- Paper: https://arxiv.org/abs/2203.04251
- Code: None

<a name="Image-Editing"></a>

# 图像编辑(Image Editing)

**Style Transformer for Image Inversion and Editing**
**图像翻转和编辑风格转换器**

- Paper: https://arxiv.org/abs/2203.07932
- Code: https://github.com/sapphire497/style-transformer

**Blended Diffusion for Text-driven Editing of Natural Images**
**混合扩散用于自然图像文本驱动的编辑**

- Paper: https://arxiv.org/abs/2111.14818
- Code: https://github.com/omriav/blended-diffusion

**SemanticStyleGAN: Learning Compositional Generative Priors for Controllable Image Synthesis and Editing**
**语义风格生成器（SemanticStyleGAN）：控制图像生成和编辑的可控制图像合成与编辑**

- Homepage: https://semanticstylegan.github.io/

- Paper: https://arxiv.org/abs/2112.02236
- Demo: https://semanticstylegan.github.io/videos/demo.mp4

<a name="LLV"></a>

# Low-level Vision

**ISNAS-DIP: Image-Specific Neural Architecture Search for Deep Image Prior**
**ISNAS-DIP: 针对深度图像优先的图像特定神经架构搜索**

- Paper: https://arxiv.org/abs/2111.15362
- Code: None

**Restormer: Efficient Transformer for High-Resolution Image Restoration**
**重建者：高效的用于高分辨率图像恢复的Transformer**

- Paper: https://arxiv.org/abs/2111.09881
- Code: https://github.com/swz30/Restormer

**Robust Equivariant Imaging: a fully unsupervised framework for learning to image from noisy and partial measurements**
**鲁棒等方差成像：一个完全无监督的框架，用于从噪声和部分测量中学习图像**

- Paper(Oral): https://arxiv.org/abs/2111.12855
- Code: https://github.com/edongdongchen/REI

<a name="Super-Resolution"></a>

# 超分辨率(Super-Resolution)

## 图像超分辨率(Image Super-Resolution)

**Learning the Degradation Distribution for Blind Image Super-Resolution**
**学习盲图像超分辨率中的退化分布**

- Paper: https://arxiv.org/abs/2203.04962
- Code: https://github.com/greatlog/UnpairedSR

## 视频超分辨率(Video Super-Resolution)

**BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment**
**BasicVSR++：通过增强传播和对齐来改进视频超分辨率**

- Paper: https://arxiv.org/abs/2104.13371
- Code: https://github.com/open-mmlab/mmediting
- Code: https://github.com/ckkelvinchan/BasicVSR_PlusPlus
- 中文解读：https://mp.weixin.qq.com/s/HZTwYfphixyLHxlbCAxx4g

**Look Back and Forth: Video Super-Resolution with Explicit Temporal Difference Modeling**
**回溯和向前：具有显式时间差异的 Video 超分辨率**

- Paper: https://arxiv.org/abs/2204.07114
- Code: None

**A New Dataset and Transformer for Stereoscopic Video Super-Resolution**
**新数据集和Transformer用于立体视频超分辨率**

- Paper: https://arxiv.org/abs/2204.10039
- Code: https://github.com/H-deep/Trans-SVSR/
- Dataset: http://shorturl.at/mpwGX

<a name="Deblur"></a>

# 去模糊(Deblur)

## 图像去模糊(Image Deblur)

**Learning to Deblur using Light Field Generated and Real Defocus Images**
**学习使用生成光场和真实去模糊图像**

- Homepage: http://lyruan.com/Projects/DRBNet/
- Paper(Oral): https://arxiv.org/abs/2204.00442

- Code: https://github.com/lingyanruan/DRBNet

<a name="3D-Point-Cloud"></a>

# 3D点云(3D Point Cloud)

**Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling**
**点BERT: 使用遮罩点模型进行预训练的3D点云转换器**

- Homepage: https://point-bert.ivg-research.xyz/

- Paper: https://arxiv.org/abs/2111.14819
- Code: https://github.com/lulutang0608/Point-BERT

**A Unified Query-based Paradigm for Point Cloud Understanding**
**统一基于查询的点云理解范式**

- Paper: https://arxiv.org/abs/2203.01252
- Code: None 

**CrossPoint: Self-Supervised Cross-Modal Contrastive Learning for 3D Point Cloud Understanding**
**CrossPoint：自监督的跨模态对比学习用于3D点云理解**

- Paper: https://arxiv.org/abs/2203.00680
- Code: https://github.com/MohamedAfham/CrossPoint

**PointCLIP: Point Cloud Understanding by CLIP**
**PointCLIP: 由CLIP解释的点云理解**

- Paper: https://arxiv.org/abs/2112.02413
- Code: https://github.com/ZrrSkywalker/PointCLIP

**Fast Point Transformer**
**快速点积变换器**

- Homepage: http://cvlab.postech.ac.kr/research/FPT/
- Paper: https://arxiv.org/abs/2112.04702
- Code: https://github.com/POSTECH-CVLab/FastPointTransformer

**RCP: Recurrent Closest Point for Scene Flow Estimation on 3D Point Clouds**
**RCP：针对3D点云场景流估计的循环最近点**

- Paper: https://arxiv.org/abs/2205.11028
- Code: https://github.com/gxd1994/RCP

**The Devil is in the Pose: Ambiguity-free 3D Rotation-invariant Learning via Pose-aware Convolution**
**恶魔在姿势：通过姿势感知的无歧义3D旋转不变性学习**

- Paper: https://arxiv.org/abs/2205.15210
- Code: https://github.com/GostInShell/PaRI-Conv 

<a name="3D-Object-Detection"></a>

# 3D目标检测(3D Object Detection)

**Not All Points Are Equal: Learning Highly Efficient Point-based Detectors for 3D LiDAR Point Clouds**
**不是所有点都相等：学习用于3D LiDAR点云的高效点检测算法**

- Paper(Oral): https://arxiv.org/abs/2203.11139

- Code: https://github.com/yifanzhang713/IA-SSD

- Demo: https://www.youtube.com/watch?v=3jP2o9KXunA

**BoxeR: Box-Attention for 2D and 3D Transformers**
**盒子推理：2D和3D变压器**
- Paper: https://arxiv.org/abs/2111.13087
- Code: https://github.com/kienduynguyen/BoxeR
- 中文解读：https://mp.weixin.qq.com/s/UnUJJBwcAsRgz6TnQf_b7w

**Embracing Single Stride 3D Object Detector with Sparse Transformer**
**拥抱单步3D对象检测与稀疏变换器**

- Paper: https://arxiv.org/abs/2112.06375

- Code: https://github.com/TuSimple/SST

**Canonical Voting: Towards Robust Oriented Bounding Box Detection in 3D Scenes** 

- Paper: https://arxiv.org/abs/2011.12001
- Code: https://github.com/qq456cvb/CanonicalVoting

**MonoDTR: Monocular 3D Object Detection with Depth-Aware Transformer**
**MonoDTR: 单目3D物体检测与深度感知的Transformer**

- Paper: https://arxiv.org/abs/2203.10981
- Code: https://github.com/kuanchihhuang/MonoDTR

**HyperDet3D: Learning a Scene-conditioned 3D Object Detector**
**半监督学习：场景条件下的3D对象检测**

- Paper: https://arxiv.org/abs/2204.05599
- Code: None

**OccAM's Laser: Occlusion-based Attribution Maps for 3D Object Detectors on LiDAR Data**
**Occam 激光：基于遮挡的3D物体检测器在LiDAR数据上的掩膜分配图**

- Paper: https://arxiv.org/abs/2204.06577
- Code: https://github.com/dschinagl/occam

**DAIR-V2X: A Large-Scale Dataset for Vehicle-Infrastructure Cooperative 3D Object Detection**
**DAIR-V2X：用于车辆基础设施协同大规模3D物体检测的大规模数据集**

- Homepage: https://thudair.baai.ac.cn/index
- Paper: https://arxiv.org/abs/2204.05575
- Code: https://github.com/AIR-THU/DAIR-V2X

**Ithaca365: Dataset and Driving Perception under Repeated and Challenging Weather Conditions**
**Ithaca365：在反复且具有挑战性的天气条件下，数据集和驱动感知**

- Homepage: https://ithaca365.mae.cornell.edu/

- Paper: https://arxiv.org/abs/2208.01166

<a name="3DSS"></a>

# 3D语义分割(3D Semantic Segmentation)

**Scribble-Supervised LiDAR Semantic Segmentation**
**手绘监督LiDAR语义分割**

- Paper: https://arxiv.org/abs/2203.08537
- Dataset: https://github.com/ouenal/scribblekitti

**Stratified Transformer for 3D Point Cloud Segmentation**
**用于3D点云分段的随机变换器**

- Paper: https://arxiv.org/pdf/2203.14508.pdf
- Code: https://github.com/dvlab-research/Stratified-Transformer

# 3D实例分割(3D Instance Segmentation)

**Ithaca365: Dataset and Driving Perception under Repeated and Challenging Weather Conditions**
**Ithaca365：在反复和具有挑战性的天气条件下，数据集和驾驶感知**

- Homepage: https://ithaca365.mae.cornell.edu/

- Paper: https://arxiv.org/abs/2208.01166

<a name="3D-Object-Tracking"></a>

# 3D目标跟踪(3D Object Tracking)

**Beyond 3D Siamese Tracking: A Motion-Centric Paradigm for 3D Single Object Tracking in Point Clouds**
**超越3D对比：基于运动的3D单目标跟踪在点云中的3D**

- Paper: https://arxiv.org/abs/2203.01730
- Code: https://github.com/Ghostish/Open3DSOT

**PTTR: Relational 3D Point Cloud Object Tracking with Transformer**
**点云关系建模与Transformer的3D目标跟踪**

- Paper: https://arxiv.org/abs/2112.02857
- Code: https://github.com/Jasonkks/PTTR 

<a name="3D-Human-Pose-Estimation"></a>

# 3D人体姿态估计(3D Human Pose Estimation)

**MHFormer: Multi-Hypothesis Transformer for 3D Human Pose Estimation**
**MHFormer: 多重假设变体用于3D人体姿态估计**

- Paper: https://arxiv.org/abs/2111.12707

- Code: https://github.com/Vegetebird/MHFormer

- 中文解读: https://zhuanlan.zhihu.com/p/439459426

**MixSTE: Seq2seq Mixed Spatio-Temporal Encoder for 3D Human Pose Estimation in Video**
**MixSTE: 3D 人类姿态估计的序列到序列混合空间-时间编码器**

- Paper: https://arxiv.org/abs/2203.00859
- Code: None

**Distribution-Aware Single-Stage Models for Multi-Person 3D Pose Estimation**
**多模态3D姿态估计的分布感知单阶段模型**

- Paper: https://arxiv.org/abs/2203.07697
- Code: None
- 中文解读：https://mp.weixin.qq.com/s/L_F28IFLXvs5R4V9TTUpRw

**BEV: Putting People in their Place: Monocular Regression of 3D People in Depth**
**BEV：置人于其位：深度中3D人物的单目回归**

- Homepage: https://arthur151.github.io/BEV/BEV.html
- Paper: https://arxiv.org/abs/2112.08274
- Code: https://github.com/Arthur151/ROMP
- Dataset: https://github.com/Arthur151/Relative_Human
- Demo: https://www.youtube.com/watch?v=Q62fj_6AxRI

<a name="3DSSC"></a>

# 3D语义场景补全(3D Semantic Scene Completion)

**MonoScene: Monocular 3D Semantic Scene Completion**
**单目场景：单目3D语义场景补全**

- Paper: https://arxiv.org/abs/2112.00726
- Code: https://github.com/cv-rits/MonoScene

<a name="3D-R"></a>

# 3D重建(3D Reconstruction)

**BANMo: Building Animatable 3D Neural Models from Many Casual Videos**
**BANMo：从许多日常视频中构建可动画的3D神经网络模型**

- Homepage: https://banmo-www.github.io/
- Paper: https://arxiv.org/abs/2112.12761
- Code: https://github.com/facebookresearch/banmo
- 中文解读：https://mp.weixin.qq.com/s/NMHP8-xWwrX40vpGx55Qew

<a name="ReID"></a>

# 行人重识别(Person Re-identification)

**NFormer: Robust Person Re-identification with Neighbor Transformer**
**NFormer: 用于强大的人重新识别的邻居自注意力**

- Paper: https://arxiv.org/abs/2204.09331
- Code: https://github.com/haochenheheda/NFormer

<a name="COD"></a>

# 伪装物体检测(Camouflaged Object Detection)

**Zoom In and Out: A Mixed-scale Triplet Network for Camouflaged Object Detection**
**缩放进去和出来：用于伪装目标检测的混合尺度三元组网络**

- Paper: https://arxiv.org/abs/2203.02688
- Code: https://github.com/lartpang/ZoomNet

<a name="Depth-Estimation"></a>

# 深度估计(Depth Estimation)

## 单目深度估计

**NeW CRFs: Neural Window Fully-connected CRFs for Monocular Depth Estimation**
**NeW CRFs：用于单目深度估计的神经窗口全连接CRFs**

- Paper: https://arxiv.org/abs/2203.01502
- Code: None

**OmniFusion: 360 Monocular Depth Estimation via Geometry-Aware Fusion**
**OmniFusion: 通过几何感知的融合实现360度单目深度估计**

- Paper: https://arxiv.org/abs/2203.00838
- Code: None

**Toward Practical Self-Supervised Monocular Indoor Depth Estimation**
**朝着实用自监督单目室内深度估计**

- Paper: https://arxiv.org/abs/2112.02306
- Code: None

**P3Depth: Monocular Depth Estimation with a Piecewise Planarity Prior**
**P3Depth: 单目深度估计与分片平面约束**

- Paper: https://arxiv.org/abs/2204.02091
- Code: https://github.com/SysCV/P3Depth

**Multi-Frame Self-Supervised Depth with Transformers**
**多帧自监督深度学习**

- Homepage: https://sites.google.com/tri.global/depthformer

- Paper: https://arxiv.org/abs/2204.07616
- Code: None

<a name="Stereo-Matching"></a>

# 立体匹配(Stereo Matching)

**ACVNet: Attention Concatenation Volume for Accurate and Efficient Stereo Matching**
**ACVNet：用于准确高效双目立体匹配的注意力卷积**

- Paper: https://arxiv.org/abs/2203.02146
- Code: https://github.com/gangweiX/ACVNet

<a name="FM"></a>

# 特征匹配(Feature Matching)

**ClusterGNN: Cluster-based Coarse-to-Fine Graph Neural Network for Efficient Feature Matching**
**聚类 GNN：基于聚类的粗到细图神经网络，用于高效的特征匹配**

- Paper: https://arxiv.org/abs/2204.11700
- Code: None

<a name="Lane-Detection"></a>

# 车道线检测(Lane Detection)

**Rethinking Efficient Lane Detection via Curve Modeling**
**重新思考通过曲线建模来提高高效车道检测的方法**

- Paper: https://arxiv.org/abs/2203.02431
- Code: https://github.com/voldemortX/pytorch-auto-drive
- Demo：https://user-images.githubusercontent.com/32259501/148680744-a18793cd-f437-461f-8c3a-b909c9931709.mp4

**A Keypoint-based Global Association Network for Lane Detection**
**基于关键点的全局关联网络 lanes 检测**

- Paper: https://arxiv.org/abs/2204.07335
- Code: https://github.com/Wolfwjs/GANet

<a name="Optical-Flow-Estimation"></a>

# 光流估计(Optical Flow Estimation)

**Imposing Consistency for Optical Flow Estimation**
**强制一致性光学流动估计**

- Paper: https://arxiv.org/abs/2204.07262
- Code: None

**Deep Equilibrium Optical Flow Estimation**
**深度平衡光学流估计**

- Paper: https://arxiv.org/abs/2204.08442
- Code: https://github.com/locuslab/deq-flow

**GMFlow: Learning Optical Flow via Global Matching**
**GMFlow：通过全局匹配学习光学流动**

- Paper(Oral): https://arxiv.org/abs/2111.13680
- Code: https://github.com/haofeixu/gmflow

<a name="Image-Inpainting"></a>

# 图像修复(Image Inpainting)

**Incremental Transformer Structure Enhanced Image Inpainting with Masking Positional Encoding**
**增强型Transformer结构用于图像修复的逐行卷积**

- Paper: https://arxiv.org/abs/2203.00867

- Code: https://github.com/DQiaole/ZITS_inpainting

<a name="Image-Retrieval"></a>

# 图像检索(Image Retrieval)

**Correlation Verification for Image Retrieval**
**图像检索的相关性验证**

- Paper(Oral): https://arxiv.org/abs/2204.01458
- Code: https://github.com/sungonce/CVNet

<a name="Face-Recognition"></a>

# 人脸识别(Face Recognition)

**AdaFace: Quality Adaptive Margin for Face Recognition**
**AdaFace：面部识别的高质量自适应边框**

- Paper(Oral): https://arxiv.org/abs/2204.00964 
- Code: https://github.com/mk-minchul/AdaFace

<a name="Crowd-Counting"></a>

# 人群计数(Crowd Counting)

**Leveraging Self-Supervision for Cross-Domain Crowd Counting**
**利用自监督学习实现跨领域聚类**

- Paper: https://arxiv.org/abs/2103.16291
- Code: None

<a name="Medical-Image"></a>

# 医学图像(Medical Image)

**BoostMIS: Boosting Medical Image Semi-supervised Learning with Adaptive Pseudo Labeling and Informative Active Annotation**
**提升MIS: 使用自适应伪标签和有信息的活动注释来提高医学图像的半监督学习**

- Paper: https://arxiv.org/abs/2203.02533
- Code: None

**Anti-curriculum Pseudo-labelling for Semi-supervised Medical Image Classification**
**半监督医学图像分类的抗课程伪标签**

- Paper: https://arxiv.org/abs/2111.12918
- Code: https://github.com/FBLADL/ACPL

**DiRA: Discriminative, Restorative, and Adversarial Learning for Self-supervised Medical Image Analysis**
**DiRA：用于自监督医学图像分析的判别性、恢复性和对抗性学习**

- Paper: https://arxiv.org/abs/2204.10437

- Code: https://github.com/JLiangLab/DiRA

<a name="Video Generation"></a>

# 视频生成(Video Generation)

**StyleGAN-V: A Continuous Video Generator with the Price, Image Quality and Perks of StyleGAN2**
**StyleGAN-V：一个带有StyleGAN2价格、图像质量和特权的连续视频生成器。**

- Homepage: https://universome.github.io/stylegan-v
- Paper: https://arxiv.org/abs/2112.14683

- Code: https://github.com/universome/stylegan-v

- Demo: https://kaust-cair.s3.amazonaws.com/stylegan-v/stylegan-v.mp4

<a name="Scene-Graph-Generation"></a>

# 场景图生成(Scene Graph Generation)

 **SGTR: End-to-end Scene Graph Generation with Transformer**

- Paper: https://arxiv.org/abs/2112.12970
- Code: None

<a name="R-VOS"></a>

# 参考视频目标分割(Referring Video Object Segmentation)

**Language as Queries for Referring Video Object Segmentation**
**语言作为视频对象分割的查询**

- Paper: https://arxiv.org/abs/2201.00487
- Code:  https://github.com/wjn922/ReferFormer

**ReSTR: Convolution-free Referring Image Segmentation Using Transformers**
**无监督：使用变压器进行无约束的引用图像分割**

- Paper: https://arxiv.org/abs/2203.16768
- Code: None

<a name="GR"></a>

# 步态识别(Gait Recognition)

**Gait Recognition in the Wild with Dense 3D Representations and A Benchmark**
**在野外进行人行走识别，具有稠密的三维表示和基准测试**

- Homepage: https://gait3d.github.io/
- Paper: https://arxiv.org/abs/2204.02569
- Code: https://github.com/Gait3D/Gait3D-Benchmark

<a name="ST"></a>

# 风格迁移(Style Transfer)

**StyleMesh: Style Transfer for Indoor 3D Scene Reconstructions**
**StyleMesh: 风格迁移用于室内3D场景重建**

- Homepage: https://lukashoel.github.io/stylemesh/
- Paper: https://arxiv.org/abs/2112.01530

- Code: https://github.com/lukasHoel/stylemesh
- Demo：https://www.youtube.com/watch?v=ZqgiTLcNcks

<a name="AD"></a>

# 异常检测(Anomaly Detection)

**UBnormal: New Benchmark for Supervised Open-Set Video Anomaly Detection**
**UBnormal：监督开放集视频异常检测的新基准**

- Paper: https://arxiv.org/abs/2111.08644

- Dataset: https://github.com/lilygeorgescu/UBnormal

**Self-Supervised Predictive Convolutional Attentive Block for Anomaly Detection**
**自监督预测卷积注意力块用于异常检测**

- Paper(Oral): https://arxiv.org/abs/2111.09099
- Code: https://github.com/ristea/sspcab

对抗样本)<a name="AE"></a>

# 对抗样本(Adversarial Examples)

**Shadows can be Dangerous: Stealthy and Effective Physical-world Adversarial Attack by Natural Phenomenon**
**影子危险：自然现象对物理世界的高效隐身攻击**

- Paper: https://arxiv.org/abs/2203.03818
- Code: https://github.com/hncszyq/ShadowAttack

**LAS-AT: Adversarial Training with Learnable Attack Strategy**
**LAS-AT: 学习攻击策略的对抗性训练**

- Paper(Oral): https://arxiv.org/abs/2203.06616
- Code: https://github.com/jiaxiaojunQAQ/LAS-AT

**Segment and Complete: Defending Object Detectors against Adversarial Patch Attacks with Robust Patch Detection**
**分割和完整：通过鲁棒的补丁检测防御对抗性补丁攻击的对象检测器**

- Paper: https://arxiv.org/abs/2112.04532
- Code: https://github.com/joellliu/SegmentAndComplete

<a name="WSOL"></a>

# 弱监督物体检测(Weakly Supervised Object Localization)

**Weakly Supervised Object Localization as Domain Adaption**
**弱监督目标域自适应**

- Paper: https://arxiv.org/abs/2203.01714
- Code: https://github.com/zh460045050/DA-WSOL_CVPR2022

<a name="ROD"></a>

# 雷达目标检测(Radar Object Detection)

**Exploiting Temporal Relations on Radar Perception for Autonomous Driving**
**利用雷达感知中的时空关系实现自动驾驶**

- Paper: https://arxiv.org/abs/2204.01184
- Code: None

<a name="HSI"></a>

# 高光谱图像重建(Hyperspectral Image Reconstruction)

**Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction**
**带 mask 指导的用于高光谱图像重建的卷积转换器**

- Paper: https://arxiv.org/abs/2111.07910
- Code: https://github.com/caiyuanhao1998/MST

<a name="Image-Stitching"></a>

# 图像拼接(Image Stitching)

**Deep Rectangling for Image Stitching: A Learning Baseline**
**深度对齐用于图像拼接：学习基线**

- Paper(Oral): https://arxiv.org/abs/2203.03831

- Code: https://github.com/nie-lang/DeepRectangling
- Dataset: https://github.com/nie-lang/DeepRectangling
- 中文解读：https://mp.weixin.qq.com/s/lp5AnrtO_9urp-Fv6Z0l2Q

<a name="Watermarking"></a>

# 水印(Watermarking)

**Deep 3D-to-2D Watermarking: Embedding Messages in 3D Meshes and Extracting Them from 2D Renderings**
**深度3D转2D水印：将3D模型中的信息嵌入到2D渲染中并提取出来**

- Paper: https://arxiv.org/abs/2104.13450
- Code: None

<a name="AC"></a>

# Action Counting

**TransRAC: Encoding Multi-scale Temporal Correlation with Transformers for Repetitive Action Counting**
**TRAC：使用变分自注意力机制编码多尺度时序相关性以实现重复动作计数**

- Paper(Oral): https://arxiv.org/abs/2204.01018
- Dataset: https://svip-lab.github.io/dataset/RepCount_dataset.html
- Code: https://github.com/SvipRepetitionCounting/TransRAC

<a name="GSR"></a>

# Grounded Situation Recognition

**Collaborative Transformers for Grounded Situation Recognition**
**合作变压器 for  grounded situation recognition**

- Paper: https://arxiv.org/abs/2203.16518
- Code: https://github.com/jhcho99/CoFormer

<a name="ZSL"></a>

# Zero-shot Learning

**Unseen Classes at a Later Time? No Problem**
**未知的类别何时可见？没问题**

- Paper: https://arxiv.org/abs/2203.16517
- Code: https://github.com/sumitramalagi/Unseen-classes-at-a-later-time

<a name="DeepFakes"></a>

# DeepFakes

**Detecting Deepfakes with Self-Blended Images**
**检测深度伪造现象通过自混合图像**

- Paper(Oral): https://arxiv.org/abs/2204.08376

- Code: https://github.com/mapooon/SelfBlendedImages

<a name="Datasets"></a>

# 数据集(Datasets)

**It's About Time: Analog Clock Reading in the Wild**
**关于时间：野外模拟时钟读取**

- Homepage: https://charigyang.github.io/abouttime/
- Paper: https://arxiv.org/abs/2111.09162
- Code: https://github.com/charigyang/itsabouttime
- Demo: https://youtu.be/cbiMACA6dRc

**Toward Practical Self-Supervised Monocular Indoor Depth Estimation**
**朝着实用自监督单目室内深度估计**

- Paper: https://arxiv.org/abs/2112.02306
- Code: None

**Kubric: A scalable dataset generator**
**库布里克：可扩展的数据集生成器**

- Paper: https://arxiv.org/abs/2203.03570
- Code: https://github.com/google-research/kubric
- 中文解读：https://mp.weixin.qq.com/s/mJ8HzY6C0GifxsErJIS3Mg

**Scribble-Supervised LiDAR Semantic Segmentation**
**手绘监督LiDAR语义分割**

- Paper: https://arxiv.org/abs/2203.08537
- Dataset: https://github.com/ouenal/scribblekitti

**Deep Rectangling for Image Stitching: A Learning Baseline**
**深度对齐用于图像拼接：学习基础**

- Paper(Oral): https://arxiv.org/abs/2203.03831
- Code: https://github.com/nie-lang/DeepRectangling
- Dataset: https://github.com/nie-lang/DeepRectangling
- 中文解读：https://mp.weixin.qq.com/s/lp5AnrtO_9urp-Fv6Z0l2Q

**ObjectFolder 2.0: A Multisensory Object Dataset for Sim2Real Transfer**
**ObjectFolder 2.0：一个用于模拟真实环境的多感官对象数据集**

- Homepage: https://ai.stanford.edu/~rhgao/objectfolder2.0/
- Paper: https://arxiv.org/abs/2204.02389
- Dataset: https://github.com/rhgao/ObjectFolder
- Demo：https://youtu.be/e5aToT3LkRA

**Shape from Polarization for Complex Scenes in the Wild**
**来自极地的形状，用于野外复杂场景的形状**

- Homepage: https://chenyanglei.github.io/sfpwild/index.html
- Paper: https://arxiv.org/abs/2112.11377
- Code: https://github.com/ChenyangLEI/sfp-wild

**Visible-Thermal UAV Tracking: A Large-Scale Benchmark and New Baseline**
**可见-热无人机追踪：大规模基准和新的基准**

- Homepage: https://zhang-pengyu.github.io/DUT-VTUAV/
- Paper: https://arxiv.org/abs/2204.04120

**TransRAC: Encoding Multi-scale Temporal Correlation with Transformers for Repetitive Action Counting**
**TRAC：使用变分自注意力机制编码多尺度时序相关性以实现重复动作计数**

- Paper(Oral): https://arxiv.org/abs/2204.01018
- Dataset: https://svip-lab.github.io/dataset/RepCount_dataset.html
- Code: https://github.com/SvipRepetitionCounting/TransRAC

**FineDiving: A Fine-grained Dataset for Procedure-aware Action Quality Assessment**
**精细潜水：一个用于程序感知动作质量评估的细粒度数据集**

- Paper(Oral): https://arxiv.org/abs/2204.03646
- Dataset: https://github.com/xujinglin/FineDiving
- Code: https://github.com/xujinglin/FineDiving
- 中文解读：https://mp.weixin.qq.com/s/8t12Y34eMNwvJr8PeryWXg

**Aesthetic Text Logo Synthesis via Content-aware Layout Inferring**
**美学文本逻辑合成通过内容感知布局推断**

- Paper: https://arxiv.org/abs/2204.02701
- Dataset: https://github.com/yizhiwang96/TextLogoLayout
- Code: https://github.com/yizhiwang96/TextLogoLayout

**DAIR-V2X: A Large-Scale Dataset for Vehicle-Infrastructure Cooperative 3D Object Detection**
**DAIR-V2X：用于车辆基础设施协同3D目标检测的大规模数据集**

- Homepage: https://thudair.baai.ac.cn/index
- Paper: https://arxiv.org/abs/2204.05575
- Code: https://github.com/AIR-THU/DAIR-V2X

**A New Dataset and Transformer for Stereoscopic Video Super-Resolution**
**新数据集和变压器用于立体视频超分辨率**

- Paper: https://arxiv.org/abs/2204.10039
- Code: https://github.com/H-deep/Trans-SVSR/
- Dataset: http://shorturl.at/mpwGX

**Putting People in their Place: Monocular Regression of 3D People in Depth**
**置人于其位：深度3D人物单眼回归**

- Homepage: https://arthur151.github.io/BEV/BEV.html
- Paper: https://arxiv.org/abs/2112.08274

- Code:https://github.com/Arthur151/ROMP
- Dataset: https://github.com/Arthur151/Relative_Human

**UBnormal: New Benchmark for Supervised Open-Set Video Anomaly Detection**
**UBnormal：监督开放集视频异常检测的新基准**

- Paper: https://arxiv.org/abs/2111.08644
- Dataset: https://github.com/lilygeorgescu/UBnormal

**DanceTrack: Multi-Object Tracking in Uniform Appearance and Diverse Motion**
**舞曲：在统一外观和多样动作中的多目标跟踪**

- Homepage: https://dancetrack.github.io
- Paper: https://arxiv.org/abs/2111.14690
- Dataset: https://github.com/DanceTrack/DanceTrack

**Visual Abductive Reasoning**
**视觉演绎推理**

- Paper: https://arxiv.org/abs/2203.14040
- Code: https://github.com/leonnnop/VAR

**Large-scale Video Panoptic Segmentation in the Wild: A Benchmark**
**野外的大规模视频全景分割：一个基准**

- Paper: https://github.com/VIPSeg-Dataset/VIPSeg-Dataset/blob/main/VIPSeg2022.pdf
- Code: https://github.com/VIPSeg-Dataset/VIPSeg-Dataset
- Dataset: https://github.com/VIPSeg-Dataset/VIPSeg-Dataset

**Ithaca365: Dataset and Driving Perception under Repeated and Challenging Weather Conditions**
**伊卡哈克365：在反复和具有挑战性的天气条件下，数据集和驾驶感知**

- Homepage: https://ithaca365.mae.cornell.edu/

- Paper: https://arxiv.org/abs/2208.01166

<a name="New-Tasks"></a>

# 新任务(New Task)

**Language-based Video Editing via Multi-Modal Multi-Level Transformer**
**基于语言的多模态多层次Transformer视频编辑**

- Paper: https://arxiv.org/abs/2104.01122
- Code: None

**It's About Time: Analog Clock Reading in the Wild**
**关于时间：野生环境中的模拟时钟读取**

- Homepage: https://charigyang.github.io/abouttime/
- Paper: https://arxiv.org/abs/2111.09162
- Code: https://github.com/charigyang/itsabouttime
- Demo: https://youtu.be/cbiMACA6dRc

**Splicing ViT Features for Semantic Appearance Transfer**
**拼接ViT特征以进行语义外观转移**

- Homepage: https://splice-vit.github.io/
- Paper: https://arxiv.org/abs/2201.00424
- Code: https://github.com/omerbt/Splice

**Visual Abductive Reasoning**
**视觉演绎推理**

- Paper: https://arxiv.org/abs/2203.14040
- Code: https://github.com/leonnnop/VAR

<a name="Others"></a>

# 其他(Others)

**Kubric: A scalable dataset generator**
**库布里克：可扩展的数据集生成器**

- Paper: https://arxiv.org/abs/2203.03570
- Code: https://github.com/google-research/kubric
- 中文解读：https://mp.weixin.qq.com/s/mJ8HzY6C0GifxsErJIS3Mg

**X-Trans2Cap: Cross-Modal Knowledge Transfer using Transformer for 3D Dense Captioning**
**X-Trans2Cap: 使用Transformer进行3D稠密补全的跨模态知识传递**

- Paper: https://arxiv.org/abs/2203.00843
- Code: https://github.com/CurryYuan/X-Trans2Cap

**Balanced MSE for Imbalanced Visual Regression**
**平衡不平衡视觉回归的均方误差**

- Paper(Oral): https://arxiv.org/abs/2203.16427
- Code: https://github.com/jiawei-ren/BalancedMSE

**SNUG: Self-Supervised Neural Dynamic Garments**
**SNUG：自监督神经动态衣物**

- Homepage: http://mslab.es/projects/SNUG/
- Paper(Oral): https://arxiv.org/abs/2204.02219
- Code: https://github.com/isantesteban/snug

**Shape from Polarization for Complex Scenes in the Wild**
**来自极光的野外复杂场景的形状**

- Homepage: https://chenyanglei.github.io/sfpwild/index.html
- Paper: https://arxiv.org/abs/2112.11377
- Code: https://github.com/ChenyangLEI/sfp-wild

**LASER: LAtent SpacE Rendering for 2D Visual Localization**
**激光：LAtent空间2D视觉定位渲染**

- Paper(Oral): https://arxiv.org/abs/2204.00157
- Code: None

**Single-Photon Structured Light**
**单光子结构光**

- Paper(Oral): https://arxiv.org/abs/2204.05300
- Code: None

**3DeformRS: Certifying Spatial Deformations on Point Clouds**
**3DeformRS：点云中空间变形的安全性证明**

- Paper: https://arxiv.org/abs/2204.05687
- Code: None

**Aesthetic Text Logo Synthesis via Content-aware Layout Inferring**
**美学文本生成通过内容感知布局推断**

- Paper: https://arxiv.org/abs/2204.02701
- Dataset: https://github.com/yizhiwang96/TextLogoLayout
- Code: https://github.com/yizhiwang96/TextLogoLayout

**Self-Supervised Predictive Learning: A Negative-Free Method for Sound Source Localization in Visual Scenes**
**负值无监督预测学习：用于视觉场景中声音源定位的负值无监督预测方法**

- Paper: https://arxiv.org/abs/2203.13412
- Code: https://github.com/zjsong/SSPL

**Robust and Accurate Superquadric Recovery: a Probabilistic Approach**
**鲁棒且准确的超级quadric恢复：一种概率方法**

- Paper(Oral): https://arxiv.org/abs/2111.14517
- Code: https://github.com/bmlklwx/EMS-superquadric_fitting

**Towards Bidirectional Arbitrary Image Rescaling: Joint Optimization and Cycle Idempotence**
**向双向任意图像缩放：联合优化和循环对称性**

- Paper: https://arxiv.org/abs/2203.00911
- Code: None

**Not All Tokens Are Equal: Human-centric Visual Analysis via Token Clustering Transformer**
**不是所有的令牌都是相等的：通过令牌聚类和Transformer进行以人为中心的视觉分析**

- Paper(Oral): https://arxiv.org/abs/2204.08680
- Code: https://github.com/zengwang430521/TCFormer

**DeepDPM: Deep Clustering With an Unknown Number of Clusters**
**DeepDPM：未知数量聚类的深度聚类**

- Paper: https://arxiv.org/abs/2203.14309
- Code: https://github.com/BGU-CS-VIL/DeepDPM

**ZeroCap: Zero-Shot Image-to-Text Generation for Visual-Semantic Arithmetic**
**零重载：用于视觉语义算术的零样本图像文本生成**

- Paper: https://arxiv.org/abs/2111.14447
- Code: https://github.com/YoadTew/zero-shot-image-to-text

**Proto2Proto: Can you recognize the car, the way I do?**
**proto2proto：你能识别我正在认的这辆车吗？**

- Paper: https://arxiv.org/abs/2204.11830
- Code: https://github.com/archmaester/proto2proto

**Putting People in their Place: Monocular Regression of 3D People in Depth**
**置人于其位：深度3D人物单眼回归**

- Homepage: https://arthur151.github.io/BEV/BEV.html
- Paper: https://arxiv.org/abs/2112.08274
- Code:https://github.com/Arthur151/ROMP
- Dataset: https://github.com/Arthur151/Relative_Human

**Light Field Neural Rendering**
**光场神经渲染**

- Homepage: https://light-field-neural-rendering.github.io/
- Paper(Oral): https://arxiv.org/abs/2112.09687
- Code: https://github.com/google-research/google-research/tree/master/light_field_neural_rendering

**Neural Texture Extraction and Distribution for Controllable Person Image Synthesis**
**神经纹理提取和分布用于控制人物图像生成**

- Paper: https://arxiv.org/abs/2204.06160
- Code: https://github.com/RenYurui/Neural-Texture-Extraction-Distribution

**Locality-Aware Inter-and Intra-Video Reconstruction for Self-Supervised Correspondence Learning**
**基于局部感知的视频内和外重建用于自监督对应学习**

- Paper: https://arxiv.org/abs/2203.14333
- Code: https://github.com/0liliulei/LIIR  
