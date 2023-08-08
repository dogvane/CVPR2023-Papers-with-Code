# CVPR 2023 论文和开源项目合集(Papers with Code)

[CVPR 2023](https://openaccess.thecvf.com/CVPR2023?day=all) 论文和开源项目合集(papers with code)！

**25.78% = 2360 / 9155**
**25.78% = 2360 / 9155**

CVPR 2023 decisions are now available on OpenReview! This year, wereceived a record number of **9155** submissions (a 12% increase over CVPR 2022), and accepted **2360** papers, for a 25.78% acceptance rate.

> 注0：项目来自于 https://github.com/amusi/CVPR2023-Papers-with-Code， 当前项目将原文里的标题用翻译工具转为中文，未做修订，仅作参考

> 注1：欢迎各位大佬提交issue，分享CVPR 2023论文和开源项目！
>
> 注2：关于往年CV顶会论文以及其他优质CV论文和大盘点，详见： https://github.com/amusi/daily-paper-computer-vision
>
> - [CVPR 2019](CVPR2019-Papers-with-Code.md)
> - [CVPR 2020](CVPR2020-Papers-with-Code.md)
> - [CVPR 2021](CVPR2021-Papers-with-Code.md)
> - [CVPR 2022](CVPR2022-Papers-with-Code.md)

如果你想了解最新最优质的的CV论文、开源项目和学习资料，欢迎扫码加入【CVer学术交流群】！互相学习，一起进步~ 

![](CVer学术交流群.png)

# 【CVPR 2023 论文开源目录】

- [Backbone](#Backbone)
- [CLIP](#CLIP)
- [MAE](#MAE)
- [GAN](#GAN)
- [GNN](#GNN)
- [MLP](#MLP)
- [NAS](#NAS)
- [OCR](#OCR)
- [NeRF](#NeRF)
- [DETR](#DETR)
- [Prompt](#Prompt)
- [Diffusion Models(扩散模型)](#Diffusion)
- [Avatars](#Avatars)
- [ReID(重识别)](#ReID)
- [长尾分布(Long-Tail)](#Long-Tail)
- [Vision Transformer](#Vision-Transformer)
- [视觉和语言(Vision-Language)](#VL)
- [自监督学习(Self-supervised Learning)](#SSL)
- [数据增强(Data Augmentation)](#DA)
- [目标检测(Object Detection)](#Object-Detection)
- [目标跟踪(Visual Tracking)](#VT)
- [语义分割(Semantic Segmentation)](#Semantic-Segmentation)
- [实例分割(Instance Segmentation)](#Instance-Segmentation)
- [全景分割(Panoptic Segmentation)](#Panoptic-Segmentation)
- [医学图像分割(Medical Image Segmentation)](#MIS)
- [视频目标分割(Video Object Segmentation)](#VOS)
- [视频实例分割(Video Instance Segmentation)](#VIS)
- [参考图像分割(Referring Image Segmentation)](#RIS)
- [图像抠图(Image Matting)](#Matting)
- [图像编辑(Image Editing)](#Image-Editing)
- [Low-level Vision](#LLV)
- [超分辨率(Super-Resolution)](#SR)
- [去噪(Denoising)](#Denoising)
- [去模糊(Deblur)](#Deblur)
- [3D点云(3D Point Cloud)](#3D-Point-Cloud)
- [3D目标检测(3D Object Detection)](#3DOD)
- [3D语义分割(3D Semantic Segmentation)](#3DSS)
- [3D目标跟踪(3D Object Tracking)](#3D-Object-Tracking)
- [3D语义场景补全(3D Semantic Scene Completion)](#3DSSC)
- [3D配准(3D Registration)](#3D-Registration)
- [3D人体姿态估计(3D Human Pose Estimation)](#3D-Human-Pose-Estimation)
- [3D人体Mesh估计(3D Human Mesh Estimation)](#3D-Human-Pose-Estimation)
- [医学图像(Medical Image)](#Medical-Image)
- [图像生成(Image Generation)](#Image-Generation)
- [视频生成(Video Generation)](#Video-Generation)
- [视频理解(Video Understanding)](#Video-Understanding)
- [行为检测(Action Detection)](#Action-Detection)
- [文本检测(Text Detection)](#Text-Detection)
- [知识蒸馏(Knowledge Distillation)](#KD)
- [模型剪枝(Model Pruning)](#Pruning)
- [图像压缩(Image Compression)](#IC)
- [异常检测(Anomaly Detection)](#AD)
- [三维重建(3D Reconstruction)](#3D-Reconstruction)
- [深度估计(Depth Estimation)](#Depth-Estimation)
- [轨迹预测(Trajectory Prediction)](#TP)
- [车道线检测(Lane Detection)](#Lane-Detection)
- [图像描述(Image Captioning)](#Image-Captioning)
- [视觉问答(Visual Question Answering)](#VQA)
- [手语识别(Sign Language Recognition)](#SLR)
- [视频预测(Video Prediction)](#Video-Prediction)
- [新视点合成(Novel View Synthesis)](#NVS)
- [Zero-Shot Learning(零样本学习)](#ZSL)
- [立体匹配(Stereo Matching)](#Stereo-Matching)
- [特征匹配(Feature Matching)](#Feature-Matching)
- [场景图生成(Scene Graph Generation)](#SGG)
- [隐式神经表示(Implicit Neural Representations)](#INR)
- [图像质量评价(Image Quality Assessment)](#IQA)
- [数据集(Datasets)](#Datasets)
- [新任务(New Tasks)](#New-Tasks)
- [其他(Others)](#Others)

<a name="Backbone"></a>

# Backbone

**Integrally Pre-Trained Transformer Pyramid Networks** 

- Paper: https://arxiv.org/abs/2211.12735
- Code: https://github.com/sunsmarterjie/iTPN

**Stitchable Neural Networks**
**缝合神经网络**

- Homepage: https://snnet.github.io/
- Paper: https://arxiv.org/abs/2302.06586
- Code: https://github.com/ziplab/SN-Net

**Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks**
**跑步，不走路：追逐更高的FLOPS，为更快的神经网络加速**

- Paper: https://arxiv.org/abs/2303.03667
- Code: https://github.com/JierunChen/FasterNet 

**BiFormer: Vision Transformer with Bi-Level Routing Attention**
**BiFormer: 具有双层路由注意力机制的视觉变分器**

- Paper: None
- Code: https://github.com/rayleizhu/BiFormer 

**DeepMAD: Mathematical Architecture Design for Deep Convolutional Neural Network**
**深度MAD: 深度卷积神经网络的数学架构设计**

- Paper: https://arxiv.org/abs/2303.02165
- Code: https://github.com/alibaba/lightweight-neural-architecture-search 

**Vision Transformer with Super Token Sampling**
**具有超令牌采样的视觉变压器**

- Paper: https://arxiv.org/abs/2211.11167
- Code: https://github.com/hhb072/SViT

**Hard Patches Mining for Masked Image Modeling**
**硬质补丁挖掘用于遮罩图像建模**

- Paper: None
- Code: None

**SMPConv: Self-moving Point Representations for Continuous Convolution**
**SMPConv: 连续卷积的自移动点表示**

- Paper: https://arxiv.org/abs/2304.02330
- Code: https://github.com/sangnekim/SMPConv

**Making Vision Transformers Efficient from A Token Sparsification View**
**从令牌稀疏化的角度优化视觉变压器**

- Paper: https://arxiv.org/abs/2303.08685
- Code: https://github.com/changsn/STViT-R 

<a name="CLIP"></a>

# CLIP

**GALIP: Generative Adversarial CLIPs for Text-to-Image Synthesis**
**GALIP：用于文本到图像生成的生成对抗网络**

- Paper: https://arxiv.org/abs/2301.12959
- Code: https://github.com/tobran/GALIP

**DeltaEdit: Exploring Text-free Training for Text-driven Image Manipulation**
**ΔEdit：探索无需文本驱动的图像编辑**

- Paper: https://arxiv.org/abs/2303.06285
- Code: https://github.com/Yueming6568/DeltaEdit 

<a name="MAE"></a>

# MAE

**Learning 3D Representations from 2D Pre-trained Models via Image-to-Point Masked Autoencoders** 

- Paper: https://arxiv.org/abs/2212.06785
- Code: https://github.com/ZrrSkywalker/I2P-MAE

**Generic-to-Specific Distillation of Masked Autoencoders**
**隐藏的自动编码器泛化到特定**

- Paper: https://arxiv.org/abs/2302.14771
- Code: https://github.com/pengzhiliang/G2SD

<a name="GAN"></a>

# GAN

**DeltaEdit: Exploring Text-free Training for Text-driven Image Manipulation**
**ΔEdit：探索无文本训练，用于文本驱动的图像操作**

- Paper: https://arxiv.org/abs/2303.06285
- Code: https://github.com/Yueming6568/DeltaEdit 

<a name="NeRF"></a>

# NeRF

**NoPe-NeRF: Optimising Neural Radiance Field with No Pose Prior**
**NoPe-NeRF: 优化神经辐射场，无需姿态先验**

- Home: https://nope-nerf.active.vision/
- Paper: https://arxiv.org/abs/2212.07388
- Code: None

**Latent-NeRF for Shape-Guided Generation of 3D Shapes and Textures**
**潜在的NeRF用于形状引导生成3D形状和纹理**

- Paper: https://arxiv.org/abs/2211.07600
- Code: https://github.com/eladrich/latent-nerf

**NeRF in the Palm of Your Hand: Corrective Augmentation for Robotics via Novel-View Synthesis**
**手掌中的NeRF：通过新颖的视图合成为机器人提供校正增强**

- Paper: https://arxiv.org/abs/2301.08556
- Code: None

**Panoptic Lifting for 3D Scene Understanding with Neural Fields**
**带有神经场的3D场景理解中的全景提升**

- Homepage: https://nihalsid.github.io/panoptic-lifting/
- Paper: https://arxiv.org/abs/2212.09802
- Code: None

**NeRFLiX: High-Quality Neural View Synthesis by Learning a Degradation-Driven Inter-viewpoint MiXer**
**NeRFLiX: 由学习退化驱动的交互式多视角器**

- Homepage: https://redrock303.github.io/nerflix/
- Paper: https://arxiv.org/abs/2303.06919 
- Code: None

**HNeRV: A Hybrid Neural Representation for Videos**
**HNeRV：一种用于视频的混合神经表示**

- Homepage: https://haochen-rye.github.io/HNeRV
- Paper: https://arxiv.org/abs/2304.02633
- Code: https://github.com/haochen-rye/HNeRV

<a name="DETR"></a>

# DETR

**DETRs with Hybrid Matching**
**DETRs with Hybrid Matching**

- Paper: https://arxiv.org/abs/2207.13080
- Code: https://github.com/HDETR

<a name="Prompt"></a>

# Prompt

**Diversity-Aware Meta Visual Prompting**
**多样性感知的元视觉提示**

- Paper: https://arxiv.org/abs/2303.08138
- Code: https://github.com/shikiw/DAM-VP 

<a name="NAS"></a>

# NAS

**PA&DA: Jointly Sampling PAth and DAta for Consistent NAS**
**PA&DA：联合采样PA路径和DAta以实现一致的NAS**

- Paper: https://arxiv.org/abs/2302.14772
- Code: https://github.com/ShunLu91/PA-DA

<a name="Avatars"></a>

# Avatars

**Structured 3D Features for Reconstructing Relightable and Animatable Avatars**
**结构化的3D特征用于重建可重用和可动画化角色**

- Homepage: https://enriccorona.github.io/s3f/
- Paper: https://arxiv.org/abs/2212.06820
- Code: None
- Demo: https://www.youtube.com/watch?v=mcZGcQ6L-2s

**Learning Personalized High Quality Volumetric Head Avatars from Monocular RGB Videos**
**从单目RGB视频中学取个性化高质量体积头戴式表盘**

- Homepage: https://augmentedperception.github.io/monoavatar/
- Paper: https://arxiv.org/abs/2304.01436

<a name="ReID"></a>

# ReID(重识别)

**Clothing-Change Feature Augmentation for Person Re-Identification**
**服装变更特征增强用于人员重新识别**

- Paper: None
- Code: None

**MSINet: Twins Contrastive Search of Multi-Scale Interaction for Object ReID**
**MSINet：多尺度交互对比搜索对象的重新识别**

- Paper: https://arxiv.org/abs/2303.07065
- Code: https://github.com/vimar-gu/MSINet

**Shape-Erased Feature Learning for Visible-Infrared Person Re-Identification**
**形状消去特征学习可视红外人体重新识别**

- Paper: https://arxiv.org/abs/2304.04205
- Code: None

**Large-scale Training Data Search for Object Re-identification**
**大规模训练数据搜索用于对象识别**

- Paper: https://arxiv.org/abs/2303.16186
- Code: https://github.com/yorkeyao/SnP 

<a name="Diffusion"></a>

# Diffusion Models(扩散模型)

**Video Probabilistic Diffusion Models in Projected Latent Space** 

- Homepage: https://sihyun.me/PVDM/
- Paper: https://arxiv.org/abs/2302.07685
- Code: https://github.com/sihyun-yu/PVDM

**Solving 3D Inverse Problems using Pre-trained 2D Diffusion Models**
**使用预训练的2D扩散模型解决3D反演问题**

- Paper: https://arxiv.org/abs/2211.10655
- Code: None

**Imagic: Text-Based Real Image Editing with Diffusion Models**
**异想天开：基于扩散模型的文本驱动式图像编辑**

- Homepage: https://imagic-editing.github.io/
- Paper: https://arxiv.org/abs/2210.09276
- Code: None

**Parallel Diffusion Models of Operator and Image for Blind Inverse Problems**
**盲目问题中的算子异质扩散模型**

- Paper: https://arxiv.org/abs/2211.10656
- Code: None

**DiffRF: Rendering-guided 3D Radiance Field Diffusion**
**DiffRF: 基于渲染的3D辐射场扩散**

- Homepage: https://sirwyver.github.io/DiffRF/
- Paper: https://arxiv.org/abs/2212.01206
- Code: None

**MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation**
**MM-Diffusion：学习联合音频和视频生成的多模态扩散模型**

- Paper: https://arxiv.org/abs/2212.09478
- Code: https://github.com/researchmm/MM-Diffusion

**HouseDiffusion: Vector Floorplan Generation via a Diffusion Model with Discrete and Continuous Denoising**
**HouseDiffusion：通过具有离散和连续去噪的扩散模型生成矢量 floorplan。**

- Homepage: https://aminshabani.github.io/housediffusion/
- Paper: https://arxiv.org/abs/2211.13287
- Code: https://github.com/aminshabani/house_diffusion 

**TrojDiff: Trojan Attacks on Diffusion Models with Diverse Targets**
**TrojDiff：针对具有多样目标的扩散模型的 Trojan 攻击**

- Paper: https://arxiv.org/abs/2303.05762
- Code: https://github.com/chenweixin107/TrojDiff

**Back to the Source: Diffusion-Driven Adaptation to Test-Time Corruption**
**回到原文：基于扩散的适应性测试时间损坏**

- Paper: https://arxiv.org/abs/2207.03442
- Code: https://github.com/shiyegao/DDA 

**DR2: Diffusion-based Robust Degradation Remover for Blind Face Restoration**
**DR2：基于扩散的盲人脸修复增强去除器**

- Paper: https://arxiv.org/abs/2303.06885
- Code: None

**Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion**
**跟踪与速度：通过引导轨迹扩散可控制的行人动画**

- Homepage: https://nv-tlabs.github.io/trace-pace/
- Paper: https://arxiv.org/abs/2304.01893
- Code: None

**Generative Diffusion Prior for Unified Image Restoration and Enhancement**
**生成对抗扩散优先用于统一图像恢复和增强**

- Paper: https://arxiv.org/abs/2304.01247
- Code: None

**Conditional Image-to-Video Generation with Latent Flow Diffusion Models**
**条件图像到视频生成：潜在流动扩散模型**

- Paper: https://arxiv.org/abs/2303.13744
- Code: https://github.com/nihaomiao/CVPR23_LFDM 

<a name="Long-Tail"></a>

# 长尾分布(Long-Tail)

**Long-Tailed Visual Recognition via Self-Heterogeneous Integration with Knowledge Excavation**
**长尾视觉识别通过自下而上的异质整合与知识挖掘**

- Paper: https://arxiv.org/abs/2304.01279
- Code: None

<a name="Vision-Transformer"></a>

# Vision Transformer

**Integrally Pre-Trained Transformer Pyramid Networks** 

- Paper: https://arxiv.org/abs/2211.12735
- Code: https://github.com/sunsmarterjie/iTPN

**Mask3D: Pre-training 2D Vision Transformers by Learning Masked 3D Priors**
**Mask3D: 使用遮罩3D生成器进行预训练的2D视觉变压器**

- Homepage: https://niessnerlab.org/projects/hou2023mask3d.html
- Paper: https://arxiv.org/abs/2302.14746
- Code: None

**Learning Trajectory-Aware Transformer for Video Super-Resolution**
**学习轨迹感知的变分自注意力器用于视频超分辨率**

- Paper: https://arxiv.org/abs/2204.04216
- Code: https://github.com/researchmm/TTVSR

**Vision Transformers are Parameter-Efficient Audio-Visual Learners**
**视觉变分器是一种参数高效的音频-视觉学习者**

- Homepage: https://yanbo.ml/project_page/LAVISH/
- Code: https://github.com/GenjiB/LAVISH

**Where We Are and What We're Looking At: Query Based Worldwide Image Geo-localization Using Hierarchies and Scenes**
**我们所处位置以及我们关注的焦点：基于层次和场景的全世界图像地理定位查询**

- Paper: https://arxiv.org/abs/2303.04249
- Code: None

**DSVT: Dynamic Sparse Voxel Transformer with Rotated Sets**
**DSVT：动态稀疏 voxel 转换器 with 旋转集**

- Paper: https://arxiv.org/abs/2301.06051
- Code: https://github.com/Haiyang-W/DSVT

**DeepSolo: Let Transformer Decoder with Explicit Points Solo for Text Spotting**
**DeepSolo：为文本检测提供显式点生成器**

- Paper: https://arxiv.org/abs/2211.10772
- Code link: https://github.com/ViTAE-Transformer/DeepSolo

**BiFormer: Vision Transformer with Bi-Level Routing Attention**
**BiFormer: 具有双层路由注意力机制的视觉Transformer**

- Paper: https://arxiv.org/abs/2303.08810
- Code: https://github.com/rayleizhu/BiFormer

**Vision Transformer with Super Token Sampling**
**具有超令牌采样的Transformer视觉模型**

- Paper: https://arxiv.org/abs/2211.11167
- Code: https://github.com/hhb072/SViT

**BEVFormer v2: Adapting Modern Image Backbones to Bird's-Eye-View Recognition via Perspective Supervision**
**BEVFormer v2：通过视角监督调整现代图像骨干网络以实现 bird's-eye-view 识别**

- Paper: https://arxiv.org/abs/2211.10439
- Code: None

**BAEFormer: Bi-directional and Early Interaction Transformers for Bird’s Eye View Semantic Segmentation**
**BAEFormer: 双向和早期交互变压器用于鸟瞰式语义分割**

- Paper: None
- Code: None

**Visual Dependency Transformers: Dependency Tree Emerges from Reversed Attention**
**视觉依赖变体：依赖树从反转注意力中产生**

- Paper: https://arxiv.org/abs/2304.03282
- Code: None

**Making Vision Transformers Efficient from A Token Sparsification View**
**从令牌稀疏化的角度优化视觉Transformer**

- Paper: https://arxiv.org/abs/2303.08685
- Code: https://github.com/changsn/STViT-R 

<a name="VL"></a>

# 视觉和语言(Vision-Language)

**GIVL: Improving Geographical Inclusivity of Vision-Language Models with Pre-Training Methods**
**GIVL: 改进视觉语言模型在地理包容性方面的表现，通过预训练方法**

- Paper: https://arxiv.org/abs/2301.01893
- Code: None

**Teaching Structured Vision&Language Concepts to Vision&Language Models**
**教授结构化视觉与语言概念给视觉与语言模型**

- Paper: https://arxiv.org/abs/2211.11733
- Code: None

**Uni-Perceiver v2: A Generalist Model for Large-Scale Vision and Vision-Language Tasks**
**Uni-Perceiver v2：大规模视觉和视觉语言任务的泛模型**

- Paper: https://arxiv.org/abs/2211.09808
- Code: https://github.com/fundamentalvision/Uni-Perceiver

**Towards Generalisable Video Moment Retrieval: Visual-Dynamic Injection to Image-Text Pre-Training**
**向可扩展性视频 moment 检索：视觉动态注入到图像文本预训练**

- Paper: https://arxiv.org/abs/2303.00040
- Code: None

**CapDet: Unifying Dense Captioning and Open-World Detection Pretraining**
**捕获目标检测：统一稠密标注和开放世界检测预训练**

- Paper: https://arxiv.org/abs/2303.02489
- Code: None

**FAME-ViL: Multi-Tasking Vision-Language Model for Heterogeneous Fashion Tasks**
**FAME-ViL：多任务视觉语言模型，用于异质时尚任务**

- Paper: https://arxiv.org/abs/2303.02483
- Code: None

**Meta-Explore: Exploratory Hierarchical Vision-and-Language Navigation Using Scene Object Spectrum Grounding**
**元探索：使用场景对象谱聚类进行探究式层次视觉语言导航**

- Homepage: https://rllab-snu.github.io/projects/Meta-Explore/doc.html
- Paper: https://arxiv.org/abs/2303.04077
- Code: None

**All in One: Exploring Unified Video-Language Pre-training**
**万物皆可统一：探索统一视频语言预训练**

- Paper: https://arxiv.org/abs/2203.07303
- Code: https://github.com/showlab/all-in-one

**Position-guided Text Prompt for Vision Language Pre-training**
**面向位置的文本提示，用于视觉语言预训练**

- Paper: https://arxiv.org/abs/2212.09737
- Code: https://github.com/sail-sg/ptp

**EDA: Explicit Text-Decoupling and Dense Alignment for 3D Visual Grounding**
**EDA：显式文本解耦和稠密对齐 3D 视觉定位**

- Paper: https://arxiv.org/abs/2209.14941
- Code: https://github.com/yanmin-wu/EDA

**CapDet: Unifying Dense Captioning and Open-World Detection Pretraining**
**捕获目标检测：统一稠密标注和开放世界检测预训练**

- Paper: https://arxiv.org/abs/2303.02489
- Code: None

**FAME-ViL: Multi-Tasking Vision-Language Model for Heterogeneous Fashion Tasks**
**FAME-ViL：多任务视觉语言模型，用于异质时尚任务**

- Paper: https://arxiv.org/abs/2303.02483
- Code: https://github.com/BrandonHanx/FAME-ViL

**Align and Attend: Multimodal Summarization with Dual Contrastive Losses**
**对齐和参加：带有双对比损失的多元摘要**

- Homepage: https://boheumd.github.io/A2Summ/
- Paper: https://arxiv.org/abs/2303.07284
- Code: https://github.com/boheumd/A2Summ

**Multi-Modal Representation Learning with Text-Driven Soft Masks**
**多模态表示学习与文本驱动的软掩码**

- Paper: https://arxiv.org/abs/2304.00719
- Code: None

**Learning to Name Classes for Vision and Language Models**
**学习为视觉和语言模型命名类别**

- Paper: https://arxiv.org/abs/2304.01830
- Code: None

<a name="Object-Detection"></a>

# 目标检测(Object Detection)

**YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors**
**YOLOv7: 训练有素的教学套件为实时物体检测创造了新境界**

- Paper: https://arxiv.org/abs/2207.02696
- Code: https://github.com/WongKinYiu/yolov7

**DETRs with Hybrid Matching**
**DETR with Hybrid Matching**

- Paper: https://arxiv.org/abs/2207.13080
- Code: https://github.com/HDETR

**Enhanced Training of Query-Based Object Detection via Selective Query Recollection**
**基于选择性查询回调的查询为基础对象检测的增强训练**

- Paper: https://arxiv.org/abs/2212.07593
- Code: https://github.com/Fangyi-Chen/SQR

**Object-Aware Distillation Pyramid for Open-Vocabulary Object Detection**
**面向对象的蒸馏 Pyramid 模型 Open-Vocabulary 对象检测**

- Paper: https://arxiv.org/abs/2303.05892
- Code: https://github.com/LutingWang/OADP

<a name="VT"></a>

# 目标跟踪(Object Tracking)

**Simple Cues Lead to a Strong Multi-Object Tracker**
**简单的提示导致一个强大的多目标跟踪器**

- Paper: https://arxiv.org/abs/2206.04656
- Code: None

**Joint Visual Grounding and Tracking with Natural Language Specification**
**联合视觉基础和跟踪与自然语言规格**

- Paper: https://arxiv.org/abs/2303.12027
- Code: https://github.com/lizhou-cs/JointNLT 

<a name="Semantic-Segmentation"></a>

# 语义分割(Semantic Segmentation)

**Efficient Semantic Segmentation by Altering Resolutions for Compressed Videos**
**高效的语义分割通过改变压缩视频的分辨率来实现**

- Paper: https://arxiv.org/abs/2303.07224
- Code: https://github.com/THU-LYJ-Lab/AR-Seg

**FREDOM: Fairness Domain Adaptation Approach to Semantic Scene Understanding**
**自由：公平领域自适应方法用于语义场景理解**

- Paper: https://arxiv.org/abs/2304.02135
- Code: https://github.com/uark-cviu/FREDOM

<a name="MIS"></a>

# 医学图像分割(Medical Image Segmentation)

**Label-Free Liver Tumor Segmentation**
**标签无肝肿瘤分割**

- Paper: https://arxiv.org/abs/2303.14869
- Code: https://github.com/MrGiovanni/SyntheticTumors

**Directional Connectivity-based Segmentation of Medical Images**
**基于方向连接的医学图像分割**

- Paper: https://arxiv.org/abs/2304.00145
- Code: https://github.com/Zyun-Y/DconnNet

**Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation**
**双向复制粘贴半监督医学图像分割**

- Paper: https://arxiv.org/abs/2305.00673
- Code: https://github.com/DeepMed-Lab-ECNU/BCP

**Devil is in the Queries: Advancing Mask Transformers for Real-world Medical Image Segmentation and Out-of-Distribution Localization**
**恶魔在查询中：用于真实医学图像分割和异构局部化的先进面具变压器**

- Paper: https://arxiv.org/abs/2304.00212
- Code: None

**Fair Federated Medical Image Segmentation via Client Contribution Estimation**
**公平联合医学图像分割通过客户端贡献估计**

- Paper: https://arxiv.org/abs/2303.16520
- Code: https://github.com/NVIDIA/NVFlare/tree/dev/research/fed-ce

**Ambiguous Medical Image Segmentation using Diffusion Models**
**模糊医学图像分割使用扩散模型**

- Homepage: https://aimansnigdha.github.io/cimd/
- Paper: https://arxiv.org/abs/2304.04745
- Code: https://github.com/aimansnigdha/Ambiguous-Medical-Image-Segmentation-using-Diffusion-Models

**Orthogonal Annotation Benefits Barely-supervised Medical Image Segmentation**
**线性注释：几乎没有监督的医学图像分割**

- Paper: https://arxiv.org/abs/2303.13090
- Code: https://github.com/HengCai-NJU/DeSCO

**MagicNet: Semi-Supervised Multi-Organ Segmentation via Magic-Cube Partition and Recovery**
**MagicNet: 半监督多器官分割通过魔方划分和恢复**

- Paper: https://arxiv.org/abs/2301.01767
- Code: https://github.com/DeepMed-Lab-ECNU/MagicNet

**MCF: Mutual Correction Framework for Semi-Supervised Medical Image Segmentation**
**MCF: 半监督医疗图像分割相互校正框架**

- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Wang_MCF_Mutual_Correction_Framework_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.html
- Code: https://github.com/WYC-321/MCF

**Rethinking Few-Shot Medical Segmentation: A Vector Quantization View**
**重新思考少量样本医疗分割：一个向量化视角**

- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Huang_Rethinking_Few-Shot_Medical_Segmentation_A_Vector_Quantization_View_CVPR_2023_paper.html
- Code: None

**Pseudo-label Guided Contrastive Learning for Semi-supervised Medical Image Segmentation**
**伪标签引导对比学习半监督医学图像分割**

- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Basak_Pseudo-Label_Guided_Contrastive_Learning_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.html
- Code: https://github.com/hritam-98/PatchCL-MedSeg

**SDC-UDA: Volumetric Unsupervised Domain Adaptation Framework for Slice-Direction Continuous Cross-Modality Medical Image Segmentation**
**SDC-UDA：用于医学图像分割的卷积无监督领域自适应框架**

- Paper: https://arxiv.org/abs/2305.11012
- Code: None

**DoNet: Deep De-overlapping Network for Cytology Instance Segmentation**
**DoNet: 深度重叠网络用于细胞学实例分割**

- Paper: https://arxiv.org/abs/2303.14373
- Code: https://github.com/DeepDoNet/DoNet

<a name="VOS"></a>

# 视频目标分割（Video Object Segmentation）

**Two-shot Video Object Segmentation**
**双击视频对象分割**

- Paper: https://arxiv.org/abs/2303.12078
- Code: https://github.com/yk-pku/Two-shot-Video-Object-Segmentation

 **Under Video Object Segmentation Section**

- Paper: https://arxiv.org/abs/2303.07815
- Code: None

<a name="VIS"></a>

# 视频实例分割(Video Instance Segmentation)

**Mask-Free Video Instance Segmentation**
**无 mask 视频实例分割**

- Paper: https://arxiv.org/abs/2303.15904
- Code: https://github.com/SysCV/MaskFreeVis 

<a name="RIS"></a>

# 参考图像分割(Referring Image Segmentation )

**PolyFormer: Referring Image Segmentation as Sequential Polygon Generation**
**PolyFormer：将图像分割视为序列多边形生成**

- Paper: https://arxiv.org/abs/2302.07387 

- Code: None

<a name="3D-Point-Cloud"></a>

# 3D点云(3D-Point-Cloud)

**Physical-World Optical Adversarial Attacks on 3D Face Recognition**
**物理世界对3D人脸识别的光学对抗攻击**

- Paper: https://arxiv.org/abs/2205.13412
- Code: https://github.com/PolyLiYJ/SLAttack.git

**IterativePFN: True Iterative Point Cloud Filtering**
**迭代PFN：真实的迭代点云过滤**

- Paper: https://arxiv.org/abs/2304.01529
- Code: https://github.com/ddsediri/IterativePFN

**Attention-based Point Cloud Edge Sampling**
**基于注意力的点云边缘采样**

- Homepage: https://junweizheng93.github.io/publications/APES/APES.html 
- Paper: https://arxiv.org/abs/2302.14673
- Code: https://github.com/JunweiZheng93/APES

<a name="3DOD"></a>

# 3D目标检测(3D Object Detection)

**DSVT: Dynamic Sparse Voxel Transformer with Rotated Sets**
**DSVT：动态稀疏立方体变换器**

- Paper: https://arxiv.org/abs/2301.06051
- Code: https://github.com/Haiyang-W/DSVT 

**FrustumFormer: Adaptive Instance-aware Resampling for Multi-view 3D Detection**
**FrustumFormer: 适应性实例感测多视角3D检测**

- Paper:  https://arxiv.org/abs/2301.04467
- Code: None

**3D Video Object Detection with Learnable Object-Centric Global Optimization**
**3D 视频对象检测与可学习的目标中心全局优化**

- Paper: None
- Code: None

**Hierarchical Supervision and Shuffle Data Augmentation for 3D Semi-Supervised Object Detection**
**分层监督和随机数据增强对于3D半监督目标检测**

- Paper: https://arxiv.org/abs/2304.01464
- Code: https://github.com/azhuantou/HSSDA

<a name="3DOD"></a>

# 3D语义分割(3D Semantic Segmentation)

**Less is More: Reducing Task and Model Complexity for 3D Point Cloud Semantic Segmentation**
**少即是多：为3D点云语义分割减少任务和模型复杂性**

- Paper: https://arxiv.org/abs/2303.11203
- Code: https://github.com/l1997i/lim3d 

<a name="3DSSC"></a>

# 3D语义场景补全(3D Semantic Scene Completion)

- Paper: https://arxiv.org/abs/2302.12251
- Code: https://github.com/NVlabs/VoxFormer 

<a name="3D-Registration"></a>

# 3D配准(3D Registration)

**Robust Outlier Rejection for 3D Registration with Variational Bayes**
**稳健的异常排斥对于3D配准使用变分贝叶斯**

- Paper: https://arxiv.org/abs/2304.01514
- Code: https://github.com/Jiang-HB/VBReg

<a name="3D-Human-Pose-Estimation"></a>

# 3D人体姿态估计(3D Human Pose Estimation)

<a name="3D-Human-Mesh-Estimation"></a>

# 3D人体Mesh估计(3D Human Mesh Estimation)

**3D Human Mesh Estimation from Virtual Markers**
**3D 人体网格估计从虚拟标记**

- Paper: https://arxiv.org/abs/2303.11726
- Code: https://github.com/ShirleyMaxx/VirtualMarker 

<a name="LLV"></a>

# Low-level Vision

**Causal-IR: Learning Distortion Invariant Representation for Image Restoration from A Causality Perspective**
**因果-IR：从因果视角学习图像恢复的失真不变表示**

- Paper: https://arxiv.org/abs/2303.06859
- Code: https://github.com/lixinustc/Casual-IR-DIL 

**Burstormer: Burst Image Restoration and Enhancement Transformer**
**暴风雨：暴风雨图像恢复和增强变压器**

- Paper: https://arxiv.org/abs/2304.01194
- Code: http://github.com/akshaydudhane16/Burstormer

<a name="SR"></a>

# 超分辨率(Video Super-Resolution)

**Super-Resolution Neural Operator**
**超级分辨率神经算子**

- Paper: https://arxiv.org/abs/2303.02584
- Code: https://github.com/2y7c3/Super-Resolution-Neural-Operator 

## 视频超分辨率

**Learning Trajectory-Aware Transformer for Video Super-Resolution**
**学习轨迹感知的变分自注意力器用于视频超分辨率**

- Paper: https://arxiv.org/abs/2204.04216

- Code: https://github.com/researchmm/TTVSR

Denoising<a name="Denoising"></a>

# 去噪(Denoising)

## 图像去噪(Image Denoising)

**Masked Image Training for Generalizable Deep Image Denoising**
**隐藏图像训练用于一般深度图像去噪**

- Paper- : https://arxiv.org/abs/2303.13132
- Code: https://github.com/haoyuc/MaskedDenoising 

<a name="Image-Generation"></a>

# 图像生成(Image Generation)

**GALIP: Generative Adversarial CLIPs for Text-to-Image Synthesis**
**GALIP: 生成对抗网络CLIPs用于文本到图像合成**

- Paper: https://arxiv.org/abs/2301.12959
- Code: https://github.com/tobran/GALIP 

**MAGE: MAsked Generative Encoder to Unify Representation Learning and Image Synthesis**
**MAGE：要求生成对抗网络统一表示学习和图像生成**

- Paper: https://arxiv.org/abs/2211.09117
- Code: https://github.com/LTH14/mage

**Toward Verifiable and Reproducible Human Evaluation for Text-to-Image Generation**
**面向可验证和可重复的人类评估文本到图像生成**

- Paper: https://arxiv.org/abs/2304.01816
- Code: None

**Few-shot Semantic Image Synthesis with Class Affinity Transfer**
**少样本语义图像生成与类别迁移**

- Paper: https://arxiv.org/abs/2304.02321
- Code: None

**TopNet: Transformer-based Object Placement Network for Image Compositing**
**顶级网络：基于Transformer的图像合成物体放置网络**

- Paper: https://arxiv.org/abs/2304.03372
- Code: None

<a name="Video-Generation"></a>

# 视频生成(Video Generation)

**MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation**
**MM-Diffusion：学习联合音频和视频生成的多模态扩散模型**

- Paper: https://arxiv.org/abs/2212.09478
- Code: https://github.com/researchmm/MM-Diffusion

**Conditional Image-to-Video Generation with Latent Flow Diffusion Models**
**条件图像到视频生成与潜在流动扩散模型**

- Paper: https://arxiv.org/abs/2303.13744
- Code: https://github.com/nihaomiao/CVPR23_LFDM 

<a name="Video-Understanding"></a>

# 视频理解(Video Understanding)

**Learning Transferable Spatiotemporal Representations from Natural Script Knowledge**
**从自然脚本知识中学习可转移的时空表示**

- Paper: https://arxiv.org/abs/2209.15280
- Code: https://github.com/TencentARC/TVTS

**Frame Flexible Network**
**框架灵活网络**

- Paper: https://arxiv.org/abs/2303.14817
- Code: https://github.com/BeSpontaneous/FFN

**Masked Motion Encoding for Self-Supervised Video Representation Learning**
**遮蔽运动编码用于自监督视频表示学习**

- Paper: https://arxiv.org/abs/2210.06096
- Code: https://github.com/XinyuSun/MME

**MARLIN: Masked Autoencoder for facial video Representation LearnING**
**玛琳：面部视频表征的遮蔽自动编码器学习**

- Paper: https://arxiv.org/abs/2211.06627
- Code: https://github.com/ControlNet/MARLIN 

<a name="Action-Detection"></a>

# 行为检测(Action Detection)

**TriDet: Temporal Action Detection with Relative Boundary Modeling**
**TriDet: 基于相对边界模型的时间动作检测**

- Paper: https://arxiv.org/abs/2303.07347
- Code: https://github.com/dingfengshi/TriDet 

<a name="Text-Detection"></a>

# 文本检测(Text Detection)

**DeepSolo: Let Transformer Decoder with Explicit Points Solo for Text Spotting**
**DeepSolo：带有显式点对文本检测的Transformer解码器**

- Paper: https://arxiv.org/abs/2211.10772
- Code link: https://github.com/ViTAE-Transformer/DeepSolo

<a name="KD"></a>

# 知识蒸馏(Knowledge Distillation)

**Learning to Retain while Acquiring: Combating Distribution-Shift in Adversarial Data-Free Knowledge Distillation**
**在对抗性数据无监督知识蒸馏中学会保留：对抗分布漂移的防御**

- Paper: https://arxiv.org/abs/2302.14290
- Code: None

**Generic-to-Specific Distillation of Masked Autoencoders**
**隐藏变量自动编码器的泛化到特定蒸馏**

- Paper: https://arxiv.org/abs/2302.14771
- Code: https://github.com/pengzhiliang/G2SD

<a name="Pruning"></a>

# 模型剪枝(Model Pruning)

**DepGraph: Towards Any Structural Pruning**
**DepGraph：迈向任何结构剪枝**

- Paper: https://arxiv.org/abs/2301.12900
- Code: https://github.com/VainF/Torch-Pruning 

<a name="IC"></a>

# 图像压缩(Image Compression)

**Context-Based Trit-Plane Coding for Progressive Image Compression**
**基于内容的渐进图像压缩**

- Paper: https://arxiv.org/abs/2303.05715
- Code: https://github.com/seungminjeon-github/CTC

<a name="AD"></a>

# 异常检测(Anomaly Detection)

**Deep Feature In-painting for Unsupervised Anomaly Detection in X-ray Images**
**深度特征回填在X射线图像的无监督异常检测中的应用**

- Paper: https://arxiv.org/abs/2111.13495
- Code: https://github.com/tiangexiang/SQUID 

<a name="3D-Reconstruction"></a>

# 三维重建(3D Reconstruction)

**OReX: Object Reconstruction from Planar Cross-sections Using Neural Fields**
**OReX：从平面截面中使用神经场进行物体重构**

- Paper: https://arxiv.org/abs/2211.12886
- Code: None

**SparsePose: Sparse-View Camera Pose Regression and Refinement**
**稀疏姿态：稀疏视图相机姿态回归和细化**

- Paper: https://arxiv.org/abs/2211.16991
- Code: None

**NeuDA: Neural Deformable Anchor for High-Fidelity Implicit Surface Reconstruction**
**神经可塑锚：用于高精度隐式表面重构的高维神经网络**

- Paper: https://arxiv.org/abs/2303.02375
- Code: None

**Vid2Avatar: 3D Avatar Reconstruction from Videos in the Wild via Self-supervised Scene Decomposition**
**Vid2Avatar：通过自监督场景分解从野外视频重建3D Avatar**

- Homepage: https://moygcc.github.io/vid2avatar/
- Paper: https://arxiv.org/abs/2302.11566
- Code: https://github.com/MoyGcc/vid2avatar
- Demo: https://youtu.be/EGi47YeIeGQ

**To fit or not to fit: Model-based Face Reconstruction and Occlusion Segmentation from Weak Supervision**
**是否符合：基于模型的面部重建和遮盖分割弱监督**

- Paper: https://arxiv.org/abs/2106.09614
- Code: https://github.com/unibas-gravis/Occlusion-Robust-MoFA

**Structural Multiplane Image: Bridging Neural View Synthesis and 3D Reconstruction**
**结构多层图像：连接神经视图合成和3D重建**

- Paper: https://arxiv.org/abs/2303.05937
- Code: None

**3D Cinemagraphy from a Single Image**
**单张图像的3D电影短片**

- Homepage: https://xingyi-li.github.io/3d-cinemagraphy/
- Paper: https://arxiv.org/abs/2303.05724
- Code: https://github.com/xingyi-li/3d-cinemagraphy

**Revisiting Rotation Averaging: Uncertainties and Robust Losses**
**重新回顾旋转平均：不确定的性和鲁棒损失**

- Paper: https://arxiv.org/abs/2303.05195
- Code https://github.com/zhangganlin/GlobalSfMpy 

**FFHQ-UV: Normalized Facial UV-Texture Dataset for 3D Face Reconstruction**
**FFHQ-UV: 用于3D人脸重建的标准化面部UV纹理数据集**

- Paper: https://arxiv.org/abs/2211.13874
- Code: https://github.com/csbhr/FFHQ-UV 

**A Hierarchical Representation Network for Accurate and Detailed Face Reconstruction from In-The-Wild Images**
**一种从野外图像中进行准确详细人脸重建的层次表示网络**

- Homepage: https://younglbw.github.io/HRN-homepage/ 

- Paper: https://arxiv.org/abs/2302.14434
- Code: https://github.com/youngLBW/HRN

<a name="Depth-Estimation"></a>

# 深度估计(Depth Estimation)

**Lite-Mono: A Lightweight CNN and Transformer Architecture for Self-Supervised Monocular Depth Estimation**
**轻量级单通道深度估计：一个用于自监督单通道深度估计的轻量级CNN和Transformer架构**

- Paper: https://arxiv.org/abs/2211.13202
- Code: https://github.com/noahzn/Lite-Mono 

<a name="TP"></a>

# 轨迹预测(Trajectory Prediction)

**IPCC-TP: Utilizing Incremental Pearson Correlation Coefficient for Joint Multi-Agent Trajectory Prediction**
**IPCC-TP：联合多智能体轨迹预测的增量Pearson相关系数**

- Paper:  https://arxiv.org/abs/2303.00575
- Code: None

**EqMotion: Equivariant Multi-agent Motion Prediction with Invariant Interaction Reasoning**
**均衡多智能体运动预测与不变交互推理**

- Paper: https://arxiv.org/abs/2303.10876
- Code: https://github.com/MediaBrain-SJTU/EqMotion 

<a name="Lane-Detection"></a>

# 车道线检测(Lane Detection)

**Anchor3DLane: Learning to Regress 3D Anchors for Monocular 3D Lane Detection**
**锚3D通道：学习为单目3D车道检测训练3D锚点**

- Paper: https://arxiv.org/abs/2301.02371
- Code: https://github.com/tusen-ai/Anchor3DLane

**BEV-LaneDet: An Efficient 3D Lane Detection Based on Virtual Camera via Key-Points**
**BEV-LaneDet：基于虚拟相机的关键点高效3D车道检测**

- Paper:  https://arxiv.org/abs/2210.06006v3 
- Code:  https://github.com/gigo-team/bev_lane_det 

<a name="Image-Captioning"></a>

# 图像描述(Image Captioning)

**ConZIC: Controllable Zero-shot Image Captioning by Sampling-Based Polishing**
**ConZIC：基于采样的可控制零样本图像标题识别**

- Paper: https://arxiv.org/abs/2303.02437
- Code: Node

**Cross-Domain Image Captioning with Discriminative Finetuning**
**跨域图像标题识别与有监督微调**

- Paper: https://arxiv.org/abs/2304.01662
- Code: None

**Model-Agnostic Gender Debiased Image Captioning**
**模型无关的性别偏置图像标题生成**

- Paper: https://arxiv.org/abs/2304.03693
- Code: None

<a name="VQA"></a>

# 视觉问答(Visual Question Answering)

**MixPHM: Redundancy-Aware Parameter-Efficient Tuning for Low-Resource Visual Question Answering**
**混合PHM：针对低资源视觉问答的冗余性感知参数高效调整**

- Paper:  https://arxiv.org/abs/2303.01239
- Code: https://github.com/jingjing12110/MixPHM

<a name="SLR"></a>

# 手语识别(Sign Language Recognition)

**Continuous Sign Language Recognition with Correlation Network**
**连续手语识别与相关网络**

Paper: https://arxiv.org/abs/2303.03202

Code: https://github.com/hulianyuyy/CorrNet

<a name="Video-Prediction"></a>

# 视频预测(Video Prediction)

**MOSO: Decomposing MOtion, Scene and Object for Video Prediction**
**MOSO: 分解运动、场景和对象，用于视频预测**

- Paper: https://arxiv.org/abs/2303.03684
- Code: https://github.com/anonymous202203/MOSO

<a name="NVS"></a>

# 新视点合成(Novel View Synthesis)

 **3D Video Loops from Asynchronous Input**

- Homepage: https://limacv.github.io/VideoLoop3D_web/
- Paper: https://arxiv.org/abs/2303.05312
- Code: https://github.com/limacv/VideoLoop3D 

<a name="ZSL"></a>

# Zero-Shot Learning(零样本学习)

**Bi-directional Distribution Alignment for Transductive Zero-Shot Learning**
**双向分布对齐对于导电零样本学习**

- Paper: https://arxiv.org/abs/2303.08698
- Code: https://github.com/Zhicaiwww/Bi-VAEGAN

**Semantic Prompt for Few-Shot Learning**
**少样本学习语义提示**

- Paper: None
- Code: None

<a name="Stereo-Matching"></a>

# 立体匹配(Stereo Matching)

**Iterative Geometry Encoding Volume for Stereo Matching**
**迭代几何编码体积用于立体匹配**

- Paper: https://arxiv.org/abs/2303.06615
- Code: https://github.com/gangweiX/IGEV

**Learning the Distribution of Errors in Stereo Matching for Joint Disparity and Uncertainty Estimation**
**联合差异和不确定性估计中，学习立体匹配中错误分布**

- Paper: https://arxiv.org/abs/2304.00152
- Code: None

<a name="Feature-Matching"></a>

# 特征匹配(Feature Matching)

**Adaptive Spot-Guided Transformer for Consistent Local Feature Matching**
**适应性空间引导的Transformer模型用于一致的局部特征匹配**

- Homepage: [https://astr2023.github.io](https://astr2023.github.io/) 
- Paper: https://arxiv.org/abs/2303.16624
- Code: https://github.com/ASTR2023/ASTR

<a name="SGG"></a>

# 场景图生成(Scene Graph Generation)

**Prototype-based Embedding Network for Scene Graph Generation**
**基于原型表征的网络用于场景图生成**

- Paper: https://arxiv.org/abs/2303.07096
- Code: None

<a name="INR"></a>

# 隐式神经表示(Implicit Neural Representations)

**Polynomial Implicit Neural Representations For Large Diverse Datasets**
**多项式隐式神经表示对于大规模多样化数据集**

- Paper: https://arxiv.org/abs/2303.11424
- Code: https://github.com/Rajhans0/Poly_INR

<a name="IQA"></a>

# 图像质量评价(Image Quality Assessment)

**Re-IQA: Unsupervised Learning for Image Quality Assessment in the Wild**
**重新计算：在野外进行图像质量评估的无监督学习**

- Paper: https://arxiv.org/abs/2304.00451
- Code: None

<a name="Datasets"></a>

# 数据集(Datasets)

**Human-Art: A Versatile Human-Centric Dataset Bridging Natural and Artificial Scenes**
**人类艺术：一个将自然和人工场景结合在一起的多功能人类中心化数据集**

- Paper: https://arxiv.org/abs/2303.02760
- Code: None

**Align and Attend: Multimodal Summarization with Dual Contrastive Losses**
**对齐和参加：具有双对比损失的多模态摘要**

- Homepage: https://boheumd.github.io/A2Summ/
- Paper: https://arxiv.org/abs/2303.07284
- Code: https://github.com/boheumd/A2Summ

**GeoNet: Benchmarking Unsupervised Adaptation across Geographies**
**地理网络：地理环境中的无监督适应性基准研究**

- Homepage: https://tarun005.github.io/GeoNet/
- Paper: https://arxiv.org/abs/2303.15443

**CelebV-Text: A Large-Scale Facial Text-Video Dataset**
**赛博-文本：大规模面部文本-视频数据集**

- Homepage: https://celebv-text.github.io/
- Paper: https://arxiv.org/abs/2303.14717

<a name="Others"></a>

# 其他(Others)

**Interactive Segmentation as Gaussian Process Classification**
**交互式分割作为高斯过程分类**

- Paper: https://arxiv.org/abs/2302.14578
- Code: None

**Backdoor Attacks Against Deep Image Compression via Adaptive Frequency Trigger**
**针对深度图像压缩的暗网攻击通过自适应频率触发**

- Paper: https://arxiv.org/abs/2302.14677
- Code: None

**SplineCam: Exact Visualization and Characterization of Deep Network Geometry and Decision Boundaries**
**SplineCam：深度网络几何和决策边界精确可视化和表征**

- Homepage: http://bit.ly/splinecam
- Paper: https://arxiv.org/abs/2302.12828
- Code: None

**SCOTCH and SODA: A Transformer Video Shadow Detection Framework**
**SCOTCH 和 SODA：一种用于视频阴影检测的 Transformer 框架**

- Paper: https://arxiv.org/abs/2211.06885
- Code: None

**DeepMapping2: Self-Supervised Large-Scale LiDAR Map Optimization**
**DeepMapping2: 深度学习自监督大规模LiDAR地图优化**

- Homepage: https://ai4ce.github.io/DeepMapping2/
- Paper: https://arxiv.org/abs/2212.06331
- None: https://github.com/ai4ce/DeepMapping2

**RelightableHands: Efficient Neural Relighting of Articulated Hand Models**
**可重新调整的手：高效的手部关节模型重新定位**

- Homepage: https://sh8.io/#/relightable_hands
- Paper: https://arxiv.org/abs/2302.04866
- Code: None

**Token Turing Machines**
**令牌图灵机**

- Paper: https://arxiv.org/abs/2211.09119
- Code: None

**Single Image Backdoor Inversion via Robust Smoothed Classifiers**
**单张图像背景门通过鲁棒平滑分类器**

- Paper: https://arxiv.org/abs/2303.00215
- Code: https://github.com/locuslab/smoothinv

**To fit or not to fit: Model-based Face Reconstruction and Occlusion Segmentation from Weak Supervision**
**是否适合：基于模型的面部重建和遮挡分割**

- Paper: https://arxiv.org/abs/2106.09614
- Code: https://github.com/unibas-gravis/Occlusion-Robust-MoFA

**HOOD: Hierarchical Graphs for Generalized Modelling of Clothing Dynamics**
**HOOD：用于衣物动力学一般建模的分层图形**

- Homepage: https://dolorousrtur.github.io/hood/
- Paper: https://arxiv.org/abs/2212.07242
- Code: https://github.com/dolorousrtur/hood
- Demo: https://www.youtube.com/watch?v=cBttMDPrUYY

**A Whac-A-Mole Dilemma: Shortcuts Come in Multiples Where Mitigating One Amplifies Others**
**挥棒子困境：简化的方法在多个层面上存在，一个简化的方法会强化其他简化的方法。**

- Paper: https://arxiv.org/abs/2212.04825
- Code: https://github.com/facebookresearch/Whac-A-Mole.git

**RelightableHands: Efficient Neural Relighting of Articulated Hand Models**
**重新定位手部模型：高效的手部关节重新定位**

- Homepage: https://sh8.io/#/relightable_hands
- Paper: https://arxiv.org/abs/2302.04866
- Code: None
- Demo: https://sh8.io/static/media/teacher_video.923d87957fe0610730c2.mp4

**Neuro-Modulated Hebbian Learning for Fully Test-Time Adaptation**
**神经调节的Hebbian学习用于完全测试时间适应**

- Paper: https://arxiv.org/abs/2303.00914
- Code: None

**Demystifying Causal Features on Adversarial Examples and Causal Inoculation for Robust Network by Adversarial Instrumental Variable Regression**
**解码对抗性样本和因果接种对鲁棒网络的因果特征**

- Paper: https://arxiv.org/abs/2303.01052
- Code: None

**UniDexGrasp: Universal Robotic Dexterous Grasping via Learning Diverse Proposal Generation and Goal-Conditioned Policy**
**UniDexGrasp: 统一机器人灵活抓取通过学习多样化的建议生成和目标条件策略**

- Paper: https://arxiv.org/abs/2303.00938
- Code: None

**Disentangling Orthogonal Planes for Indoor Panoramic Room Layout Estimation with Cross-Scale Distortion Awareness**
**解开两正交平面为室内全景房间布局估计的二维扭曲感知**

- Paper: https://arxiv.org/abs/2303.00971
- Code: https://github.com/zhijieshen-bjtu/DOPNet

**Learning Neural Parametric Head Models**
**学习神经参数高维头模型**

- Homepage: https://simongiebenhain.github.io/NPHM)
- Paper: https://arxiv.org/abs/2212.02761
- Code: None

**A Meta-Learning Approach to Predicting Performance and Data Requirements**
**元学习方法预测性能和数据需求**

- Paper: https://arxiv.org/abs/2303.01598
- Code: None

**MACARONS: Mapping And Coverage Anticipation with RGB Online Self-Supervision**
**MACARONS：使用RGB在线自监督实现图层映射和覆盖预测**

- Homepage: https://imagine.enpc.fr/~guedona/MACARONS/
- Paper: https://arxiv.org/abs/2303.03315
- Code: None

**Masked Images Are Counterfactual Samples for Robust Fine-tuning**
**遮罩图像是用于稳健微调的反事实样本**

- Paper: https://arxiv.org/abs/2303.03052
- Code: None

**HairStep: Transfer Synthetic to Real Using Strand and Depth Maps for Single-View 3D Hair Modeling**
**hairstep：使用丝线和深度图将合成头发模型转换为真实效果**

- Paper: https://arxiv.org/abs/2303.02700
- Code: None

**Decompose, Adjust, Compose: Effective Normalization by Playing with Frequency for Domain Generalization**
**分解，调整，组合：通过改变频率进行域推广的有效归一化**

- Paper: https://arxiv.org/abs/2303.02328
- Code: None

**Gradient Norm Aware Minimization Seeks First-Order Flatness and Improves Generalization**
**梯度范数感知最小化寻求一阶平滑性并改进泛化**

- Paper: https://arxiv.org/abs/2303.03108
- Code: None

**Unlearnable Clusters: Towards Label-agnostic Unlearnable Examples**
**无监督学习聚类：指向标签无关的无监督学习示例**

- Paper: https://arxiv.org/abs/2301.01217
- Code: https://github.com/jiamingzhang94/Unlearnable-Clusters 

**Where We Are and What We're Looking At: Query Based Worldwide Image Geo-localization Using Hierarchies and Scenes**
**我们所处之处和我们所关注的：基于层次和场景的查询全球图像地理定位**

- Paper: https://arxiv.org/abs/2303.04249
- Code: None

**UniHCP: A Unified Model for Human-Centric Perceptions**
**UniHCP：统一的人类感知模型**

- Paper: https://arxiv.org/abs/2303.02936
- Code: https://github.com/OpenGVLab/UniHCP

**CUDA: Convolution-based Unlearnable Datasets**
**CUDA：基于卷积的不学习数据集**

- Paper: https://arxiv.org/abs/2303.04278
- Code: https://github.com/vinusankars/Convolution-based-Unlearnability

**Masked Images Are Counterfactual Samples for Robust Fine-tuning**
**遮蔽图像是对齐的样本用于稳健的微调**

- Paper: https://arxiv.org/abs/2303.03052
- Code: None

**AdaptiveMix: Robust Feature Representation via Shrinking Feature Space**
**自适应混合：通过收缩特征空间实现稳健特征表示**

- Paper: https://arxiv.org/abs/2303.01559
- Code: https://github.com/WentianZhang-ML/AdaptiveMix 

**Physical-World Optical Adversarial Attacks on 3D Face Recognition**
**针对3D人脸识别的物理世界光学对抗攻击**

- Paper: https://arxiv.org/abs/2205.13412
- Code: https://github.com/PolyLiYJ/SLAttack.git

**DPE: Disentanglement of Pose and Expression for General Video Portrait Editing**
**DPE： pose 和表达的解码**

- Paper: https://arxiv.org/abs/2301.06281
- Code: https://carlyx.github.io/DPE/ 

**SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation**
**悲伤谈话者：为时尚化音频驱动的单张图像聊天机器人学习真实的3D运动系数**

- Paper: https://arxiv.org/abs/2211.12194
- Code: https://github.com/Winfredy/SadTalker

**Intrinsic Physical Concepts Discovery with Object-Centric Predictive Models**
**内生物理概念发现与以对象为中心的预测模型**

- Paper: None
- Code: None

**Sharpness-Aware Gradient Matching for Domain Generalization**
**深度可分离梯度匹配在领域泛化**

- Paper: None
- Code: https://github.com/Wang-pengfei/SAGM

**Mind the Label-shift for Augmentation-based Graph Out-of-distribution Generalization**
**注意：基于增强的图卷积自监督泛化**

- Paper: None
- Code: None

**Blind Video Deflickering by Neural Filtering with a Flawed Atlas**
**盲视频去抖动通过神经滤波器与有缺陷的地图**

- Homepage:  https://chenyanglei.github.io/deflicker 
- Paper: None
- Code: None

**RiDDLE: Reversible and Diversified De-identification with Latent Encryptor**
**RiDDLE：可逆和多元化的反向身份识别与潜在加密器**

- Paper: None
- Code:  https://github.com/ldz666666/RiDDLE 

**PoseExaminer: Automated Testing of Out-of-Distribution Robustness in Human Pose and Shape Estimation**
**姿势评估器：人类姿势和形状估计中的分布式外模式鲁棒性自动测试**

- Paper: https://arxiv.org/abs/2303.07337
- Code: None

**Upcycling Models under Domain and Category Shift**
**领域和类别迁移下的升级模型**

- Paper: https://arxiv.org/abs/2303.07110
- Code: https://github.com/ispc-lab/GLC

**Modality-Agnostic Debiasing for Single Domain Generalization**
**模式无关的消元处理单领域泛化**

- Paper: https://arxiv.org/abs/2303.07123
- Code: None

**Progressive Open Space Expansion for Open-Set Model Attribution**
**为Open-Set模型分配渐进式开放空间扩展**

- Paper: https://arxiv.org/abs/2303.06877
- Code: None

**Dynamic Neural Network for Multi-Task Learning Searching across Diverse Network Topologies**
**动态神经网络在多任务学习中搜索 diverse网络拓扑**

- Paper: https://arxiv.org/abs/2303.06856
- Code: None

**GFPose: Learning 3D Human Pose Prior with Gradient Fields**
**GFPose: 学习3D人体姿态优先与梯度场**

- Paper: https://arxiv.org/abs/2212.08641
- Code: https://github.com/Embracing/GFPose 

**PRISE: Demystifying Deep Lucas-Kanade with Strongly Star-Convex Constraints for Multimodel Image Alignment**
**PRISE：通过强星凸约束多模态图像对齐解开深度洛卡西德**

- Paper: https://arxiv.org/abs/2303.11526
- Code: https://github.com/Zhang-VISLab

**Sketch2Saliency: Learning to Detect Salient Objects from Human Drawings**
**轮廓2识别：从人类绘画中检测突出对象**

- Paper: https://arxiv.org/abs/2303.11502
- Code: None

**Boundary Unlearning**
**边界消融**

- Paper: https://arxiv.org/abs/2303.11570
- Code: None

**ImageNet-E: Benchmarking Neural Network Robustness via Attribute Editing**
**图像网络-E：通过属性编辑测试神经网络的鲁棒性**

- Paper: https://arxiv.org/abs/2303.17096
- Code: https://github.com/alibaba/easyrobust

**Zero-shot Model Diagnosis**
**零样本模型诊断**

- Paper: https://arxiv.org/abs/2303.15441
- Code: None

**GeoNet: Benchmarking Unsupervised Adaptation across Geographies**
**GeoNet：无监督适应性在地理空间上的基准测试**

- Homepage: https://tarun005.github.io/GeoNet/
- Paper: https://arxiv.org/abs/2303.15443

**Quantum Multi-Model Fitting**
**量子多模型拟合**

- Paper: https://arxiv.org/abs/2303.15444
- Code: https://github.com/FarinaMatteo/qmmf

**DivClust: Controlling Diversity in Deep Clustering**
**DivClust：控制深度聚类中的多样性**

- Paper: https://arxiv.org/abs/2304.01042
- Code: None

**Neural Volumetric Memory for Visual Locomotion Control**
**神经体积记忆视觉定位控制**

- Homepage: https://rchalyang.github.io/NVM
- Paper: https://arxiv.org/abs/2304.01201
- Code: https://rchalyang.github.io/NVM

**MonoHuman: Animatable Human Neural Field from Monocular Video**
**MonoHuman: 从单眼视频中的可动画人类神经场**

- Homepage: https://yzmblog.github.io/projects/MonoHuman/
- Paper: https://arxiv.org/abs/2304.02001
- Code: https://github.com/Yzmblog/MonoHuman

**Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion**
**追踪与速度：通过引导轨迹扩散控制的行人动画**

- Homepage: https://nv-tlabs.github.io/trace-pace/
- Paper: https://arxiv.org/abs/2304.01893
- Code: None

**Bridging the Gap between Model Explanations in Partially Annotated Multi-label Classification**
**部分注释多标签分类模型中的模型解释 gap 之间的桥梁**

- Paper: https://arxiv.org/abs/2304.01804
- Code: None

**HyperCUT: Video Sequence from a Single Blurry Image using Unsupervised Ordering**
**HyperCUT：从单个模糊图像中生成视频序列的无监督排序**

- Paper: https://arxiv.org/abs/2304.01686
- Code: None

**On the Stability-Plasticity Dilemma of Class-Incremental Learning**
**在类别递增学习中的稳定性-塑性困境**

- Paper: https://arxiv.org/abs/2304.01663
- Code: None

**Defending Against Patch-based Backdoor Attacks on Self-Supervised Learning**
**防御基于补丁的自我监督学习中的补丁后门攻击**

- Paper: https://arxiv.org/abs/2304.01482
- Code: None

**VNE: An Effective Method for Improving Deep Representation by Manipulating Eigenvalue Distribution**
**VNE：通过操纵特征值分布的有效方法来改进深度表示**

- Paper: https://arxiv.org/abs/2304.01434
- Code: https://github.com/jaeill/CVPR23-VNE

**Detecting and Grounding Multi-Modal Media Manipulation**
**检测和定位多模态媒体操作**

- Homepage: https://rshaojimmy.github.io/Projects/MultiModal-DeepFake
- Paper: https://arxiv.org/abs/2304.02556
- Code: https://github.com/rshaojimmy/MultiModal-DeepFake

**Meta-causal Learning for Single Domain Generalization**
**元因果学习用于单领域推广**

- Paper: https://arxiv.org/abs/2304.03709
- Code: None

**Disentangling Writer and Character Styles for Handwriting Generation**
**解开手写风格生成中作家与角色的风格分离**

- Paper: https://arxiv.org/abs/2303.14736
- Code: https://github.com/dailenson/SDT

**DexArt: Benchmarking Generalizable Dexterous Manipulation with Articulated Objects**
**DexArt：使用关节对象进行灵活度基准测试**

- Homepage: https://www.chenbao.tech/dexart/

- Code: https://github.com/Kami-code/dexart-release

**Hidden Gems: 4D Radar Scene Flow Learning Using Cross-Modal Supervision**
**隐藏的宝藏：使用跨模态监督的4D雷达场景流学习**

- Homepage: https://toytiny.github.io/publication/23-cmflow-cvpr/index.html 
- Paper: https://arxiv.org/abs/2303.00462
- Code: https://github.com/Toytiny/CMFlow

**Marching-Primitives: Shape Abstraction from Signed Distance Function**
**Marching-Primitives: 从签名距离函数中提取形状**

- Paper: https://arxiv.org/abs/2303.13190
- Code: https://github.com/ChirikjianLab/Marching-Primitives

**Towards Trustable Skin Cancer Diagnosis via Rewriting Model's Decision**
**皮肤癌诊断的可靠途径是通过重写模型的决策**

- Paper: https://arxiv.org/abs/2303.00885
- Code: None
