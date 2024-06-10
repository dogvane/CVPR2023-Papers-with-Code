# CVPR 2023 论文和开源项目合集(Papers with Code)

[CVPR 2023](https://openaccess.thecvf.com/CVPR2023?day=all) 论文和开源项目合集(papers with code)！

**25.78% = 2360 / 9155**
**25.78% 等于 2360 除以 9155。**

CVPR 2023 decisions are now available on OpenReview! This year, wereceived a record number of **9155** submissions (a 12% increase over CVPR 2022), and accepted **2360** papers, for a 25.78% acceptance rate.


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
**可缝制的神经网络**

- Homepage: https://snnet.github.io/
- Paper: https://arxiv.org/abs/2302.06586
- Code: https://github.com/ziplab/SN-Net

**Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks**
**快跑，别走：追求更高的FLOPS以实现更快的神经网络**

- Paper: https://arxiv.org/abs/2303.03667
- Code: https://github.com/JierunChen/FasterNet 

**BiFormer: Vision Transformer with Bi-Level Routing Attention**
**BiFormer：具有双级路由注意力的视觉Transformer**

- Paper: None
- Code: https://github.com/rayleizhu/BiFormer 

**DeepMAD: Mathematical Architecture Design for Deep Convolutional Neural Network**
**DeepMAD：深度卷积神经网络数学架构设计**

- Paper: https://arxiv.org/abs/2303.02165
- Code: https://github.com/alibaba/lightweight-neural-architecture-search 

**Vision Transformer with Super Token Sampling**
**超级标记采样视觉Transformer**

- Paper: https://arxiv.org/abs/2211.11167
- Code: https://github.com/hhb072/SViT

**Hard Patches Mining for Masked Image Modeling**
**硬质块状物采矿用于掩码图像建模**

- Paper: None
- Code: None

**SMPConv: Self-moving Point Representations for Continuous Convolution**
**SMPConv：用于连续卷积的自动移动点表示**

- Paper: https://arxiv.org/abs/2304.02330
- Code: https://github.com/sangnekim/SMPConv

**Making Vision Transformers Efficient from A Token Sparsification View**
**从词元稀疏化视角提高视觉Transformer的效率**

- Paper: https://arxiv.org/abs/2303.08685
- Code: https://github.com/changsn/STViT-R 

<a name="CLIP"></a>

# CLIP

**GALIP: Generative Adversarial CLIPs for Text-to-Image Synthesis**
**GALIP：用于文本到图像合成的生成对抗CLIPs**

- Paper: https://arxiv.org/abs/2301.12959
- Code: https://github.com/tobran/GALIP

**DeltaEdit: Exploring Text-free Training for Text-driven Image Manipulation**
**DeltaEdit：探索基于文本驱动的图像处理的文本无关训练**

- Paper: https://arxiv.org/abs/2303.06285
- Code: https://github.com/Yueming6568/DeltaEdit 

<a name="MAE"></a>

# MAE

**Learning 3D Representations from 2D Pre-trained Models via Image-to-Point Masked Autoencoders** 

- Paper: https://arxiv.org/abs/2212.06785
- Code: https://github.com/ZrrSkywalker/I2P-MAE

**Generic-to-Specific Distillation of Masked Autoencoders**
**通用到特定蒸馏的掩码自动编码器**

- Paper: https://arxiv.org/abs/2302.14771
- Code: https://github.com/pengzhiliang/G2SD

<a name="GAN"></a>

# GAN

**DeltaEdit: Exploring Text-free Training for Text-driven Image Manipulation**
**DeltaEdit：探索基于文本驱动的图像处理的无文本训练方法**

- Paper: https://arxiv.org/abs/2303.06285
- Code: https://github.com/Yueming6568/DeltaEdit 

<a name="NeRF"></a>

# NeRF

**NoPe-NeRF: Optimising Neural Radiance Field with No Pose Prior**
**NoPe-NeRF：无需姿态先验的神经辐射场优化**

- Home: https://nope-nerf.active.vision/
- Paper: https://arxiv.org/abs/2212.07388
- Code: None

**Latent-NeRF for Shape-Guided Generation of 3D Shapes and Textures**
**潜式NeRF用于形状引导的3D形状和纹理生成**

- Paper: https://arxiv.org/abs/2211.07600
- Code: https://github.com/eladrich/latent-nerf

**NeRF in the Palm of Your Hand: Corrective Augmentation for Robotics via Novel-View Synthesis**
**掌中NeRF：通过新颖视角合成实现的机器人校正增强**

- Paper: https://arxiv.org/abs/2301.08556
- Code: None

**Panoptic Lifting for 3D Scene Understanding with Neural Fields**
**全景提升用于基于神经场的3D场景理解**

- Homepage: https://nihalsid.github.io/panoptic-lifting/
- Paper: https://arxiv.org/abs/2212.09802
- Code: None

**NeRFLiX: High-Quality Neural View Synthesis by Learning a Degradation-Driven Inter-viewpoint MiXer**
**NeRFLiX：通过学习一个由降级驱动的跨视角混合器实现高质量神经视图合成**

- Homepage: https://redrock303.github.io/nerflix/
- Paper: https://arxiv.org/abs/2303.06919 
- Code: None

**HNeRV: A Hybrid Neural Representation for Videos**
**HNeRV：一种用于视频的混合神经网络表示**

- Homepage: https://haochen-rye.github.io/HNeRV
- Paper: https://arxiv.org/abs/2304.02633
- Code: https://github.com/haochen-rye/HNeRV

<a name="DETR"></a>

# DETR

**DETRs with Hybrid Matching**
**混合匹配的DETRs**

- Paper: https://arxiv.org/abs/2207.13080
- Code: https://github.com/HDETR

<a name="Prompt"></a>

# Prompt

**Diversity-Aware Meta Visual Prompting**
**多样性感知元视觉提示**

- Paper: https://arxiv.org/abs/2303.08138
- Code: https://github.com/shikiw/DAM-VP 

<a name="NAS"></a>

# NAS

**PA&DA: Jointly Sampling PAth and DAta for Consistent NAS**
**PA&DA：联合采样路径和数据以实现一致的神经架构搜索**

- Paper: https://arxiv.org/abs/2302.14772
- Code: https://github.com/ShunLu91/PA-DA

<a name="Avatars"></a>

# Avatars

**Structured 3D Features for Reconstructing Relightable and Animatable Avatars**
**用于重建可重光照和可动画化的虚拟角色的结构化3D特征**

- Homepage: https://enriccorona.github.io/s3f/
- Paper: https://arxiv.org/abs/2212.06820
- Code: None
- Demo: https://www.youtube.com/watch?v=mcZGcQ6L-2s

**Learning Personalized High Quality Volumetric Head Avatars from Monocular RGB Videos**
**从单目RGB视频中学习个性化高质量体积头部虚拟形象**

- Homepage: https://augmentedperception.github.io/monoavatar/
- Paper: https://arxiv.org/abs/2304.01436

<a name="ReID"></a>

# ReID(重识别)

**Clothing-Change Feature Augmentation for Person Re-Identification**
**服装更换特征增强的人体重识别**

- Paper: None
- Code: None

**MSINet: Twins Contrastive Search of Multi-Scale Interaction for Object ReID**
**MSINet：多尺度交互对象重识别的孪生对比搜索**

- Paper: https://arxiv.org/abs/2303.07065
- Code: https://github.com/vimar-gu/MSINet

**Shape-Erased Feature Learning for Visible-Infrared Person Re-Identification**
**形状擦除可见光-红外人体重识别特征学习**

- Paper: https://arxiv.org/abs/2304.04205
- Code: None

**Large-scale Training Data Search for Object Re-identification**
**大规模训练数据搜索用于物体再识别**

- Paper: https://arxiv.org/abs/2303.16186
- Code: https://github.com/yorkeyao/SnP 

<a name="Diffusion"></a>

# Diffusion Models(扩散模型)

**Video Probabilistic Diffusion Models in Projected Latent Space** 

- Homepage: https://sihyun.me/PVDM/
- Paper: https://arxiv.org/abs/2302.07685
- Code: https://github.com/sihyun-yu/PVDM

**Solving 3D Inverse Problems using Pre-trained 2D Diffusion Models**
**使用预训练的二维扩散模型解决三维逆问题**

- Paper: https://arxiv.org/abs/2211.10655
- Code: None

**Imagic: Text-Based Real Image Editing with Diffusion Models**
**Imagic：基于文本的实时图像编辑与扩散模型**

- Homepage: https://imagic-editing.github.io/
- Paper: https://arxiv.org/abs/2210.09276
- Code: None

**Parallel Diffusion Models of Operator and Image for Blind Inverse Problems**
**并行算子和图像的盲反问题并行扩散模型**

- Paper: https://arxiv.org/abs/2211.10656
- Code: None

**DiffRF: Rendering-guided 3D Radiance Field Diffusion**
**DiffRF：渲染引导的3D辐射场扩散**

- Homepage: https://sirwyver.github.io/DiffRF/
- Paper: https://arxiv.org/abs/2212.01206
- Code: None

**MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation**
**MM-Diffusion：学习用于联合音频和视频生成的多模态扩散模型**

- Paper: https://arxiv.org/abs/2212.09478
- Code: https://github.com/researchmm/MM-Diffusion

**HouseDiffusion: Vector Floorplan Generation via a Diffusion Model with Discrete and Continuous Denoising**
**HouseDiffusion：通过具有离散和连续去噪的扩散模型生成矢量平面图**

- Homepage: https://aminshabani.github.io/housediffusion/
- Paper: https://arxiv.org/abs/2211.13287
- Code: https://github.com/aminshabani/house_diffusion 

**TrojDiff: Trojan Attacks on Diffusion Models with Diverse Targets**
**TrojDiff：针对具有多样化目标的扩散模型上的木马攻击**

- Paper: https://arxiv.org/abs/2303.05762
- Code: https://github.com/chenweixin107/TrojDiff

**Back to the Source: Diffusion-Driven Adaptation to Test-Time Corruption**
**回到源头：针对测试时干扰的扩散驱动自适应**

- Paper: https://arxiv.org/abs/2207.03442
- Code: https://github.com/shiyegao/DDA 

**DR2: Diffusion-based Robust Degradation Remover for Blind Face Restoration**
**DR2：基于扩散的鲁棒退化去除器，用于盲脸恢复**

- Paper: https://arxiv.org/abs/2303.06885
- Code: None

**Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion**
**轨迹与步速：通过引导轨迹扩散实现可控的行人动画**

- Homepage: https://nv-tlabs.github.io/trace-pace/
- Paper: https://arxiv.org/abs/2304.01893
- Code: None

**Generative Diffusion Prior for Unified Image Restoration and Enhancement**
**统一图像修复与增强的生成扩散先验**

- Paper: https://arxiv.org/abs/2304.01247
- Code: None

**Conditional Image-to-Video Generation with Latent Flow Diffusion Models**
**条件图像到视频生成与潜在流扩散模型**

- Paper: https://arxiv.org/abs/2303.13744
- Code: https://github.com/nihaomiao/CVPR23_LFDM 

<a name="Long-Tail"></a>

# 长尾分布(Long-Tail)

**Long-Tailed Visual Recognition via Self-Heterogeneous Integration with Knowledge Excavation**
**通过自异构集成与知识挖掘的尾部视觉识别**

- Paper: https://arxiv.org/abs/2304.01279
- Code: None

<a name="Vision-Transformer"></a>

# Vision Transformer

**Integrally Pre-Trained Transformer Pyramid Networks** 

- Paper: https://arxiv.org/abs/2211.12735
- Code: https://github.com/sunsmarterjie/iTPN

**Mask3D: Pre-training 2D Vision Transformers by Learning Masked 3D Priors**
**Mask3D：通过学习掩码3D先验对2D视觉Transformer进行预训练**

- Homepage: https://niessnerlab.org/projects/hou2023mask3d.html
- Paper: https://arxiv.org/abs/2302.14746
- Code: None

**Learning Trajectory-Aware Transformer for Video Super-Resolution**
**学习轨迹感知的Transformer用于视频超分辨率**

- Paper: https://arxiv.org/abs/2204.04216
- Code: https://github.com/researchmm/TTVSR

**Vision Transformers are Parameter-Efficient Audio-Visual Learners**
**视觉Transformer是参数高效的视听学习者。**

- Homepage: https://yanbo.ml/project_page/LAVISH/
- Code: https://github.com/GenjiB/LAVISH

**Where We Are and What We're Looking At: Query Based Worldwide Image Geo-localization Using Hierarchies and Scenes**
**我们在哪里以及我们在看什么：基于查询的全球图像地理定位，利用层次和场景**

- Paper: https://arxiv.org/abs/2303.04249
- Code: None

**DSVT: Dynamic Sparse Voxel Transformer with Rotated Sets**
**DSVT：具有旋转集合的动态稀疏体素转换器**

- Paper: https://arxiv.org/abs/2301.06051
- Code: https://github.com/Haiyang-W/DSVT

**DeepSolo: Let Transformer Decoder with Explicit Points Solo for Text Spotting**
**DeepSolo：让具有显式点的Transformer解码器独立用于文本检测**

- Paper: https://arxiv.org/abs/2211.10772
- Code link: https://github.com/ViTAE-Transformer/DeepSolo

**BiFormer: Vision Transformer with Bi-Level Routing Attention**
**BiFormer：具有双级路由注意力的视觉Transformer**

- Paper: https://arxiv.org/abs/2303.08810
- Code: https://github.com/rayleizhu/BiFormer

**Vision Transformer with Super Token Sampling**
**超采样超级标记的视觉Transformer**

- Paper: https://arxiv.org/abs/2211.11167
- Code: https://github.com/hhb072/SViT

**BEVFormer v2: Adapting Modern Image Backbones to Bird's-Eye-View Recognition via Perspective Supervision**
**BEVFormer v2：通过透视监督将现代图像骨干适应于鸟瞰图识别**

- Paper: https://arxiv.org/abs/2211.10439
- Code: None

**BAEFormer: Bi-directional and Early Interaction Transformers for Bird’s Eye View Semantic Segmentation**
**BAEFormer：用于鸟瞰视图语义分割的双向和早期交互Transformer**

- Paper: None
- Code: None

**Visual Dependency Transformers: Dependency Tree Emerges from Reversed Attention**
**视觉依赖转换器：依赖树从反转注意力中涌现**

- Paper: https://arxiv.org/abs/2304.03282
- Code: None

**Making Vision Transformers Efficient from A Token Sparsification View**
**从标记稀疏化视角提高视觉Transformer的效率**

- Paper: https://arxiv.org/abs/2303.08685
- Code: https://github.com/changsn/STViT-R 

<a name="VL"></a>

# 视觉和语言(Vision-Language)

**GIVL: Improving Geographical Inclusivity of Vision-Language Models with Pre-Training Methods**
**GIVL：通过预训练方法提升视觉-语言模型的地理包容性**

- Paper: https://arxiv.org/abs/2301.01893
- Code: None

**Teaching Structured Vision&Language Concepts to Vision&Language Models**
**教授结构化视觉与语言概念给视觉与语言模型**

- Paper: https://arxiv.org/abs/2211.11733
- Code: None

**Uni-Perceiver v2: A Generalist Model for Large-Scale Vision and Vision-Language Tasks**
**Uni-Perceiver v2：适用于大规模视觉和视觉语言任务的通用模型**

- Paper: https://arxiv.org/abs/2211.09808
- Code: https://github.com/fundamentalvision/Uni-Perceiver

**Towards Generalisable Video Moment Retrieval: Visual-Dynamic Injection to Image-Text Pre-Training**
**向通用视频瞬间检索迈进：视觉-动态注入到图像-文本预训练**

- Paper: https://arxiv.org/abs/2303.00040
- Code: None

**CapDet: Unifying Dense Captioning and Open-World Detection Pretraining**
**CapDet：统一密集描述和开放世界检测预训练**

- Paper: https://arxiv.org/abs/2303.02489
- Code: None

**FAME-ViL: Multi-Tasking Vision-Language Model for Heterogeneous Fashion Tasks**
**FAME-ViL：用于异构时尚任务的多任务视觉-语言模型**

- Paper: https://arxiv.org/abs/2303.02483
- Code: None

**Meta-Explore: Exploratory Hierarchical Vision-and-Language Navigation Using Scene Object Spectrum Grounding**
**元探索：利用场景物体频谱定位的探索性层次视觉和语言导航**

- Homepage: https://rllab-snu.github.io/projects/Meta-Explore/doc.html
- Paper: https://arxiv.org/abs/2303.04077
- Code: None

**All in One: Exploring Unified Video-Language Pre-training**
**一应俱全：探索统一的视频-语言预训练**

- Paper: https://arxiv.org/abs/2203.07303
- Code: https://github.com/showlab/all-in-one

**Position-guided Text Prompt for Vision Language Pre-training**
**位置引导的视觉语言预训练文本提示**

- Paper: https://arxiv.org/abs/2212.09737
- Code: https://github.com/sail-sg/ptp

**EDA: Explicit Text-Decoupling and Dense Alignment for 3D Visual Grounding**
**EDA：显式文本解耦和密集对齐用于3D视觉定位**

- Paper: https://arxiv.org/abs/2209.14941
- Code: https://github.com/yanmin-wu/EDA

**CapDet: Unifying Dense Captioning and Open-World Detection Pretraining**
**CapDet：统一密集标题和开放世界检测预训练**

- Paper: https://arxiv.org/abs/2303.02489
- Code: None

**FAME-ViL: Multi-Tasking Vision-Language Model for Heterogeneous Fashion Tasks**
**FAME-ViL：用于异构时尚任务的多元任务视觉-语言模型**

- Paper: https://arxiv.org/abs/2303.02483
- Code: https://github.com/BrandonHanx/FAME-ViL

**Align and Attend: Multimodal Summarization with Dual Contrastive Losses**
**对齐与关注：具有双重对比损失的跨模态摘要**

- Homepage: https://boheumd.github.io/A2Summ/
- Paper: https://arxiv.org/abs/2303.07284
- Code: https://github.com/boheumd/A2Summ

**Multi-Modal Representation Learning with Text-Driven Soft Masks**
**多模态表示学习与文本驱动软掩码**

- Paper: https://arxiv.org/abs/2304.00719
- Code: None

**Learning to Name Classes for Vision and Language Models**
**学习为视觉和语言模型命名类别的技巧**

- Paper: https://arxiv.org/abs/2304.01830
- Code: None

<a name="Object-Detection"></a>

# 目标检测(Object Detection)

**YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors**
**YOLOv7：可训练的免费包集为实时物体检测器创造了新的最高水平**

- Paper: https://arxiv.org/abs/2207.02696
- Code: https://github.com/WongKinYiu/yolov7

**DETRs with Hybrid Matching**
**混合匹配的DETRs**

- Paper: https://arxiv.org/abs/2207.13080
- Code: https://github.com/HDETR

**Enhanced Training of Query-Based Object Detection via Selective Query Recollection**
**通过选择性查询回忆增强基于查询的目标检测训练**

- Paper: https://arxiv.org/abs/2212.07593
- Code: https://github.com/Fangyi-Chen/SQR

**Object-Aware Distillation Pyramid for Open-Vocabulary Object Detection**
**开放词汇物体检测的对象感知蒸馏金字塔**

- Paper: https://arxiv.org/abs/2303.05892
- Code: https://github.com/LutingWang/OADP

<a name="VT"></a>

# 目标跟踪(Object Tracking)

**Simple Cues Lead to a Strong Multi-Object Tracker**
**简单提示引导强大的多目标追踪器**

- Paper: https://arxiv.org/abs/2206.04656
- Code: None

**Joint Visual Grounding and Tracking with Natural Language Specification**
**基于自然语言指定的联合视觉定位和跟踪**

- Paper: https://arxiv.org/abs/2303.12027
- Code: https://github.com/lizhou-cs/JointNLT 

<a name="Semantic-Segmentation"></a>

# 语义分割(Semantic Segmentation)

**Efficient Semantic Segmentation by Altering Resolutions for Compressed Videos**
**高效通过调整压缩视频分辨率进行语义分割**

- Paper: https://arxiv.org/abs/2303.07224
- Code: https://github.com/THU-LYJ-Lab/AR-Seg

**FREDOM: Fairness Domain Adaptation Approach to Semantic Scene Understanding**
**公平域适应语义场景理解方法**

- Paper: https://arxiv.org/abs/2304.02135
- Code: https://github.com/uark-cviu/FREDOM

<a name="MIS"></a>

# 医学图像分割(Medical Image Segmentation)

**Label-Free Liver Tumor Segmentation**
**无标记肝肿瘤分割**

- Paper: https://arxiv.org/abs/2303.14869
- Code: https://github.com/MrGiovanni/SyntheticTumors

**Directional Connectivity-based Segmentation of Medical Images**
**基于方向连通性的医学图像分割**

- Paper: https://arxiv.org/abs/2304.00145
- Code: https://github.com/Zyun-Y/DconnNet

**Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation**
**双向复制粘贴用于半监督医学图像分割**

- Paper: https://arxiv.org/abs/2305.00673
- Code: https://github.com/DeepMed-Lab-ECNU/BCP

**Devil is in the Queries: Advancing Mask Transformers for Real-world Medical Image Segmentation and Out-of-Distribution Localization**
**查询中的恶魔：推进用于现实世界医学图像分割和分布外定位的掩码转换器**

- Paper: https://arxiv.org/abs/2304.00212
- Code: None

**Fair Federated Medical Image Segmentation via Client Contribution Estimation**
**公平联邦医疗图像分割通过客户端贡献估计**

- Paper: https://arxiv.org/abs/2303.16520
- Code: https://github.com/NVIDIA/NVFlare/tree/dev/research/fed-ce

**Ambiguous Medical Image Segmentation using Diffusion Models**
**使用扩散模型进行模糊医学图像分割**

- Homepage: https://aimansnigdha.github.io/cimd/
- Paper: https://arxiv.org/abs/2304.04745
- Code: https://github.com/aimansnigdha/Ambiguous-Medical-Image-Segmentation-using-Diffusion-Models

**Orthogonal Annotation Benefits Barely-supervised Medical Image Segmentation**
**正交注释几乎无监督医学图像分割的益处**

- Paper: https://arxiv.org/abs/2303.13090
- Code: https://github.com/HengCai-NJU/DeSCO

**MagicNet: Semi-Supervised Multi-Organ Segmentation via Magic-Cube Partition and Recovery**
**MagicNet：通过魔方分割和恢复的半监督多器官分割**

- Paper: https://arxiv.org/abs/2301.01767
- Code: https://github.com/DeepMed-Lab-ECNU/MagicNet

**MCF: Mutual Correction Framework for Semi-Supervised Medical Image Segmentation**
**MCF：半监督医学图像分割的相互校正框架**

- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Wang_MCF_Mutual_Correction_Framework_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.html
- Code: https://github.com/WYC-321/MCF

**Rethinking Few-Shot Medical Segmentation: A Vector Quantization View**
**重新思考小样本医学分割：矢量量化视角**

- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Huang_Rethinking_Few-Shot_Medical_Segmentation_A_Vector_Quantization_View_CVPR_2023_paper.html
- Code: None

**Pseudo-label Guided Contrastive Learning for Semi-supervised Medical Image Segmentation**
**伪标签引导的对比学习方法在半监督医学图像分割中的应用**

- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Basak_Pseudo-Label_Guided_Contrastive_Learning_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.html
- Code: https://github.com/hritam-98/PatchCL-MedSeg

**SDC-UDA: Volumetric Unsupervised Domain Adaptation Framework for Slice-Direction Continuous Cross-Modality Medical Image Segmentation**
**SDC-UDA：切片方向连续跨模态医学图像分割的体积无监督域适应框架**

- Paper: https://arxiv.org/abs/2305.11012
- Code: None

**DoNet: Deep De-overlapping Network for Cytology Instance Segmentation**
**DoNet：针对细胞学实例分割的深度去重叠网络**

- Paper: https://arxiv.org/abs/2303.14373
- Code: https://github.com/DeepDoNet/DoNet

<a name="VOS"></a>

# 视频目标分割（Video Object Segmentation）

**Two-shot Video Object Segmentation**
**两帧视频目标分割**

- Paper: https://arxiv.org/abs/2303.12078
- Code: https://github.com/yk-pku/Two-shot-Video-Object-Segmentation

 **Under Video Object Segmentation Section**

- Paper: https://arxiv.org/abs/2303.07815
- Code: None

<a name="VIS"></a>

# 视频实例分割(Video Instance Segmentation)

**Mask-Free Video Instance Segmentation**
**无遮挡视频实例分割**

- Paper: https://arxiv.org/abs/2303.15904
- Code: https://github.com/SysCV/MaskFreeVis 

<a name="RIS"></a>

# 参考图像分割(Referring Image Segmentation )

**PolyFormer: Referring Image Segmentation as Sequential Polygon Generation**
**PolyFormer：将指代图像分割视为顺序多边形生成**

- Paper: https://arxiv.org/abs/2302.07387 

- Code: None

<a name="3D-Point-Cloud"></a>

# 3D点云(3D-Point-Cloud)

**Physical-World Optical Adversarial Attacks on 3D Face Recognition**
**物理世界光学对抗攻击对3D人脸识别**

- Paper: https://arxiv.org/abs/2205.13412
- Code: https://github.com/PolyLiYJ/SLAttack.git

**IterativePFN: True Iterative Point Cloud Filtering**
**迭代PFN：真，迭代点云滤波**

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
**DSVT：旋转集动态稀疏体素转换器**

- Paper: https://arxiv.org/abs/2301.06051
- Code: https://github.com/Haiyang-W/DSVT 

**FrustumFormer: Adaptive Instance-aware Resampling for Multi-view 3D Detection**
**FrustumFormer：多视图3D检测的自适应实例感知重采样**

- Paper:  https://arxiv.org/abs/2301.04467
- Code: None

**3D Video Object Detection with Learnable Object-Centric Global Optimization**
**3D视频目标检测：具有可学习以目标为中心的全局优化的方法**

- Paper: None
- Code: None

**Hierarchical Supervision and Shuffle Data Augmentation for 3D Semi-Supervised Object Detection**
**分层监督和随机数据增强用于3D半监督目标检测**

- Paper: https://arxiv.org/abs/2304.01464
- Code: https://github.com/azhuantou/HSSDA

<a name="3DOD"></a>

# 3D语义分割(3D Semantic Segmentation)

**Less is More: Reducing Task and Model Complexity for 3D Point Cloud Semantic Segmentation**
**少即是多：降低3D点云语义分割的任务和模型复杂性**

- Paper: https://arxiv.org/abs/2303.11203
- Code: https://github.com/l1997i/lim3d 

<a name="3DSSC"></a>

# 3D语义场景补全(3D Semantic Scene Completion)

- Paper: https://arxiv.org/abs/2302.12251
- Code: https://github.com/NVlabs/VoxFormer 

<a name="3D-Registration"></a>

# 3D配准(3D Registration)

**Robust Outlier Rejection for 3D Registration with Variational Bayes**
**基于变分贝叶斯的三维配准的鲁棒异常值拒绝**

- Paper: https://arxiv.org/abs/2304.01514
- Code: https://github.com/Jiang-HB/VBReg

<a name="3D-Human-Pose-Estimation"></a>

# 3D人体姿态估计(3D Human Pose Estimation)

<a name="3D-Human-Mesh-Estimation"></a>

# 3D人体Mesh估计(3D Human Mesh Estimation)

**3D Human Mesh Estimation from Virtual Markers**
**从虚拟标记中进行3D人体网格估计**

- Paper: https://arxiv.org/abs/2303.11726
- Code: https://github.com/ShirleyMaxx/VirtualMarker 

<a name="LLV"></a>

# Low-level Vision

**Causal-IR: Learning Distortion Invariant Representation for Image Restoration from A Causality Perspective**
**因果-图像恢复：从因果角度学习失真不变表示**

- Paper: https://arxiv.org/abs/2303.06859
- Code: https://github.com/lixinustc/Casual-IR-DIL 

**Burstormer: Burst Image Restoration and Enhancement Transformer**
**Burstormer：爆发图像恢复和增强转换器**

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
**学习轨迹感知的Transformer用于视频超分辨率**

- Paper: https://arxiv.org/abs/2204.04216

- Code: https://github.com/researchmm/TTVSR

Denoising<a name="Denoising"></a>

# 去噪(Denoising)

## 图像去噪(Image Denoising)

**Masked Image Training for Generalizable Deep Image Denoising**
**掩码图像训练以实现通用的深度图像去噪**

- Paper- : https://arxiv.org/abs/2303.13132
- Code: https://github.com/haoyuc/MaskedDenoising 

<a name="Image-Generation"></a>

# 图像生成(Image Generation)

**GALIP: Generative Adversarial CLIPs for Text-to-Image Synthesis**
**GALIP：用于文本到图像合成的生成对抗CLIPs**

- Paper: https://arxiv.org/abs/2301.12959
- Code: https://github.com/tobran/GALIP 

**MAGE: MAsked Generative Encoder to Unify Representation Learning and Image Synthesis**
**MAGE：面向统一表示学习和图像合成的MAsked生成编码器**

- Paper: https://arxiv.org/abs/2211.09117
- Code: https://github.com/LTH14/mage

**Toward Verifiable and Reproducible Human Evaluation for Text-to-Image Generation**
**迈向可验证和可重现的文本到图像生成的人评方法**

- Paper: https://arxiv.org/abs/2304.01816
- Code: None

**Few-shot Semantic Image Synthesis with Class Affinity Transfer**
**少量样本语义图像合成与类关联迁移**

- Paper: https://arxiv.org/abs/2304.02321
- Code: None

**TopNet: Transformer-based Object Placement Network for Image Compositing**
**TopNet：基于Transformer的对象放置网络用于图像合成**

- Paper: https://arxiv.org/abs/2304.03372
- Code: None

<a name="Video-Generation"></a>

# 视频生成(Video Generation)

**MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation**
**MM-Diffusion：学习多模态扩散模型以实现音频和视频的联合生成**

- Paper: https://arxiv.org/abs/2212.09478
- Code: https://github.com/researchmm/MM-Diffusion

**Conditional Image-to-Video Generation with Latent Flow Diffusion Models**
**基于潜在流扩散模型的条件图像到视频生成**

- Paper: https://arxiv.org/abs/2303.13744
- Code: https://github.com/nihaomiao/CVPR23_LFDM 

<a name="Video-Understanding"></a>

# 视频理解(Video Understanding)

**Learning Transferable Spatiotemporal Representations from Natural Script Knowledge**
**从自然脚本知识中学习可迁移的时空表征**

- Paper: https://arxiv.org/abs/2209.15280
- Code: https://github.com/TencentARC/TVTS

**Frame Flexible Network**
**框架灵活网络**

- Paper: https://arxiv.org/abs/2303.14817
- Code: https://github.com/BeSpontaneous/FFN

**Masked Motion Encoding for Self-Supervised Video Representation Learning**
**掩码运动编码用于自监督视频表征学习**

- Paper: https://arxiv.org/abs/2210.06096
- Code: https://github.com/XinyuSun/MME

**MARLIN: Masked Autoencoder for facial video Representation LearnING**
**MARLIN：用于面部视频表征学习的掩码自动编码器**

- Paper: https://arxiv.org/abs/2211.06627
- Code: https://github.com/ControlNet/MARLIN 

<a name="Action-Detection"></a>

# 行为检测(Action Detection)

**TriDet: Temporal Action Detection with Relative Boundary Modeling**
**TriDet：基于相对边界建模的时间动作检测**

- Paper: https://arxiv.org/abs/2303.07347
- Code: https://github.com/dingfengshi/TriDet 

<a name="Text-Detection"></a>

# 文本检测(Text Detection)

**DeepSolo: Let Transformer Decoder with Explicit Points Solo for Text Spotting**
**DeepSolo：让带有显式点的Transformer解码器独立用于文本检测**

- Paper: https://arxiv.org/abs/2211.10772
- Code link: https://github.com/ViTAE-Transformer/DeepSolo

<a name="KD"></a>

# 知识蒸馏(Knowledge Distillation)

**Learning to Retain while Acquiring: Combating Distribution-Shift in Adversarial Data-Free Knowledge Distillation**
**在学习的同时保留：对抗无数据知识蒸馏中的分布偏移对抗**

- Paper: https://arxiv.org/abs/2302.14290
- Code: None

**Generic-to-Specific Distillation of Masked Autoencoders**
**通用到特定蒸馏的掩码自编码器**

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
**基于上下文的Trit-Plane编码用于渐进式图像压缩**

- Paper: https://arxiv.org/abs/2303.05715
- Code: https://github.com/seungminjeon-github/CTC

<a name="AD"></a>

# 异常检测(Anomaly Detection)

**Deep Feature In-painting for Unsupervised Anomaly Detection in X-ray Images**
**深度特征内插法在X射线图像中的无监督异常检测**

- Paper: https://arxiv.org/abs/2111.13495
- Code: https://github.com/tiangexiang/SQUID 

<a name="3D-Reconstruction"></a>

# 三维重建(3D Reconstruction)

**OReX: Object Reconstruction from Planar Cross-sections Using Neural Fields**
**OReX：利用神经网络场从平面截面进行物体重建**

- Paper: https://arxiv.org/abs/2211.12886
- Code: None

**SparsePose: Sparse-View Camera Pose Regression and Refinement**
**稀疏姿态：稀疏视角相机姿态回归与优化**

- Paper: https://arxiv.org/abs/2211.16991
- Code: None

**NeuDA: Neural Deformable Anchor for High-Fidelity Implicit Surface Reconstruction**
**NeuDA：高保真隐式表面重建的神经可变形锚点**

- Paper: https://arxiv.org/abs/2303.02375
- Code: None

**Vid2Avatar: 3D Avatar Reconstruction from Videos in the Wild via Self-supervised Scene Decomposition**
**Vid2Avatar：通过自监督场景分解从野外视频中重建3D虚拟形象**

- Homepage: https://moygcc.github.io/vid2avatar/
- Paper: https://arxiv.org/abs/2302.11566
- Code: https://github.com/MoyGcc/vid2avatar
- Demo: https://youtu.be/EGi47YeIeGQ

**To fit or not to fit: Model-based Face Reconstruction and Occlusion Segmentation from Weak Supervision**
**适合还是不适合：基于弱监督的模型人脸重建和遮挡分割**

- Paper: https://arxiv.org/abs/2106.09614
- Code: https://github.com/unibas-gravis/Occlusion-Robust-MoFA

**Structural Multiplane Image: Bridging Neural View Synthesis and 3D Reconstruction**
**结构多平面图像：连接神经视图合成与三维重建**

- Paper: https://arxiv.org/abs/2303.05937
- Code: None

**3D Cinemagraphy from a Single Image**
**单张图片制作的3D动态图**

- Homepage: https://xingyi-li.github.io/3d-cinemagraphy/
- Paper: https://arxiv.org/abs/2303.05724
- Code: https://github.com/xingyi-li/3d-cinemagraphy

**Revisiting Rotation Averaging: Uncertainties and Robust Losses**
**重新审视旋转平均：不确定性和鲁棒损失**

- Paper: https://arxiv.org/abs/2303.05195
- Code https://github.com/zhangganlin/GlobalSfMpy 

**FFHQ-UV: Normalized Facial UV-Texture Dataset for 3D Face Reconstruction**
**FFHQ-UV：用于3D人脸重建的归一化面部UV纹理数据集**

- Paper: https://arxiv.org/abs/2211.13874
- Code: https://github.com/csbhr/FFHQ-UV 

**A Hierarchical Representation Network for Accurate and Detailed Face Reconstruction from In-The-Wild Images**
**一种从野外图像中精确且详细重建人脸的分层表示网络**

- Homepage: https://younglbw.github.io/HRN-homepage/ 

- Paper: https://arxiv.org/abs/2302.14434
- Code: https://github.com/youngLBW/HRN

<a name="Depth-Estimation"></a>

# 深度估计(Depth Estimation)

**Lite-Mono: A Lightweight CNN and Transformer Architecture for Self-Supervised Monocular Depth Estimation**
**Lite-Mono：一种用于自监督单目深度估计的轻量级CNN和Transformer架构**

- Paper: https://arxiv.org/abs/2211.13202
- Code: https://github.com/noahzn/Lite-Mono 

<a name="TP"></a>

# 轨迹预测(Trajectory Prediction)

**IPCC-TP: Utilizing Incremental Pearson Correlation Coefficient for Joint Multi-Agent Trajectory Prediction**
**IPCC-TP：利用增量皮尔逊相关系数进行联合多智能体轨迹预测**

- Paper:  https://arxiv.org/abs/2303.00575
- Code: None

**EqMotion: Equivariant Multi-agent Motion Prediction with Invariant Interaction Reasoning**
**EqMotion：具有不变交互推理的等变多智能体运动预测**

- Paper: https://arxiv.org/abs/2303.10876
- Code: https://github.com/MediaBrain-SJTU/EqMotion 

<a name="Lane-Detection"></a>

# 车道线检测(Lane Detection)

**Anchor3DLane: Learning to Regress 3D Anchors for Monocular 3D Lane Detection**
**Anchor3DLane：学习回归单目3D车道检测的3D锚点**

- Paper: https://arxiv.org/abs/2301.02371
- Code: https://github.com/tusen-ai/Anchor3DLane

**BEV-LaneDet: An Efficient 3D Lane Detection Based on Virtual Camera via Key-Points**
**BEV-LaneDet：基于虚拟相机和关键点的有效3D车道检测**

- Paper:  https://arxiv.org/abs/2210.06006v3 
- Code:  https://github.com/gigo-team/bev_lane_det 

<a name="Image-Captioning"></a>

# 图像描述(Image Captioning)

**ConZIC: Controllable Zero-shot Image Captioning by Sampling-Based Polishing**
**基于采样优化的可控零样本图像描述生成**

- Paper: https://arxiv.org/abs/2303.02437
- Code: Node

**Cross-Domain Image Captioning with Discriminative Finetuning**
**跨域图像标题生成与判别性微调**

- Paper: https://arxiv.org/abs/2304.01662
- Code: None

**Model-Agnostic Gender Debiased Image Captioning**
**模型无关的性别偏差图像标题生成**

- Paper: https://arxiv.org/abs/2304.03693
- Code: None

<a name="VQA"></a>

# 视觉问答(Visual Question Answering)

**MixPHM: Redundancy-Aware Parameter-Efficient Tuning for Low-Resource Visual Question Answering**
**MixPHM：针对低资源视觉问答的冗余感知参数高效调优**

- Paper:  https://arxiv.org/abs/2303.01239
- Code: https://github.com/jingjing12110/MixPHM

<a name="SLR"></a>

# 手语识别(Sign Language Recognition)

**Continuous Sign Language Recognition with Correlation Network**
**连续手语识别与相关性网络**

Paper: https://arxiv.org/abs/2303.03202

Code: https://github.com/hulianyuyy/CorrNet

<a name="Video-Prediction"></a>

# 视频预测(Video Prediction)

**MOSO: Decomposing MOtion, Scene and Object for Video Prediction**
**MOSO：分解运动、场景和物体以进行视频预测**

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
**双向分布对齐用于归纳零样本学习**

- Paper: https://arxiv.org/abs/2303.08698
- Code: https://github.com/Zhicaiwww/Bi-VAEGAN

**Semantic Prompt for Few-Shot Learning**
**语义提示用于小样本学习**

- Paper: None
- Code: None

<a name="Stereo-Matching"></a>

# 立体匹配(Stereo Matching)

**Iterative Geometry Encoding Volume for Stereo Matching**
**迭代几何编码体积用于立体匹配**

- Paper: https://arxiv.org/abs/2303.06615
- Code: https://github.com/gangweiX/IGEV

**Learning the Distribution of Errors in Stereo Matching for Joint Disparity and Uncertainty Estimation**
**学习联合视差和不确定性估计中的立体匹配错误分布**

- Paper: https://arxiv.org/abs/2304.00152
- Code: None

<a name="Feature-Matching"></a>

# 特征匹配(Feature Matching)

**Adaptive Spot-Guided Transformer for Consistent Local Feature Matching**
**自适应点引导变压器用于一致的局部特征匹配**

- Homepage: [https://astr2023.github.io](https://astr2023.github.io/) 
- Paper: https://arxiv.org/abs/2303.16624
- Code: https://github.com/ASTR2023/ASTR

<a name="SGG"></a>

# 场景图生成(Scene Graph Generation)

**Prototype-based Embedding Network for Scene Graph Generation**
**基于原型嵌入的场景图生成网络**

- Paper: https://arxiv.org/abs/2303.07096
- Code: None

<a name="INR"></a>

# 隐式神经表示(Implicit Neural Representations)

**Polynomial Implicit Neural Representations For Large Diverse Datasets**
**多项式隐式神经网络表示法用于大型多样化数据集**

- Paper: https://arxiv.org/abs/2303.11424
- Code: https://github.com/Rajhans0/Poly_INR

<a name="IQA"></a>

# 图像质量评价(Image Quality Assessment)

**Re-IQA: Unsupervised Learning for Image Quality Assessment in the Wild**
**Re-IQA：用于野外观测图像质量评估的无监督学习方法**

- Paper: https://arxiv.org/abs/2304.00451
- Code: None

<a name="Datasets"></a>

# 数据集(Datasets)

**Human-Art: A Versatile Human-Centric Dataset Bridging Natural and Artificial Scenes**
**人艺：一个多功能以人为中心的数据库，连接自然场景与人工场景**

- Paper: https://arxiv.org/abs/2303.02760
- Code: None

**Align and Attend: Multimodal Summarization with Dual Contrastive Losses**
**对齐与关注：双对比损失的跨模态摘要**

- Homepage: https://boheumd.github.io/A2Summ/
- Paper: https://arxiv.org/abs/2303.07284
- Code: https://github.com/boheumd/A2Summ

**GeoNet: Benchmarking Unsupervised Adaptation across Geographies**
**GeoNet：跨地理区域的无监督适应基准测试**

- Homepage: https://tarun005.github.io/GeoNet/
- Paper: https://arxiv.org/abs/2303.15443

**CelebV-Text: A Large-Scale Facial Text-Video Dataset**
**CelebV-Text：一个大规模的人脸文本-视频数据集**

- Homepage: https://celebv-text.github.io/
- Paper: https://arxiv.org/abs/2303.14717

<a name="Others"></a>

# 其他(Others)

**Interactive Segmentation as Gaussian Process Classification**
**交互式分割作为高斯过程分类**

- Paper: https://arxiv.org/abs/2302.14578
- Code: None

**Backdoor Attacks Against Deep Image Compression via Adaptive Frequency Trigger**
**通过自适应频率触发的针对深度图像压缩的后门攻击**

- Paper: https://arxiv.org/abs/2302.14677
- Code: None

**SplineCam: Exact Visualization and Characterization of Deep Network Geometry and Decision Boundaries**
**SplineCam：深度网络几何和决策边界的精确可视化和表征**

- Homepage: http://bit.ly/splinecam
- Paper: https://arxiv.org/abs/2302.12828
- Code: None

**SCOTCH and SODA: A Transformer Video Shadow Detection Framework**
**苏格兰和苏打：一种Transformer视频阴影检测框架**

- Paper: https://arxiv.org/abs/2211.06885
- Code: None

**DeepMapping2: Self-Supervised Large-Scale LiDAR Map Optimization**
**DeepMapping2：自监督大规模激光雷达地图优化**

- Homepage: https://ai4ce.github.io/DeepMapping2/
- Paper: https://arxiv.org/abs/2212.06331
- None: https://github.com/ai4ce/DeepMapping2

**RelightableHands: Efficient Neural Relighting of Articulated Hand Models**
**可重光照的手模型：高效的神经可重光照方法**

- Homepage: https://sh8.io/#/relightable_hands
- Paper: https://arxiv.org/abs/2302.04866
- Code: None

**Token Turing Machines**
**图灵机令牌**

- Paper: https://arxiv.org/abs/2211.09119
- Code: None

**Single Image Backdoor Inversion via Robust Smoothed Classifiers**
**单图像后门反演通过鲁棒平滑分类器**

- Paper: https://arxiv.org/abs/2303.00215
- Code: https://github.com/locuslab/smoothinv

**To fit or not to fit: Model-based Face Reconstruction and Occlusion Segmentation from Weak Supervision**
**适合还是不适合：基于模型的弱监督人脸重建和遮挡分割**

- Paper: https://arxiv.org/abs/2106.09614
- Code: https://github.com/unibas-gravis/Occlusion-Robust-MoFA

**HOOD: Hierarchical Graphs for Generalized Modelling of Clothing Dynamics**
**标题：用于广义服装动态建模的分层图（HOOD）**

- Homepage: https://dolorousrtur.github.io/hood/
- Paper: https://arxiv.org/abs/2212.07242
- Code: https://github.com/dolorousrtur/hood
- Demo: https://www.youtube.com/watch?v=cBttMDPrUYY

**A Whac-A-Mole Dilemma: Shortcuts Come in Multiples Where Mitigating One Amplifies Others**
**“打地鼠困境：在缓解一种问题的同时，多重捷径会放大其他问题”**

- Paper: https://arxiv.org/abs/2212.04825
- Code: https://github.com/facebookresearch/Whac-A-Mole.git

**RelightableHands: Efficient Neural Relighting of Articulated Hand Models**
**可重光照的手模型：高效神经重光照的关节手模型**

- Homepage: https://sh8.io/#/relightable_hands
- Paper: https://arxiv.org/abs/2302.04866
- Code: None
- Demo: https://sh8.io/static/media/teacher_video.923d87957fe0610730c2.mp4

**Neuro-Modulated Hebbian Learning for Fully Test-Time Adaptation**
**神经调节的赫布学习实现全测试时适应**

- Paper: https://arxiv.org/abs/2303.00914
- Code: None

**Demystifying Causal Features on Adversarial Examples and Causal Inoculation for Robust Network by Adversarial Instrumental Variable Regression**
**揭示对抗样本上的因果特征以及通过对抗工具变量回归进行因果接种以实现鲁棒网络的去神秘化**

- Paper: https://arxiv.org/abs/2303.01052
- Code: None

**UniDexGrasp: Universal Robotic Dexterous Grasping via Learning Diverse Proposal Generation and Goal-Conditioned Policy**
**UniDexGrasp：通过学习多样化的提案生成和目标条件策略实现通用机器人灵巧抓取**

- Paper: https://arxiv.org/abs/2303.00938
- Code: None

**Disentangling Orthogonal Planes for Indoor Panoramic Room Layout Estimation with Cross-Scale Distortion Awareness**
**室内全景房间布局估计中的正交平面解耦及跨尺度畸变感知**

- Paper: https://arxiv.org/abs/2303.00971
- Code: https://github.com/zhijieshen-bjtu/DOPNet

**Learning Neural Parametric Head Models**
**学习神经参数头部模型**

- Homepage: https://simongiebenhain.github.io/NPHM)
- Paper: https://arxiv.org/abs/2212.02761
- Code: None

**A Meta-Learning Approach to Predicting Performance and Data Requirements**
**一种用于预测性能和数据需求的元学习方法**

- Paper: https://arxiv.org/abs/2303.01598
- Code: None

**MACARONS: Mapping And Coverage Anticipation with RGB Online Self-Supervision**
**马卡龙：基于RGB在线自监督的映射和覆盖预测**

- Homepage: https://imagine.enpc.fr/~guedona/MACARONS/
- Paper: https://arxiv.org/abs/2303.03315
- Code: None

**Masked Images Are Counterfactual Samples for Robust Fine-tuning**
**掩码图像是用于鲁棒微调的反事实样本**

- Paper: https://arxiv.org/abs/2303.03052
- Code: None

**HairStep: Transfer Synthetic to Real Using Strand and Depth Maps for Single-View 3D Hair Modeling**
**HairStep：利用单视图3D毛发建模中的单股和深度图将合成毛发转换为真实毛发**

- Paper: https://arxiv.org/abs/2303.02700
- Code: None

**Decompose, Adjust, Compose: Effective Normalization by Playing with Frequency for Domain Generalization**
**分解、调整、组合：通过频率操作实现有效的域泛化归一化**

- Paper: https://arxiv.org/abs/2303.02328
- Code: None

**Gradient Norm Aware Minimization Seeks First-Order Flatness and Improves Generalization**
**梯度范数感知最小化寻求一阶平坦性并提升泛化能力。**

- Paper: https://arxiv.org/abs/2303.03108
- Code: None

**Unlearnable Clusters: Towards Label-agnostic Unlearnable Examples**
**不可学习簇：朝向无标签不可学习示例**

- Paper: https://arxiv.org/abs/2301.01217
- Code: https://github.com/jiamingzhang94/Unlearnable-Clusters 

**Where We Are and What We're Looking At: Query Based Worldwide Image Geo-localization Using Hierarchies and Scenes**
**我们所在的位置和我们所观察的对象：利用层次和场景进行基于查询的全球图像地理定位**

- Paper: https://arxiv.org/abs/2303.04249
- Code: None

**UniHCP: A Unified Model for Human-Centric Perceptions**
**UniHCP：以人为本感知的统一模型**

- Paper: https://arxiv.org/abs/2303.02936
- Code: https://github.com/OpenGVLab/UniHCP

**CUDA: Convolution-based Unlearnable Datasets**
**基于卷积的不可学习数据集**

- Paper: https://arxiv.org/abs/2303.04278
- Code: https://github.com/vinusankars/Convolution-based-Unlearnability

**Masked Images Are Counterfactual Samples for Robust Fine-tuning**
**带遮罩的图像是用于鲁棒微调的反事实样本**

- Paper: https://arxiv.org/abs/2303.03052
- Code: None

**AdaptiveMix: Robust Feature Representation via Shrinking Feature Space**
**自适应混合：通过收缩特征空间实现鲁棒的特征表示**

- Paper: https://arxiv.org/abs/2303.01559
- Code: https://github.com/WentianZhang-ML/AdaptiveMix 

**Physical-World Optical Adversarial Attacks on 3D Face Recognition**
**物理世界对3D人脸识别的光学对抗攻击**

- Paper: https://arxiv.org/abs/2205.13412
- Code: https://github.com/PolyLiYJ/SLAttack.git

**DPE: Disentanglement of Pose and Expression for General Video Portrait Editing**
**DPE：通用视频肖像编辑中的姿态与表情解耦**

- Paper: https://arxiv.org/abs/2301.06281
- Code: https://carlyx.github.io/DPE/ 

**SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation**
**SadTalker：学习用于风格化音频驱动的单图像说话人脸动画的逼真3D运动系数**

- Paper: https://arxiv.org/abs/2211.12194
- Code: https://github.com/Winfredy/SadTalker

**Intrinsic Physical Concepts Discovery with Object-Centric Predictive Models**
**以对象为中心的预测模型进行内在物理概念发现**

- Paper: None
- Code: None

**Sharpness-Aware Gradient Matching for Domain Generalization**
**领域泛化中的锐度感知梯度匹配**

- Paper: None
- Code: https://github.com/Wang-pengfei/SAGM

**Mind the Label-shift for Augmentation-based Graph Out-of-distribution Generalization**
**注意基于增强的图异常分布泛化中的标签偏移问题**

- Paper: None
- Code: None

**Blind Video Deflickering by Neural Filtering with a Flawed Atlas**
**基于有缺陷图集的神经网络滤波盲视频去闪烁**

- Homepage:  https://chenyanglei.github.io/deflicker 
- Paper: None
- Code: None

**RiDDLE: Reversible and Diversified De-identification with Latent Encryptor**
**RiDDLE：带有潜在加密器的可逆和多样化脱敏**

- Paper: None
- Code:  https://github.com/ldz666666/RiDDLE 

**PoseExaminer: Automated Testing of Out-of-Distribution Robustness in Human Pose and Shape Estimation**
**姿态检查器：人体姿态和形状估计中分布外鲁棒性的自动测试**

- Paper: https://arxiv.org/abs/2303.07337
- Code: None

**Upcycling Models under Domain and Category Shift**
**基于领域和类别变化的升级改造模型**

- Paper: https://arxiv.org/abs/2303.07110
- Code: https://github.com/ispc-lab/GLC

**Modality-Agnostic Debiasing for Single Domain Generalization**
**单域泛化中的模态无关去偏**

- Paper: https://arxiv.org/abs/2303.07123
- Code: None

**Progressive Open Space Expansion for Open-Set Model Attribution**
**开放式模型归因的渐进式开放空间扩展**

- Paper: https://arxiv.org/abs/2303.06877
- Code: None

**Dynamic Neural Network for Multi-Task Learning Searching across Diverse Network Topologies**
**动态神经网络用于多任务学习，跨越不同的网络拓扑结构。**

- Paper: https://arxiv.org/abs/2303.06856
- Code: None

**GFPose: Learning 3D Human Pose Prior with Gradient Fields**
**GFPose：利用梯度场学习3D人体姿态先验**

- Paper: https://arxiv.org/abs/2212.08641
- Code: https://github.com/Embracing/GFPose 

**PRISE: Demystifying Deep Lucas-Kanade with Strongly Star-Convex Constraints for Multimodel Image Alignment**
**PRISE：通过强星凸约束解开深度Lucas-Kanade的多模型图像配准之谜**

- Paper: https://arxiv.org/abs/2303.11526
- Code: https://github.com/Zhang-VISLab

**Sketch2Saliency: Learning to Detect Salient Objects from Human Drawings**
**Sketch2Saliency：从人类绘画中学习检测显著物体**

- Paper: https://arxiv.org/abs/2303.11502
- Code: None

**Boundary Unlearning**
**边界去学习**

- Paper: https://arxiv.org/abs/2303.11570
- Code: None

**ImageNet-E: Benchmarking Neural Network Robustness via Attribute Editing**
**ImageNet-E：通过属性编辑评估神经网络鲁棒性的基准测试**

- Paper: https://arxiv.org/abs/2303.17096
- Code: https://github.com/alibaba/easyrobust

**Zero-shot Model Diagnosis**
**零样本模型诊断**

- Paper: https://arxiv.org/abs/2303.15441
- Code: None

**GeoNet: Benchmarking Unsupervised Adaptation across Geographies**
**GeoNet：地理范围下无监督自适应的基准测试**

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
**神经体积记忆用于视觉运动控制**

- Homepage: https://rchalyang.github.io/NVM
- Paper: https://arxiv.org/abs/2304.01201
- Code: https://rchalyang.github.io/NVM

**MonoHuman: Animatable Human Neural Field from Monocular Video**
**单目视频中的可动画化人脑神经场**

- Homepage: https://yzmblog.github.io/projects/MonoHuman/
- Paper: https://arxiv.org/abs/2304.02001
- Code: https://github.com/Yzmblog/MonoHuman

**Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion**
**轨迹与速度：通过引导轨迹扩散的可控行人动画**

- Homepage: https://nv-tlabs.github.io/trace-pace/
- Paper: https://arxiv.org/abs/2304.01893
- Code: None

**Bridging the Gap between Model Explanations in Partially Annotated Multi-label Classification**
**弥合部分标注多标签分类中模型解释的差距**

- Paper: https://arxiv.org/abs/2304.01804
- Code: None

**HyperCUT: Video Sequence from a Single Blurry Image using Unsupervised Ordering**
**HyperCUT：使用无监督排序从单张模糊图像生成视频序列**

- Paper: https://arxiv.org/abs/2304.01686
- Code: None

**On the Stability-Plasticity Dilemma of Class-Incremental Learning**
**关于类增量学习的稳定性-塑性困境**

- Paper: https://arxiv.org/abs/2304.01663
- Code: None

**Defending Against Patch-based Backdoor Attacks on Self-Supervised Learning**
**防御基于补丁的自监督学习后门攻击**

- Paper: https://arxiv.org/abs/2304.01482
- Code: None

**VNE: An Effective Method for Improving Deep Representation by Manipulating Eigenvalue Distribution**
**VNE：通过操纵特征值分布来提高深度表示的有效方法**

- Paper: https://arxiv.org/abs/2304.01434
- Code: https://github.com/jaeill/CVPR23-VNE

**Detecting and Grounding Multi-Modal Media Manipulation**
**检测和定位多模态媒体操纵**

- Homepage: https://rshaojimmy.github.io/Projects/MultiModal-DeepFake
- Paper: https://arxiv.org/abs/2304.02556
- Code: https://github.com/rshaojimmy/MultiModal-DeepFake

**Meta-causal Learning for Single Domain Generalization**
**元因果学习用于单域泛化**

- Paper: https://arxiv.org/abs/2304.03709
- Code: None

**Disentangling Writer and Character Styles for Handwriting Generation**
**解开书写风格与角色风格以实现手写生成**

- Paper: https://arxiv.org/abs/2303.14736
- Code: https://github.com/dailenson/SDT

**DexArt: Benchmarking Generalizable Dexterous Manipulation with Articulated Objects**
**DexArt：使用可动对象进行泛化灵活操作的基准测试**

- Homepage: https://www.chenbao.tech/dexart/

- Code: https://github.com/Kami-code/dexart-release

**Hidden Gems: 4D Radar Scene Flow Learning Using Cross-Modal Supervision**
**隐藏宝石：利用跨模态监督进行4D雷达场景流学习**

- Homepage: https://toytiny.github.io/publication/23-cmflow-cvpr/index.html 
- Paper: https://arxiv.org/abs/2303.00462
- Code: https://github.com/Toytiny/CMFlow

**Marching-Primitives: Shape Abstraction from Signed Distance Function**
**行进原语：从符号距离函数中提取形状抽象**

- Paper: https://arxiv.org/abs/2303.13190
- Code: https://github.com/ChirikjianLab/Marching-Primitives

**Towards Trustable Skin Cancer Diagnosis via Rewriting Model's Decision**
**通过重写模型决策实现可信赖的皮肤癌诊断**

- Paper: https://arxiv.org/abs/2303.00885
- Code: None
