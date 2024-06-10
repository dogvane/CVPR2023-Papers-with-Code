# CVPR 2024 论文和开源项目合集(Papers with Code)

CVPR 2024 decisions are now available on OpenReview！

> 注0：项目来自于 https://github.com/amusi/CVPR2024-Papers-with-Code， 当前项目将原文里的标题用翻译工具转为中文，未做修订，仅作参考

> 注1：欢迎各位大佬提交issue，分享CVPR 2024论文和开源项目！
>
> 注2：关于往年CV顶会论文以及其他优质CV论文和大盘点，详见： https://github.com/amusi/daily-paper-computer-vision
>
> - [CVPR 2019](CVPR2019-Papers-with-Code.md)
> - [CVPR 2020](CVPR2020-Papers-with-Code.md)
> - [CVPR 2021](CVPR2021-Papers-with-Code.md)
> - [CVPR 2022](CVPR2022-Papers-with-Code.md)
> - [CVPR 2023](CVPR2022-Papers-with-Code.md)

欢迎扫码加入【CVer学术交流群】，这是最大的计算机视觉AI知识星球！每日更新，第一时间分享最新最前沿的计算机视觉、AI绘画、图像处理、深度学习、自动驾驶、医疗影像和AIGC等方向的学习资料，学起来！

![](CVer学术交流群.png)

# 【CVPR 2024 论文开源目录】

- [3DGS(Gaussian Splatting)](#3DGS)
- [Avatars](#Avatars)
- [Backbone](#Backbone)
- [CLIP](#CLIP)
- [MAE](#MAE)
- [Embodied AI](#Embodied-AI)
- [GAN](#GAN)
- [GNN](#GNN)
- [多模态大语言模型(MLLM)](#MLLM)
- [大语言模型(LLM)](#LLM)
- [NAS](#NAS)
- [OCR](#OCR)
- [NeRF](#NeRF)
- [DETR](#DETR)
- [Prompt](#Prompt)
- [扩散模型(Diffusion Models)](#Diffusion)
- [ReID(重识别)](#ReID)
- [长尾分布(Long-Tail)](#Long-Tail)
- [Vision Transformer](#Vision-Transformer)
- [视觉和语言(Vision-Language)](#VL)
- [自监督学习(Self-supervised Learning)](#SSL)
- [数据增强(Data Augmentation)](#DA)
- [目标检测(Object Detection)](#Object-Detection)
- [异常检测(Anomaly Detection)](#Anomaly-Detection)
- [目标跟踪(Visual Tracking)](#VT)
- [语义分割(Semantic Segmentation)](#Semantic-Segmentation)
- [实例分割(Instance Segmentation)](#Instance-Segmentation)
- [全景分割(Panoptic Segmentation)](#Panoptic-Segmentation)
- [医学图像(Medical Image)](#MI)
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
- [自动驾驶(Autonomous Driving)](#Autonomous-Driving)
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
- [3D生成(3D Generation)](#3D-Generation)
- [视频理解(Video Understanding)](#Video-Understanding)
- [行为检测(Action Detection)](#Action-Detection)
- [文本检测(Text Detection)](#Text-Detection)
- [知识蒸馏(Knowledge Distillation)](#KD)
- [模型剪枝(Model Pruning)](#Pruning)
- [图像压缩(Image Compression)](#IC)
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
- [视频质量评价(Video Quality Assessment)](#Video-Quality-Assessment)
- [数据集(Datasets)](#Datasets)
- [新任务(New Tasks)](#New-Tasks)
- [其他(Others)](#Others)

<a name="3DGS"></a>

# 3DGS(Gaussian Splatting)

**Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering**
**Scaffold-GS：结构化3D高斯函数，用于视图自适应渲染**

- Homepage: https://city-super.github.io/scaffold-gs/
- Paper: https://arxiv.org/abs/2312.00109
- Code: https://github.com/city-super/Scaffold-GS

**GPS-Gaussian: Generalizable Pixel-wise 3D Gaussian Splatting for Real-time Human Novel View Synthesis**
**GPS-Gaussian：可泛化的像素级3D高斯分层技术，用于实时生成人类新颖视角合成**

- Homepage: https://shunyuanzheng.github.io/GPS-Gaussian 
- Paper: https://arxiv.org/abs/2312.02155
- Code: https://github.com/ShunyuanZheng/GPS-Gaussian

**GaussianAvatar: Towards Realistic Human Avatar Modeling from a Single Video via Animatable 3D Gaussians**
**高斯头像：通过可动3D高斯实现从单个视频中生成逼真的人类头像建模**

- Paper: https://arxiv.org/abs/2312.02134
- Code: https://github.com/huliangxiao/GaussianAvatar

**GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting**
**高斯编辑器：利用高斯喷溅技术实现快速可控的3D编辑**

- Paper: https://arxiv.org/abs/2311.14521
- Code: https://github.com/buaacyw/GaussianEditor 

**Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction**
**可变形3D高斯函数用于高保真单目动态场景重建**

- Homepage: https://ingra14m.github.io/Deformable-Gaussians/ 
- Paper: https://arxiv.org/abs/2309.13101
- Code: https://github.com/ingra14m/Deformable-3D-Gaussians

**SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes**
**SC-GS：用于可编辑动态场景的稀疏控制高斯喷溅**

- Homepage: https://yihua7.github.io/SC-GS-web/ 
- Paper: https://arxiv.org/abs/2312.14937
- Code: https://github.com/yihua7/SC-GS

**Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis**
**时空高斯特征喷溅技术用于实时动态视图合成**

- Homepage: https://oppo-us-research.github.io/SpacetimeGaussians-website/ 
- Paper: https://arxiv.org/abs/2312.16812
- Code: https://github.com/oppo-us-research/SpacetimeGaussians

**DNGaussian: Optimizing Sparse-View 3D Gaussian Radiance Fields with Global-Local Depth Normalization**
**DNGaussian：通过全局-局部深度归一化优化稀疏视图3D高斯辐射场**

- Homepage: https://fictionarry.github.io/DNGaussian/
- Paper: https://arxiv.org/abs/2403.06912
- Code: https://github.com/Fictionarry/DNGaussian

**4D Gaussian Splatting for Real-Time Dynamic Scene Rendering**
**实时动态场景渲染的4D高斯散斑技术**

- Paper: https://arxiv.org/abs/2310.08528
- Code: https://github.com/hustvl/4DGaussians

**GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging 2D and 3D Diffusion Models**
**高斯梦者：通过连接二维和三维扩散模型实现从文本到3D高斯的快速生成**

- Paper: https://arxiv.org/abs/2310.08529
- Code: https://github.com/hustvl/GaussianDreamer

<a name="Avatars"></a>

# Avatars

**GaussianAvatar: Towards Realistic Human Avatar Modeling from a Single Video via Animatable 3D Gaussians**
**高斯头像：通过可动画的3D高斯实现从单个视频到逼真的人像建模**

- Paper: https://arxiv.org/abs/2312.02134
- Code: https://github.com/huliangxiao/GaussianAvatar

**Real-Time Simulated Avatar from Head-Mounted Sensors**
**实时模拟头部佩戴传感器生成的虚拟形象**

- Homepage: https://www.zhengyiluo.com/SimXR/
- Paper: https://arxiv.org/abs/2403.06862

<a name="Backbone"></a>

# Backbone

**RepViT: Revisiting Mobile CNN From ViT Perspective**
**RepViT：从ViT视角重新审视移动CNN**

- Paper: https://arxiv.org/abs/2307.09283
- Code: https://github.com/THU-MIG/RepViT

**TransNeXt: Robust Foveal Visual Perception for Vision Transformers**
**TransNeXt：针对视觉Transformer的鲁棒性黄斑视觉感知**

- Paper: https://arxiv.org/abs/2311.17132
- Code: https://github.com/DaiShiResearch/TransNeXt

<a name="CLIP"></a>

# CLIP

**Alpha-CLIP: A CLIP Model Focusing on Wherever You Want**
**Alpha-CLIP：一个聚焦于您所想之处的CLIP模型**

- Paper: https://arxiv.org/abs/2312.03818
- Code: https://github.com/SunzeY/AlphaCLIP

**FairCLIP: Harnessing Fairness in Vision-Language Learning**
**公平CLIP：在视觉-语言学习中利用公平性**

- Paper: https://arxiv.org/abs/2403.19949
- Code: https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP

<a name="MAE"></a>

# MAE

<a name="Embodied-AI"></a>

# Embodied AI

**EmbodiedScan: A Holistic Multi-Modal 3D Perception Suite Towards Embodied AI**
**具身扫描：面向具身人工智能的全方位多模态3D感知套件**

- Homepage: https://tai-wang.github.io/embodiedscan/
- Paper: https://arxiv.org/abs/2312.16170
- Code: https://github.com/OpenRobotLab/EmbodiedScan

**MP5: A Multi-modal Open-ended Embodied System in Minecraft via Active Perception**
**MP5：通过主动感知在Minecraft中的多模态开放式具身系统**

- Homepage: https://iranqin.github.io/MP5.github.io/ 
- Paper: https://arxiv.org/abs/2312.07472
- Code: https://github.com/IranQin/MP5

**LEMON: Learning 3D Human-Object Interaction Relation from 2D Images**
**柠檬：从二维图像中学习3D人-物交互关系**

- Paper: https://arxiv.org/abs/2312.08963
- Code: https://github.com/yyvhang/lemon_3d 

<a name="GAN"></a>

# GAN

<a name="OCR"></a>

# OCR

**An Empirical Study of Scaling Law for OCR**
**OCR缩放定律的实证研究**

- Paper: https://arxiv.org/abs/2401.00028
- Code: https://github.com/large-ocr-model/large-ocr-model.github.io

**ODM: A Text-Image Further Alignment Pre-training Approach for Scene Text Detection and Spotting**
**ODM：一种用于场景文本检测和定位的文本-图像进一步对齐预训练方法**

- Paper: https://arxiv.org/abs/2403.00303
- Code: https://github.com/PriNing/ODM 

<a name="NeRF"></a>

# NeRF

**PIE-NeRF🍕: Physics-based Interactive Elastodynamics with NeRF**
**PIE-NeRF🍕：基于物理的交互式弹性动力学与NeRF**

- Paper: https://arxiv.org/abs/2311.13099
- Code: https://github.com/FYTalon/pienerf/ 

<a name="DETR"></a>

# DETR

**DETRs Beat YOLOs on Real-time Object Detection**
**DETR在实时目标检测上击败了YOLOs**

- Paper: https://arxiv.org/abs/2304.08069
- Code: https://github.com/lyuwenyu/RT-DETR

**Salience DETR: Enhancing Detection Transformer with Hierarchical Salience Filtering Refinement**
**显著性DETR：通过层次显著性过滤精炼增强检测Transformer**

- Paper: https://arxiv.org/abs/2403.16131
- Code: https://github.com/xiuqhou/Salience-DETR

<a name="Prompt"></a>

# Prompt

<a name="MLLM"></a>

# 多模态大语言模型(MLLM)

**mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration**
**mPLUG-Owl2：通过模态协作革新多模态大型语言模型**

- Paper: https://arxiv.org/abs/2311.04257
- Code: https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2

**Link-Context Learning for Multimodal LLMs**
**多模态LLM的链接上下文学习**

- Paper: https://arxiv.org/abs/2308.07891
- Code: https://github.com/isekai-portal/Link-Context-Learning/tree/main 

**OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation**
**OPERA：通过过度信任惩罚和反思-分配缓解多模态大型语言模型中的幻觉**

- Paper: https://arxiv.org/abs/2311.17911
- Code: https://github.com/shikiw/OPERA

**Making Large Multimodal Models Understand Arbitrary Visual Prompts**
**制作能够理解任意视觉提示的大型多模态模型**

- Homepage: https://vip-llava.github.io/ 
- Paper: https://arxiv.org/abs/2312.00784

**Pink: Unveiling the power of referential comprehension for multi-modal llms**
**粉红色：揭示多模态LLMs中参照理解的力量**

- Paper: https://arxiv.org/abs/2310.00582
- Code: https://github.com/SY-Xuan/Pink

**Chat-UniVi: Unified Visual Representation Empowers Large Language Models with Image and Video Understanding**
**Chat-UniVi：统一视觉表示通过图像和视频理解赋能大型语言模型**

- Paper: https://arxiv.org/abs/2311.08046
- Code: https://github.com/PKU-YuanGroup/Chat-UniVi

**OneLLM: One Framework to Align All Modalities with Language**
**OneLLM：一个框架，将所有模态与语言对齐**

- Paper: https://arxiv.org/abs/2312.03700
- Code: https://github.com/csuhan/OneLLM

<a name="LLM"></a>

# 大语言模型(LLM)

**VTimeLLM: Empower LLM to Grasp Video Moments**
**VTimeLLM：赋予LLM把握视频瞬间的能力**

- Paper: https://arxiv.org/abs/2311.18445
- Code: https://github.com/huangb23/VTimeLLM 

<a name="NAS"></a>

# NAS

<a name="ReID"></a>

# ReID(重识别)

**Magic Tokens: Select Diverse Tokens for Multi-modal Object Re-Identification**
**魔法令牌：为多模态物体重识别选择多样化的令牌**

- Paper: https://arxiv.org/abs/2403.10254
- Code: https://github.com/924973292/EDITOR 

**Noisy-Correspondence Learning for Text-to-Image Person Re-identification**
**文本到图像人物重识别的噪声对应学习**

- Paper: https://arxiv.org/abs/2308.09911

- Code : https://github.com/QinYang79/RDE 

<a name="Diffusion"></a>

# 扩散模型(Diffusion Models)

**InstanceDiffusion: Instance-level Control for Image Generation**
**实例扩散：图像生成中的实例级控制**

- Homepage: https://people.eecs.berkeley.edu/~xdwang/projects/InstDiff/

- Paper: https://arxiv.org/abs/2402.03290
- Code: https://github.com/frank-xwang/InstanceDiffusion

**Residual Denoising Diffusion Models**
**残差去噪扩散模型**

- Paper: https://arxiv.org/abs/2308.13712
- Code: https://github.com/nachifur/RDDM

**DeepCache: Accelerating Diffusion Models for Free**
**DeepCache：免费加速扩散模型**

- Paper: https://arxiv.org/abs/2312.00858
- Code: https://github.com/horseee/DeepCache

**DEADiff: An Efficient Stylization Diffusion Model with Disentangled Representations**
**DEADiff：一种具有解耦表示的高效风格扩散模型**

- Homepage: https://tianhao-qi.github.io/DEADiff/ 

- Paper: https://arxiv.org/abs/2403.06951
- Code: https://github.com/Tianhao-Qi/DEADiff_code

**SVGDreamer: Text Guided SVG Generation with Diffusion Model**
**SVGDreamer：基于扩散模型的文本引导SVG生成**

- Paper: https://arxiv.org/abs/2312.16476
- Code: https://ximinng.github.io/SVGDreamer-project/

**InteractDiffusion: Interaction-Control for Text-to-Image Diffusion Model**
**交互式扩散：文本到图像扩散模型的交互控制**

- Paper: https://arxiv.org/abs/2312.05849
- Code: https://github.com/jiuntian/interactdiffusion

**MMA-Diffusion: MultiModal Attack on Diffusion Models**
**MMA-Diffusion：对扩散模型的跨模态攻击**

- Paper: https://arxiv.org/abs/2311.17516
- Code: https://github.com/yangyijune/MMA-Diffusion

**VMC: Video Motion Customization using Temporal Attention Adaption for Text-to-Video Diffusion Models**
**视频运动定制：利用时间注意力自适应的文本到视频扩散模型**

- Homeoage: https://video-motion-customization.github.io/ 
- Paper: https://arxiv.org/abs/2312.00845
- Code: https://github.com/HyeonHo99/Video-Motion-Customization

<a name="Vision-Transformer"></a>

# Vision Transformer

**TransNeXt: Robust Foveal Visual Perception for Vision Transformers**
**TransNeXt：为视觉Transformer提供鲁棒的黄斑视觉感知**

- Paper: https://arxiv.org/abs/2311.17132
- Code: https://github.com/DaiShiResearch/TransNeXt

**RepViT: Revisiting Mobile CNN From ViT Perspective**
**RepViT：从ViT视角重新审视移动CNN**

- Paper: https://arxiv.org/abs/2307.09283
- Code: https://github.com/THU-MIG/RepViT

**A General and Efficient Training for Transformer via Token Expansion**
**通过词元扩展进行通用且高效的Transformer训练**

- Paper: https://arxiv.org/abs/2404.00672
- Code: https://github.com/Osilly/TokenExpansion 

<a name="VL"></a>

# 视觉和语言(Vision-Language)

**PromptKD: Unsupervised Prompt Distillation for Vision-Language Models**
**提示KD：用于视觉-语言模型的无监督提示蒸馏**

- Paper: https://arxiv.org/abs/2403.02781
- Code: https://github.com/zhengli97/PromptKD

**FairCLIP: Harnessing Fairness in Vision-Language Learning**
**公平CLIP：在视觉语言学习中利用公平性**

- Paper: https://arxiv.org/abs/2403.19949
- Code: https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP

<a name="Object-Detection"></a>

# 目标检测(Object Detection)

**DETRs Beat YOLOs on Real-time Object Detection**
**DETRs在实时目标检测方面击败了YOLOs**

- Paper: https://arxiv.org/abs/2304.08069
- Code: https://github.com/lyuwenyu/RT-DETR

**Boosting Object Detection with Zero-Shot Day-Night Domain Adaptation**
**利用零样本日夜间域适应增强目标检测**

- Paper: https://arxiv.org/abs/2312.01220
- Code: https://github.com/ZPDu/Boosting-Object-Detection-with-Zero-Shot-Day-Night-Domain-Adaptation 

**YOLO-World: Real-Time Open-Vocabulary Object Detection**
**YOLO-World：实时开放词汇物体检测**

- Paper: https://arxiv.org/abs/2401.17270
- Code: https://github.com/AILab-CVC/YOLO-World

**Salience DETR: Enhancing Detection Transformer with Hierarchical Salience Filtering Refinement**
**显著性DETR：通过分层显著性滤波优化提升检测Transformer**

- Paper: https://arxiv.org/abs/2403.16131
- Code: https://github.com/xiuqhou/Salience-DETR

<a name="Anomaly-Detection"></a>

# 异常检测(Anomaly Detection)

**Anomaly Heterogeneity Learning for Open-set Supervised Anomaly Detection**
**开放集监督异常检测中的异常异质性学习**

- Paper: https://arxiv.org/abs/2310.12790
- Code: https://github.com/mala-lab/AHL

<a name="VT"></a>

# 目标跟踪(Object Tracking)

**Delving into the Trajectory Long-tail Distribution for Muti-object Tracking**
**深入探究多目标跟踪中的轨迹长尾分布**

- Paper: https://arxiv.org/abs/2403.04700
- Code: https://github.com/chen-si-jia/Trajectory-Long-tail-Distribution-for-MOT 

<a name="Semantic-Segmentation"></a>

# 语义分割(Semantic Segmentation)

**Stronger, Fewer, & Superior: Harnessing Vision Foundation Models for Domain Generalized Semantic Segmentation**
**更强、更少、更优越：利用视觉基础模型实现领域泛化语义分割**

- Paper: https://arxiv.org/abs/2312.04265
- Code: https://github.com/w1oves/Rein

**SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation**
**开放词汇语义分割的简单编码器-解码器：SED**

- Paper: https://arxiv.org/abs/2311.15537
- Code: https://github.com/xb534/SED 

<a name="MI"></a>

# 医学图像(Medical Image)

**Feature Re-Embedding: Towards Foundation Model-Level Performance in Computational Pathology**
**特征再嵌入：迈向计算病理学基础模型级别的性能**

- Paper: https://arxiv.org/abs/2402.17228
- Code: https://github.com/DearCaat/RRT-MIL

**VoCo: A Simple-yet-Effective Volume Contrastive Learning Framework for 3D Medical Image Analysis**
**VoCo：一种简单而有效的3D医学图像分析体积对比学习框架**

- Paper: https://arxiv.org/abs/2402.17300
- Code: https://github.com/Luffy03/VoCo

**ChAda-ViT : Channel Adaptive Attention for Joint Representation Learning of Heterogeneous Microscopy Images**
**ChAda-ViT：异构显微镜图像联合表示学习的通道自适应注意力**

- Paper: https://arxiv.org/abs/2311.15264
- Code: https://github.com/nicoboou/chada_vit 

<a name="MIS"></a>

# 医学图像分割(Medical Image Segmentation)



<a name="Autonomous-Driving"></a>

# 自动驾驶(Autonomous Driving)

**UniPAD: A Universal Pre-training Paradigm for Autonomous Driving**
**UniPAD：自动驾驶的通用预训练范式**

- Paper: https://arxiv.org/abs/2310.08370
- Code: https://github.com/Nightmare-n/UniPAD

**Cam4DOcc: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications**
**Cam4DOcc：自动驾驶应用中仅使用摄像头进行4D占用预测的基准测试**

- Paper: https://arxiv.org/abs/2311.17663
- Code: https://github.com/haomo-ai/Cam4DOcc

**Memory-based Adapters for Online 3D Scene Perception**
**基于内存的在线3D场景感知适配器**

- Paper: https://arxiv.org/abs/2403.06974
- Code: https://github.com/xuxw98/Online3D

**Symphonize 3D Semantic Scene Completion with Contextual Instance Queries**
**将3D语义场景补全与上下文实例查询同步化**

- Paper: https://arxiv.org/abs/2306.15670
- Code: https://github.com/hustvl/Symphonies

**A Real-world Large-scale Dataset for Roadside Cooperative Perception**
**真实世界大规模道路侧协同感知数据集**

- Paper: https://arxiv.org/abs/2403.10145
- Code: https://github.com/AIR-THU/DAIR-RCooper

**Adaptive Fusion of Single-View and Multi-View Depth for Autonomous Driving**
**单视和多视深度自适应融合用于自动驾驶**

- Paper: https://arxiv.org/abs/2403.07535
- Code: https://github.com/Junda24/AFNet

**Traffic Scene Parsing through the TSP6K Dataset**
**通过TSP6K数据集进行交通场景解析**

- Paper: https://arxiv.org/pdf/2303.02835.pdf
- Code: https://github.com/PengtaoJiang/TSP6K 

<a name="3D-Point-Cloud"></a>

# 3D点云(3D-Point-Cloud)



<a name="3DOD"></a>

# 3D目标检测(3D Object Detection)

**PTT: Point-Trajectory Transformer for Efficient Temporal 3D Object Detection**
**PTT：高效时序3D目标检测的点-轨迹变换器**

- Paper: https://arxiv.org/abs/2312.08371
- Code: https://github.com/kuanchihhuang/PTT

**UniMODE: Unified Monocular 3D Object Detection**
**UniMODE：统一单目3D目标检测**

- Paper: https://arxiv.org/abs/2402.18573

<a name="3DOD"></a>

# 3D语义分割(3D Semantic Segmentation)

<a name="Image-Editing"></a>

# 图像编辑(Image Editing)

**Edit One for All: Interactive Batch Image Editing**
**一键编辑：交互式批量图像编辑**

- Homepage: https://thaoshibe.github.io/edit-one-for-all 
- Paper: https://arxiv.org/abs/2401.10219
- Code: https://github.com/thaoshibe/edit-one-for-all

<a name="Video-Editing"></a>

# 视频编辑(Video Editing)

**MaskINT: Video Editing via Interpolative Non-autoregressive Masked Transformers**
**MaskINT：通过插值非自回归掩码变换器进行视频编辑**

- Homepage:  [https://maskint.github.io](https://maskint.github.io/) 

- Paper: https://arxiv.org/abs/2312.12468

<a name="LLV"></a>

# Low-level Vision

**Residual Denoising Diffusion Models**
**残差去噪扩散模型**

- Paper: https://arxiv.org/abs/2308.13712
- Code: https://github.com/nachifur/RDDM

**Boosting Image Restoration via Priors from Pre-trained Models**
**通过预训练模型先验信息增强图像恢复**

- Paper: https://arxiv.org/abs/2403.06793

<a name="SR"></a>

# 超分辨率(Super-Resolution)

**SeD: Semantic-Aware Discriminator for Image Super-Resolution**
**SeD：图像超分辨率中的语义感知判别器**

- Paper: https://arxiv.org/abs/2402.19387
- Code: https://github.com/lbc12345/SeD

**APISR: Anime Production Inspired Real-World Anime Super-Resolution**
**APISR：受动画制作启发的现实世界动画超分辨率**

- Paper: https://arxiv.org/abs/2403.01598
- Code: https://github.com/Kiteretsu77/APISR 

<a name="Denoising"></a>

# 去噪(Denoising)

## 图像去噪(Image Denoising)

<a name="3D-Human-Pose-Estimation"></a>

# 3D人体姿态估计(3D Human Pose Estimation)

**Hourglass Tokenizer for Efficient Transformer-Based 3D Human Pose Estimation**
**沙漏分词器用于高效基于Transformer的3D人体姿态估计**

- Paper: https://arxiv.org/abs/2311.12028
- Code: https://github.com/NationalGAILab/HoT 

<a name="Image-Generation"></a>

# 图像生成(Image Generation)

**InstanceDiffusion: Instance-level Control for Image Generation**
**实例扩散：图像生成中的实例级控制**

- Homepage: https://people.eecs.berkeley.edu/~xdwang/projects/InstDiff/

- Paper: https://arxiv.org/abs/2402.03290
- Code: https://github.com/frank-xwang/InstanceDiffusion

**ECLIPSE: A Resource-Efficient Text-to-Image Prior for Image Generations**
**ECLIPSE：一种高效利用资源的文本到图像生成先验**

- Homepage: https://eclipse-t2i.vercel.app/
- Paper: https://arxiv.org/abs/2312.04655

- Code: https://github.com/eclipse-t2i/eclipse-inference

**Instruct-Imagen: Image Generation with Multi-modal Instruction**
**指令-图像：多模态指令下的图像生成**

- Paper: https://arxiv.org/abs/2401.01952

**Residual Denoising Diffusion Models**
**残差去噪扩散模型**

- Paper: https://arxiv.org/abs/2308.13712
- Code: https://github.com/nachifur/RDDM

**UniGS: Unified Representation for Image Generation and Segmentation**
**UniGS：图像生成与分割的统一表示**

- Paper: https://arxiv.org/abs/2312.01985

**Multi-Instance Generation Controller for Text-to-Image Synthesis**
**多实例生成控制器，用于文本到图像合成**

- Paper: https://arxiv.org/abs/2402.05408
- Code: https://github.com/limuloo/migc

**SVGDreamer: Text Guided SVG Generation with Diffusion Model**
**SVGDreamer：基于扩散模型的文本引导SVG生成**

- Paper: https://arxiv.org/abs/2312.16476
- Code: https://ximinng.github.io/SVGDreamer-project/

**InteractDiffusion: Interaction-Control for Text-to-Image Diffusion Model**
**交互扩散：文本到图像扩散模型的交互控制**

- Paper: https://arxiv.org/abs/2312.05849
- Code: https://github.com/jiuntian/interactdiffusion

**Ranni: Taming Text-to-Image Diffusion for Accurate Prompt Following**
**Ranni：驯服文本到图像扩散，实现准确提示跟随**

- Paper: https://arxiv.org/abs/2311.17002
- Code: https://github.com/ali-vilab/Ranni

<a name="Video-Generation"></a>

# 视频生成(Video Generation)

**Vlogger: Make Your Dream A Vlog**
**视频博主：让你的梦想成为一档视频博客**

- Paper: https://arxiv.org/abs/2401.09414
- Code: https://github.com/Vchitect/Vlogger

**VBench: Comprehensive Benchmark Suite for Video Generative Models**
**VBench：视频生成模型的全面基准测试套件**

- Homepage: https://vchitect.github.io/VBench-project/ 
- Paper: https://arxiv.org/abs/2311.17982
- Code: https://github.com/Vchitect/VBench

**VMC: Video Motion Customization using Temporal Attention Adaption for Text-to-Video Diffusion Models**
**视频运动定制：利用时间注意力自适应的文本到视频扩散模型**

- Homeoage: https://video-motion-customization.github.io/ 
- Paper: https://arxiv.org/abs/2312.00845
- Code: https://github.com/HyeonHo99/Video-Motion-Customization

<a name="3D-Generation"></a>

# 3D生成

**CityDreamer: Compositional Generative Model of Unbounded 3D Cities**
**城市梦想家：无限3D城市的构图生成模型**

- Homepage: https://haozhexie.com/project/city-dreamer/ 
- Paper: https://arxiv.org/abs/2309.00610
- Code: https://github.com/hzxie/city-dreamer

**LucidDreamer: Towards High-Fidelity Text-to-3D Generation via Interval Score Matching**
**清醒梦境者：通过区间得分匹配实现高保真文本到3D生成**

- Paper: https://arxiv.org/abs/2311.11284
- Code: https://github.com/EnVision-Research/LucidDreamer 

<a name="Video-Understanding"></a>

# 视频理解(Video Understanding)

**MVBench: A Comprehensive Multi-modal Video Understanding Benchmark**
**MVBench：一个全面的跨模态视频理解基准**

- Paper: https://arxiv.org/abs/2311.17005
- Code: https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2 

<a name="KD"></a>

# 知识蒸馏(Knowledge Distillation)

**Logit Standardization in Knowledge Distillation**
**知识蒸馏中的Logit标准化**

- Paper: https://arxiv.org/abs/2403.01427
- Code: https://github.com/sunshangquan/logit-standardization-KD

**Efficient Dataset Distillation via Minimax Diffusion**
**通过最小-最大扩散进行高效数据集蒸馏**

- Paper: https://arxiv.org/abs/2311.15529
- Code: https://github.com/vimar-gu/MinimaxDiffusion

<a name="Stereo-Matching"></a>

# 立体匹配(Stereo Matching)

**Neural Markov Random Field for Stereo Matching**
**神经马尔可夫随机场用于立体匹配**

- Paper: https://arxiv.org/abs/2403.11193
- Code: https://github.com/aeolusguan/NMRF 

<a name="SGG"></a>

# 场景图生成(Scene Graph Generation)

**HiKER-SGG: Hierarchical Knowledge Enhanced Robust Scene Graph Generation**
**HiKER-SGG：层次知识增强鲁棒场景图生成**

- Homepage: https://zhangce01.github.io/HiKER-SGG/ 
- Paper : https://arxiv.org/abs/2403.12033
- Code: https://github.com/zhangce01/HiKER-SGG

<a name="Video-Quality-Assessment"></a>

# 视频质量评价(Video Quality Assessment)

**KVQ: Kaleidoscope Video Quality Assessment for Short-form Videos**
**KVQ：短视频的万花筒视频质量评估**

- Homepage: https://lixinustc.github.io/projects/KVQ/ 

- Paper: https://arxiv.org/abs/2402.07220
- Code: https://github.com/lixinustc/KVQ-Challenge-CVPR-NTIRE2024

<a name="Datasets"></a>

# 数据集(Datasets)

**A Real-world Large-scale Dataset for Roadside Cooperative Perception**
**现实世界大规模道路侧协同感知数据集**

- Paper: https://arxiv.org/abs/2403.10145
- Code: https://github.com/AIR-THU/DAIR-RCooper

**Traffic Scene Parsing through the TSP6K Dataset**
**通过TSP6K数据集进行交通场景解析**

- Paper: https://arxiv.org/pdf/2303.02835.pdf
- Code: https://github.com/PengtaoJiang/TSP6K 

<a name="Others"></a>

# 其他(Others)

**Object Recognition as Next Token Prediction**
**对象识别作为下一个标记预测**

- Paper: https://arxiv.org/abs/2312.02142
- Code: https://github.com/kaiyuyue/nxtp

**ParameterNet: Parameters Are All You Need for Large-scale Visual Pretraining of Mobile Networks**
**ParameterNet：参数即是所有，用于移动网络大规模视觉预训练**

- Paper: https://arxiv.org/abs/2306.14525
- Code: https://parameternet.github.io/ 

**Seamless Human Motion Composition with Blended Positional Encodings**
**无缝的人体运动合成与混合位置编码**

- Paper: https://arxiv.org/abs/2402.15509
- Code: https://github.com/BarqueroGerman/FlowMDM 

**LL3DA: Visual Interactive Instruction Tuning for Omni-3D Understanding, Reasoning, and Planning**
**LL3DA：用于全3D理解、推理和规划的视觉交互式指令调优**

- Homepage:  https://ll3da.github.io/ 

- Paper: https://arxiv.org/abs/2311.18651
- Code: https://github.com/Open3DA/LL3DA

 **CLOVA: A Closed-LOop Visual Assistant with Tool Usage and Update**

- Homepage: https://clova-tool.github.io/ 
- Paper: https://arxiv.org/abs/2312.10908

**MoMask: Generative Masked Modeling of 3D Human Motions**
**MoMask：3D人体动作的生成式掩码建模**

- Paper: https://arxiv.org/abs/2312.00063
- Code: https://github.com/EricGuo5513/momask-codes

 **Amodal Ground Truth and Completion in the Wild**

- Homepage: https://www.robots.ox.ac.uk/~vgg/research/amodal/ 
- Paper: https://arxiv.org/abs/2312.17247
- Code: https://github.com/Championchess/Amodal-Completion-in-the-Wild

**Improved Visual Grounding through Self-Consistent Explanations**
**通过自洽解释提升视觉定位**

- Paper: https://arxiv.org/abs/2312.04554
- Code: https://github.com/uvavision/SelfEQ

**ImageNet-D: Benchmarking Neural Network Robustness on Diffusion Synthetic Object**
**ImageNet-D：在扩散合成物体上基准测试神经网络鲁棒性**

- Homepage: https://chenshuang-zhang.github.io/imagenet_d/
- Paper: https://arxiv.org/abs/2403.18775
- Code: https://github.com/chenshuang-zhang/imagenet_d

**Learning from Synthetic Human Group Activities**
**从合成人类群体活动中学习**

- Homepage: https://cjerry1243.github.io/M3Act/ 
- Paper  https://arxiv.org/abs/2306.16772
- Code: https://github.com/cjerry1243/M3Act

**A Cross-Subject Brain Decoding Framework**
**跨学科大脑解码框架**

- Homepage: https://littlepure2333.github.io/MindBridge/
- Paper: https://arxiv.org/abs/2404.07850
- Code: https://github.com/littlepure2333/MindBridge

**Multi-Task Dense Prediction via Mixture of Low-Rank Experts**
**通过低秩专家混合的多任务密集预测**

- Paper : https://arxiv.org/abs/2403.17749
- Code: https://github.com/YuqiYang213/MLoRE

**Contrastive Mean-Shift Learning for Generalized Category Discovery**
**对比均值漂移学习用于广义类别发现**

- Homepage: https://postech-cvlab.github.io/cms/ 
- Paper: https://arxiv.org/abs/2404.09451
- Code: https://github.com/sua-choi/CMS
  
