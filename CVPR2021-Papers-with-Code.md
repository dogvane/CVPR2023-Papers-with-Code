# CVPR 2021 论文和开源项目合集(Papers with Code)

[CVPR 2021](http://cvpr2021.thecvf.com/) 论文和开源项目合集(papers with code)！

CVPR 2021 收录列表：http://cvpr2021.thecvf.com/sites/default/files/2021-03/accepted_paper_ids.txt

> 注1：欢迎各位大佬提交issue，分享CVPR 2021论文和开源项目！
>
> 注2：关于往年CV顶会论文以及其他优质CV论文和大盘点，详见： https://github.com/amusi/daily-paper-computer-vision

如果你想了解最新最优质的的CV论文、开源项目和学习资料，欢迎扫码加入【CVer学术交流群】！互相学习，一起进步~ 

![](CVer学术交流群.png)

## 【CVPR 2021 论文开源目录】

- [Best Paper](#Best-Paper)
- [Backbone](#Backbone)
- [NAS](#NAS)
- [GAN](#GAN)
- [VAE](#VAE)
- [Visual Transformer](#Visual-Transformer)
- [Regularization](#Regularization)
- [SLAM](#SLAM)
- [长尾分布(Long-Tailed)](#Long-Tailed)
- [数据增广(Data Augmentation)](#DA)
- [无监督/自监督(Self-Supervised)](#Un/Self-Supervised)
- [半监督(Semi-Supervised)](#Semi-Supervised)
- [胶囊网络(Capsule Network)](#Capsule-Network)
- [图像分类(Image Classification](#Image-Classification)
- [2D目标检测(Object Detection)](#Object-Detection)
- [单/多目标跟踪(Object Tracking)](#Object-Tracking)
- [语义分割(Semantic Segmentation)](#Semantic-Segmentation)
- [实例分割(Instance Segmentation)](#Instance-Segmentation)
- [全景分割(Panoptic Segmentation)](#Panoptic-Segmentation)
- [医学图像分割(Medical Image Segmentation)](#Medical-Image-Segmentation)
- [视频目标分割(Video-Object-Segmentation)](#VOS)
- [交互式视频目标分割(Interactive-Video-Object-Segmentation)](#IVOS)
- [显著性检测(Saliency Detection)](#Saliency-Detection)
- [伪装物体检测(Camouflaged Object Detection)](#Camouflaged-Object-Detection)
- [协同显著性检测(Co-Salient Object Detection)](#CoSOD)
- [图像抠图(Image Matting)](#Matting)
- [行人重识别(Person Re-identification)](#Re-ID)
- [行人搜索(Person Search)](#Person-Search)
- [视频理解/行为识别(Video Understanding)](#Video-Understanding)
- [人脸识别(Face Recognition)](#Face-Recognition)
- [人脸检测(Face Detection)](#Face-Detection)
- [人脸活体检测(Face Anti-Spoofing)](#Face-Anti-Spoofing)
- [Deepfake检测(Deepfake Detection)](#Deepfake-Detection)
- [人脸年龄估计(Age-Estimation)](#Age-Estimation)
- [人脸表情识别(Facial-Expression-Recognition)](#FER)
- [Deepfakes](#Deepfakes)
- [人体解析(Human Parsing)](#Human-Parsing)
- [2D/3D人体姿态估计(2D/3D Human Pose Estimation)](#Human-Pose-Estimation)
- [动物姿态估计(Animal Pose Estimation)](#Animal-Pose-Estimation)
- [手部姿态估计(Hand Pose Estimation)](#Hand-Pose-Estimation)
- [Human Volumetric Capture](#Human-Volumetric-Capture)
- [场景文本识别(Scene Text Recognition)](#Scene-Text-Recognition)
- [图像压缩(Image Compression)](#Image-Compression)
- [模型压缩/剪枝/量化](#Model-Compression)
- [知识蒸馏(Knowledge Distillation)](#KD)
- [超分辨率(Super-Resolution)](#Super-Resolution)
- [去雾(Dehazing)](#Dehazing)
- [图像恢复(Image Restoration)](#Image-Restoration)
- [图像补全(Image Inpainting)](#Image-Inpainting)
- [图像编辑(Image Editing)](#Image-Editing)
- [图像描述(Image Captioning)](#Image-Captioning)
- [字体生成(Font Generation)](#Font-Generation)
- [图像匹配(Image Matching)](#Image-Matching)
- [图像融合(Image Blending)](#Image-Blending)
- [反光去除(Reflection Removal)](#Reflection-Removal)
- [3D点云分类(3D Point Clouds Classification)](#3D-C)
- [3D目标检测(3D Object Detection)](#3D-Object-Detection)
- [3D语义分割(3D Semantic Segmentation)](#3D-Semantic-Segmentation)
- [3D全景分割(3D Panoptic Segmentation)](#3D-Panoptic-Segmentation)
- [3D目标跟踪(3D Object Tracking)](#3D-Object-Tracking)
- [3D点云配准(3D Point Cloud Registration)](#3D-PointCloud-Registration)
- [3D点云补全(3D-Point-Cloud-Completion)](#3D-Point-Cloud-Completion)
- [3D重建(3D Reconstruction)](#3D-Reconstruction)
- [6D位姿估计(6D Pose Estimation)](#6D-Pose-Estimation)
- [相机姿态估计(Camera Pose Estimation)](#Camera-Pose-Estimation)
- [深度估计(Depth Estimation)](#Depth-Estimation)
- [立体匹配(Stereo Matching)](#Stereo-Matching)
- [光流估计(Flow Estimation)](#Flow-Estimation)
- [车道线检测(Lane Detection)](#Lane-Detection)
- [轨迹预测(Trajectory Prediction)](#Trajectory-Prediction)
- [人群计数(Crowd Counting)](#Crowd-Counting)
- [对抗样本(Adversarial-Examples)](#AE)
- [图像检索(Image Retrieval)](#Image-Retrieval)
- [视频检索(Video Retrieval)](#Video-Retrieval)
- [跨模态检索(Cross-modal Retrieval)](#Cross-modal-Retrieval) 
- [Zero-Shot Learning](#Zero-Shot-Learning)
- [联邦学习(Federated Learning)](#Federated-Learning)
- [视频插帧(Video Frame Interpolation)](#Video-Frame-Interpolation)
- [视觉推理(Visual Reasoning)](#Visual-Reasoning)
- [图像合成(Image Synthesis)](#Image-Synthesis)
- [视图合成(Visual Synthesis)](#Visual-Synthesis)
- [风格迁移(Style Transfer)](#Style-Transfer)
- [布局生成(Layout Generation)](#Layout-Generation)
- [Domain Generalization](#Domain-Generalization)
- [Domain Adaptation](#Domain-Adaptation)
- [Open-Set](#Open-Set)
- [Adversarial Attack](#Adversarial-Attack)
- ["人-物"交互(HOI)检测](#HOI)
- [阴影去除(Shadow Removal)](#Shadow-Removal)
- [虚拟试衣(Virtual Try-On)](#Virtual-Try-On)
- [标签噪声(Label Noise)](#Label-Noise)
- [视频稳像(Video Stabilization)](#Video-Stabilization)
- [数据集(Datasets)](#Datasets)
- [其他(Others)](#Others)
- [待添加(TODO)](#TO-DO)
- [不确定中没中(Not Sure)](#Not-Sure)

<a name="Best-Paper"></a>

# Best Paper

**GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields**
**长颈鹿：将场景表示为组成生成神经网络特征场的图结构**

- Homepage: https://m-niemeyer.github.io/project-pages/giraffe/index.html
- Paper(Oral): https://arxiv.org/abs/2011.12100

- Code: https://github.com/autonomousvision/giraffe

- Demo: http://www.youtube.com/watch?v=fIaDXC-qRSg&vq=hd1080&autoplay=1

<a name="Backbone"></a>

# Backbone

**HR-NAS: Searching Efficient High-Resolution Neural Architectures with Lightweight Transformers**
**HR-NAS：利用轻量级变压器搜索高效的高分辨率神经架构**

- Paper(Oral): https://arxiv.org/abs/2106.06560

- Code: https://github.com/dingmyu/HR-NAS

**BCNet: Searching for Network Width with Bilaterally Coupled Network**
**BCNet：使用具有双向耦合网络的搜索网络宽度**

- Paper: https://arxiv.org/abs/2105.10533
- Code: None

**Decoupled Dynamic Filter Networks**
**分离动态滤波器网络**

- Homepage: https://thefoxofsky.github.io/project_pages/ddf
- Paper: https://arxiv.org/abs/2104.14107
- Code: https://github.com/thefoxofsky/DDF

**Lite-HRNet: A Lightweight High-Resolution Network**
**Lite-HRNet：轻量级高分辨率网络**

- Paper: https://arxiv.org/abs/2104.06403
- https://github.com/HRNet/Lite-HRNet

**CondenseNet V2: Sparse Feature Reactivation for Deep Networks**
**紧凑网络V2：稀疏特征激活对于深度网络**

- Paper: https://arxiv.org/abs/2104.04382

- Code: https://github.com/jianghaojun/CondenseNetV2

**Diverse Branch Block: Building a Convolution as an Inception-like Unit**
**Diverse Branch Block：将构建卷积作为异构分支单元**

- Paper: https://arxiv.org/abs/2103.13425

- Code: https://github.com/DingXiaoH/DiverseBranchBlock

**Scaling Local Self-Attention For Parameter Efficient Visual Backbones**
**缩放局部自注意力以实现参数高效的视觉骨干网络**

- Paper(Oral): https://arxiv.org/abs/2103.12731

- Code: None

**ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network**
**ReXNet：在卷积神经网络中的减小表示瓶颈**

- Paper: https://arxiv.org/abs/2007.00992
- Code:  https://github.com/clovaai/rexnet

**Involution: Inverting the Inherence of Convolution for Visual Recognition**
**革命：扭转卷积对于视觉识别的继承**

- Paper: https://github.com/d-li14/involution
- Code: https://arxiv.org/abs/2103.06255

**Coordinate Attention for Efficient Mobile Network Design**
**高效移动网络设计的协注意力**

- Paper:  https://arxiv.org/abs/2103.02907
- Code: https://github.com/Andrew-Qibin/CoordAttention

**Inception Convolution with Efficient Dilation Search**
**基于efficient dilation的卷积搜索**

- Paper:  https://arxiv.org/abs/2012.13587 
- Code: https://github.com/yifan123/IC-Conv

**RepVGG: Making VGG-style ConvNets Great Again**
**RepVGG：再次让VGG-风格卷积神经网络变得伟大**

- Paper: https://arxiv.org/abs/2101.03697
- Code: https://github.com/DingXiaoH/RepVGG

<a name="NAS"></a>

# NAS

**HR-NAS: Searching Efficient High-Resolution Neural Architectures with Lightweight Transformers**
**HR-NAS：使用轻量级变压器搜索高效的高分辨率神经架构**

- Paper(Oral): https://arxiv.org/abs/2106.06560

- Code: https://github.com/dingmyu/HR-NAS

**BCNet: Searching for Network Width with Bilaterally Coupled Network**
**BCNet：使用具有相互连接的网络搜索网络宽度**

- Paper: https://arxiv.org/abs/2105.10533
- Code: None

**ViPNAS: Efficient Video Pose Estimation via Neural Architecture Search**
**ViPNAS：通过神经架构搜索实现高效的视频姿态估计**

- Paper: ttps://arxiv.org/abs/2105.10154
- Code: None

**Combined Depth Space based Architecture Search For Person Re-identification**
**深度空间多模态架构基于人物重新识别**

- Paper: https://arxiv.org/abs/2104.04163
- Code: None

**DiNTS: Differentiable Neural Network Topology Search for 3D Medical Image Segmentation**
**DiNTS：可微分神经网络拓扑搜索3D医学图像分割**

- Paper(Oral): https://arxiv.org/abs/2103.15954
- Code: None

**HR-NAS: Searching Efficient High-Resolution Neural Architectures with Transformers**
**HR-NAS：使用变压器高效搜索高分辨率神经架构**

- Paper(Oral): None
- Code: https://github.com/dingmyu/HR-NAS

**Neural Architecture Search with Random Labels**
**神经架构搜索与随机标签**

- Paper: https://arxiv.org/abs/2101.11834
- Code: None

**Towards Improving the Consistency, Efficiency, and Flexibility of Differentiable Neural Architecture Search**
**为了改进不同iable神经架构搜索的一致性、效率和灵活性**

- Paper: https://arxiv.org/abs/2101.11342
- Code: None

**Joint-DetNAS: Upgrade Your Detector with NAS, Pruning and Dynamic Distillation**
**联合DetNAS：用NAS、剪枝和动态蒸馏升级您的检测器**

- Paper:  https://arxiv.org/abs/2105.12971 
- Code: None

**Prioritized Architecture Sampling with Monto-Carlo Tree Search**
**带有蒙特卡洛树搜索的优先架构采样**

- Paper: https://arxiv.org/abs/2103.11922
- Code: https://github.com/xiusu/NAS-Bench-Macro

**Contrastive Neural Architecture Search with Neural Architecture Comparators**
**对比神经网络架构搜索与神经架构比较器**

- Paper: https://arxiv.org/abs/2103.05471
- Code: https://github.com/chenyaofo/CTNAS

**AttentiveNAS: Improving Neural Architecture Search via Attentive** 

- Paper: https://arxiv.org/abs/2011.09011
- Code: None

**ReNAS: Relativistic Evaluation of Neural Architecture Search**
**ReNAS: 相对论评估神经架构搜索**

- Paper: https://arxiv.org/abs/1910.01523
- Code: None

**HourNAS: Extremely Fast Neural Architecture**
**小时纳米：极端快速神经架构**

- Paper: https://arxiv.org/abs/2005.14446
- Code: None

**Searching by Generating: Flexible and Efficient One-Shot NAS with Architecture Generator**
**搜索通过生成：灵活高效的一键NAS架构生成器**

- Paper: https://arxiv.org/abs/2103.07289
- Code: https://github.com/eric8607242/SGNAS

**OPANAS: One-Shot Path Aggregation Network Architecture Search for Object Detection**
**OPANAS：一次运行路径聚合网络结构搜索目标检测**

- Paper: https://arxiv.org/abs/2103.04507
- Code: https://github.com/VDIGPKU/OPANAS

**Inception Convolution with Efficient Dilation Search**
**基于efficient dilation搜索的Inception卷积**

- Paper:  https://arxiv.org/abs/2012.13587 
- Code: None

<a name="GAN"></a>

# GAN

**High-Resolution Photorealistic Image Translation in Real-Time: A Laplacian Pyramid Translation Network**
**实时高分辨率照片风格迁移：拉普拉斯金字塔转换网络**

- Paper: https://arxiv.org/abs/2105.09188
- Code: https://github.com/csjliang/LPTN
- Dataset: https://github.com/csjliang/LPTN

**DG-Font: Deformable Generative Networks for Unsupervised Font Generation**
**DG-Font: 可变形生成网络，用于无监督字体生成**

- Paper: https://arxiv.org/abs/2104.03064

- Code: https://github.com/ecnuycxie/DG-Font

**PD-GAN: Probabilistic Diverse GAN for Image Inpainting**
**PD-GAN：概率多样性生成对抗网络用于图像修复**

- Paper: https://arxiv.org/abs/2105.02201
- Code: https://github.com/KumapowerLIU/PD-GAN

**StyleMapGAN: Exploiting Spatial Dimensions of Latent in GAN for Real-time Image Editing**
**StyleMapGAN: 利用潜在空间中的空间维度进行实时图像编辑**

- Paper: https://arxiv.org/abs/2104.14754
- Code: https://github.com/naver-ai/StyleMapGAN
- Demo Video: https://youtu.be/qCapNyRA_Ng

**Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality Artistic Style Transfer**
**起草和修订：用于快速高质量艺术风格转换的拉普拉斯金字塔网络**

- Paper: https://arxiv.org/abs/2104.05376
- Code: https://github.com/PaddlePaddle/PaddleGAN/

**Regularizing Generative Adversarial Networks under Limited Data**
**生成对抗网络（GAN）在有限数据下的正则化**

- Homepage: https://hytseng0509.github.io/lecam-gan/
- Paper: https://faculty.ucmerced.edu/mhyang/papers/cvpr2021_gan_limited_data.pdf
- Code: https://github.com/google/lecam-gan

**Towards Real-World Blind Face Restoration with Generative Facial Prior**
**面向现实世界的人脸修复与生成面部优先**

- Paper: https://arxiv.org/abs/2101.04061
- Code: None

**TediGAN: Text-Guided Diverse Image Generation and Manipulation**
**TediGAN：文本指导的多样图像生成和操作**

- Homepage: https://xiaweihao.com/projects/tedigan/

- Paper: https://arxiv.org/abs/2012.03308
- Code: https://github.com/weihaox/TediGAN

**Generative Hierarchical Features from Synthesizing Image**
**合成图像生成层次特征**

- Homepage: https://genforce.github.io/ghfeat/

- Paper(Oral): https://arxiv.org/abs/2007.10379
- Code: https://github.com/genforce/ghfeat

**Teachers Do More Than Teach: Compressing Image-to-Image Models**
**教师的作用不仅仅局限于传授知识：压缩图像到图像模型**

- Paper: https://arxiv.org/abs/2103.03467
- Code: https://github.com/snap-research/CAT

**HistoGAN: Controlling Colors of GAN-Generated and Real Images via Color Histograms**
**HistoGAN: 通过颜色直方图控制GAN生成的和真实图像的颜色**

- Paper: https://arxiv.org/abs/2011.11731
- Code: https://github.com/mahmoudnafifi/HistoGAN

**pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis**
**π-GAN：3D 感知的图像生成器**

- Homepage: https://marcoamonteiro.github.io/pi-GAN-website/

- Paper(Oral): https://arxiv.org/abs/2012.00926
- Code: None

**DivCo: Diverse Conditional Image Synthesis via Contrastive Generative Adversarial Network**
**DivCo: 多样化的条件图像生成通过对比生成对抗网络**

- Paper: https://arxiv.org/abs/2103.07893
- Code: None

**Diverse Semantic Image Synthesis via Probability Distribution Modeling**
**多样化的语义图像生成通过概率分布建模**

- Paper: https://arxiv.org/abs/2103.06878
- Code: https://github.com/tzt101/INADE.git

**LOHO: Latent Optimization of Hairstyles via Orthogonalization**
**LOHO：通过正则化优化发型的潜在优化**

- Paper: https://arxiv.org/abs/2103.03891
- Code: None

**PISE: Person Image Synthesis and Editing with Decoupled GAN**
**PISE：与解耦GAN相结合的人像合成与编辑**

- Paper: https://arxiv.org/abs/2103.04023
- Code: https://github.com/Zhangjinso/PISE

**DeFLOCNet: Deep Image Editing via Flexible Low-level Controls**
**DeFLOCNet：通过灵活的低级控制实现深度图像编辑**

- Paper: http://raywzy.com/
- Code: http://raywzy.com/

**PD-GAN: Probabilistic Diverse GAN for Image Inpainting**
**PD-GAN：概率多样性生成对抗网络用于图像修复**

- Paper: http://raywzy.com/
- Code: http://raywzy.com/

**Efficient Conditional GAN Transfer with Knowledge Propagation across Classes**
**高效条件生成对抗网络迁移与类别间知识传播**

- Paper: https://www.researchgate.net/publication/349309756_Efficient_Conditional_GAN_Transfer_with_Knowledge_Propagation_across_Classes
- Code: http://github.com/mshahbazi72/cGANTransfer

**Exploiting Spatial Dimensions of Latent in GAN for Real-time Image Editing**
**利用潜在空间维度在GAN中实现实时图像编辑**

- Paper: None
- Code: None

**Hijack-GAN: Unintended-Use of Pretrained, Black-Box GANs**
**劫持-GAN：未计划的预训练黑色盒GAN**

- Paper: https://arxiv.org/abs/2011.14107
- Code: None

**Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation**
**编码风格：风格迁移图像到图像的编码器**

- Homepage: https://eladrich.github.io/pixel2style2pixel/
- Paper: https://arxiv.org/abs/2008.00951
- Code: https://github.com/eladrich/pixel2style2pixel

**A 3D GAN for Improved Large-pose Facial Recognition**
**改进的大型人脸识别的3D GAN**

- Paper: https://arxiv.org/abs/2012.10545
- Code: None

**HumanGAN: A Generative Model of Humans Images**
**人类生成对抗网络：人类图像的生成模型**

- Paper: https://arxiv.org/abs/2103.06902
- Code: None

**ID-Unet: Iterative Soft and Hard Deformation for View Synthesis**
**ID-Unet: 迭代式软硬变形用于视图合成**

- Paper: https://arxiv.org/abs/2103.02264
- Code: https://github.com/MingyuY/Iterative-view-synthesis

**CoMoGAN: continuous model-guided image-to-image translation**
**CoMoGAN：连续模型引导的图像间转换**

- Paper(Oral): https://arxiv.org/abs/2103.06879
- Code: https://github.com/cv-rits/CoMoGAN

**Training Generative Adversarial Networks in One Stage**
**训练生成对抗网络在一步中**

- Paper: https://arxiv.org/abs/2103.00430
- Code: None

**Closed-Form Factorization of Latent Semantics in GANs**
**GANs 中的潜在语义封闭式因式分解**

- Homepage: https://genforce.github.io/sefa/
- Paper(Oral): https://arxiv.org/abs/2007.06600
- Code: https://github.com/genforce/sefa

**Anycost GANs for Interactive Image Synthesis and Editing**
**任何成本的 GANs 用于交互式图像生成和编辑**

- Paper: https://arxiv.org/abs/2103.03243
- Code: https://github.com/mit-han-lab/anycost-gan

**Image-to-image Translation via Hierarchical Style Disentanglement**
**图像到图像的翻译通过层次风格解绑**

- Paper: https://arxiv.org/abs/2103.01456
- Code: https://github.com/imlixinyang/HiSD

<a name="VAE"></a>

# VAE

**Soft-IntroVAE: Analyzing and Improving Introspective Variational Autoencoders**
**软-引入VAE：分析并改进内省变分自动编码器**

- Homepage: https://taldatech.github.io/soft-intro-vae-web/

- Paper: https://arxiv.org/abs/2012.13253
- Code: https://github.com/taldatech/soft-intro-vae-pytorch

<a name="Visual Transformer"></a>

# Visual Transformer

**1. End-to-End Human Pose and Mesh Reconstruction with Transformers**
**1. 端到端的人体姿态和网格重构：Transformer**

- Paper: https://arxiv.org/abs/2012.09760
- Code: https://github.com/microsoft/MeshTransformer

**2. Temporal-Relational CrossTransformers for Few-Shot Action Recognition**
**2. 用于少量数据动作识别的时间-关系跨变换器**

- Paper: https://arxiv.org/abs/2101.06184
- Code: https://github.com/tobyperrett/trx

**3. Kaleido-BERT：Vision-Language Pre-training on Fashion Domain**
**3. Kaleido-BERT：在时尚领域的视觉语言预训练**

- Paper: https://arxiv.org/abs/2103.16110
- Code: https://github.com/mczhuge/Kaleido-BERT

**4. HOTR: End-to-End Human-Object Interaction Detection with Transformers**
**4. HOTR：使用变压器进行端到端的人机对象交互检测**

- Paper: https://arxiv.org/abs/2104.13682
- Code: https://github.com/kakaobrain/HOTR

**5. Multi-Modal Fusion Transformer for End-to-End Autonomous Driving**
**5. 多模态融合变压器用于端到端自动驾驶**

- Paper: https://arxiv.org/abs/2104.09224
- Code: https://github.com/autonomousvision/transfuser

**6. Pose Recognition with Cascade Transformers**
**6. 带有Cascade变量的转换器进行姿态识别**

- Paper: https://arxiv.org/abs/2104.06976

- Code: https://github.com/mlpc-ucsd/PRTR

**7. Variational Transformer Networks for Layout Generation**
**7. 变分自注意力网络用于布局生成**

- Paper: https://arxiv.org/abs/2104.02416
- Code: None

**8. LoFTR: Detector-Free Local Feature Matching with Transformers**
**8. LoFTR: 无需检测器的局部特征匹配与转换器**

- Homepage: https://zju3dv.github.io/loftr/
- Paper: https://arxiv.org/abs/2104.00680
- Code: https://github.com/zju3dv/LoFTR

**9. Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers**
**9. 从序列到序列重新思考语义分割**

- Paper: https://arxiv.org/abs/2012.15840
- Code: https://github.com/fudan-zvg/SETR

**10. Thinking Fast and Slow: Efficient Text-to-Visual Retrieval with Transformers**
**10. 思考快与慢：使用变压器实现高效的文本到视觉检索**

- Paper: https://arxiv.org/abs/2103.16553
- Code: None

**11. Transformer Tracking**
**11. 转换器跟踪**

- Paper: https://arxiv.org/abs/2103.15436
- Code: https://github.com/chenxin-dlut/TransT

**12. HR-NAS: Searching Efficient High-Resolution Neural Architectures with Transformers**
**12. HR-NAS：使用变压器在高效高分辨率神经架构中进行搜索**

- Paper(Oral):  https://arxiv.org/abs/2106.06560 
- Code: https://github.com/dingmyu/HR-NAS

**13. MIST: Multiple Instance Spatial Transformer**
**13. Mist: 多实例空间转换器**

- Paper: https://arxiv.org/abs/1811.10725
- Code: None

**14. Multimodal Motion Prediction with Stacked Transformers**
**14. 多模态运动预测 with 堆叠的变压器**

- Paper: https://arxiv.org/abs/2103.11624
- Code: https://decisionforce.github.io/mmTransformer

**15. Revamping cross-modal recipe retrieval with hierarchical Transformers and self-supervised learning**
**15. 改进跨模态 recipe 检索，使用分层 Transformers 和自监督学习**

- Paper: https://www.amazon.science/publications/revamping-cross-modal-recipe-retrieval-with-hierarchical-transformers-and-self-supervised-learning

- Code: https://github.com/amzn/image-to-recipe-transformers

**16. Transformer Meets Tracker: Exploiting Temporal Context for Robust Visual Tracking**
**16. 变压器与跟踪器：利用时间上下文来实现稳健视觉跟踪**

- Paper(Oral): https://arxiv.org/abs/2103.11681

- Code: https://github.com/594422814/TransformerTrack

**17. Pre-Trained Image Processing Transformer**
**17. 预训练图像处理变压器**

- Paper:  https://arxiv.org/abs/2012.00364 
- Code: None

**18. End-to-End Video Instance Segmentation with Transformers**
**18. 端到端视频实例分割与变压器**

- Paper(Oral): https://arxiv.org/abs/2011.14503
- Code: https://github.com/Epiphqny/VisTR

**19. UP-DETR: Unsupervised Pre-training for Object Detection with Transformers**
**19. UP-DETR: 无监督预训练，用于对象检测的 Transformer**

- Paper(Oral): https://arxiv.org/abs/2011.09094
- Code: https://github.com/dddzg/up-detr

**20. End-to-End Human Object Interaction Detection with HOI Transformer**
**20. 端到端的人机对象交互检测与HOI Transformer**

- Paper: https://arxiv.org/abs/2103.04503
- Code: https://github.com/bbepoch/HoiTransformer

**21. Transformer Interpretability Beyond Attention Visualization** 

- Paper: https://arxiv.org/abs/2012.09838
- Code: https://github.com/hila-chefer/Transformer-Explainability

**22. Diverse Part Discovery: Occluded Person Re-Identification With Part-Aware Transformer**
**22. 多重部分发现：部分感知Transformer进行遮挡人重识别**

- Paper: None
- Code: None

**23. LayoutTransformer: Scene Layout Generation With Conceptual and Spatial Diversity**
**23. LayoutTransformer：具有概念性和空间多样性的场景布局生成**

- Paper: None
- Code: None

**24. Line Segment Detection Using Transformers without Edges**
**24. 利用Transformer进行无边直线段检测**

- Paper(Oral): https://arxiv.org/abs/2101.01909
- Code: None

**25. MaX-DeepLab: End-to-End Panoptic Segmentation With Mask Transformers**
**MaX-DeepLab：使用遮罩变压器的端到端全景分割**

- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Wang_MaX-DeepLab_End-to-End_Panoptic_Segmentation_With_Mask_Transformers_CVPR_2021_paper.html
- Code: None

**26. SSTVOS: Sparse Spatiotemporal Transformers for Video Object Segmentation**
**26. SSTVOS：稀疏时空变分器用于视频对象分割**

- Paper(Oral): https://arxiv.org/abs/2101.08833
- Code: https://github.com/dukebw/SSTVOS

**27. Facial Action Unit Detection With Transformers**
**27. 面部动作单元检测与Transformer**

- Paper: None
- Code: None

**28. Clusformer: A Transformer Based Clustering Approach to Unsupervised Large-Scale Face and Visual Landmark Recognition**
**28. Clusformer：一种基于Transformer的聚类方法，用于大规模无监督人脸和视觉地标识别**

- Paper: None
- Code: None

**29. Lesion-Aware Transformers for Diabetic Retinopathy Grading**
**29. 糖尿病视网膜病变分级中的病损感知变压器**

- Paper: None
- Code: None

**30. Topological Planning With Transformers for Vision-and-Language Navigation**
**30. 拓扑规划中的变压器用于视觉与语言导航**

- Paper: https://arxiv.org/abs/2012.05292
- Code: None

**31. Adaptive Image Transformer for One-Shot Object Detection**
**31. 用于单例目标检测的适应性图像变换器**

- Paper: None
- Code: None

**32. Multi-Stage Aggregated Transformer Network for Temporal Language Localization in Videos**
**32. 多阶段聚合Transformer网络用于视频中的时序语言定位**

- Paper: None
- Code: None

**33. Taming Transformers for High-Resolution Image Synthesis**
**33. 驯服变分器以实现高分辨率图像合成**

- Homepage: https://compvis.github.io/taming-transformers/
- Paper(Oral): https://arxiv.org/abs/2012.09841
- Code: https://github.com/CompVis/taming-transformers

**34. Self-Supervised Video Hashing via Bidirectional Transformers**
**34. 基于双向变换器的自监督视频哈希**

- Paper: None
- Code: None

**35. Point 4D Transformer Networks for Spatio-Temporal Modeling in Point Cloud Videos**
**35. 点云视频时空建模的点云变分器网络（D Point4D Transformer Networks）**

- Paper(Oral): https://hehefan.github.io/pdfs/p4transformer.pdf
- Code: None

**36. Gaussian Context Transformer**
**36. 加权高斯上下文转换器**

- Paper: None
- Code: None

**37. General Multi-Label Image Classification With Transformers**
**37. 带变量的多标签图像分类使用转换器**

- Paper: https://arxiv.org/abs/2011.14027
- Code: None

**38. Bottleneck Transformers for Visual Recognition**
**38. 瓶颈 transformers for 视觉识别**

- Paper: https://arxiv.org/abs/2101.11605
- Code: None

**39. VLN BERT: A Recurrent Vision-and-Language BERT for Navigation**
**39. VLN BERT：用于导航的循环视觉和语言 BERT**

- Paper(Oral): https://arxiv.org/abs/2011.13922
- Code: https://github.com/YicongHong/Recurrent-VLN-BERT

**40. Less Is More: ClipBERT for Video-and-Language Learning via Sparse Sampling**
**40. 少即是多：通过稀疏采样实现视频和语言学习的ClipBERT**

- Paper(Oral): https://arxiv.org/abs/2102.06183
- Code: https://github.com/jayleicn/ClipBERT

**41. Self-attention based Text Knowledge Mining for Text Detection**
**41. 基于自注意力文本知识图谱的文本检测**

- Paper: None
- Code: https://github.com/CVI-SZU/STKM

**42. SSAN: Separable Self-Attention Network for Video Representation Learning**
**42. SSAN：分离的 Self-Attention 网络 for 视频表示学习**

- Paper: None
- Code: None

**43. Scaling Local Self-Attention For Parameter Efficient Visual Backbones**
**43. 缩放局部自注意力以实现参数高效的视觉骨干网络**

- Paper(Oral): https://arxiv.org/abs/2103.12731

- Code: None

<a name="Regularization"></a>

# Regularization

**Regularizing Neural Networks via Adversarial Model Perturbation**
**通过对抗模型对神经网络进行正则化**

- Paper: https://arxiv.org/abs/2010.04925
- Code: https://github.com/hiyouga/AMP-Regularizer

<a name="SLAM"></a>

# SLAM

**Differentiable SLAM-net: Learning Particle SLAM for Visual Navigation**
**可微分SLAM-net：用于视觉导航的粒子SLAM学习**

- Paper: https://arxiv.org/abs/2105.07593
- Code: None

**Generalizing to the Open World: Deep Visual Odometry with Online Adaptation**
**推广到开放世界：在线自适应深度视觉里程计**

- Paper: https://arxiv.org/abs/2103.15279
- Code: https://arxiv.org/abs/2103.15279

<a name="Long-Tailed"></a>

# 长尾分布(Long-Tailed)

**Adversarial Robustness under Long-Tailed Distribution**
**对抗鲁棒性在长尾分布下的**

- Paper(Oral): https://arxiv.org/abs/2104.02703
- Code: https://github.com/wutong16/Adversarial_Long-Tail 

**Distribution Alignment: A Unified Framework for Long-tail Visual Recognition**
**分布对齐：一个用于长尾视觉识别的统一框架**

- Paper: https://arxiv.org/abs/2103.16370
- Code: https://github.com/Megvii-BaseDetection/DisAlign

**Adaptive Class Suppression Loss for Long-Tail Object Detection**
**适应性类别抑制损失 for 长尾目标检测**

- Paper: https://arxiv.org/abs/2104.00885
- Code: https://github.com/CASIA-IVA-Lab/ACSL

**Contrastive Learning based Hybrid Networks for Long-Tailed Image Classification**
**基于对比学习的混合网络长尾图像分类**

- Paper: https://arxiv.org/abs/2103.14267
- Code: None

<a name="DA"></a>

# 数据增广(Data Augmentation)

**Scale-aware Automatic Augmentation for Object Detection**
**scale-aware 自动扩展物体检测**

- Paper: https://arxiv.org/abs/2103.17220

- Code: https://github.com/Jia-Research-Lab/SA-AutoAug

<a name="Un/Self-Supervised"></a>

# 无监督/自监督(Un/Self-Supervised)

**Domain-Specific Suppression for Adaptive Object Detection**
**针对自适应目标检测的领域特定抑制**

- Paper: https://arxiv.org/abs/2105.03570
- Code: None

**A Large-Scale Study on Unsupervised Spatiotemporal Representation Learning**
**一种大规模的空间-时间表示学习无监督研究**

- Paper: https://arxiv.org/abs/2104.14558

- Code: https://github.com/facebookresearch/SlowFast

**Unsupervised Multi-Source Domain Adaptation for Person Re-Identification**
**无监督多源领域自适应 for 人物重新识别**

- Paper: https://arxiv.org/abs/2104.12961
- Code: None

**Self-supervised Video Representation Learning by Context and Motion Decoupling**
**基于上下文和运动分解的自监督视频表示学习**

- Paper: https://arxiv.org/abs/2104.00862
- Code: None

**Removing the Background by Adding the Background: Towards Background Robust Self-supervised Video Representation Learning**
**移除背景，添加背景：走向背景鲁棒的自监督视频表示学习**

- Homepage: https://fingerrec.github.io/index_files/jinpeng/papers/CVPR2021/project_website.html
- Paper: https://arxiv.org/abs/2009.05769
- Code: https://github.com/FingerRec/BE

**Spatially Consistent Representation Learning**
**空间一致表示学习**

- Paper: https://arxiv.org/abs/2103.06122
- Code: None

**VideoMoCo: Contrastive Video Representation Learning with Temporally Adversarial Examples**
**视频MoCo：具有时间对抗样式的对比视频表示学习**

- Paper: https://arxiv.org/abs/2103.05905
- Code: https://github.com/tinapan-pt/VideoMoCo

**Exploring Simple Siamese Representation Learning**
**探索简单的闽南表示学习**

- Paper(Oral): https://arxiv.org/abs/2011.10566
- Code: None

**Dense Contrastive Learning for Self-Supervised Visual Pre-Training**
**稠密对比学习用于自监督视觉预训练**

- Paper(Oral): https://arxiv.org/abs/2011.09157
- Code: https://github.com/WXinlong/DenseCL

<a name="Semi-Supervised"></a>

# 半监督学习(Semi-Supervised )

**Instant-Teaching: An End-to-End Semi-Supervised Object Detection Framework**
**即时教学：一个端到端的半监督目标检测框架**

- 作者单位: 阿里巴巴

- Paper: https://arxiv.org/abs/2103.11402
- Code: None

**Adaptive Consistency Regularization for Semi-Supervised Transfer Learning**
**半监督迁移学习的自适应一致性正则化**

- Paper: https://arxiv.org/abs/2103.02193
- Code: https://github.com/SHI-Labs/Semi-Supervised-Transfer-Learning

<a name="Capsule-Network"></a>

# 胶囊网络(Capsule Network)

**Capsule Network is Not More Robust than Convolutional Network**
**胶囊网络不再比卷积网络更强大。**

- Paper: https://arxiv.org/abs/2103.15459
- Code: None

<a name="Image-Classification"></a>

# 图像分类(Image Classification)

**Correlated Input-Dependent Label Noise in Large-Scale Image Classification**
**相关输入依赖标签噪声在大规模图像分类**

- Paper(Oral): https://arxiv.org/abs/2105.10305
- Code: https://github.com/google/uncertainty-baselines/tree/master/baselines/imagenet

<a name="Object-Detection"></a>

# 2D目标检测(Object Detection)

## 2D目标检测

**1. Scaled-YOLOv4: Scaling Cross Stage Partial Network**
**1. Scaled-YOLOv4：缩放跨阶段部分网络**

- 作者单位: 中央研究院, 英特尔, 静宜大学
- Paper: https://arxiv.org/abs/2011.08036
- Code: https://github.com/WongKinYiu/ScaledYOLOv4
- 中文解读: [YOLOv4官方改进版来了！55.8% AP！速度最高达1774 FPS，Scaled-YOLOv4正式开源！](https://mp.weixin.qq.com/s/AcrJPNoAVhn8cGBUGK7ekA)

**2. You Only Look One-level Feature**
**2. 您只看一个一级特征**

- 作者单位: 中科院, 国科大, 旷视科技
- Paper: https://arxiv.org/abs/2103.09460
- Code: https://github.com/megvii-model/YOLOF
- 中文解读: [CVPR 2021 | 没有FPN！中科院&旷视提出YOLOF：你只需看一层特征](https://mp.weixin.qq.com/s/EJqAG1gTVaP2icI6QL742A)

**3. Sparse R-CNN: End-to-End Object Detection with Learnable Proposals**
**3.稀疏 R-CNN：学习可提议的端到端目标检测**

- 作者单位: 香港大学, 同济大学, 字节跳动AI Lab, 加利福尼亚大学伯克利分校
- Paper: https://arxiv.org/abs/2011.12450
- Code: https://github.com/PeizeSun/SparseR-CNN
- 中文解读: [目标检测新范式！港大同济伯克利提出Sparse R-CNN，代码刚刚开源！](https://mp.weixin.qq.com/s/P2Zgh1wTqf8L2976El5nfQ)

**4. End-to-End Object Detection with Fully Convolutional Network**
**4. 端到端目标检测与全卷积网络**

- 作者单位: 旷视科技, 西安交通大学
- Paper: https://arxiv.org/abs/2012.03544
- Code: https://github.com/Megvii-BaseDetection/DeFCN

**5. Dynamic Head: Unifying Object Detection Heads with Attentions**
**5. 动态头部：将对象检测头与注意力统一**

- 作者单位: 微软
- Paper: https://arxiv.org/abs/2106.08322
- Code: https://github.com/microsoft/DynamicHead
- 中文解读: [60.6 AP！打破COCO记录！微软提出DyHead：将注意力与目标检测Heads统一](https://mp.weixin.qq.com/s/uYPUqVXwNau71VAYW3bYIA)

**6. Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection**
**6. 推广的 Focal Loss V2：为稠密目标检测学习可靠的局部定位质量估计**

- 作者单位: 南京理工大学, Momenta, 南京大学, 清华大学
- Paper: https://arxiv.org/abs/2011.12885
- Code: https://github.com/implus/GFocalV2
- 中文解读：[CVPR 2021 | GFLV2：目标检测良心技术，无Cost涨点！](https://mp.weixin.qq.com/s/JB7k3NwXU-cDueg6w9mghQ)

**7. UP-DETR: Unsupervised Pre-training for Object Detection with Transformers**
**7. UP-DETR: 无监督预训练，用于物体检测的 Transformers**

- 作者单位: 华南理工大学, 腾讯微信AI
- Paper(Oral): https://arxiv.org/abs/2011.09094
- Code: https://github.com/dddzg/up-detr
- 中文解读: [CVPR 2021 Oral | Transformer再发力！华南理工和微信提出UP-DETR：无监督预训练检测器](https://mp.weixin.qq.com/s/Hprp7B16SGFhVEKXfKiRBQ)

**8. MobileDets: Searching for Object Detection Architectures for Mobile Accelerators**
**8. MobileDets: 寻找适用于移动加速器的物体检测架构**

- 作者单位: 威斯康星大学, 谷歌

- Paper: https://openaccess.thecvf.com/content/CVPR2021/papers/Xiong_MobileDets_Searching_for_Object_Detection_Architectures_for_Mobile_Accelerators_CVPR_2021_paper.pdf
- Code: https://github.com/tensorflow/models/tree/master/research/object_detection

**9. Tracking Pedestrian Heads in Dense Crowd**
**9. 跟踪拥挤人群中的人头**

- 作者单位: 雷恩第一大学
- Homepage: https://project.inria.fr/crowdscience/project/dense-crowd-head-tracking/
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Sundararaman_Tracking_Pedestrian_Heads_in_Dense_Crowd_CVPR_2021_paper.html
- Code1: https://github.com/Sentient07/HeadHunter
- Code2: https://github.com/Sentient07/HeadHunter%E2%80%93T
- Dataset: https://project.inria.fr/crowdscience/project/dense-crowd-head-tracking/

**10. Joint-DetNAS: Upgrade Your Detector with NAS, Pruning and Dynamic Distillation**
**10. 联合DetNAS：用NAS、剪枝和动态蒸馏升级你的检测器**

- 作者单位: 香港科技大学, 华为诺亚
- Paper:  https://arxiv.org/abs/2105.12971 
- Code: None

**11. PSRR-MaxpoolNMS: Pyramid Shifted MaxpoolNMS with Relationship Recovery**
**11. PSRR-MaxpoolNMS: 金字塔移动最大池化NMS与关系恢复**

- 作者单位: A*star, 四川大学,  南洋理工大学
- Paper: https://arxiv.org/abs/2105.12990
- Code: None

**12. IQDet: Instance-wise Quality Distribution Sampling for Object Detection**
**12. IQDet：用于对象检测的实例质量分布采样**

- 作者单位: 旷视科技
- Paper: https://arxiv.org/abs/2104.06936
- Code: None

**13. Multi-Scale Aligned Distillation for Low-Resolution Detection**
**13. 多尺度融合蒸馏低分辨率检测**

- 作者单位: 香港中文大学, Adobe研究院, 思谋科技
- Paper: https://jiaya.me/papers/ms_align_distill_cvpr21.pdf
- Code: https://github.com/Jia-Research-Lab/MSAD

**14. Adaptive Class Suppression Loss for Long-Tail Object Detection**
**14. 适应性类别抑制损失，用于长尾目标检测**

- 作者单位: 中科院, 国科大, ObjectEye, 北京大学, 鹏城实验室, Nexwise

- Paper: https://arxiv.org/abs/2104.00885
- Code: https://github.com/CASIA-IVA-Lab/ACSL

**15. VarifocalNet: An IoU-aware Dense Object Detector**
**15. VarifocalNet：一种基于IoU的稠密对象检测器**

- 作者单位: 昆士兰科技大学, 昆士兰大学
- Paper(Oral): https://arxiv.org/abs/2008.13367
- Code: https://github.com/hyz-xmaster/VarifocalNet

**16. OTA: Optimal Transport Assignment for Object Detection**
**16. OTA：最优传输分配对象检测**

- 作者单位: 早稻田大学, 旷视科技

- Paper: https://arxiv.org/abs/2103.14259
- Code: https://github.com/Megvii-BaseDetection/OTA

**17. Distilling Object Detectors via Decoupled Features**
**17. 通过分离特征来提取目标检测器**

- 作者单位: 华为诺亚, 悉尼大学
- Paper: https://arxiv.org/abs/2103.14475
- Code: https://github.com/ggjy/DeFeat.pytorch

**18. Robust and Accurate Object Detection via Adversarial Learning**
**18. 鲁棒且准确的通过对抗学习实现物体检测**

- 作者单位: 谷歌, UCLA, UCSC

- Paper: https://arxiv.org/abs/2103.13886

- Code: None

**19. OPANAS: One-Shot Path Aggregation Network Architecture Search for Object Detection**
**19. OPANAS：用于检测对象的单次路径聚合网络结构搜索**

- 作者单位: 北京大学, Anyvision, 石溪大学
- Paper: https://arxiv.org/abs/2103.04507
- Code: https://github.com/VDIGPKU/OPANAS

**20. Multiple Instance Active Learning for Object Detection**
**多实例活动检测**

- 作者单位: 国科大, 华为诺亚, 清华大学
- Paper: https://openaccess.thecvf.com/content/CVPR2021/papers/Yuan_Multiple_Instance_Active_Learning_for_Object_Detection_CVPR_2021_paper.pdf
- Code: https://github.com/yuantn/MI-AOD

**21. Towards Open World Object Detection**
**21. 面向开放世界目标检测**

- 作者单位: 印度理工学院, MBZUAI, 澳大利亚国立大学, 林雪平大学
- Paper(Oral): https://arxiv.org/abs/2103.02603
- Code: https://github.com/JosephKJ/OWOD

**22. RankDetNet: Delving Into Ranking Constraints for Object Detection**
**22. RankDetNet：深入挖掘对象检测中的排名约束**

- 作者单位: 赛灵思
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Liu_RankDetNet_Delving_Into_Ranking_Constraints_for_Object_Detection_CVPR_2021_paper.html
- Code: None

## 旋转目标检测

**23. Dense Label Encoding for Boundary Discontinuity Free Rotation Detection**
**23. 稠密标签编码用于边界不连续性自由旋转检测**

- 作者单位: 上海交通大学, 国科大
- Paper: https://arxiv.org/abs/2011.09670
- Code1: https://github.com/Thinklab-SJTU/DCL_RetinaNet_Tensorflow
- Code2: https://github.com/yangxue0827/RotationDetection 

**24. ReDet: A Rotation-equivariant Detector for Aerial Object Detection**
**24. ReDet: 空中目标检测的旋转等价探测器**

- 作者单位: 武汉大学

- Paper: https://arxiv.org/abs/2103.07733
- Code: https://github.com/csuhan/ReDet

**25. Beyond Bounding-Box: Convex-Hull Feature Adaptation for Oriented and Densely Packed Object Detection**
**25. 超越边界框：面向和密集包装对象的凸包特征自适应检测**

- 作者单位: 国科大, 清华大学
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Guo_Beyond_Bounding-Box_Convex-Hull_Feature_Adaptation_for_Oriented_and_Densely_Packed_CVPR_2021_paper.html
- Code: https://github.com/SDL-GuoZonghao/BeyondBoundingBox

## Few-Shot目标检测

**26. Accurate Few-Shot Object Detection With Support-Query Mutual Guidance and Hybrid Loss**
**26. 精确的少样本物体检测支持查询互相关引导和混合损失**

- 作者单位: 复旦大学, 同济大学, 浙江大学

- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Accurate_Few-Shot_Object_Detection_With_Support-Query_Mutual_Guidance_and_Hybrid_CVPR_2021_paper.html
- Code: None

**27. Adaptive Image Transformer for One-Shot Object Detection**
**27. 适应性图像变换器用于一例目标检测**

- 作者单位: 中央研究院, 台湾AI Labs 
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Adaptive_Image_Transformer_for_One-Shot_Object_Detection_CVPR_2021_paper.html
- Code: None

**28. Dense Relation Distillation with Context-aware Aggregation for Few-Shot Object Detection**
**28. 稠密关联蒸馏与上下文感知的聚合对于少样本目标检测**

- 作者单位: 北京大学, 北邮
- Paper: https://arxiv.org/abs/2103.17115
- Code: https://github.com/hzhupku/DCNet 

**29. Semantic Relation Reasoning for Shot-Stable Few-Shot Object Detection**
**29. 针对稳定少量的多目标检测的语义关系推理**

- 作者单位: 卡内基梅隆大学(CMU)

- Paper: https://arxiv.org/abs/2103.01903
- Code: None

**30. FSCE: Few-Shot Object Detection via Contrastive Proposal Encoding**
**30. FSCE：通过对比提议编码实现少样本物体检测**

- 作者单位: 南加利福尼亚大学, 旷视科技
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Sun_FSCE_Few-Shot_Object_Detection_via_Contrastive_Proposal_Encoding_CVPR_2021_paper.html
- Code:  https://github.com/MegviiDetection/FSCE 

**31. Hallucination Improves Few-Shot Object Detection**
**31. 错觉改善了少数样本目标检测**

- 作者单位: 伊利诺伊大学厄巴纳-香槟分校
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Hallucination_Improves_Few-Shot_Object_Detection_CVPR_2021_paper.html
- Code: https://github.com/pppplin/HallucFsDet

**32. Few-Shot Object Detection via Classification Refinement and Distractor Retreatment**
**32. 少样本物体检测通过分类细化和干扰者修复**

- 作者单位: 新加坡国立大学, SIMTech
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Li_Few-Shot_Object_Detection_via_Classification_Refinement_and_Distractor_Retreatment_CVPR_2021_paper.html
- Code: None

**33. Generalized Few-Shot Object Detection Without Forgetting**
**33. 具有遗忘的泛化少样本目标检测**

- 作者单位: 旷视科技
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Fan_Generalized_Few-Shot_Object_Detection_Without_Forgetting_CVPR_2021_paper.html
- Code: None

**34. Transformation Invariant Few-Shot Object Detection**
**34. 变形式迁移少数样本目标检测**

- 作者单位: 华为诺亚方舟实验室

- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Li_Transformation_Invariant_Few-Shot_Object_Detection_CVPR_2021_paper.html
- Code: None

**35. UniT: Unified Knowledge Transfer for Any-Shot Object Detection and Segmentation**
**35. UniT: 统一知识传递任意目标检测和分割**

- 作者单位: 不列颠哥伦比亚大学, Vector AI, CIFAR AI Chair
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Khandelwal_UniT_Unified_Knowledge_Transfer_for_Any-Shot_Object_Detection_and_Segmentation_CVPR_2021_paper.html
- Code: https://github.com/ubc-vision/UniT

**36. Beyond Max-Margin: Class Margin Equilibrium for Few-Shot Object Detection**
**36. 除了最大margin：用于少样本目标检测的类别边际平衡**

- 作者单位: 国科大, 厦门大学, 鹏城实验室
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Li_Beyond_Max-Margin_Class_Margin_Equilibrium_for_Few-Shot_Object_Detection_CVPR_2021_paper.html
- Code: https://github.com/Bohao-Lee/CME

## 半监督目标检测

 **37. Points As Queries: Weakly Semi-Supervised Object Detection by Points]**

- 作者单位: 旷视科技, 复旦大学
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Points_As_Queries_Weakly_Semi-Supervised_Object_Detection_by_Points_CVPR_2021_paper.html
- Code: None

**38. Data-Uncertainty Guided Multi-Phase Learning for Semi-Supervised Object Detection**
**38. 数据不确定性与半监督目标检测的多阶段学习**

- 作者单位: 清华大学
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Data-Uncertainty_Guided_Multi-Phase_Learning_for_Semi-Supervised_Object_Detection_CVPR_2021_paper.html
- Code: None

**39. Positive-Unlabeled Data Purification in the Wild for Object Detection**
**39. 野外中正则化无标签数据清洗用于目标检测**

- 作者单位: 华为诺亚方舟实验室, 悉尼大学, 北京大学

- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Guo_Positive-Unlabeled_Data_Purification_in_the_Wild_for_Object_Detection_CVPR_2021_paper.html
- Code: None

**40. Interactive Self-Training With Mean Teachers for Semi-Supervised Object Detection**
**40. 半监督目标检测中的交互式自训练与均值教师**

- 作者单位: 阿里巴巴, 香港理工大学
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Yang_Interactive_Self-Training_With_Mean_Teachers_for_Semi-Supervised_Object_Detection_CVPR_2021_paper.html
- Code: None

**41. Instant-Teaching: An End-to-End Semi-Supervised Object Detection Framework**
**41. 即时教学：一个端到端的半监督目标检测框架**

- 作者单位: 阿里巴巴
- Paper: https://arxiv.org/abs/2103.11402
- Code: None

**42. Humble Teachers Teach Better Students for Semi-Supervised Object Detection**
**42. 谦虚的教师在半监督目标检测中教会更好的学生**

- 作者单位:  卡内基梅隆大学(CMU), 亚马逊
- Homepage: https://yihet.com/humble-teacher
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Tang_Humble_Teachers_Teach_Better_Students_for_Semi-Supervised_Object_Detection_CVPR_2021_paper.html
- Code: https://github.com/lryta/HumbleTeacher

**43. Interpolation-Based Semi-Supervised Learning for Object Detection**
**43. 基于插值的半监督学习物体检测**

- 作者单位: 首尔大学, 阿尔托大学等
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Jeong_Interpolation-Based_Semi-Supervised_Learning_for_Object_Detection_CVPR_2021_paper.html
- Code: https://github.com/soo89/ISD-SSD

# 域自适应目标检测

**44. Domain-Specific Suppression for Adaptive Object Detection**
**44. 针对自适应目标检测的领域特定抑制**

- 作者单位: 中科院, 寒武纪, 国科大
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Domain-Specific_Suppression_for_Adaptive_Object_Detection_CVPR_2021_paper.html
- Code: None

**45. MeGA-CDA: Memory Guided Attention for Category-Aware Unsupervised Domain Adaptive Object Detection**
**45. MeGA-CDA：类别感知的无监督领域自适应对象检测的内存指导注意力**

- 作者单位: 约翰斯·霍普金斯大学, 梅赛德斯—奔驰
- Paper: https://arxiv.org/abs/2103.04224
- Code: None

**46. Unbiased Mean Teacher for Cross-Domain Object Detection**
**46. 无偏均值教师：跨领域对象检测**

- 作者单位: 电子科技大学, ETH Zurich
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Deng_Unbiased_Mean_Teacher_for_Cross-Domain_Object_Detection_CVPR_2021_paper.html
- Code: https://github.com/kinredon/umt

**47. I^3Net: Implicit Instance-Invariant Network for Adapting One-Stage Object Detectors**
**47. I^3Net：用于调整一阶段目标检测器的隐式实例无关网络**

- 作者单位: 香港大学, 厦门大学, Deepwise AI Lab
- Paper: https://arxiv.org/abs/2103.13757
- Code: None 

## 自监督目标检测

**48. There Is More Than Meets the Eye: Self-Supervised Multi-Object Detection and Tracking With Sound by Distilling Multimodal Knowledge**
**48. 除了看穿一切，还有更多：通过提炼多模态知识进行自监督多目标检测和跟踪**

- 作者单位: 弗莱堡大学
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Valverde_There_Is_More_Than_Meets_the_Eye_Self-Supervised_Multi-Object_Detection_CVPR_2021_paper.html
- Code: http://rl.uni-freiburg.de/research/multimodal-distill

**49. Instance Localization for Self-supervised Detection Pretraining**
**49. 实例本地化用于自监督检测预训练**

- 作者单位: 香港中文大学, 微软亚洲研究院
- Paper: https://arxiv.org/abs/2102.08318
- Code: https://github.com/limbo0000/InstanceLoc

## 弱监督目标检测

**50. Informative and Consistent Correspondence Mining for Cross-Domain Weakly Supervised Object Detection**
**50. 用于跨领域弱监督目标检测的信息抽取和一致性对应挖掘**

- 作者单位: 北航, 鹏城实验室, 商汤科技
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Hou_Informative_and_Consistent_Correspondence_Mining_for_Cross-Domain_Weakly_Supervised_Object_CVPR_2021_paper.html
- Code: None

**51. DAP: Detection-Aware Pre-training with Weak Supervision** 

- 作者单位: UIUC, 微软
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Zhong_DAP_Detection-Aware_Pre-Training_With_Weak_Supervision_CVPR_2021_paper.html
- Code: None

## 其他

**52. Open-Vocabulary Object Detection Using Captions**
**52. 利用标题进行开放词汇对象检测**

- 作者单位：Snap, 哥伦比亚大学

- Paper(Oral): https://openaccess.thecvf.com/content/CVPR2021/html/Zareian_Open-Vocabulary_Object_Detection_Using_Captions_CVPR_2021_paper.html
- Code: https://github.com/alirezazareian/ovr-cnn

**53. Depth From Camera Motion and Object Detection**
**53. 深度来自相机运动和物体检测**

- 作者单位:  密歇根大学, SIAI

- Paper: https://arxiv.org/abs/2103.01468
- Code: https://github.com/griffbr/ODMD
- Dataset: https://github.com/griffbr/ODMD

**54. Unsupervised Object Detection With LIDAR Clues**
**54. 无监督目标检测与LiDAR线索**

- 作者单位: 商汤科技, 国科大, 中科大
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Tian_Unsupervised_Object_Detection_With_LIDAR_Clues_CVPR_2021_paper.html
- Code: None

**55. GAIA: A Transfer Learning System of Object Detection That Fits Your Needs**
**55. GAIA：一个适用于物体检测的迁移学习系统，符合您的需求**

- 作者单位: 国科大, 北理, 中科院, 商汤科技
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Bu_GAIA_A_Transfer_Learning_System_of_Object_Detection_That_Fits_CVPR_2021_paper.html
- Code: https://github.com/GAIA-vision/GAIA-det

**56. General Instance Distillation for Object Detection**
**56. 对象检测的总体实例蒸馏**

- 作者单位: 旷视科技, 北航
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Dai_General_Instance_Distillation_for_Object_Detection_CVPR_2021_paper.html
- Code: None

**57. AQD: Towards Accurate Quantized Object Detection**
**57. AQD：迈向准确的量化目标检测**

- 作者单位: 蒙纳士大学, 阿德莱德大学, 华南理工大学
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Chen_AQD_Towards_Accurate_Quantized_Object_Detection_CVPR_2021_paper.html
- Code: https://github.com/aim-uofa/model-quantization

**58. Scale-Aware Automatic Augmentation for Object Detection**
**58. 针对对象检测的规模感知自动扩展**

- 作者单位: 香港中文大学, 字节跳动AI Lab, 思谋科技
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Scale-Aware_Automatic_Augmentation_for_Object_Detection_CVPR_2021_paper.html
- Code: https://github.com/Jia-Research-Lab/SA-AutoAug

**59. Equalization Loss v2: A New Gradient Balance Approach for Long-Tailed Object Detection**
**59. 均衡损失 v2：用于长尾目标检测的新梯度平衡方法**

- 作者单位: 同济大学, 商汤科技, 清华大学
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Tan_Equalization_Loss_v2_A_New_Gradient_Balance_Approach_for_Long-Tailed_CVPR_2021_paper.html
- Code: https://github.com/tztztztztz/eqlv2

**60. Class-Aware Robust Adversarial Training for Object Detection**
**60. 针对对象检测的类别感知鲁棒对抗训练**

- 作者单位: 哥伦比亚大学,  中央研究院 
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Class-Aware_Robust_Adversarial_Training_for_Object_Detection_CVPR_2021_paper.html
- Code: None

**61. Improved Handling of Motion Blur in Online Object Detection**
**61. 改进在线物体检测中的运动模糊处理**

- 作者单位: 伦敦大学学院
- Homepage: http://visual.cs.ucl.ac.uk/pubs/handlingMotionBlur/
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Sayed_Improved_Handling_of_Motion_Blur_in_Online_Object_Detection_CVPR_2021_paper.html
- Code: None

**62. Multiple Instance Active Learning for Object Detection**
**62. 多实例主动学习**

- 作者单位: 国科大, 华为诺亚
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Yuan_Multiple_Instance_Active_Learning_for_Object_Detection_CVPR_2021_paper.html
- Code: https://github.com/yuantn/MI-AOD

**63. Neural Auto-Exposure for High-Dynamic Range Object Detection**
**63. 神经元自适应曝光对于高动态范围目标检测**

- 作者单位: Algolux, 普林斯顿大学
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Onzon_Neural_Auto-Exposure_for_High-Dynamic_Range_Object_Detection_CVPR_2021_paper.html
- Code: None

**64. Generalizable Pedestrian Detection: The Elephant in the Room**
**64. 通用行人检测：遮蔽的房间里的大象**

- 作者单位: IIAI, 阿尔托大学
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Hasan_Generalizable_Pedestrian_Detection_The_Elephant_in_the_Room_CVPR_2021_paper.html
- Code: https://github.com/hasanirtiza/Pedestron

**65. Neural Auto-Exposure for High-Dynamic Range Object Detection**
**65. 神经元自动曝光高动态范围目标检测**

- 作者单位: Algolux, 普林斯顿大学
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Onzon_Neural_Auto-Exposure_for_High-Dynamic_Range_Object_Detection_CVPR_2021_paper.html
- Code: None

<a name="Object-Tracking"></a>

# 单/多目标跟踪(Object Tracking)

## 单目标跟踪

**LightTrack: Finding Lightweight Neural Networks for Object Tracking via One-Shot Architecture Search**
**LightTrack: 通过一次搜索构建轻量级神经网络来进行目标跟踪**

- Paper: https://arxiv.org/abs/2104.14545

- Code: https://github.com/researchmm/LightTrack

**Towards More Flexible and Accurate Object Tracking with Natural Language: Algorithms and Benchmark**
**面向更加灵活和准确的物体跟踪的自然语言算法与基准**

- Homepage: https://sites.google.com/view/langtrackbenchmark/

- Paper: https://arxiv.org/abs/2103.16746
- Evaluation Toolkit: https://github.com/wangxiao5791509/TNL2K_evaluation_toolkit
- Demo Video: https://www.youtube.com/watch?v=7lvVDlkkff0&ab_channel=XiaoWang 

**IoU Attack: Towards Temporally Coherent Black-Box Adversarial Attack for Visual Object Tracking**
**IoU Attack：朝着时间一致的黑色盒对抗攻击，用于视觉对象跟踪**

- Paper: https://arxiv.org/abs/2103.14938
- Code: https://github.com/VISION-SJTU/IoUattack

**Graph Attention Tracking**
**图结构注意力跟踪**

- Paper: https://arxiv.org/abs/2011.11204
- Code: https://github.com/ohhhyeahhh/SiamGAT

**Rotation Equivariant Siamese Networks for Tracking**
**旋转等价性二维人脸网络用于跟踪**

- Paper: https://arxiv.org/abs/2012.13078
- Code: None

**Track to Detect and Segment: An Online Multi-Object Tracker**
**跟踪检测与分割：在线多对象跟踪器**

- Homepage: https://jialianwu.com/projects/TraDeS.html
- Paper: None
- Code: None

**Transformer Meets Tracker: Exploiting Temporal Context for Robust Visual Tracking**
**变压器与跟踪器：利用时空上下文实现稳健视觉跟踪**

- Paper(Oral): https://arxiv.org/abs/2103.11681

- Code: https://github.com/594422814/TransformerTrack

**Transformer Tracking**
**变压器跟踪**

- Paper: https://arxiv.org/abs/2103.15436
- Code: https://github.com/chenxin-dlut/TransT

## 多目标跟踪

**Tracking Pedestrian Heads in Dense Crowd**
**跟踪拥挤人群中的行人头部**

- Homepage: https://project.inria.fr/crowdscience/project/dense-crowd-head-tracking/
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Sundararaman_Tracking_Pedestrian_Heads_in_Dense_Crowd_CVPR_2021_paper.html
- Code1: https://github.com/Sentient07/HeadHunter
- Code2: https://github.com/Sentient07/HeadHunter%E2%80%93T
- Dataset: https://project.inria.fr/crowdscience/project/dense-crowd-head-tracking/

**Multiple Object Tracking with Correlation Learning**
**多对象跟踪与相关学习**

- Paper: https://arxiv.org/abs/2104.03541
- Code: None

**Probabilistic Tracklet Scoring and Inpainting for Multiple Object Tracking**
**概率轨迹点评分和修复多对象跟踪**

- Paper: https://arxiv.org/abs/2012.02337
- Code: None

**Learning a Proposal Classifier for Multiple Object Tracking**
**学习多对象跟踪的提案分类器**

- Paper: https://arxiv.org/abs/2103.07889
- Code: https://github.com/daip13/LPC_MOT.git

**Track to Detect and Segment: An Online Multi-Object Tracker**
**跟踪检测和分割：在线多对象跟踪器**

- Homepage: https://jialianwu.com/projects/TraDeS.html
- Paper: https://arxiv.org/abs/2103.08808
- Code: https://github.com/JialianW/TraDeS

<a name="Semantic-Segmentation"></a>

# 语义分割(Semantic Segmentation)

**1. HyperSeg: Patch-wise Hypernetwork for Real-time Semantic Segmentation**
**1. HyperSeg：用于实时语义分割的补全网络**

- 作者单位: Facebook AI, 巴伊兰大学, 特拉维夫大学

- Homepage: https://nirkin.com/hyperseg/
- Paper: https://openaccess.thecvf.com/content/CVPR2021/papers/Nirkin_HyperSeg_Patch-Wise_Hypernetwork_for_Real-Time_Semantic_Segmentation_CVPR_2021_paper.pdf

- Code: https://github.com/YuvalNirkin/hyperseg

**2. Rethinking BiSeNet For Real-time Semantic Segmentation**
**2. 重新思考 BiSeNet 对于实时语义分割的新方法**

- 作者单位: 美团

- Paper: https://arxiv.org/abs/2104.13188

- Code: https://github.com/MichaelFan01/STDC-Seg

**3. Progressive Semantic Segmentation**
**3. 进步性语义分割**

- 作者单位: VinAI Research, VinUniversity, 阿肯色大学, 石溪大学
- Paper: https://arxiv.org/abs/2104.03778
- Code: https://github.com/VinAIResearch/MagNet

**4. Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers**
**4. 从序列到序列的角度重新思考语义分割，使用变压器**

- 作者单位: 复旦大学, 牛津大学, 萨里大学, 腾讯优图, Facebook AI
- Homepage: https://fudan-zvg.github.io/SETR
- Paper: https://arxiv.org/abs/2012.15840
- Code: https://github.com/fudan-zvg/SETR

**5. Capturing Omni-Range Context for Omnidirectional Segmentation**
**5. 捕获全方位范围上下文以进行多方向分割**

- 作者单位: 卡尔斯鲁厄理工学院, 卡尔·蔡司, 华为
- Paper: https://arxiv.org/abs/2103.05687
- Code: None

**6. Learning Statistical Texture for Semantic Segmentation**
**6. 学习统计纹理用于语义分割**

- 作者单位: 北航, 商汤科技
- Paper: https://arxiv.org/abs/2103.04133
- Code: None

**7. InverseForm: A Loss Function for Structured Boundary-Aware Segmentation**
**7. InverseForm: 结构化边界感知的分割损失函数**

- 作者单位: 高通AI研究院
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Borse_InverseForm_A_Loss_Function_for_Structured_Boundary-Aware_Segmentation_CVPR_2021_paper.html
- Code: None

**8. DCNAS: Densely Connected Neural Architecture Search for Semantic Image Segmentation**
**8. DCNAS：密集连接神经架构，用于语义图像分割搜索**

- 作者单位: Joyy Inc, 快手, 北航等
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_DCNAS_Densely_Connected_Neural_Architecture_Search_for_Semantic_Image_Segmentation_CVPR_2021_paper.html
- Code: None

## 弱监督语义分割

**9. Railroad Is Not a Train: Saliency As Pseudo-Pixel Supervision for Weakly Supervised Semantic Segmentation**
**9. 铁路不是火车：弱监督语义分割的显著性作为伪像素监督**

- 作者单位: 延世大学, 成均馆大学
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Lee_Railroad_Is_Not_a_Train_Saliency_As_Pseudo-Pixel_Supervision_for_CVPR_2021_paper.html
- Code: https://github.com/halbielee/EPS

**10. Background-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation**
**10. 背景感知的池化和对噪声敏感的损失 weakly-supervised semantic segmentation**

- 作者单位: 延世大学
- Homepage:  https://cvlab.yonsei.ac.kr/projects/BANA/ 
- Paper: https://arxiv.org/abs/2104.00905
- Code: None

**11. Non-Salient Region Object Mining for Weakly Supervised Semantic Segmentation**
**11. 非显著区域对象挖掘：弱监督语义分割**

- 作者单位: 南京理工大学, MBZUAI, 电子科技大学, 阿德莱德大学, 悉尼科技大学

- Paper: https://arxiv.org/abs/2103.14581
- Code: https://github.com/NUST-Machine-Intelligence-Laboratory/nsrom

**12. Embedded Discriminative Attention Mechanism for Weakly Supervised Semantic Segmentation**
**12. 用于弱监督语义分割的嵌入式判别性注意力机制**

- 作者单位: 北京理工大学, 美团
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Wu_Embedded_Discriminative_Attention_Mechanism_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2021_paper.html
- Code: https://github.com/allenwu97/EDAM

**13. BBAM: Bounding Box Attribution Map for Weakly Supervised Semantic and Instance Segmentation**
**13. BBAM：弱监督语义和实例分割的边界框归一化映射**

- 作者单位: 首尔大学
- Paper: https://arxiv.org/abs/2103.08907
- Code: https://github.com/jbeomlee93/BBAM

## 半监督语义分割

**14. Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision**
**14. 半监督语义分割与交叉伪监督**

- 作者单位: 北京大学, 微软亚洲研究院
- Paper: https://arxiv.org/abs/2106.01226
- Code: https://github.com/charlesCXK/TorchSemiSeg

**15. Semi-supervised Domain Adaptation based on Dual-level Domain Mixing for Semantic Segmentation**
**15. 半监督领域自适应基于双层次领域混合语义分割**

- 作者单位: 华为, 大连理工大学, 北京大学
- Paper: https://arxiv.org/abs/2103.04705
- Code: None

**16. Semi-Supervised Semantic Segmentation With Directional Context-Aware Consistency**
**16. 半监督语义分割与方向上下文感知一致性**

- 作者单位: 香港中文大学, 思谋科技, 牛津大学
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Lai_Semi-Supervised_Semantic_Segmentation_With_Directional_Context-Aware_Consistency_CVPR_2021_paper.html
- Code: None

**17. Semantic Segmentation With Generative Models: Semi-Supervised Learning and Strong Out-of-Domain Generalization**
**17. 语义分割与生成模型：半监督学习和强领域外推广**

- 作者单位: NVIDIA, 多伦多大学, 耶鲁大学, MIT, Vector Institute
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Li_Semantic_Segmentation_With_Generative_Models_Semi-Supervised_Learning_and_Strong_Out-of-Domain_CVPR_2021_paper.html
- Code: https://nv-tlabs.github.io/semanticGAN/

**18. Three Ways To Improve Semantic Segmentation With Self-Supervised Depth Estimation**
**18. 三种改进语义分割的方法之一：自监督深度估计**

- 作者单位: ETH Zurich, 伯恩大学, 鲁汶大学
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Hoyer_Three_Ways_To_Improve_Semantic_Segmentation_With_Self-Supervised_Depth_Estimation_CVPR_2021_paper.html
- Code: https://github.com/lhoyer/improving_segmentation_with_selfsupervised_depth

## 域自适应语义分割

**19. Cluster, Split, Fuse, and Update: Meta-Learning for Open Compound Domain Adaptive Semantic Segmentation**
**19. 集群、划分、融合和更新：用于开放复合领域自适应语义分割的元学习**

- 作者单位: ETH Zurich, 鲁汶大学, 电子科技大学

- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Gong_Cluster_Split_Fuse_and_Update_Meta-Learning_for_Open_Compound_Domain_CVPR_2021_paper.html
- Code: None

**20. Source-Free Domain Adaptation for Semantic Segmentation**
**20. 源免费域适应语义分割**

- 作者单位: 华东师范大学
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Liu_Source-Free_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2021_paper.html
- Code: None

**21. Uncertainty Reduction for Model Adaptation in Semantic Segmentation**
**21. 不确定性降低在语义分割中的模型自适应**

- 作者单位: Idiap Research Institute, EPFL, 日内瓦大学
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/S_Uncertainty_Reduction_for_Model_Adaptation_in_Semantic_Segmentation_CVPR_2021_paper.html
- Code: https://git.io/JthPp

**22. Self-Supervised Augmentation Consistency for Adapting Semantic Segmentation**
**22. 自监督增强一致性用于适应语义分割**

- 作者单位: 达姆施塔特工业大学, hessian.AI
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Araslanov_Self-Supervised_Augmentation_Consistency_for_Adapting_Semantic_Segmentation_CVPR_2021_paper.html
- Code: https://github.com/visinf/da-sac

**23. RobustNet: Improving Domain Generalization in Urban-Scene Segmentation via Instance Selective Whitening**
**23. RobustNet：通过实例选择性消融来改进城市景观分割中的领域泛化**

- 作者单位: LG AI研究院, KAIST等
- Paper: https://arxiv.org/abs/2103.15597
- Code: https://github.com/shachoi/RobustNet

**24. Coarse-to-Fine Domain Adaptive Semantic Segmentation with Photometric Alignment and Category-Center Regularization**
**24. 粗到细领域自适应语义分割与光度对齐和类别中心正则化**

- 作者单位: 香港大学, 深睿医疗
- Paper: https://arxiv.org/abs/2103.13041
- Code: None

**25. MetaCorrection: Domain-aware Meta Loss Correction for Unsupervised Domain Adaptation in Semantic Segmentation**
**25. 元学习：用于语义分割的无监督领域自适应的领域感知元损失校正**

- 作者单位: 香港城市大学, 百度
- Paper: https://arxiv.org/abs/2103.05254
- Code: https://github.com/cyang-cityu/MetaCorrection

**26. Multi-Source Domain Adaptation with Collaborative Learning for Semantic Segmentation**
**26. 多源领域自适应与合作学习用于语义分割**

- 作者单位: 华为云, 华为诺亚, 大连理工大学
- Paper: https://arxiv.org/abs/2103.04717
- Code: None

**27. Prototypical Pseudo Label Denoising and Target Structure Learning for Domain Adaptive Semantic Segmentation**
**27. 典型伪标签去噪与目标结构学习：领域自适应语义分割**

- 作者单位: 中国科学技术大学, 微软亚洲研究院
- Paper: https://arxiv.org/abs/2101.10979
- Code: https://github.com/microsoft/ProDA

**28. DANNet: A One-Stage Domain Adaptation Network for Unsupervised Nighttime Semantic Segmentation**
**28. DANet：一种用于无监督夜间语义分割的一阶段领域自适应网络**

- 作者单位: 南卡罗来纳大学, 天远视科技
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Wu_DANNet_A_One-Stage_Domain_Adaptation_Network_for_Unsupervised_Nighttime_Semantic_CVPR_2021_paper.html
- Code: https://github.com/W-zx-Y/DANNet

## Few-Shot语义分割

**29. Scale-Aware Graph Neural Network for Few-Shot Semantic Segmentation**
**29. 用于少量多目标语义分割的规模感知图神经网络**

- 作者单位: MBZUAI, IIAI, 哈工大
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Xie_Scale-Aware_Graph_Neural_Network_for_Few-Shot_Semantic_Segmentation_CVPR_2021_paper.html
- Code: None

**30. Anti-Aliasing Semantic Reconstruction for Few-Shot Semantic Segmentation**
**30. 对抗平滑语义重建用于少样本语义分割**

- 作者单位: 国科大, 清华大学
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Liu_Anti-Aliasing_Semantic_Reconstruction_for_Few-Shot_Semantic_Segmentation_CVPR_2021_paper.html
- Code: https://github.com/Bibkiller/ASR 

## 无监督语义分割

**31. PiCIE: Unsupervised Semantic Segmentation Using Invariance and Equivariance in Clustering**
**31. PiCIE：利用聚类中的不变性和等方性进行无监督语义分割**

- 作者单位: UT-Austin, 康奈尔大学
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Cho_PiCIE_Unsupervised_Semantic_Segmentation_Using_Invariance_and_Equivariance_in_Clustering_CVPR_2021_paper.html
- Code: https:// github.com/janghyuncho/PiCIE

## 视频语义分割

**32. VSPW: A Large-scale Dataset for Video Scene Parsing in the Wild**
**32. VSPW：用于野外视频场景解析的大规模数据集**

- 作者单位: 浙江大学, 百度, 悉尼科技大学
- Homepage: https://www.vspwdataset.com/
- Paper: https://www.vspwdataset.com/CVPR2021__miao.pdf
- GitHub: https://github.com/sssdddwww2/vspw_dataset_download

## 其它

**33. Continual Semantic Segmentation via Repulsion-Attraction of Sparse and Disentangled Latent Representations**
**33. 持续语义分割通过稀疏和分离的潜在表示的排斥-吸引力**

- 作者单位: 帕多瓦大学

- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Michieli_Continual_Semantic_Segmentation_via_Repulsion-Attraction_of_Sparse_and_Disentangled_Latent_CVPR_2021_paper.html
- Code: https://lttm.dei.unipd.it/paper_data/SDR/

**34. Exploit Visual Dependency Relations for Semantic Segmentation**
**34. 利用视觉依赖关系进行语义分割**

- 作者单位: 伊利诺伊大学芝加哥分校
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Liu_Exploit_Visual_Dependency_Relations_for_Semantic_Segmentation_CVPR_2021_paper.html
- Code: None

**35. Revisiting Superpixels for Active Learning in Semantic Segmentation With Realistic Annotation Costs**
**35. 重新审视超像素在语义分割中的主动学习与真实注释成本**

- 作者单位: Institute for Infocomm Research, 新加坡国立大学
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Cai_Revisiting_Superpixels_for_Active_Learning_in_Semantic_Segmentation_With_Realistic_CVPR_2021_paper.html
- Code: None

**36. PLOP: Learning without Forgetting for Continual Semantic Segmentation**
**36. PLOP: 持续语义分割的学习无遗忘**

- 作者单位: 索邦大学, Heuritech, Datakalab, Valeo.ai 
- Paper: https://arxiv.org/abs/2011.11390
- Code: https://github.com/arthurdouillard/CVPR2021_PLOP

**37. 3D-to-2D Distillation for Indoor Scene Parsing**
**37. 3D至2D蒸馏在室内场景解析中的应用**

- 作者单位: 香港中文大学, 香港大学
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Liu_3D-to-2D_Distillation_for_Indoor_Scene_Parsing_CVPR_2021_paper.html
- Code: None

**38. Bidirectional Projection Network for Cross Dimension Scene Understanding**
**38. 双向投影网络用于多维度场景理解**

- 作者单位: 香港中文大学, 牛津大学等
- Paper(Oral): https://arxiv.org/abs/2103.14326
- Code: https://github.com/wbhu/BPNet

**39. PointFlow: Flowing Semantics Through Points for Aerial Image Segmentation**
**39. PointFlow：通过飞行图像进行分割的流动语义**

- 作者单位: 北京大学, 中科院, 国科大, ETH Zurich, 商汤科技等

- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Li_PointFlow_Flowing_Semantics_Through_Points_for_Aerial_Image_Segmentation_CVPR_2021_paper.html
- Code: https://github.com/lxtGH/PFSegNets

<a name="Instance-Segmentation"></a>

# 实例分割(Instance Segmentation)

**DCT-Mask: Discrete Cosine Transform Mask Representation for Instance Segmentation**
**DCT-Mask:离散余弦变换 mask 表示，用于实例分割**

- Paper: https://arxiv.org/abs/2011.09876
- Code: https://github.com/aliyun/DCT-Mask

**Incremental Few-Shot Instance Segmentation**
**逐步减少样本数量实例分割**

- Paper: https://arxiv.org/abs/2105.05312
- Code: https://github.com/danganea/iMTFA

**A^2-FPN: Attention Aggregation based Feature Pyramid Network for Instance Segmentation**
**A^2-FPN: 基于注意力的特征金字塔网络实例分割**

- Paper: https://arxiv.org/abs/2105.03186
- Code: None

**RefineMask: Towards High-Quality Instance Segmentation with Fine-Grained Features**
**优化遮罩：使用细粒度特征实现高质量实例分割**

- Paper: https://arxiv.org/abs/2104.08569
- Code: https://github.com/zhanggang001/RefineMask/

**Look Closer to Segment Better: Boundary Patch Refinement for Instance Segmentation**
**更深入地分割：实例分割的边界补丁细化**

- Paper: https://arxiv.org/abs/2104.05239
- Code:  https://github.com/tinyalpha/BPR 

**Multi-Scale Aligned Distillation for Low-Resolution Detection**
**多尺度蒸馏低分辨率检测**

- Paper: https://jiaya.me/papers/ms_align_distill_cvpr21.pdf

- Code: https://github.com/Jia-Research-Lab/MSAD

**Boundary IoU: Improving Object-Centric Image Segmentation Evaluation**
**边界IoU：改进对象中心图像分割评估**

- Homepage: https://bowenc0221.github.io/boundary-iou/
- Paper: https://arxiv.org/abs/2103.16562

- Code: https://github.com/bowenc0221/boundary-iou-api

**Deep Occlusion-Aware Instance Segmentation with Overlapping BiLayers**
**深度遮挡感实例分割与重叠双层**

- Paper: https://arxiv.org/abs/2103.12340

- Code: https://github.com/lkeab/BCNet 

**Zero-shot instance segmentation（Not Sure）**
**零样本实例分割（不确定）**

- Paper: None
- Code: https://github.com/CVPR2021-pape-id-1395/CVPR2021-paper-id-1395

## 视频实例分割

**STMask: Spatial Feature Calibration and Temporal Fusion for Effective One-stage Video Instance Segmentation**
**STMask：用于有效的一级视频实例分割的空间特征校准和时间融合**

- Paper: http://www4.comp.polyu.edu.hk/~cslzhang/papers.htm
- Code: https://github.com/MinghanLi/STMask

**End-to-End Video Instance Segmentation with Transformers**
**端到端视频实例分割与变压器**

- Paper(Oral): https://arxiv.org/abs/2011.14503
- Code: https://github.com/Epiphqny/VisTR

<a name="Panoptic-Segmentation"></a>

# 全景分割(Panoptic Segmentation)

**ViP-DeepLab: Learning Visual Perception with Depth-aware Video Panoptic Segmentation**
**ViP-DeepLab：使用深度感知的视频全景分割学习视觉感知**

- Paper: https://arxiv.org/abs/2012.05258
- Code: https://github.com/joe-siyuan-qiao/ViP-DeepLab
- Dataset: https://github.com/joe-siyuan-qiao/ViP-DeepLab

**Part-aware Panoptic Segmentation**
**部分感知的全景分割**

- Paper: https://arxiv.org/abs/2106.06351
- Code: https://github.com/tue-mps/panoptic_parts
- Dataset: https://github.com/tue-mps/panoptic_parts

**Exemplar-Based Open-Set Panoptic Segmentation Network**
**基于示例的开集像素分割网络**

- Homepage: https://cv.snu.ac.kr/research/EOPSN/
- Paper: https://arxiv.org/abs/2105.08336
- Code: https://github.com/jd730/EOPSN

**MaX-DeepLab: End-to-End Panoptic Segmentation With Mask Transformers**
**MaX-DeepLab：使用遮罩变分器的端到端全景分割**

- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Wang_MaX-DeepLab_End-to-End_Panoptic_Segmentation_With_Mask_Transformers_CVPR_2021_paper.html
- Code: None

**Panoptic Segmentation Forecasting**
**Panoptic 分割预测**

- Paper: https://arxiv.org/abs/2104.03962
- Code: https://github.com/nianticlabs/panoptic-forecasting

**Fully Convolutional Networks for Panoptic Segmentation**
**全卷积网络用于全景分割**

- Paper: https://arxiv.org/abs/2012.00720

- Code: https://github.com/yanwei-li/PanopticFCN

**Cross-View Regularization for Domain Adaptive Panoptic Segmentation**
**跨视角正则化域自适应全视角分割**

- Paper: https://arxiv.org/abs/2103.02584
- Code: None

<a name="Medical-Image-Segmentation"></a>

# 医学图像分割

**1. Learning Calibrated Medical Image Segmentation via Multi-Rater Agreement Modeling**
**1. 通过多评者协议进行学习，实现医学图像分割校准**

- 作者单位: 腾讯天衍实验室, 北京同仁医院
- Paper(Best Paper Candidate): https://openaccess.thecvf.com/content/CVPR2021/html/Ji_Learning_Calibrated_Medical_Image_Segmentation_via_Multi-Rater_Agreement_Modeling_CVPR_2021_paper.html
- Code: https://github.com/jiwei0921/MRNet/

**2. Every Annotation Counts: Multi-Label Deep Supervision for Medical Image Segmentation**
**2. 每个注释都算数：多标签深度监督用于医学图像分割**

- 作者单位: 卡尔斯鲁厄理工学院, 卡尔·蔡司等
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Reiss_Every_Annotation_Counts_Multi-Label_Deep_Supervision_for_Medical_Image_Segmentation_CVPR_2021_paper.html
- Code: None

**3. FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space**
**3. FedDG: 基于连续频率空间的医学图像分割的联邦领域泛化**

- 作者单位: 香港中文大学, 香港理工大学
- Paper: https://arxiv.org/abs/2103.06030
- Code: https://github.com/liuquande/FedDG-ELCFS

**4. DiNTS: Differentiable Neural Network Topology Search for 3D Medical Image Segmentation**
**4. DiNTS：用于3D医学图像分割的可微神经网络拓扑搜索**

- 作者单位: 约翰斯·霍普金斯大大学, NVIDIA
- Paper(Oral): https://arxiv.org/abs/2103.15954
- Code: None

**5. DARCNN: Domain Adaptive Region-Based Convolutional Neural Network for Unsupervised Instance Segmentation in Biomedical Images**
**5. DARCNN：用于生物医学图像的无监督实例分割领域自适应区域卷积神经网络**

- 作者单位: 斯坦福大学

- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Hsu_DARCNN_Domain_Adaptive_Region-Based_Convolutional_Neural_Network_for_Unsupervised_Instance_CVPR_2021_paper.html
- Code: None

<a name="VOS"></a>

# 视频目标分割(Video-Object-Segmentation)

**Learning Position and Target Consistency for Memory-based Video Object Segmentation**
**基于记忆的视觉对象分割学习的位置和目标一致性**

- Paper: https://arxiv.org/abs/2104.04329
- Code: None

**SSTVOS: Sparse Spatiotemporal Transformers for Video Object Segmentation**
**SSTVOS：稀疏时空转换器用于视频对象分割**

- Paper(Oral): https://arxiv.org/abs/2101.08833
- Code: https://github.com/dukebw/SSTVOS

<a name="IVOS"></a>

# 交互式视频目标分割(Interactive-Video-Object-Segmentation)

**Modular Interactive Video Object Segmentation: Interaction-to-Mask, Propagation and Difference-Aware Fusion**
**模块化交互式视频对象分割：交互到遮罩、传播和差异感知的融合**

- Homepage: https://hkchengrex.github.io/MiVOS/

- Paper: https://arxiv.org/abs/2103.07941

- Code: https://github.com/hkchengrex/MiVOS
- Demo: https://hkchengrex.github.io/MiVOS/video.html#partb

**Learning to Recommend Frame for Interactive Video Object Segmentation in the Wild**
**学习在野外为交互式视频对象分割推荐框架**

- Paper: https://arxiv.org/abs/2103.10391

- Code: https://github.com/svip-lab/IVOS-W

<a name="Saliency-Detection"></a>

# 显著性检测(Saliency Detection)

**Uncertainty-aware Joint Salient Object and Camouflaged Object Detection**
**不确定性感知的联合显著物体和伪装物体检测**

- Paper: https://arxiv.org/abs/2104.02628

- Code: https://github.com/JingZhang617/Joint_COD_SOD

**Deep RGB-D Saliency Detection with Depth-Sensitive Attention and Automatic Multi-Modal Fusion**
**深度 RGB-D 显著性检测与深度敏感关注和自动多模态融合**

- Paper(Oral): https://arxiv.org/abs/2103.11832
- Code: https://github.com/sunpeng1996/DSA2F

<a name="Camouflaged-Object-Detection"></a>

# 伪装物体检测(Camouflaged Object Detection)

**Uncertainty-aware Joint Salient Object and Camouflaged Object Detection**
**联合显著物体和伪装物体检测**

- Paper: https://arxiv.org/abs/2104.02628

- Code: https://github.com/JingZhang617/Joint_COD_SOD

<a name="CoSOD"></a>

# 协同显著性检测(Co-Salient Object Detection)

**Group Collaborative Learning for Co-Salient Object Detection**
**小组合作学习中的共显性目标检测**

- Paper: https://arxiv.org/abs/2104.01108
- Code: https://github.com/fanq15/GCoNet

<a name="Matting"></a>

# 协同显著性检测(Image Matting)

**Semantic Image Matting**
**语义图像匹配**

- Paper: https://arxiv.org/abs/2104.08201
- Code: https://github.com/nowsyn/SIM
- Dataset: https://github.com/nowsyn/SIM

<a name="Re-ID"></a>

# 行人重识别(Person Re-identification)

**Generalizable Person Re-identification with Relevance-aware Mixture of Experts**
**具有相关性感知的专家混合的领域自适应人机识别**

- Paper: https://arxiv.org/abs/2105.09156
- Code: None

**Unsupervised Multi-Source Domain Adaptation for Person Re-Identification**
**无监督多源领域自适应用于人物识别**

- Paper: https://arxiv.org/abs/2104.12961
- Code: None

**Combined Depth Space based Architecture Search For Person Re-identification**
**深度空间联合架构寻找人重新识别**

- Paper: https://arxiv.org/abs/2104.04163
- Code: None

<a name="Person-Search"></a>

# 行人搜索(Person Search)

**Anchor-Free Person Search**
**无锚自由人搜索**

- Paper: https://arxiv.org/abs/2103.11617
- Code: https://github.com/daodaofr/AlignPS
- Interpretation: [首个无需锚框（Anchor-Free）的行人搜索框架 | CVPR 2021](https://mp.weixin.qq.com/s/iqJkgp0JBanmeBPyHUkb-A)

<a name="Video-Understanding"></a>

# 视频理解/行为识别(Video Understanding)

**Temporal-Relational CrossTransformers for Few-Shot Action Recognition**
**时空关联的交叉变换器在少样本动作识别中**

- Paper: https://arxiv.org/abs/2101.06184
- Code: https://github.com/tobyperrett/trx

**FrameExit: Conditional Early Exiting for Efficient Video Recognition**
**框架退出：用于高效视频识别的条件的早期退出**

- Paper(Oral): https://arxiv.org/abs/2104.13400
- Code: None

**No frame left behind: Full Video Action Recognition**
**没有遗漏任何帧：完整视频动作识别**

- Paper: https://arxiv.org/abs/2103.15395
- Code: None

**Learning Salient Boundary Feature for Anchor-free Temporal Action Localization**
**学习突出边界特征以无锚时间动作局部定位**

- Paper: https://arxiv.org/abs/2103.13137
- Code: None

**Temporal Context Aggregation Network for Temporal Action Proposal Refinement**
**针对时间动作提议的 Temporal 上下文聚合网络**

- Paper: https://arxiv.org/abs/2103.13141
- Code: None
- Interpretation: [CVPR 2021 | TCANet：最强时序动作提名修正网络](https://mp.weixin.qq.com/s/UOWMfpTljkyZznHtpkQBhA)

**ACTION-Net: Multipath Excitation for Action Recognition**
**网络-Net：用于动作识别的多路径激发**

- Paper: https://arxiv.org/abs/2103.07372
- Code: https://github.com/V-Sense/ACTION-Net

**Removing the Background by Adding the Background: Towards Background Robust Self-supervised Video Representation Learning**
**移除背景，添加背景：朝着背景鲁棒的自我监督视频表示学习**

- Homepage: https://fingerrec.github.io/index_files/jinpeng/papers/CVPR2021/project_website.html
- Paper: https://arxiv.org/abs/2009.05769
- Code: https://github.com/FingerRec/BE

**TDN: Temporal Difference Networks for Efficient Action Recognition**
**TDN：用于有效动作识别的时间差网络**

- Paper: https://arxiv.org/abs/2012.10071
- Code: https://github.com/MCG-NJU/TDN

<a name="Face-Recognition"></a>

# 人脸识别(Face Recognition)

**A 3D GAN for Improved Large-pose Facial Recognition**
**3D GAN用于改进大规模人脸识别**

- Paper: https://arxiv.org/abs/2012.10545
- Code: None

**MagFace: A Universal Representation for Face Recognition and Quality Assessment**
**MagFace：面部识别和质量评估的通用表示**

- Paper(Oral): https://arxiv.org/abs/2103.06627
- Code: https://github.com/IrvingMeng/MagFace

**WebFace260M: A Benchmark Unveiling the Power of Million-Scale Deep Face Recognition**
**WebFace260M：揭示百万规模深度人脸识别力量的基准**

- Homepage: https://www.face-benchmark.org/
- Paper: https://arxiv.org/abs/2103.04098 
- Dataset: https://www.face-benchmark.org/

**When Age-Invariant Face Recognition Meets Face Age Synthesis: A Multi-Task Learning Framework**
**当不变年龄人脸识别与面部年龄合成相遇时：一种多任务学习框架**

- Paper(Oral): https://arxiv.org/abs/2103.01520
- Code: https://github.com/Hzzone/MTLFace
- Dataset: https://github.com/Hzzone/MTLFace

<a name="Face-Detection"></a>

# 人脸检测(Face Detection)

**HLA-Face: Joint High-Low Adaptation for Low Light Face Detection**
**HLA-Face：低光面部检测的高低适应联合**

- Homepage: https://daooshee.github.io/HLA-Face-Website/
- Paper: https://arxiv.org/abs/2104.01984
- Code: https://github.com/daooshee/HLA-Face-Code

**CRFace: Confidence Ranker for Model-Agnostic Face Detection Refinement**
**CRFace：模型无关面部检测自信度排名器**

- Paper: https://arxiv.org/abs/2103.07017
- Code: None

<a name="Face-Anti-Spoofing"></a>

# 人脸活体检测(Face Anti-Spoofing)

**Cross Modal Focal Loss for RGBD Face Anti-Spoofing**
**跨模态焦点损失用于RGBD人脸对抗伪造成**

- Paper: https://arxiv.org/abs/2103.00948
- Code: None

<a name="Deepfake-Detection"></a>

# Deepfake检测(Deepfake Detection)

**Spatial-Phase Shallow Learning: Rethinking Face Forgery Detection in Frequency Domain**
**空间相位浅层学习：重新思考频率域中的面部伪造检测**

- Paper：https://arxiv.org/abs/2103.01856
- Code: None

**Multi-attentional Deepfake Detection**
**多注意力深度伪造检测**

- Paper：https://arxiv.org/abs/2103.02406
- Code: None

<a name="Age-Estimation"></a>

# 人脸年龄估计(Age Estimation)

**Continuous Face Aging via Self-estimated Residual Age Embedding**
**通过自估残余年龄嵌入的连续面部衰老**

- Paper: https://arxiv.org/abs/2105.00020
- Code: None

**PML: Progressive Margin Loss for Long-tailed Age Classification**
**PML: 渐进式长尾年龄分类的渐进式边际损失**

- Paper: https://arxiv.org/abs/2103.02140
- Code: None

<a name="FER"></a>

# 人脸表情识别(Facial Expression Recognition)

**Affective Processes: stochastic modelling of temporal context for emotion and facial expression recognition**
**情感过程：对情感和面部表情识别的时空语境的随机建模**

- Paper: https://arxiv.org/abs/2103.13372
- Code: None

<a name="Deepfakes"></a>

# Deepfakes

**MagDR: Mask-guided Detection and Reconstruction for Defending Deepfakes**
**MagDR：用于防御深度伪造的 mask 引导检测与重建**

- Paper: https://arxiv.org/abs/2103.14211
- Code: None

<a name="Human-Parsing"></a>

# 人体解析(Human Parsing)

**Differentiable Multi-Granularity Human Representation Learning for Instance-Aware Human Semantic Parsing**
**不同步多粒度人类表示学习实例感知语义解析**

- Paper: https://arxiv.org/abs/2103.04570
- Code: https://github.com/tfzhou/MG-HumanParsing

<a name="Human-Pose-Estimation"></a>

# 2D/3D人体姿态估计(2D/3D Human Pose Estimation)

## 2D 人体姿态估计

**ViPNAS: Efficient Video Pose Estimation via Neural Architecture Search**
**通过神经架构搜索实现高效的视频姿态估计：ViPNAS**

- Paper: ttps://arxiv.org/abs/2105.10154
- Code: None

**When Human Pose Estimation Meets Robustness: Adversarial Algorithms and Benchmarks**
**人体姿态估计与鲁棒性：对抗性算法和基准**

- Paper: https://arxiv.org/abs/2105.06152
- Code: None

**Pose Recognition with Cascade Transformers**
**带有级联变形的姿态识别**

- Paper: https://arxiv.org/abs/2104.06976

- Code: https://github.com/mlpc-ucsd/PRTR

**DCPose: Deep Dual Consecutive Network for Human Pose Estimation**
**DCPose: 深度双重连续网络用于人体姿态估计**

-  Paper: https://arxiv.org/abs/2103.07254
- Code: https://github.com/Pose-Group/DCPose 

## 3D 人体姿态估计

**End-to-End Human Pose and Mesh Reconstruction with Transformers**
**端到端的人体姿态和网格重构**

- Paper: https://arxiv.org/abs/2012.09760
- Code: https://github.com/microsoft/MeshTransformer

**PoseAug: A Differentiable Pose Augmentation Framework for 3D Human Pose Estimation**
**PoseAug: 3D 人体姿态估计的差分框架**

- Paper(Oral): https://arxiv.org/abs/2105.02465

- Code: https://github.com/jfzhang95/PoseAug

**Camera-Space Hand Mesh Recovery via Semantic Aggregation and Adaptive 2D-1D Registration**
**通过语义聚合和自适应2D-1D对齐的相机空间手部网格重构**

- Paper: https://arxiv.org/abs/2103.02845
- Code: https://github.com/SeanChenxy/HandMesh

**Monocular 3D Multi-Person Pose Estimation by Integrating Top-Down and Bottom-Up Networks**
**单目3D多人体位估计通过将自顶向下和自底向上网络集成**

- Paper: https://arxiv.org/abs/2104.01797
- https://github.com/3dpose/3D-Multi-Person-Pose

**HybrIK: A Hybrid Analytical-Neural Inverse Kinematics Solution for 3D Human Pose and Shape Estimation**
**HybrIK：用于3D人体姿势和形状估计的混合分析-神经逆运动解决方案**

- Homepage: https://jeffli.site/HybrIK/ 
- Paper: https://arxiv.org/abs/2011.14672
- Code: https://github.com/Jeff-sjtu/HybrIK

<a name="Animal-Pose-Estimation"></a>

# 动物姿态估计(Animal Pose Estimation)

**From Synthetic to Real: Unsupervised Domain Adaptation for Animal Pose Estimation**
**从合成到真实：动物姿态估计的无监督领域自适应**

- Paper: https://arxiv.org/abs/2103.14843
- Code: None

<a name="Hand-Pose-Estimation"></a>

# 手部姿态估计(Hand Pose Estimation)

**Semi-Supervised 3D Hand-Object Poses Estimation with Interactions in Time**
**半监督3D手部姿态估计与时间交互**

- Homepage: https://stevenlsw.github.io/Semi-Hand-Object/
- Paper: https://arxiv.org/abs/2106.05266
- Code: https://github.com/stevenlsw/Semi-Hand-Object

<a name="Human-Volumetric-Capture"></a>

# Human Volumetric Capture

**POSEFusion: Pose-guided Selective Fusion for Single-view Human Volumetric Capture**
**PoseFusion：用于单视图人体体积捕获的姿势引导选择性融合**

- Homepage: http://www.liuyebin.com/posefusion/posefusion.html

- Paper(Oral): https://arxiv.org/abs/2103.15331
- Code: None

<a name="Scene-Text-Recognition"></a>

# 场景文本检测(Scene Text Detection)

**Fourier Contour Embedding for Arbitrary-Shaped Text Detection**
**傅里叶边界嵌入用于任意形状文本检测**

- Paper: https://arxiv.org/abs/2104.10442
- Code: None

<a name="Scene-Text-Recognition"></a>

# 场景文本识别(Scene Text Recognition)

**Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text Recognition**
**像人类一样阅读：用于场景文本识别的自主、双向和迭代语言模型**

- Paper: https://arxiv.org/abs/2103.06495
- Code: https://github.com/FangShancheng/ABINet

<a name="Image-Compression"></a>

# 图像压缩

**Checkerboard Context Model for Efficient Learned Image Compression**
**用于高效学习图像压缩的棋盘上下文模型**

- Paper: https://arxiv.org/abs/2103.15306
- Code: None

**Slimmable Compressive Autoencoders for Practical Neural Image Compression**
**用于实际神经图像压缩的紧致压缩自动编码器**

- Paper: https://arxiv.org/abs/2103.15726
- Code: None

**Attention-guided Image Compression by Deep Reconstruction of Compressive Sensed Saliency Skeleton**
**深度重建压缩感知骨架的注意力引导图像压缩**

- Paper: https://arxiv.org/abs/2103.15368
- Code: None

<a name="Model-Compression"></a>

# 模型压缩/剪枝/量化

**Teachers Do More Than Teach: Compressing Image-to-Image Models**
**教师不仅仅是传授知识：压缩图像到图像模型**

- Paper: https://arxiv.org/abs/2103.03467
- Code: https://github.com/snap-research/CAT

## 模型剪枝

**Dynamic Slimmable Network**
**动态可缩放网络**

- Paper: https://arxiv.org/abs/2103.13258
- Code: https://github.com/changlin31/DS-Net

## 模型量化

**Network Quantization with Element-wise Gradient Scaling**
**网络量化与元素级梯度缩放**

- Paper: https://arxiv.org/abs/2104.00903
- Code: None

**Zero-shot Adversarial Quantization**
**零样本对抗量化**

- Paper(Oral): https://arxiv.org/abs/2103.15263
- Code: https://git.io/Jqc0y

**Learnable Companding Quantization for Accurate Low-bit Neural Networks**
**可学习扩展量化对准确低位神经网络的改进**

- Paper: https://arxiv.org/abs/2103.07156
- Code: None

<a name="KD"></a>

# 知识蒸馏(Knowledge Distillation)

**Distilling Knowledge via Knowledge Review**
**通过知识审查提炼知识**

- Paper: https://arxiv.org/abs/2104.09044
- Code: https://github.com/Jia-Research-Lab/ReviewKD

**Distilling Object Detectors via Decoupled Features**
**通过分离特征对对象进行提纯**

- Paper: https://arxiv.org/abs/2103.14475
- Code: https://github.com/ggjy/DeFeat.pytorch

<a name="Super-Resolution"></a>

# 超分辨率(Super-Resolution)

**Image Super-Resolution with Non-Local Sparse Attention**
**图像超分辨率与非局部稀疏关注**

- Paper: https://openaccess.thecvf.com/content/CVPR2021/papers/Mei_Image_Super-Resolution_With_Non-Local_Sparse_Attention_CVPR_2021_paper.pdf
- Code: https://github.com/HarukiYqM/Non-Local-Sparse-Attention

**Towards Fast and Accurate Real-World Depth Super-Resolution: Benchmark Dataset and Baseline**
**朝着快速且准确地处理真实世界深度超分辨率：基准数据集和基线**

- Homepage: http://mepro.bjtu.edu.cn/resource.html
- Paper: https://arxiv.org/abs/2104.06174
- Code: None

**ClassSR: A General Framework to Accelerate Super-Resolution Networks by Data Characteristic**
**ClassSR：一种通过数据特性加速超分辨率网络的通用框架**

- Paper: https://arxiv.org/abs/2103.04039
- Code: https://github.com/Xiangtaokong/ClassSR

**AdderSR: Towards Energy Efficient Image Super-Resolution**
**AdderSR：迈向能源高效的图像超分辨率**

- Paper: https://arxiv.org/abs/2009.08891
- Code: None

<a name="Dehazing"></a>

# 去雾(Dehazing)

**Contrastive Learning for Compact Single Image Dehazing**
**对比学习用于紧凑单幅去雾**

- Paper: https://arxiv.org/abs/2104.09367
- Code: https://github.com/GlassyWu/AECR-Net

## 视频超分辨率

**Temporal Modulation Network for Controllable Space-Time Video Super-Resolution**
**时变模块用于控制空间-时间视频超分辨率**

- Paper: None
- Code: https://github.com/CS-GangXu/TMNet

<a name="Image-Restoration"></a>

# 图像恢复(Image Restoration)

**Multi-Stage Progressive Image Restoration**
**多阶段渐进式图像恢复**

- Paper: https://arxiv.org/abs/2102.02808
- Code: https://github.com/swz30/MPRNet

<a name="Image-Inpainting"></a>

# 图像补全(Image Inpainting)

**PD-GAN: Probabilistic Diverse GAN for Image Inpainting**
**PD-GAN：概率多样性生成对抗网络用于图像修复**

- Paper: https://arxiv.org/abs/2105.02201
- Code: https://github.com/KumapowerLIU/PD-GAN

**TransFill: Reference-guided Image Inpainting by Merging Multiple Color and Spatial Transformations**
**TransFill: 基于参考的图像修复技术，通过合并多个颜色和空间变换**

- Homepage: https://yzhouas.github.io/projects/TransFill/index.html
- Paper: https://arxiv.org/abs/2103.15982
- Code: None

<a name="Image-Editing"></a>

# 图像编辑(Image Editing)

**StyleMapGAN: Exploiting Spatial Dimensions of Latent in GAN for Real-time Image Editing**
**StyleMapGAN: 用于实时图像编辑的 GAN 中的潜在空间维度**

- Paper: https://arxiv.org/abs/2104.14754
- Code: https://github.com/naver-ai/StyleMapGAN
- Demo Video: https://youtu.be/qCapNyRA_Ng

**High-Fidelity and Arbitrary Face Editing**
**高精度且任意的面部编辑**

- Paper: https://arxiv.org/abs/2103.15814
- Code: None

**Anycost GANs for Interactive Image Synthesis and Editing**
**任何成本的 GANs 用于交互式图像合成与编辑**

- Paper: https://arxiv.org/abs/2103.03243
- Code: https://github.com/mit-han-lab/anycost-gan

**PISE: Person Image Synthesis and Editing with Decoupled GAN**
**PISE：使用去耦GAN进行人物图像合成与编辑**

- Paper: https://arxiv.org/abs/2103.04023
- Code: https://github.com/Zhangjinso/PISE

**DeFLOCNet: Deep Image Editing via Flexible Low-level Controls**
**DeFLOCNet: 通过灵活的低级控制实现深度图像编辑**

- Paper: http://raywzy.com/
- Code: http://raywzy.com/

**Exploiting Spatial Dimensions of Latent in GAN for Real-time Image Editing**
**利用潜在空间维度中的空间信息进行实时图像编辑的 GAN 模型**

- Paper: None
- Code: None

<a name="Image-Captioning"></a>

# 图像描述(Image Captioning)

**Towards Accurate Text-based Image Captioning with Content Diversity Exploration**
**面向准确文本图像标题生成与内容多样性探索**

- Paper: https://arxiv.org/abs/2105.03236
- Code: None

<a name="Font-Generation"></a>

# 字体生成(Font Generation)

**DG-Font: Deformable Generative Networks for Unsupervised Font Generation**
**DG-Font: 可变形生成网络，用于无监督字体生成**

- Paper: https://arxiv.org/abs/2104.03064

- Code: https://github.com/ecnuycxie/DG-Font

<a name="Image-Matching"></a>

# 图像匹配(Image Matcing)

**LoFTR: Detector-Free Local Feature Matching with Transformers**
**LoFTR: 无需特征检测的本地特征匹配与转换器**

- Homepage: https://zju3dv.github.io/loftr/
- Paper: https://arxiv.org/abs/2104.00680
- Code: https://github.com/zju3dv/LoFTR

**Convolutional Hough Matching Networks**
**卷积高斯匹配网络**

- Homapage: http://cvlab.postech.ac.kr/research/CHM/
- Paper(Oral): https://arxiv.org/abs/2103.16831
- Code: None

<a name="Image-Blending"></a>

# 图像融合(Image Blending)

**Bridging the Visual Gap: Wide-Range Image Blending**
**弥合视觉差距：多领域图像融合**

- Paper: https://arxiv.org/abs/2103.15149

- Code: https://github.com/julia0607/Wide-Range-Image-Blending

<a name="Reflection-Removal"></a>

# 反光去除(Reflection Removal)

**Robust Reflection Removal with Reflection-free Flash-only Cues**
**带有反射无闪光的稳定反射删除**

- Paper: https://arxiv.org/abs/2103.04273
- Code: https://github.com/ChenyangLEI/flash-reflection-removal

<a name="3D-C"></a>

# 3D点云分类(3D Point Clouds Classification)

**Equivariant Point Network for 3D Point Cloud Analysis**
**3D 点云分析等价点网络**

- Paper: https://arxiv.org/abs/2103.14147
- Code: None

**PAConv: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds**
**PAConv: 基于点云的动态核自适应卷积**

- Paper: https://arxiv.org/abs/2103.14635
- Code: https://github.com/CVMI-Lab/PAConv

<a name="3D-Object-Detection"></a>

# 3D目标检测(3D Object Detection)

**3D-MAN: 3D Multi-frame Attention Network for Object Detection**
**3D-MAN：用于物体检测的3D多帧注意力网络**

- Paper: https://arxiv.org/abs/2103.16054
- Code: None

**Back-tracing Representative Points for Voting-based 3D Object Detection in Point Clouds**
**回溯代表性点以用于基于投票的3D目标检测的点云**

- Paper: https://arxiv.org/abs/2104.06114
- Code: https://github.com/cheng052/BRNet

**HVPR: Hybrid Voxel-Point Representation for Single-stage 3D Object Detection**
**HVPR: 混合体素点表示对于单阶段3D对象检测**

- Homepage:  https://cvlab.yonsei.ac.kr/projects/HVPR/ 

- Paper: https://arxiv.org/abs/2104.00902
- Code:  https://github.com/cvlab-yonsei/HVPR 

**LiDAR R-CNN: An Efficient and Universal 3D Object Detector**
**LiDAR R-CNN：一种高效且通用的3D物体检测器**

- Paper: https://arxiv.org/abs/2103.15297
- Code: https://github.com/tusimple/LiDAR_RCNN

**M3DSSD: Monocular 3D Single Stage Object Detector**
**M3DSSD：单目3D单级目标检测**

- Paper: https://arxiv.org/abs/2103.13164

- Code: https://github.com/mumianyuxin/M3DSSD

**SE-SSD: Self-Ensembling Single-Stage Object Detector From Point Cloud**
**SE-SSD：点云自集成的单阶段目标检测器**

- Paper: None
- Code: https://github.com/Vegeta2020/SE-SSD

**Center-based 3D Object Detection and Tracking**
**基于中心的3D对象检测和跟踪**

- Paper: https://arxiv.org/abs/2006.11275
- Code: https://github.com/tianweiy/CenterPoint

**Categorical Depth Distribution Network for Monocular 3D Object Detection**
**单目3D物体检测的范畴深度分布网络**

- Paper: https://arxiv.org/abs/2103.01100
- Code: None

<a name="3D-Semantic-Segmentation"></a>

# 3D语义分割(3D Semantic Segmentation)

**Bidirectional Projection Network for Cross Dimension Scene Understanding**
**双向投影网络用于多尺度场景理解**

- Paper(Oral): https://arxiv.org/abs/2103.14326
- Code: https://github.com/wbhu/BPNet

**Semantic Segmentation for Real Point Cloud Scenes via Bilateral Augmentation and Adaptive Fusion**
**语义分割通过双边增强和自适应融合实现实点云场景**

- Paper: https://arxiv.org/abs/2103.07074
- Code: https://github.com/ShiQiu0419/BAAF-Net

**Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation**
**圆柱形和非对称3D卷积神经网络用于LiDAR分割**

- Paper: https://arxiv.org/abs/2011.10033
- Code:  https://github.com/xinge008/Cylinder3D 

 **Towards Semantic Segmentation of Urban-Scale 3D Point Clouds: A Dataset, Benchmarks and Challenges**

- Homepage: https://github.com/QingyongHu/SensatUrban
- Paper: http://arxiv.org/abs/2009.03137
- Code: https://github.com/QingyongHu/SensatUrban
- Dataset: https://github.com/QingyongHu/SensatUrban

<a name="3D-Panoptic-Segmentation"></a>

# 3D全景分割(3D Panoptic Segmentation)

**Panoptic-PolarNet: Proposal-free LiDAR Point Cloud Panoptic Segmentation**
**panoptic-polarnet：无需提议的LiDAR点云panoptic分割**

- Paper: https://arxiv.org/abs/2103.14962
- Code: https://github.com/edwardzhou130/Panoptic-PolarNet

<a name="3D-Object-Tracking"></a>

# 3D目标跟踪(3D Object Trancking)

**Center-based 3D Object Detection and Tracking**
**基于中心的3D对象检测和跟踪**

- Paper: https://arxiv.org/abs/2006.11275
- Code: https://github.com/tianweiy/CenterPoint

<a name="3D-PointCloud-Registration"></a>

# 3D点云配准(3D Point Cloud Registration)

**ReAgent: Point Cloud Registration using Imitation and Reinforcement Learning**
**ReAgent：使用模拟和强化学习进行点云注册**

- Paper: https://arxiv.org/abs/2103.15231
- Code: None

**PointDSC: Robust Point Cloud Registration using Deep Spatial Consistency**
**点云注册：使用深度空间一致性进行鲁棒的点云配准**

- Paper: https://arxiv.org/abs/2103.05465
- Code: https://github.com/XuyangBai/PointDSC 

**PREDATOR: Registration of 3D Point Clouds with Low Overlap**
**猎食者：具有低重叠的3D点云的注册**

- Paper: https://arxiv.org/abs/2011.13005
- Code: https://github.com/ShengyuH/OverlapPredator

<a name="3D-Point-Cloud-Completion"></a>

# 3D点云补全(3D Point Cloud Completion)

**Unsupervised 3D Shape Completion through GAN Inversion**
**通过GAN反转进行无监督3D形状补全**

- Homepage: https://junzhezhang.github.io/projects/ShapeInversion/
- Paper: https://arxiv.org/abs/2104.13366 
- Code: https://github.com/junzhezhang/shape-inversion 

**Variational Relational Point Completion Network**
**变分关联点完成网络**

- Homepage:  https://paul007pl.github.io/projects/VRCNet 
- Paper: https://arxiv.org/abs/2104.10154
- Code: https://github.com/paul007pl/VRCNet

**Style-based Point Generator with Adversarial Rendering for Point Cloud Completion**
**基于风格的点生成器在点云补全中的对抗渲染**

- Homepage: https://alphapav.github.io/SpareNet/

- Paper: https://arxiv.org/abs/2103.02535
- Code: https://github.com/microsoft/SpareNet

<a name="3D-Reconstruction"></a>

# 3D重建(3D Reconstruction)

**Learning to Aggregate and Personalize 3D Face from In-the-Wild Photo Collection**
**学习从野外照片集中汇总和个性化3D人脸**

- Paper: http://arxiv.org/abs/2106.07852
- Code: https://github.com/TencentYoutuResearch/3DFaceReconstruction-LAP

**Fully Understanding Generic Objects: Modeling, Segmentation, and Reconstruction**
**完全理解通用对象：建模、分割和重建**

- Paper: https://arxiv.org/abs/2104.00858
- Code: None

**NeuralRecon: Real-Time Coherent 3D Reconstruction from Monocular Video**
**NeuralRecon: 从单目视频实现实时3D重构**

- Homepage: https://zju3dv.github.io/neuralrecon/

- Paper(Oral): https://arxiv.org/abs/2104.00681
- Code: https://github.com/zju3dv/NeuralRecon

<a name="6D-Pose-Estimation"></a>

# 6D位姿估计(6D Pose Estimation)

**FS-Net: Fast Shape-based Network for Category-Level 6D Object Pose Estimation with Decoupled Rotation Mechanism**
**FS-Net：用于类别级别6D对象姿态估计的快速形状基础网络**

- Paper(Oral): https://arxiv.org/abs/2103.07054
- Code: https://github.com/DC1991/FS-Net

**GDR-Net: Geometry-Guided Direct Regression Network for Monocular 6D Object Pose Estimation**
**GDR-Net：用于单目6D目标位姿估计的几何引导直接回归网络**

- Paper: http://arxiv.org/abs/2102.12145
- code: https://git.io/GDR-Net

**FFB6D: A Full Flow Bidirectional Fusion Network for 6D Pose Estimation**
**FFB6D：用于6D姿态估计的全面流双向融合网络**

- Paper: https://arxiv.org/abs/2103.02242
- Code: https://github.com/ethnhe/FFB6D

<a name="Camera-Pose-Estimation"></a>

# 相机姿态估计

**Back to the Feature: Learning Robust Camera Localization from Pixels to Pose**
**回到特征：从像素到姿态学习鲁棒的相机定位**

- Paper: https://arxiv.org/abs/2103.09213
- Code: https://github.com/cvg/pixloc

<a name="Depth-Estimation"></a>

# 深度估计(Depth Estimation)

**S2R-DepthNet: Learning a Generalizable Depth-specific Structural Representation**
**S2R-DepthNet：学习一种具有深度特异性结构表示的通用深度表示**

- Paper(Oral): https://arxiv.org/abs/2104.00877
- Code: None

**Beyond Image to Depth: Improving Depth Prediction using Echoes**
**超越图像到深度：利用回声改善深度预测**

- Homepage: https://krantiparida.github.io/projects/bimgdepth.html
- Paper: https://arxiv.org/abs/2103.08468
- Code: https://github.com/krantiparida/beyond-image-to-depth

**S3: Learnable Sparse Signal Superdensity for Guided Depth Estimation**
**S3：可学习稀疏信号超密度用于引导深度估计**

- Paper: https://arxiv.org/abs/2103.02396
- Code: None

**Depth from Camera Motion and Object Detection**
**相机运动和物体检测的深度**

- Paper: https://arxiv.org/abs/2103.01468
- Code: https://github.com/griffbr/ODMD
- Dataset: https://github.com/griffbr/ODMD

<a name="Stereo-Matching"></a>

# 立体匹配(Stereo Matching)

**A Decomposition Model for Stereo Matching**
**立体匹配的分解模型**

- Paper: https://arxiv.org/abs/2104.07516
- Code: None

<a name="Flow-Estimation"></a>

# 光流估计(Flow Estimation)

**Self-Supervised Multi-Frame Monocular Scene Flow**
**基于自监督的多帧单目场景流**

- Paper: https://arxiv.org/abs/2105.02216
- Code: https://github.com/visinf/multi-mono-sf

**RAFT-3D: Scene Flow using Rigid-Motion Embeddings**
**RAFT-3D：使用刚性运动嵌入的场景流**

- Paper: https://arxiv.org/abs/2012.00726v1
- Code: None

**Learning Optical Flow From Still Images**
**从静止图像中学习光学流动**

- Homepage: https://mattpoggi.github.io/projects/cvpr2021aleotti/

- Paper: https://mattpoggi.github.io/assets/papers/aleotti2021cvpr.pdf
- Code: https://github.com/mattpoggi/depthstillation

**FESTA: Flow Estimation via Spatial-Temporal Attention for Scene Point Clouds**
**FESTA：通过空间-时间注意力为场景点云流量估计**

- Paper: https://arxiv.org/abs/2104.00798
- Code: None

<a name="Lane-Detection"></a>

# 车道线检测(Lane Detection)

**Focus on Local: Detecting Lane Marker from Bottom Up via Key Point**
**关注本地：通过角点从底部向上检测车道标记**

- Paper: https://arxiv.org/abs/2105.13680
- Code: None

**Keep your Eyes on the Lane: Real-time Attention-guided Lane Detection**
**让你的眼睛盯着车道：实时引导的车道检测**

- Paper: https://arxiv.org/abs/2010.12035
- Code: https://github.com/lucastabelini/LaneATT 

<a name="Trajectory-Prediction"></a>

# 轨迹预测(Trajectory Prediction)

**Divide-and-Conquer for Lane-Aware Diverse Trajectory Prediction**
**划分与 conquer 用于 lane-aware 多样性轨迹预测**

- Paper(Oral): https://arxiv.org/abs/2104.08277
- Code: None

<a name="Crowd-Counting"></a>

# 人群计数(Crowd Counting)

**Detection, Tracking, and Counting Meets Drones in Crowds: A Benchmark**
**检测、跟踪和计数在人群中与无人机相遇：一款基准测试**

- Paper: https://arxiv.org/abs/2105.02440

- Code: https://github.com/VisDrone/DroneCrowd

- Dataset: https://github.com/VisDrone/DroneCrowd

<a name="AE"></a>

# 对抗样本(Adversarial Examples)

**Enhancing the Transferability of Adversarial Attacks through Variance Tuning**
**通过变分调整增强对抗攻击的转移能力**

- Paper: https://arxiv.org/abs/2103.15571
- Code: https://github.com/JHL-HUST/VT

**LiBRe: A Practical Bayesian Approach to Adversarial Detection**
**LiBRe：一种实用的贝叶斯方法用于对抗性检测**

- Paper: https://arxiv.org/abs/2103.14835
- Code: None

**Natural Adversarial Examples**
**自然对抗样本**

- Paper: https://arxiv.org/abs/1907.07174
- Code: https://github.com/hendrycks/natural-adv-examples

<a name="Image-Retrieval"></a>

# 图像检索(Image Retrieval)

**StyleMeUp: Towards Style-Agnostic Sketch-Based Image Retrieval**
**风格迁移：指向风格无关的基于图的图像检索**

- Paper: https://arxiv.org/abs/2103.15706
- COde: None

**QAIR: Practical Query-efficient Black-Box Attacks for Image Retrieval**
**QAIR：用于图像检索的实际查询效率高的黑色盒攻击**

- Paper: https://arxiv.org/abs/2103.02927
- Code: None

<a name="Video-Retrieval"></a>

# 视频检索(Video Retrieval)

**On Semantic Similarity in Video Retrieval**
**在视频检索中的语义相似性**

- Paper: https://arxiv.org/abs/2103.10095

- Homepage: https://mwray.github.io/SSVR/
- Code: https://github.com/mwray/Semantic-Video-Retrieval

<a name="Cross-modal-Retrieval"></a>

# 跨模态检索(Cross-modal Retrieval)

**Cross-Modal Center Loss for 3D Cross-Modal Retrieval**
**跨模态中心损失用于3D跨模态检索**

- Paper: https://arxiv.org/abs/2008.03561
- Code: https://github.com/LongLong-Jing/Cross-Modal-Center-Loss 

**Thinking Fast and Slow: Efficient Text-to-Visual Retrieval with Transformers**
**思考，快与慢：使用变压器实现高效的文本-视觉检索**

- Paper: https://arxiv.org/abs/2103.16553
- Code: None

**Revamping cross-modal recipe retrieval with hierarchical Transformers and self-supervised learning**
**改进跨模态推荐检索策略，采用层次化Transformer和自监督学习**

- Paper: https://www.amazon.science/publications/revamping-cross-modal-recipe-retrieval-with-hierarchical-transformers-and-self-supervised-learning

- Code: https://github.com/amzn/image-to-recipe-transformers

<a name="Zero-Shot-Learning"></a>

#  Zero-Shot Learning

**Counterfactual Zero-Shot and Open-Set Visual Recognition**
**零样本和开放集可视识别**

- Paper: https://arxiv.org/abs/2103.00887
- Code: https://github.com/yue-zhongqi/gcm-cf

<a name="Federated-Learning"></a>

# 联邦学习(Federated Learning)

**FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space**
**联邦DG：通过连续频率空间上的周期性学习在医学图像分割中实现联邦领域泛化**

- Paper: https://arxiv.org/abs/2103.06030
- Code: https://github.com/liuquande/FedDG-ELCFS

<a name="Video-Frame-Interpolation"></a>

# 视频插帧(Video Frame Interpolation)

**CDFI: Compression-Driven Network Design for Frame Interpolation**
**CDFI：用于帧插值的压缩驱动网络设计**

- Paper: None
- Code: https://github.com/tding1/CDFI

**FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation**
**FLAVR：用于快速帧插值的流形可逆视频表示**

- Homepage: https://tarun005.github.io/FLAVR/

- Paper: https://arxiv.org/abs/2012.08512
- Code: https://github.com/tarun005/FLAVR

<a name="Visual-Reasoning"></a>

# 视觉推理(Visual Reasoning)

**Transformation Driven Visual Reasoning**
**变形驱动的视觉推理**

- homepage: https://hongxin2019.github.io/TVR/
- Paper: https://arxiv.org/abs/2011.13160
- Code: https://github.com/hughplay/TVR

<a name="Image-Synthesis"></a>

# 图像合成(Image Synthesis)

**GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields**
**斑马：将场景表示为构图生成神经特征场**

- Homepage: https://m-niemeyer.github.io/project-pages/giraffe/index.html
- Paper(Oral): https://arxiv.org/abs/2011.12100

- Code: https://github.com/autonomousvision/giraffe

- Demo: http://www.youtube.com/watch?v=fIaDXC-qRSg&vq=hd1080&autoplay=1

**Taming Transformers for High-Resolution Image Synthesis**
**驯服变分器以实现高分辨率图像合成**

- Homepage: https://compvis.github.io/taming-transformers/
- Paper(Oral): https://arxiv.org/abs/2012.09841
- Code: https://github.com/CompVis/taming-transformers

<a name="Visual-Synthesis"></a>

# 视图合成(View Synthesis)

**Stereo Radiance Fields (SRF): Learning View Synthesis for Sparse Views of Novel Scenes**
**立体辐射场（SRF）：学习场景稀疏视图的立体合成**

- Homepage: https://virtualhumans.mpi-inf.mpg.de/srf/
- Paper: https://arxiv.org/abs/2104.06935

**Self-Supervised Visibility Learning for Novel View Synthesis**
**自助监督可见性学习用于新颖视图合成**

- Paper: https://arxiv.org/abs/2103.15407
- Code: None

**NeX: Real-time View Synthesis with Neural Basis Expansion**
**NeX: 实时视图扩展与神经基础扩展**

- Homepage: https://nex-mpi.github.io/
- Paper(Oral): https://arxiv.org/abs/2103.05606

<a name="Style-Transfer"></a>

# 风格迁移(Style Transfer)

**Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality Artistic Style Transfer**
**起草和修订：用于快速高质量艺术风格转移的拉普拉斯金字塔网络**

- Paper: https://arxiv.org/abs/2104.05376
- Code: https://github.com/PaddlePaddle/PaddleGAN/

<a name="Layout-Generation"></a>

# 布局生成(Layout Generation)

**LayoutTransformer: Scene Layout Generation With Conceptual and Spatial Diversity**
**LayoutTransformer：具有概念性和空间多样性的场景布局生成**

- Paper: None
- Code: None

**Variational Transformer Networks for Layout Generation**
**变分自注意力网络用于布局生成**

- Paper: https://arxiv.org/abs/2104.02416
- Code: None

<a name="Domain-Generalization"></a>

# Domain Generalization

**Generalization on Unseen Domains via Inference-time Label-Preserving Target Projections**
**通过推理时标签保持目标投影进行未知领域的泛化**

- Paper(Oral): https://openaccess.thecvf.com/content/CVPR2021/papers/Pandey_Generalization_on_Unseen_Domains_via_Inference-Time_Label-Preserving_Target_Projections_CVPR_2021_paper.pdf
- Code: https://github.com/VSumanth99/InferenceTimeDG

**Generalizable Person Re-identification with Relevance-aware Mixture of Experts**
**具有相关性感知的专家混合的领域专家通用身份识别**

- Paper: https://arxiv.org/abs/2105.09156
- Code: None

**RobustNet: Improving Domain Generalization in Urban-Scene Segmentation via Instance Selective Whitening**
**RobustNet：通过实例选择性漂白在城市景观分割中提高领域泛化**

- Paper: https://arxiv.org/abs/2103.15597
- Code: https://github.com/shachoi/RobustNet

**Adaptive Methods for Real-World Domain Generalization**
**适应性方法用于领域外推**

- Paper: https://arxiv.org/abs/2103.15796
- Code: None

**FSDR: Frequency Space Domain Randomization for Domain Generalization**
**FSDR：频率空间域随机化，用于领域泛化**

- Paper: https://arxiv.org/abs/2103.02370
- Code: None

<a name="Domain-Adaptation"></a>

# Domain Adaptation

**Curriculum Graph Co-Teaching for Multi-Target Domain Adaptation**
**课程图联合教学多目标领域自适应**

- Paper: https://arxiv.org/abs/2104.00808
- Code: None

**Domain Consensus Clustering for Universal Domain Adaptation**
**域共识聚类用于通用领域自适应**

- Paper: http://reler.net/papers/guangrui_cvpr2021.pdf
- Code: https://github.com/Solacex/Domain-Consensus-Clustering 

<a name="Open-Set"></a>

# Open-Set

**Towards Open World Object Detection**
**面向开放世界目标检测**

- Paper(Oral): https://arxiv.org/abs/2103.02603
- Code: https://github.com/JosephKJ/OWOD

**Exemplar-Based Open-Set Panoptic Segmentation Network**
**基于样式的开放集像素分割网络**

- Homepage: https://cv.snu.ac.kr/research/EOPSN/
- Paper: https://arxiv.org/abs/2105.08336
- Code: https://github.com/jd730/EOPSN

**Learning Placeholders for Open-Set Recognition**
**学习开放集识别的上下文填充**

- Paper(Oral): https://arxiv.org/abs/2103.15086
- Code: None

<a name="Adversarial-Attack"></a>

# Adversarial Attack

**IoU Attack: Towards Temporally Coherent Black-Box Adversarial Attack for Visual Object Tracking**
**IoU Attack：朝着时间一致的黑色盒对抗攻击，用于视觉对象跟踪**

- Paper: https://arxiv.org/abs/2103.14938
- Code: https://github.com/VISION-SJTU/IoUattack

<a name="HOI"></a>

# "人-物"交互(HOI)检测

**HOTR: End-to-End Human-Object Interaction Detection with Transformers**
**HOTR：端到端人类目标交互检测与变压器**

- Paper: https://arxiv.org/abs/2104.13682
- Code: None

**Query-Based Pairwise Human-Object Interaction Detection with Image-Wide Contextual Information**
**基于查询的成对人类与对象交互检测，具有图像宽上下文信息**

- Paper: https://arxiv.org/abs/2103.05399
- Code: https://github.com/hitachi-rd-cv/qpic

**Reformulating HOI Detection as Adaptive Set Prediction**
**重新表述HOI检测为自适应集预测**

- Paper: https://arxiv.org/abs/2103.05983
- Code: https://github.com/yoyomimi/AS-Net

**Detecting Human-Object Interaction via Fabricated Compositional Learning**
**检测通过虚构的组合学习检测人机交互**

- Paper: https://arxiv.org/abs/2103.08214
- Code: https://github.com/zhihou7/FCL

**End-to-End Human Object Interaction Detection with HOI Transformer**
**端到端的人机对象交互检测与HOI Transformer**

- Paper: https://arxiv.org/abs/2103.04503
- Code: https://github.com/bbepoch/HoiTransformer

<a name="Shadow-Removal"></a>

# 阴影去除(Shadow Removal)

**Auto-Exposure Fusion for Single-Image Shadow Removal**
**自动曝光融合单张图像去除影子**

- Paper: https://arxiv.org/abs/2103.01255
- Code: https://github.com/tsingqguo/exposure-fusion-shadow-removal

<a name="Virtual-Try-On"></a>

# 虚拟换衣(Virtual Try-On)

**Parser-Free Virtual Try-on via Distilling Appearance Flows**
**无需解析的虚拟试穿，通过提取外观流进行**

**基于外观流蒸馏的无需人体解析的虚拟换装**
**基于外观流蒸馏的无需人体解析的虚拟换装**

- Paper: https://arxiv.org/abs/2103.04559
- Code: https://github.com/geyuying/PF-AFN 

<a name="Label-Noise"></a>

# 标签噪声(Label Noise)

**A Second-Order Approach to Learning with Instance-Dependent Label Noise**
**一种基于实例依赖标签噪声的第二阶学习方法**

- Paper(Oral): https://arxiv.org/abs/2012.11854
- Code: https://github.com/UCSC-REAL/CAL

<a name="Video-Stabilization"></a>

# 视频稳像(Video Stabilization)

**Real-Time Selfie Video Stabilization**
**实时自拍视频稳定**

- Paper: https://openaccess.thecvf.com/content/CVPR2021/papers/Yu_Real-Time_Selfie_Video_Stabilization_CVPR_2021_paper.pdf

- Code: https://github.com/jiy173/selfievideostabilization

<a name="Datasets"></a>

# 数据集(Datasets)

**Tracking Pedestrian Heads in Dense Crowd**
**在拥挤人群中追踪行人头部**

- Homepage: https://project.inria.fr/crowdscience/project/dense-crowd-head-tracking/
- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Sundararaman_Tracking_Pedestrian_Heads_in_Dense_Crowd_CVPR_2021_paper.html
- Code1: https://github.com/Sentient07/HeadHunter
- Code2: https://github.com/Sentient07/HeadHunter%E2%80%93T
- Dataset: https://project.inria.fr/crowdscience/project/dense-crowd-head-tracking/

**Part-aware Panoptic Segmentation**
**部分感知的全景分割**

- Paper: https://arxiv.org/abs/2106.06351
- Code: https://github.com/tue-mps/panoptic_parts
- Dataset: https://github.com/tue-mps/panoptic_parts

**Learning High Fidelity Depths of Dressed Humans by Watching Social Media Dance Videos**
**观看社交媒体舞蹈视频来学习人类的低频深度**

- Homepage: https://www.yasamin.page/hdnet_tiktok

- Paper(Oral): https://arxiv.org/abs/2103.03319

- Code: https://github.com/yasaminjafarian/HDNet_TikTok

- Dataset: https://www.yasamin.page/hdnet_tiktok#h.jr9ifesshn7v

**High-Resolution Photorealistic Image Translation in Real-Time: A Laplacian Pyramid Translation Network**
**实时高分辨率像素化图像转换：拉普拉斯金字塔转换网络**

- Paper: https://arxiv.org/abs/2105.09188
- Code: https://github.com/csjliang/LPTN
- Dataset: https://github.com/csjliang/LPTN

**Detection, Tracking, and Counting Meets Drones in Crowds: A Benchmark**
**检测、跟踪和计数在人群中与无人机相遇：一款基准测试**

- Paper: https://arxiv.org/abs/2105.02440

- Code: https://github.com/VisDrone/DroneCrowd

- Dataset: https://github.com/VisDrone/DroneCrowd

**Towards Good Practices for Efficiently Annotating Large-Scale Image Classification Datasets**
**面向高效标注大规模图像分类数据集的好实践**

- Homepage: https://fidler-lab.github.io/efficient-annotation-cookbook/
- Paper(Oral): https://arxiv.org/abs/2104.12690
- Code: https://github.com/fidler-lab/efficient-annotation-cookbook

论文下载链接：

**ViP-DeepLab: Learning Visual Perception with Depth-aware Video Panoptic Segmentation**
**ViP-DeepLab：使用深度感知的视频全景分割学习视觉感知**

- Paper: https://arxiv.org/abs/2012.05258
- Code: https://github.com/joe-siyuan-qiao/ViP-DeepLab
- Dataset: https://github.com/joe-siyuan-qiao/ViP-DeepLab

**Learning To Count Everything**
**学习计数一切**

- Paper: https://arxiv.org/abs/2104.08391
- Code: https://github.com/cvlab-stonybrook/LearningToCountEverything
- Dataset: https://github.com/cvlab-stonybrook/LearningToCountEverything

**Semantic Image Matting**
**语义图像匹配**

- Paper: https://arxiv.org/abs/2104.08201
- Code: https://github.com/nowsyn/SIM
- Dataset: https://github.com/nowsyn/SIM

**Towards Fast and Accurate Real-World Depth Super-Resolution: Benchmark Dataset and Baseline**
**朝着快速且准确地实现真实世界深度超分辨率：基准数据集和基线**

- Homepage: http://mepro.bjtu.edu.cn/resource.html
- Paper: https://arxiv.org/abs/2104.06174
- Code: None

**Visual Semantic Role Labeling for Video Understanding**
**视频理解中的视觉语义角色标注**

- Homepage: https://vidsitu.org/

- Paper: https://arxiv.org/abs/2104.00990
- Code: https://github.com/TheShadow29/VidSitu
- Dataset: https://github.com/TheShadow29/VidSitu

**VSPW: A Large-scale Dataset for Video Scene Parsing in the Wild**
**VSPW：用于野外视频场景解析的大规模数据集**

- Homepage: https://www.vspwdataset.com/
- Paper: https://www.vspwdataset.com/CVPR2021__miao.pdf
- GitHub: https://github.com/sssdddwww2/vspw_dataset_download

**Sewer-ML: A Multi-Label Sewer Defect Classification Dataset and Benchmark**
**下水道ML：一个多标签下水道缺陷分类数据集和基准**

- Homepage: https://vap.aau.dk/sewer-ml/
- Paper: https://arxiv.org/abs/2103.10619

**Sewer-ML: A Multi-Label Sewer Defect Classification Dataset and Benchmark**
**下水道-ML：一个多标签下水道缺陷分类数据集和基准**

- Homepage: https://vap.aau.dk/sewer-ml/

- Paper: https://arxiv.org/abs/2103.10895

**Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food**
**营养5k：迈向自动理解通用食品的营养**

- Paper: https://arxiv.org/abs/2103.03375
- Dataset: None

 **Towards Semantic Segmentation of Urban-Scale 3D Point Clouds: A Dataset, Benchmarks and Challenges**

- Homepage: https://github.com/QingyongHu/SensatUrban
- Paper: http://arxiv.org/abs/2009.03137
- Code: https://github.com/QingyongHu/SensatUrban
- Dataset: https://github.com/QingyongHu/SensatUrban

**When Age-Invariant Face Recognition Meets Face Age Synthesis: A Multi-Task Learning Framework**
**当不变形人脸识别技术与人脸年龄合成相遇时：一个多任务学习框架**

- Paper(Oral): https://arxiv.org/abs/2103.01520
- Code: https://github.com/Hzzone/MTLFace
- Dataset: https://github.com/Hzzone/MTLFace

**Depth from Camera Motion and Object Detection**
**相机运动和目标检测的深度**

- Paper: https://arxiv.org/abs/2103.01468
- Code: https://github.com/griffbr/ODMD
- Dataset: https://github.com/griffbr/ODMD

**There is More than Meets the Eye: Self-Supervised Multi-Object Detection and Tracking with Sound by Distilling Multimodal Knowledge**
**超过眼界的多目标检测和跟踪：通过蒸馏多模态知识来实现**

- Homepage: http://rl.uni-freiburg.de/research/multimodal-distill
- Paper: https://arxiv.org/abs/2103.01353
- Code: http://rl.uni-freiburg.de/research/multimodal-distill

**Scan2Cap: Context-aware Dense Captioning in RGB-D Scans**
**扫描2CAP：在RGB-D扫描中实现上下文感知的稠密标注**

- Paper: https://arxiv.org/abs/2012.02206
- Code: https://github.com/daveredrum/Scan2Cap

- Dataset: https://github.com/daveredrum/ScanRefer

**There is More than Meets the Eye: Self-Supervised Multi-Object Detection and Tracking with Sound by Distilling Multimodal Knowledge**
**超过眼见的：通过蒸馏多模态知识来实现自监督多目标检测和跟踪**

- Paper: https://arxiv.org/abs/2103.01353
- Code: http://rl.uni-freiburg.de/research/multimodal-distill
- Dataset: http://rl.uni-freiburg.de/research/multimodal-distill

<a name="Others"></a>

# 其他(Others)

**Fast and Accurate Model Scaling**
**快速且准确模型缩放**

- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Dollar_Fast_and_Accurate_Model_Scaling_CVPR_2021_paper.html

- Code: https://github.com/facebookresearch/pycls

**Learning High Fidelity Depths of Dressed Humans by Watching Social Media Dance Videos**
**观看社交媒体舞蹈视频学习高等保真度的人类的深度**

- Homepage: https://www.yasamin.page/hdnet_tiktok

- Paper(Oral): https://arxiv.org/abs/2103.03319

- Code: https://github.com/yasaminjafarian/HDNet_TikTok

- Dataset: https://www.yasamin.page/hdnet_tiktok#h.jr9ifesshn7v

**Omnimatte: Associating Objects and Their Effects in Video**
**Omnimatte：在视频中将对象及其效果关联起来**

- Homepage: https://omnimatte.github.io/

- Paper(Oral): https://arxiv.org/abs/2105.06993
- Code: https://omnimatte.github.io/#code

**Towards Good Practices for Efficiently Annotating Large-Scale Image Classification Datasets**
**走向高效注记大规模图像分类数据集的良好实践**

- Homepage: https://fidler-lab.github.io/efficient-annotation-cookbook/
- Paper(Oral): https://arxiv.org/abs/2104.12690
- Code: https://github.com/fidler-lab/efficient-annotation-cookbook

**Motion Representations for Articulated Animation**
**关节动画的运动表示**

- Paper: https://arxiv.org/abs/2104.11280
- Code: https://github.com/snap-research/articulated-animation

**Deep Lucas-Kanade Homography for Multimodal Image Alignment**
**深度卢卡斯-坎德多模式图像对齐**

- Paper: https://arxiv.org/abs/2104.11693
- Code: https://github.com/placeforyiming/CVPR21-Deep-Lucas-Kanade-Homography

**Skip-Convolutions for Efficient Video Processing**
**高效的视频处理**

- Paper: https://arxiv.org/abs/2104.11487
- Code: None

**KeypointDeformer: Unsupervised 3D Keypoint Discovery for Shape Control**
**键点生成器：用于形状控制的未经监督的3D关键点发现**

- Homepage: http://tomasjakab.github.io/KeypointDeformer

- Paper(Oral): https://arxiv.org/abs/2104.11224
- Code: https://github.com/tomasjakab/keypoint_deformer/

**Learning To Count Everything**
**学习计数一切**

- Paper: https://arxiv.org/abs/2104.08391
- Code: https://github.com/cvlab-stonybrook/LearningToCountEverything
- Dataset: https://github.com/cvlab-stonybrook/LearningToCountEverything

**SOLD2: Self-supervised Occlusion-aware Line Description and Detection**
**SOLD2：自监督遮挡感知线描述和检测**

- Paper(Oral): https://arxiv.org/abs/2104.03362
- Code: https://github.com/cvg/SOLD2

**Learning Probabilistic Ordinal Embeddings for Uncertainty-Aware Regression**
**学习概率图模型中的序维嵌入以实现不确定性回归**

- Homepage: https://li-wanhua.github.io/POEs/
- Paper:  https://arxiv.org/abs/2103.13629
- Code: https://github.com/Li-Wanhua/POEs

**LEAP: Learning Articulated Occupancy of People**
**跳跃：掌握人们的活动能力**

- Paper: https://arxiv.org/abs/2104.06849
- Code: None

**Visual Semantic Role Labeling for Video Understanding**
**视频理解中的视觉语义角色标注**

- Homepage: https://vidsitu.org/

- Paper: https://arxiv.org/abs/2104.00990
- Code: https://github.com/TheShadow29/VidSitu
- Dataset: https://github.com/TheShadow29/VidSitu

**UAV-Human: A Large Benchmark for Human Behavior Understanding with Unmanned Aerial Vehicles**
**UAV-Human: 无人机对人类行为的巨大基准**

- Paper: https://arxiv.org/abs/2104.00946
- Code: https://github.com/SUTDCV/UAV-Human 

**Video Prediction Recalling Long-term Motion Context via Memory Alignment Learning**
**视频预测通过内存对齐学习回溯长期运动上下文**

- Paper(Oral): https://arxiv.org/abs/2104.00924
- Code: None

**Fully Understanding Generic Objects: Modeling, Segmentation, and Reconstruction**
**全面理解泛型对象：建模、分割和重建**

- Paper: https://arxiv.org/abs/2104.00858
- Code: None

**Towards High Fidelity Face Relighting with Realistic Shadows**
**走向高保真人脸重新定位与真实阴影**

- Paper: https://arxiv.org/abs/2104.00825
- Code: None

**BRepNet: A topological message passing system for solid models**
**BRepNet：用于固有模型的拓扑消息传递系统**

- Paper(Oral): https://arxiv.org/abs/2104.00706
- Code: None

**Visually Informed Binaural Audio Generation without Binaural Audios**
**视觉引导的单耳音频生成-无双耳音频**

- Homepage: https://sheldontsui.github.io/projects/PseudoBinaural
- Paper: None

- GitHub: https://github.com/SheldonTsui/PseudoBinaural_CVPR2021
- Demo: https://www.youtube.com/watch?v=r-uC2MyAWQc

**Exploring intermediate representation for monocular vehicle pose estimation**
**探索中间表示对于单目车辆姿态估计**

- Paper: None
- Code: https://github.com/Nicholasli1995/EgoNet

**Tuning IR-cut Filter for Illumination-aware Spectral Reconstruction from RGB**
**调整IR-cut滤波器，用于从RGB中实现光感频谱重建**

- Paper(Oral): https://arxiv.org/abs/2103.14708
- Code: None

**Invertible Image Signal Processing**
**不可逆图像信号处理**

- Paper: https://arxiv.org/abs/2103.15061
- Code: https://github.com/yzxing87/Invertible-ISP

**Video Rescaling Networks with Joint Optimization Strategies for Downscaling and Upscaling**
**视频缩放网络与联合优化策略**

- Paper: https://arxiv.org/abs/2103.14858
- Code: None

**SceneGraphFusion: Incremental 3D Scene Graph Prediction from RGB-D Sequences**
**场景图生成器：从RGB-D序列的增量式3D场景图预测**

- Paper: https://arxiv.org/abs/2103.14898
- Code: None

**Embedding Transfer with Label Relaxation for Improved Metric Learning**
**嵌入转移与标签放松：改进度量学习**

- Paper: https://arxiv.org/abs/2103.14908
- Code: None

**Picasso: A CUDA-based Library for Deep Learning over 3D Meshes**
**皮卡罗：一个基于CUDA的深度学习库，适用于3D网格**

- Paper: https://arxiv.org/abs/2103.15076 
- Code: https://github.com/hlei-ziyan/Picasso

**Meta-Mining Discriminative Samples for Kinship Verification**
**元挖掘用于亲属验证的判别样本**

- Paper: https://arxiv.org/abs/2103.15108
- Code: None

**Cloud2Curve: Generation and Vectorization of Parametric Sketches**
**Cloud2Curve：参数草图的生成和向量化**

- Paper: https://arxiv.org/abs/2103.15536
- Code: None

**TrafficQA: A Question Answering Benchmark and an Efficient Network for Video Reasoning over Traffic Events**
**交通QA：用于交通事件视频推理的基于问题和答案的基准和有效网络**

- Paper: https://arxiv.org/abs/2103.15538
- Code: https://github.com/SUTDCV/SUTD-TrafficQA

**Abstract Spatial-Temporal Reasoning via Probabilistic Abduction and Execution**
**抽象空间-时间推理通过概率归纳和执行**

- Homepage: http://wellyzhang.github.io/project/prae.html

- Paper: https://arxiv.org/abs/2103.14230
- Code: None

**ACRE: Abstract Causal REasoning Beyond Covariation**
**ACRE：超越协变的抽象因果推理**

- Homepage: http://wellyzhang.github.io/project/acre.html

- Paper: https://arxiv.org/abs/2103.14232
- Code: None

**Confluent Vessel Trees with Accurate Bifurcations**
**精确的双分支容器树**

- Paper: https://arxiv.org/abs/2103.14268
- Code: None

**Few-Shot Human Motion Transfer by Personalized Geometry and Texture Modeling**
**少样本人体运动迁移通过个性化几何和纹理建模**

- Paper: https://arxiv.org/abs/2103.14338
- Code: https://github.com/HuangZhiChao95/FewShotMotionTransfer

**Neural Parts: Learning Expressive 3D Shape Abstractions with Invertible Neural Networks**
**神经元组件：使用可逆神经网络学习表征3D形状**

- Homepage: https://paschalidoud.github.io/neural_parts
- Paper: None 
- Code: https://github.com/paschalidoud/neural_parts 

**Knowledge Evolution in Neural Networks**
**神经网络中的知识进化**

- Paper(Oral): https://arxiv.org/abs/2103.05152
- Code: https://github.com/ahmdtaha/knowledge_evolution

**Multi-institutional Collaborations for Improving Deep Learning-based Magnetic Resonance Image Reconstruction Using Federated Learning**
**基于联邦学习的多机构协同改进深度学习磁共振图像重建**

- Paper: https://arxiv.org/abs/2103.02148
- Code: https://github.com/guopengf/FLMRCM

**SGP: Self-supervised Geometric Perception**
**SGP：自监督几何感知**

- Oral

- Paper: https://arxiv.org/abs/2103.03114
- Code: https://github.com/theNded/SGP

**Multi-institutional Collaborations for Improving Deep Learning-based Magnetic Resonance Image Reconstruction Using Federated Learning**
**基于联邦学习的改进深度学习磁共振图像重建的多机构合作**

- Paper: https://arxiv.org/abs/2103.02148
- Code: https://github.com/guopengf/FLMRCM

**Diffusion Probabilistic Models for 3D Point Cloud Generation**
**3D 点云生成扩散概率模型**

- Paper: https://arxiv.org/abs/2103.01458
- Code: https://github.com/luost26/diffusion-point-cloud

**Scan2Cap: Context-aware Dense Captioning in RGB-D Scans**
**扫描2捕捉：RGB-D 扫描的上下文感知稠密标题**

- Paper: https://arxiv.org/abs/2012.02206
- Code: https://github.com/daveredrum/Scan2Cap

- Dataset: https://github.com/daveredrum/ScanRefer

**There is More than Meets the Eye: Self-Supervised Multi-Object Detection and Tracking with Sound by Distilling Multimodal Knowledge**
**超过眼界的声音：通过蒸馏多模态知识来实现自监督多目标检测和跟踪**

- Paper: https://arxiv.org/abs/2103.01353
- Code: http://rl.uni-freiburg.de/research/multimodal-distill

- Dataset: http://rl.uni-freiburg.de/research/multimodal-distill

<a name="TO-DO"></a>

# 待添加(TODO)

- [重磅！腾讯优图20篇论文入选CVPR 2021](https://mp.weixin.qq.com/s/McAtOVh0osWZ3uppEoHC8A)
- [MePro团队三篇论文被CVPR 2021接收](https://mp.weixin.qq.com/s/GD5Zb6u_MQ8GZIAGeCGo3Q)

<a name="Not-Sure"></a>

# 不确定中没中(Not Sure)

**CT Film Recovery via Disentangling Geometric Deformation and Photometric Degradation: Simulated Datasets and Deep Models**
**通过解分离几何变形和光度退化进行CT film恢复：模拟数据集和深度模型**

- Paper: none
- Code: https://github.com/transcendentsky/Film-Recovery

**Toward Explainable Reflection Removal with Distilling and Model Uncertainty**
**解释性反射移除：蒸馏与模型不确定性的方法**

- Paper: none
- Code: https://github.com/ytpeng-aimlab/CVPR-2021-Toward-Explainable-Reflection-Removal-with-Distilling-and-Model-Uncertainty

**DeepOIS: Gyroscope-Guided Deep Optical Image Stabilizer Compensation**
**深度OIS：陀螺仪引导的深度光学图像稳定器补偿**

- Paper: none
- Code: https://github.com/lhaippp/DeepOIS

**Exploring Adversarial Fake Images on Face Manifold**
**探索面部纹理中的对抗性虚假图像**

- Paper: none
- Code: https://github.com/ldz666666/Style-atk

**Uncertainty-Aware Semi-Supervised Crowd Counting via Consistency-Regularized Surrogate Task**
**不确定性感知的半监督人群计数通过一致性正则化代理任务**

- Paper: none
- Code: https://github.com/yandamengdanai/Uncertainty-Aware-Semi-Supervised-Crowd-Counting-via-Consistency-Regularized-Surrogate-Task

**Temporal Contrastive Graph for Self-supervised Video Representation Learning**
**时间对比图卷积神经网络用于自监督视频表示学习**

- Paper: none
- Code: https://github.com/YangLiu9208/TCG

**Boosting Monocular Depth Estimation Models to High-Resolution via Context-Aware Patching**
**提升单目深度估计模型，通过上下文感知补全**

- Paper: none
- Code: https://github.com/ouranonymouscvpr/cvpr2021_ouranonymouscvpr

**Fast and Memory-Efficient Compact Bilinear Pooling**
**快速且内存高效的紧凑双线性池化**

- Paper: none
- Code: https://github.com/cvpr2021kp2/cvpr2021kp2

**Identification of Empty Shelves in Supermarkets using Domain-inspired Features with Structural Support Vector Machine**
**利用结构支持向量机在超市中识别空置货架**

- Paper: none
- Code: https://github.com/gapDetection/cvpr2021

 **Estimating A Child's Growth Potential From Cephalometric X-Ray Image via Morphology-Aware Interactive Keypoint Estimation** 

- Paper: none
- Code: https://github.com/interactivekeypoint2020/Morph

https://github.com/ShaoQiangShen/CVPR2021

https://github.com/gillesflash/CVPR2021

https://github.com/anonymous-submission1991/BaLeNAS

https://github.com/cvpr2021dcb/cvpr2021dcb

https://github.com/anonymousauthorCV/CVPR2021_PaperID_8578

https://github.com/AldrichZeng/FreqPrune

https://github.com/Anonymous-AdvCAM/Anonymous-AdvCAM

https://github.com/ddfss/datadrive-fss

