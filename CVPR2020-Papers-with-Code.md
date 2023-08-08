# CVPR2020-Code

[CVPR 2020](https://openaccess.thecvf.com/CVPR2020) 论文开源项目合集，同时欢迎各位大佬提交issue，分享CVPR 2020开源项目

**【推荐阅读】**
**【推荐阅读】**

- [CVPR 2020 virtual](http://cvpr20.com/)
- ECCV 2020 论文开源项目合集来了：https://github.com/amusi/ECCV2020-Code

- 关于往年CV顶会论文（如ECCV 2020、CVPR 2019、ICCV 2019）以及其他优质CV论文和大盘点，详见： https://github.com/amusi/daily-paper-computer-vision

**【CVPR 2020 论文开源目录】**
**【CVPR 2020 论文开源目录】**

- [CNN](#CNN)
- [图像分类](#Image-Classification)
- [视频分类](#Video-Classification)
- [目标检测](#Object-Detection)
- [3D目标检测](#3D-Object-Detection)
- [视频目标检测](#Video-Object-Detection)
- [目标跟踪](#Object-Tracking)
- [语义分割](#Semantic-Segmentation)
- [实例分割](#Instance-Segmentation)
- [全景分割](#Panoptic-Segmentation)
- [视频目标分割](#VOS)
- [超像素分割](#Superpixel)
- [交互式图像分割](#IIS)
- [NAS](#NAS)
- [GAN](#GAN)
- [Re-ID](#Re-ID)
- [3D点云（分类/分割/配准/跟踪等）](#3D-PointCloud)
- [人脸（识别/检测/重建等）](#Face)
- [人体姿态估计(2D/3D)](#Human-Pose-Estimation)
- [人体解析](#Human-Parsing)
- [场景文本检测](#Scene-Text-Detection)
- [场景文本识别](#Scene-Text-Recognition)
- [特征(点)检测和描述](#Feature)
- [超分辨率](#Super-Resolution)
- [模型压缩/剪枝](#Model-Compression)
- [视频理解/行为识别](#Action-Recognition)
- [人群计数](#Crowd-Counting)
- [深度估计](#Depth-Estimation)
- [6D目标姿态估计](#6DOF)
- [手势估计](#Hand-Pose)
- [显著性检测](#Saliency)
- [去噪](#Denoising)
- [去雨](#Deraining)
- [去模糊](#Deblurring)
- [去雾](#Dehazing)
- [特征点检测与描述](#Feature)
- [视觉问答(VQA)](#VQA)
- [视频问答(VideoQA)](#VideoQA)
- [视觉语言导航](#VLN)
- [视频压缩](#Video-Compression)
- [视频插帧](#Video-Frame-Interpolation)
- [风格迁移](#Style-Transfer)
- [车道线检测](#Lane-Detection)
- ["人-物"交互(HOI)检测](#HOI)
- [轨迹预测](#TP)
- [运动预测](#Motion-Predication)
- [光流估计](#OF)
- [图像检索](#IR)
- [虚拟试衣](#Virtual-Try-On)
- [HDR](#HDR)
- [对抗样本](#AE)
- [三维重建](#3D-Reconstructing)
- [深度补全](#DC)
- [语义场景补全](#SSC)
- [图像/视频描述](#Captioning)
- [线框解析](#WP)
- [数据集](#Datasets)
- [其他](#Others)
- [不确定中没中](#Not-Sure)

<a name="CNN"></a>

# CNN

**Exploring Self-attention for Image Recognition**
**探索自注意力在图像识别中的应用**

- 论文：https://hszhao.github.io/papers/cvpr20_san.pdf

- 代码：https://github.com/hszhao/SAN

**Improving Convolutional Networks with Self-Calibrated Convolutions**
**改进卷积神经网络通过自定义卷积**

- 主页：https://mmcheng.net/scconv/

- 论文：http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf
- 代码：https://github.com/backseason/SCNet

**Rethinking Depthwise Separable Convolutions: How Intra-Kernel Correlations Lead to Improved MobileNets**
**重新思考深度可分离卷积：内卷如何影响 MobileNet 的性能**

- 论文：https://arxiv.org/abs/2003.13549
- 代码：https://github.com/zeiss-microscopy/BSConv

<a name="Image-Classification"></a>

# 图像分类

**Interpretable and Accurate Fine-grained Recognition via Region Grouping**
**通过区域分组解释性且准确地识别细粒度**

- 论文：https://arxiv.org/abs/2005.10411

- 代码：https://github.com/zxhuang1698/interpretability-by-parts

**Compositional Convolutional Neural Networks: A Deep Architecture with Innate Robustness to Partial Occlusion**
**组合卷积神经网络：一种具有自然鲁棒性以应对部分遮挡的深度架构**

- 论文：https://arxiv.org/abs/2003.04490

- 代码：https://github.com/AdamKortylewski/CompositionalNets

**Spatially Attentive Output Layer for Image Classification**
**空间关注输出层用于图像分类**

- 论文：https://arxiv.org/abs/2004.07570 
- 代码（好像被原作者删除了）：https://github.com/ildoonet/spatially-attentive-output-layer 

<a name="Video-Classification"></a>

# 视频分类

**SmallBigNet: Integrating Core and Contextual Views for Video Classification**
**小BigNet：将核心和上下文视图整合起来进行视频分类**

- 论文：https://arxiv.org/abs/2006.14582
- 代码：https://github.com/xhl-video/SmallBigNet

<a name="Object-Detection"></a>

# 目标检测

**Overcoming Classifier Imbalance for Long-tail Object Detection with Balanced Group Softmax**
**平衡集归一化软max用于解决长尾目标检测中的类别不平衡问题**

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Overcoming_Classifier_Imbalance_for_Long-Tail_Object_Detection_With_Balanced_Group_CVPR_2020_paper.pdf
- 代码：https://github.com/FishYuLi/BalancedGroupSoftmax

**AugFPN: Improving Multi-scale Feature Learning for Object Detection**
**AugFPN：为对象检测改进多尺度特征学习**

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_AugFPN_Improving_Multi-Scale_Feature_Learning_for_Object_Detection_CVPR_2020_paper.pdf 
- 代码：https://github.com/Gus-Guo/AugFPN

**Noise-Aware Fully Webly Supervised Object Detection**
**噪声感知全监督目标检测**

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/html/Shen_Noise-Aware_Fully_Webly_Supervised_Object_Detection_CVPR_2020_paper.html
- 代码：https://github.com/shenyunhang/NA-fWebSOD/

**Learning a Unified Sample Weighting Network for Object Detection**
**学习一个用于目标检测的统一样本加权网络**

- 论文：https://arxiv.org/abs/2006.06568
- 代码：https://github.com/caiqi/sample-weighting-network

**D2Det: Towards High Quality Object Detection and Instance Segmentation**
**D2Det：朝着高质量的目标检测和实例分割**

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/papers/Cao_D2Det_Towards_High_Quality_Object_Detection_and_Instance_Segmentation_CVPR_2020_paper.pdf

- 代码：https://github.com/JialeCao001/D2Det

**Dynamic Refinement Network for Oriented and Densely Packed Object Detection**
**用于定向和密集包装对象检测的动态细化网络**

- 论文下载链接：https://arxiv.org/abs/2005.09973

- 代码和数据集：https://github.com/Anymake/DRN_CVPR2020

**Scale-Equalizing Pyramid Convolution for Object Detection**
**尺度等化金字塔卷积用于目标检测**

论文：https://arxiv.org/abs/2005.03101

代码：https://github.com/jshilong/SEPC

**Revisiting the Sibling Head in Object Detector**
**重新审视对象检测中的兄弟姐妹头**

- 论文：https://arxiv.org/abs/2003.07540

- 代码：https://github.com/Sense-X/TSD 

**Scale-equalizing Pyramid Convolution for Object Detection**
**尺度均衡的 Pyramid 卷积用于目标检测**

- 论文：暂无
- 代码：https://github.com/jshilong/SEPC 

**Detection in Crowded Scenes: One Proposal, Multiple Predictions**
**检测拥挤场景中的目标：一个提议，多个预测**

- 论文：https://arxiv.org/abs/2003.09163
- 代码：https://github.com/megvii-model/CrowdDetection

**Instance-aware, Context-focused, and Memory-efficient Weakly Supervised Object Detection**
**实例感知的弱监督目标检测**

- 论文：https://arxiv.org/abs/2004.04725
- 代码：https://github.com/NVlabs/wetectron

**Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection**
**通过自适应训练样本选择在锚点和无锚点检测之间搭建差距**

- 论文：https://arxiv.org/abs/1912.02424 
- 代码：https://github.com/sfzhang15/ATSS

**BiDet: An Efficient Binarized Object Detector**
**BiDet：一种高效的二进制化对象检测器**

- 论文：https://arxiv.org/abs/2003.03961 
- 代码：https://github.com/ZiweiWangTHU/BiDet

**Harmonizing Transferability and Discriminability for Adapting Object Detectors**
**调整对象检测器的迁移性和判别性**

- 论文：https://arxiv.org/abs/2003.06297
- 代码：https://github.com/chaoqichen/HTCN

**CentripetalNet: Pursuing High-quality Keypoint Pairs for Object Detection**
**CentripetalNet：追求高质量的关键点对物体检测**

- 论文：https://arxiv.org/abs/2003.09119
- 代码：https://github.com/KiveeDong/CentripetalNet

**Hit-Detector: Hierarchical Trinity Architecture Search for Object Detection**
**检测器：分层三体结构搜索物体检测**

- 论文：https://arxiv.org/abs/2003.11818
- 代码：https://github.com/ggjy/HitDet.pytorch

**EfficientDet: Scalable and Efficient Object Detection**
**EfficientDet: 高效且灵活的目标检测**

- 论文：https://arxiv.org/abs/1911.09070
- 代码：https://github.com/google/automl/tree/master/efficientdet 

<a name="3D-Object-Detection"></a>

# 3D目标检测

**SESS: Self-Ensembling Semi-Supervised 3D Object Detection**
**SESS：自编码的半监督3D对象检测**

- 论文： https://arxiv.org/abs/1912.11803

- 代码：https://github.com/Na-Z/sess

**Associate-3Ddet: Perceptual-to-Conceptual Association for 3D Point Cloud Object Detection**
**联合作者-3Ddet：3D 点云对象检测的感知-概念关联**

- 论文： https://arxiv.org/abs/2006.04356

- 代码：https://github.com/dleam/Associate-3Ddet

**What You See is What You Get: Exploiting Visibility for 3D Object Detection**
**所见即所得：利用可见性进行3D物体检测**

- 主页：https://www.cs.cmu.edu/~peiyunh/wysiwyg/

- 论文：https://arxiv.org/abs/1912.04986
- 代码：https://github.com/peiyunh/wysiwyg

**Learning Depth-Guided Convolutions for Monocular 3D Object Detection**
**学习深度引导的单目3D物体检测**

- 论文：https://arxiv.org/abs/1912.04799
- 代码：https://github.com/dingmyu/D4LCN

**Structure Aware Single-stage 3D Object Detection from Point Cloud**
**结构感知的单阶段3D物体检测**

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/html/He_Structure_Aware_Single-Stage_3D_Object_Detection_From_Point_Cloud_CVPR_2020_paper.html

- 代码：https://github.com/skyhehe123/SA-SSD

**IDA-3D: Instance-Depth-Aware 3D Object Detection from Stereo Vision for Autonomous Driving**
**IDA-3D：从立体视觉的自动驾驶车辆中的实例深度感知3D物体检测**

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/papers/Peng_IDA-3D_Instance-Depth-Aware_3D_Object_Detection_From_Stereo_Vision_for_Autonomous_CVPR_2020_paper.pdf

- 代码：https://github.com/swords123/IDA-3D

**Train in Germany, Test in The USA: Making 3D Object Detectors Generalize**
**在德国训练，在美国进行测试：让3D物体检测器变得通用**

- 论文：https://arxiv.org/abs/2005.08139

- 代码：https://github.com/cxy1997/3D_adapt_auto_driving

**MLCVNet: Multi-Level Context VoteNet for 3D Object Detection**
**MLCVNet：为3D对象检测提供多层上下文投票的Multi-Level Context VoteNet**

- 论文：https://arxiv.org/abs/2004.05679
- 代码：https://github.com/NUAAXQ/MLCVNet

**3DSSD: Point-based 3D Single Stage Object Detector**
**3DSSD：基于点的三维单阶段目标检测器**

- CVPR 2020 Oral

- 论文：https://arxiv.org/abs/2002.10187

- 代码：https://github.com/tomztyang/3DSSD

**Disp R-CNN: Stereo 3D Object Detection via Shape Prior Guided Instance Disparity Estimation**
**Disp R-CNN：通过形状优先引导实例差异估计实现立体3D目标检测**

- 论文：https://arxiv.org/abs/2004.03572

- 代码：https://github.com/zju3dv/disprcn

**End-to-End Pseudo-LiDAR for Image-Based 3D Object Detection**
**端到端伪激光雷达，用于基于图像的3D对象检测**

- 论文：https://arxiv.org/abs/2004.03080

- 代码：https://github.com/mileyan/pseudo-LiDAR_e2e

**DSGN: Deep Stereo Geometry Network for 3D Object Detection**
**DSGN：深度立体几何网络，用于3D目标检测**

- 论文：https://arxiv.org/abs/2001.03398
- 代码：https://github.com/chenyilun95/DSGN

**LiDAR-based Online 3D Video Object Detection with Graph-based Message Passing and Spatiotemporal Transformer Attention**
**LiDAR基于在线3D视频对象检测，基于图传递和空间-时间变压器注意力**

- 论文：https://arxiv.org/abs/2004.01389
- 代码：https://github.com/yinjunbo/3DVID

**PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection**
**PV-RCNN：3D 对象检测的点-体素特征集抽象**

- 论文：https://arxiv.org/abs/1912.13192

- 代码：https://github.com/sshaoshuai/PV-RCNN

**Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud**
**点云中的3D物体检测：基于图神经网络的点云**

- 论文：https://arxiv.org/abs/2003.01251 
- 代码：https://github.com/WeijingShi/Point-GNN 

<a name="Video-Object-Detection"></a>

# 视频目标检测

**Memory Enhanced Global-Local Aggregation for Video Object Detection**
**增强全局-局部聚合的内存视频对象检测**

论文：https://arxiv.org/abs/2003.12063

代码：https://github.com/Scalsol/mega.pytorch

<a name="Object-Tracking"></a>

# 目标跟踪

**SiamCAR: Siamese Fully Convolutional Classification and Regression for Visual Tracking**
**暹罗卡：泰国完全卷积分类与回归视觉跟踪**

- 论文：https://arxiv.org/abs/1911.07241
- 代码：https://github.com/ohhhyeahhh/SiamCAR

**D3S -- A Discriminative Single Shot Segmentation Tracker**
**D3S - 一种判别性单击分割跟踪器**

- 论文：https://arxiv.org/abs/1911.08862
- 代码：https://github.com/alanlukezic/d3s

**ROAM: Recurrently Optimizing Tracking Model**
**漫游：持续优化追踪模型**

- 论文：https://arxiv.org/abs/1907.12006

- 代码：https://github.com/skyoung/ROAM

**Siam R-CNN: Visual Tracking by Re-Detection**
**暹罗 R-CNN：通过重新检测的视觉跟踪**

- 主页：https://www.vision.rwth-aachen.de/page/siamrcnn
- 论文：https://arxiv.org/abs/1911.12836
- 论文2：https://www.vision.rwth-aachen.de/media/papers/192/siamrcnn.pdf
- 代码：https://github.com/VisualComputingInstitute/SiamR-CNN

**Cooling-Shrinking Attack: Blinding the Tracker with Imperceptible Noises**
**冷却收缩攻击：通过难以察觉的噪音使追踪器失明**

- 论文：https://arxiv.org/abs/2003.09595 
- 代码：https://github.com/MasterBin-IIAU/CSA 

**High-Performance Long-Term Tracking with Meta-Updater**
**高性能长期跟踪与元更新**

- 论文：https://arxiv.org/abs/2004.00305

- 代码：https://github.com/Daikenan/LTMU

**AutoTrack: Towards High-Performance Visual Tracking for UAV with Automatic Spatio-Temporal Regularization**
**AutoTrack：无人机高性能视觉跟踪的自动时空正则化**

- 论文：https://arxiv.org/abs/2003.12949

- 代码：https://github.com/vision4robotics/AutoTrack

**Probabilistic Regression for Visual Tracking**
**概率回归视觉跟踪**

- 论文：https://arxiv.org/abs/2003.12565
- 代码：https://github.com/visionml/pytracking

**MAST: A Memory-Augmented Self-supervised Tracker**
**MAST: 增强记忆的自我监督跟踪器**

- 论文：https://arxiv.org/abs/2002.07793
- 代码：https://github.com/zlai0/MAST

**Siamese Box Adaptive Network for Visual Tracking**
**双向适应网络用于视觉跟踪**

- 论文：https://arxiv.org/abs/2003.06761
- 代码：https://github.com/hqucv/siamban

## 多目标跟踪

**3D-ZeF: A 3D Zebrafish Tracking Benchmark Dataset**
**3D-ZeF: 3D斑马鱼追踪基准数据集**

- 主页：https://vap.aau.dk/3d-zef/
- 论文：https://arxiv.org/abs/2006.08466
- 代码：https://bitbucket.org/aauvap/3d-zef/src/master/
- 数据集：https://motchallenge.net/data/3D-ZeF20

<a name="Semantic-Segmentation"></a>

# 语义分割

**FDA: Fourier Domain Adaptation for Semantic Segmentation**
**FDA：费希尔域自适应用于语义分割**

- 论文：https://arxiv.org/abs/2004.05498

- 代码：https://github.com/YanchaoYang/FDA

**Super-BPD: Super Boundary-to-Pixel Direction for Fast Image Segmentation**
**超级BPD：用于快速图像分割的边界到像素方向**

- 论文：暂无

- 代码：https://github.com/JianqiangWan/Super-BPD

**Single-Stage Semantic Segmentation from Image Labels**
**单阶段语义分割从图像标签中进行**

- 论文：https://arxiv.org/abs/2005.08104

- 代码：https://github.com/visinf/1-stage-wseg

**Learning Texture Invariant Representation for Domain Adaptation of Semantic Segmentation**
**学习用于语义分割领域自适应的文本纹理不变表示**

- 论文：https://arxiv.org/abs/2003.00867
- 代码：https://github.com/MyeongJin-Kim/Learning-Texture-Invariant-Representation

**MSeg: A Composite Dataset for Multi-domain Semantic Segmentation**
**MSeg: 一个用于多领域语义分割的复合数据集**

- 论文：http://vladlen.info/papers/MSeg.pdf
- 代码：https://github.com/mseg-dataset/mseg-api

**CascadePSP: Toward Class-Agnostic and Very High-Resolution Segmentation via Global and Local Refinement**
**CascadePSP: 走向类A和非常高的分辨率分割，通过全局和局部细化**

- 论文：https://arxiv.org/abs/2005.02551
- 代码：https://github.com/hkchengrex/CascadePSP

**Unsupervised Intra-domain Adaptation for Semantic Segmentation through Self-Supervision**
**无监督领域内自监督迁移对语义分割的影响**

- Oral
- 论文：https://arxiv.org/abs/2004.07703
- 代码：https://github.com/feipan664/IntraDA

**Self-supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation**
**弱监督语义分割的自我监督等价注意力机制**

- 论文：https://arxiv.org/abs/2004.04581
- 代码：https://github.com/YudeWang/SEAM

**Temporally Distributed Networks for Fast Video Segmentation**
**用于快速视频分割的时变分布式网络**

- 论文：https://arxiv.org/abs/2004.01800

- 代码：https://github.com/feinanshan/TDNet

**Context Prior for Scene Segmentation**
**场景分割的上下文**

- 论文：https://arxiv.org/abs/2004.01547

- 代码：https://git.io/ContextPrior

**Strip Pooling: Rethinking Spatial Pooling for Scene Parsing**
**剥离池化：重新思考用于场景解析的空间池化**

- 论文：https://arxiv.org/abs/2003.13328

- 代码：https://github.com/Andrew-Qibin/SPNet

**Cars Can't Fly up in the Sky: Improving Urban-Scene Segmentation via Height-driven Attention Networks**
**汽车不能飞入天空：通过驱动注意力网络改进城市景观分割**

- 论文：https://arxiv.org/abs/2003.05128
- 代码：https://github.com/shachoi/HANet

**Learning Dynamic Routing for Semantic Segmentation**
**学习动态路由以进行语义分割**

- 论文：https://arxiv.org/abs/2003.10401

- 代码：https://github.com/yanwei-li/DynamicRouting

<a name="Instance-Segmentation"></a>

# 实例分割

**D2Det: Towards High Quality Object Detection and Instance Segmentation**
**D2Det：朝着高质量的目标检测和实例分割**

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/papers/Cao_D2Det_Towards_High_Quality_Object_Detection_and_Instance_Segmentation_CVPR_2020_paper.pdf

- 代码：https://github.com/JialeCao001/D2Det

**PolarMask: Single Shot Instance Segmentation with Polar Representation**
**极化遮罩：使用极化表示的单张实例分割**

- 论文：https://arxiv.org/abs/1909.13226 
- 代码：https://github.com/xieenze/PolarMask 
- 解读：https://zhuanlan.zhihu.com/p/84890413 

**CenterMask : Real-Time Anchor-Free Instance Segmentation**
**中心马锡：实时无锚实例分割**

- 论文：https://arxiv.org/abs/1911.06667 
- 代码：https://github.com/youngwanLEE/CenterMask 

**BlendMask: Top-Down Meets Bottom-Up for Instance Segmentation**
**BlendMask：自顶向下与自底向上实例分割**

- 论文：https://arxiv.org/abs/2001.00309
- 代码：https://github.com/aim-uofa/AdelaiDet

**Deep Snake for Real-Time Instance Segmentation**
**深度蛇用于实时实例分割**

- 论文：https://arxiv.org/abs/2001.01629
- 代码：https://github.com/zju3dv/snake

**Mask Encoding for Single Shot Instance Segmentation**
**单张实例分割的掩码编码**

- 论文：https://arxiv.org/abs/2003.11712

- 代码：https://github.com/aim-uofa/AdelaiDet

<a name="Panoptic-Segmentation"></a>

# 全景分割

**Video Panoptic Segmentation**
**视频 Panoptic 分割**

- 论文：https://arxiv.org/abs/2006.11339
- 代码：https://github.com/mcahny/vps
- 数据集：https://www.dropbox.com/s/ecem4kq0fdkver4/cityscapes-vps-dataset-1.0.zip?dl=0

**Pixel Consensus Voting for Panoptic Segmentation**
**像素共识投票用于全景分割**

- 论文：https://arxiv.org/abs/2004.01849
- 代码：还未公布

**BANet: Bidirectional Aggregation Network with Occlusion Handling for Panoptic Segmentation**
**BANet：双向汇聚网络与遮挡处理 for 全景分割**

论文：https://arxiv.org/abs/2003.14031

代码：https://github.com/Mooonside/BANet

<a name="VOS"></a>

# 视频目标分割

**A Transductive Approach for Video Object Segmentation**
**一种用于视频对象分割的导电方法**

- 论文：https://arxiv.org/abs/2004.07193

- 代码：https://github.com/microsoft/transductive-vos.pytorch

**State-Aware Tracker for Real-Time Video Object Segmentation**
**实时视频对象分割的元学习跟踪器**

- 论文：https://arxiv.org/abs/2003.00482

- 代码：https://github.com/MegviiDetection/video_analyst

**Learning Fast and Robust Target Models for Video Object Segmentation**
**学习快速而鲁棒的视频目标分割模型**

- 论文：https://arxiv.org/abs/2003.00908 
- 代码：https://github.com/andr345/frtm-vos

**Learning Video Object Segmentation from Unlabeled Videos**
**从无标签视频中学习视频对象分割**

- 论文：https://arxiv.org/abs/2003.05020
- 代码：https://github.com/carrierlxk/MuG

<a name="Superpixel"></a>

# 超像素分割

**Superpixel Segmentation with Fully Convolutional Networks**
**超级像素分割与全卷积网络**

- 论文：https://arxiv.org/abs/2003.12929
- 代码：https://github.com/fuy34/superpixel_fcn

<a name="IIS"></a>

# 交互式图像分割

**Interactive Object Segmentation with Inside-Outside Guidance**
**交互式对象分割与内向外指导**

- 论文下载链接：http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Interactive_Object_Segmentation_With_Inside-Outside_Guidance_CVPR_2020_paper.pdf
- 代码：https://github.com/shiyinzhang/Inside-Outside-Guidance
- 数据集：https://github.com/shiyinzhang/Pixel-ImageNet

<a name="NAS"></a>

# NAS

**AOWS: Adaptive and optimal network width search with latency constraints**
**AOWS：具有延迟约束的自适应最优网络宽度搜索**

- 论文：https://arxiv.org/abs/2005.10481
- 代码：https://github.com/bermanmaxim/AOWS

**Densely Connected Search Space for More Flexible Neural Architecture Search**
**紧密连接的搜索空间：为更灵活的神经架构搜索提供更多支持**

- 论文：https://arxiv.org/abs/1906.09607

- 代码：https://github.com/JaminFong/DenseNAS

**MTL-NAS: Task-Agnostic Neural Architecture Search towards General-Purpose Multi-Task Learning**
**MTL-NAS：面向通用多任务学习的任务无关神经架构搜索**

- 论文：https://arxiv.org/abs/2003.14058

- 代码：https://github.com/bhpfelix/MTLNAS

**FBNetV2: Differentiable Neural Architecture Search for Spatial and Channel Dimensions**
**FBNetV2：用于空间和通道维度可微神经架构搜索的差分神经架构**

- 论文下载链接：https://arxiv.org/abs/2004.05565

- 代码：https://github.com/facebookresearch/mobile-vision

**Neural Architecture Search for Lightweight Non-Local Networks**
**神经架构搜索轻量级非局部网络**

- 论文：https://arxiv.org/abs/2004.01961
- 代码：https://github.com/LiYingwei/AutoNL

**Rethinking Performance Estimation in Neural Architecture Search**
**重新思考神经架构搜索中的性能估计**

- 论文：https://arxiv.org/abs/2005.09917
- 代码：https://github.com/zhengxiawu/rethinking_performance_estimation_in_NAS
- 解读1：https://www.zhihu.com/question/372070853/answer/1035234510
- 解读2：https://zhuanlan.zhihu.com/p/111167409

**CARS: Continuous Evolution for Efficient Neural Architecture Search**
**CARS：连续进化的高效神经架构搜索**

- 论文：https://arxiv.org/abs/1909.04977 
- 代码（即将开源）：https://github.com/huawei-noah/CARS 

<a name="GAN"></a>

# GAN

**SEAN: Image Synthesis with Semantic Region-Adaptive Normalization**
**SEAN：具有语义区域自适应归一化的图像合成**

- 论文：https://arxiv.org/abs/1911.12861
- 代码：https://github.com/ZPdesu/SEAN

**Reusing Discriminators for Encoding: Towards Unsupervised Image-to-Image Translation**
**重用判别器进行编码：无监督图像到图像的转换**

- 论文地址：http://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Reusing_Discriminators_for_Encoding_Towards_Unsupervised_Image-to-Image_Translation_CVPR_2020_paper.html
- 代码地址：https://github.com/alpc91/NICE-GAN-pytorch 

**Distribution-induced Bidirectional Generative Adversarial Network for Graph Representation Learning**
**分布诱导的双向生成对抗网络用于图表示学习**

- 论文：https://arxiv.org/abs/1912.01899
- 代码：https://github.com/SsGood/DBGAN 

**PSGAN: Pose and Expression Robust Spatial-Aware GAN for Customizable Makeup Transfer**
**PSGAN：用于自定义化妆移植的姿势和表达式鲁棒空间感生成对抗网络**

- 论文：https://arxiv.org/abs/1909.06956
- 代码：https://github.com/wtjiang98/PSGAN

**Semantically Mutil-modal Image Synthesis**
**语义多模态图像合成**

- 主页：http://seanseattle.github.io/SMIS
- 论文：https://arxiv.org/abs/2003.12697
- 代码：https://github.com/Seanseattle/SMIS

**Unpaired Portrait Drawing Generation via Asymmetric Cycle Mapping**
**通过不对称周期映射生成未配对人物肖像**

- 论文：https://yiranran.github.io/files/CVPR2020_Unpaired%20Portrait%20Drawing%20Generation%20via%20Asymmetric%20Cycle%20Mapping.pdf
- 代码：https://github.com/yiranran/Unpaired-Portrait-Drawing

**Learning to Cartoonize Using White-box Cartoon Representations**
**学习使用白盒卡通化表示**

- 论文：https://github.com/SystemErrorWang/White-box-Cartoonization/blob/master/paper/06791.pdf

- 主页：https://systemerrorwang.github.io/White-box-Cartoonization/
- 代码：https://github.com/SystemErrorWang/White-box-Cartoonization
- 解读：https://zhuanlan.zhihu.com/p/117422157
- Demo视频：https://www.bilibili.com/video/av56708333

**GAN Compression: Efficient Architectures for Interactive Conditional GANs**
**GAN压缩：用于交互式条件GAN的高效架构**

- 论文：https://arxiv.org/abs/2003.08936

- 代码：https://github.com/mit-han-lab/gan-compression

**Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions**
**注意：基于卷积神经网络的生成式深度神经网络无法正确重现光谱分布**

- 论文：https://arxiv.org/abs/2003.01826 
- 代码：https://github.com/cc-hpc-itwm/UpConv 

<a name="Re-ID"></a>

# Re-ID

 **High-Order Information Matters: Learning Relation and Topology for Occluded Person Re-Identification**

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/html/Wang_High-Order_Information_Matters_Learning_Relation_and_Topology_for_Occluded_Person_CVPR_2020_paper.html
- 代码：https://github.com/wangguanan/HOReID 

**COCAS: A Large-Scale Clothes Changing Person Dataset for Re-identification**
**COCAS：用于重新识别的大规模衣物更换人物数据集**

- 论文：https://arxiv.org/abs/2005.07862

- 数据集：暂无

**Transferable, Controllable, and Inconspicuous Adversarial Attacks on Person Re-identification With Deep Mis-Ranking**
**深度误解重置对人员重新识别的转移性、可控制性和隐蔽对抗攻击**

- 论文：https://arxiv.org/abs/2004.04199

- 代码：https://github.com/whj363636/Adversarial-attack-on-Person-ReID-With-Deep-Mis-Ranking

**Pose-guided Visible Part Matching for Occluded Person ReID**
**基于姿态的可见部分匹配用于遮挡人物的重识别**

- 论文：https://arxiv.org/abs/2004.00230
- 代码：https://github.com/hh23333/PVPM

**Weakly supervised discriminative feature learning with state information for person identification**
**弱监督判别特征学习，利用状态信息进行人物识别**

- 论文：https://arxiv.org/abs/2002.11939 
- 代码：https://github.com/KovenYu/state-information 

<a name="3D-PointCloud"></a>

# 3D点云（分类/分割/配准等）

## 3D点云卷积

**PointASNL: Robust Point Clouds Processing using Nonlocal Neural Networks with Adaptive Sampling**
**PointASNL：使用自适应采样和非局部神经网络进行鲁棒的点云处理**

- 论文：https://arxiv.org/abs/2003.00492
- 代码：https://github.com/yanx27/PointASNL 

**Global-Local Bidirectional Reasoning for Unsupervised Representation Learning of 3D Point Clouds**
**全局-局部双向推理：用于无监督表示学习的三维点云**

- 论文下载链接：https://arxiv.org/abs/2003.12971

- 代码：https://github.com/raoyongming/PointGLR

**Grid-GCN for Fast and Scalable Point Cloud Learning**
**网格GCN用于快速且可扩展的点云学习**

- 论文：https://arxiv.org/abs/1912.02984

- 代码：https://github.com/Xharlie/Grid-GCN

**FPConv: Learning Local Flattening for Point Convolution**
**FPConv: 学习局部平滑的点卷积**

- 论文：https://arxiv.org/abs/2002.10701
- 代码：https://github.com/lyqun/FPConv

## 3D点云分类

**PointAugment: an Auto-Augmentation Framework for Point Cloud Classification**
**点增强：用于点云分类的自动增强框架**

- 论文：https://arxiv.org/abs/2002.10876 
- 代码（即将开源）： https://github.com/liruihui/PointAugment/ 

## 3D点云语义分割

**RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds**
**RandLA-Net：大规模点云的高效语义分割**

- 论文：https://arxiv.org/abs/1911.11236
- 代码：https://github.com/QingyongHu/RandLA-Net

- 解读：https://zhuanlan.zhihu.com/p/105433460

**Weakly Supervised Semantic Point Cloud Segmentation:Towards 10X Fewer Labels**
**弱监督语义点云分割：朝着 10 倍更少的标签**

- 论文：https://arxiv.org/abs/2004.04091

- 代码：https://github.com/alex-xun-xu/WeakSupPointCloudSeg

**PolarNet: An Improved Grid Representation for Online LiDAR Point Clouds Semantic Segmentation**
**极地网络：改进的网格表示，用于在线激光雷达点云语义分割**

- 论文：https://arxiv.org/abs/2003.14032
- 代码：https://github.com/edwardzhou130/PolarSeg

**Learning to Segment 3D Point Clouds in 2D Image Space**
**学习在二维图像空间中分割3D点云**

- 论文：https://arxiv.org/abs/2003.05593

- 代码：https://github.com/WPI-VISLab/Learning-to-Segment-3D-Point-Clouds-in-2D-Image-Space

## 3D点云实例分割

PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation

- 论文：https://arxiv.org/abs/2004.01658
- 代码：https://github.com/Jia-Research-Lab/PointGroup

## 3D点云配准

**Feature-metric Registration: A Fast Semi-supervised Approach for Robust Point Cloud Registration without Correspondences**
**特征度量注册：一种用于无配准鲁棒点云注册的快速半监督方法**

- 论文：https://arxiv.org/abs/2005.01014
- 代码：https://github.com/XiaoshuiHuang/fmr 

**D3Feat: Joint Learning of Dense Detection and Description of 3D Local Features**
**D3Feat: 联合学习稠密检测和3D局部特征描述**

- 论文：https://arxiv.org/abs/2003.03164
- 代码：https://github.com/XuyangBai/D3Feat

**RPM-Net: Robust Point Matching using Learned Features**
**RPM-Net：使用学习特征的稳健点匹配**

- 论文：https://arxiv.org/abs/2003.13479
- 代码：https://github.com/yewzijian/RPMNet 

## 3D点云补全

**Cascaded Refinement Network for Point Cloud Completion**
**级联细化网络用于点云补全**

- 论文：https://arxiv.org/abs/2004.03327
- 代码：https://github.com/xiaogangw/cascaded-point-completion

## 3D点云目标跟踪

**P2B: Point-to-Box Network for 3D Object Tracking in Point Clouds**
**P2B：点云中的3D对象跟踪点对点网络**

- 论文：https://arxiv.org/abs/2005.13888
- 代码：https://github.com/HaozheQi/P2B

## 其他

**An Efficient PointLSTM for Point Clouds Based Gesture Recognition**
**基于点云的高姿态手势识别的有效点LSTM**

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/html/Min_An_Efficient_PointLSTM_for_Point_Clouds_Based_Gesture_Recognition_CVPR_2020_paper.html
- 代码：https://github.com/Blueprintf/pointlstm-gesture-recognition-pytorch

<a name="Face"></a>

# 人脸

## 人脸识别

**CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition**
**课程脸：深度人脸识别自适应课程损失**

- 论文：https://arxiv.org/abs/2004.00288

- 代码：https://github.com/HuangYG123/CurricularFace

**Learning Meta Face Recognition in Unseen Domains**
**学习在未知领域的元面部识别**

- 论文：https://arxiv.org/abs/2003.07733
- 代码：https://github.com/cleardusk/MFR
- 解读：https://mp.weixin.qq.com/s/YZoEnjpnlvb90qSI3xdJqQ 

## 人脸检测

## 人脸活体检测

**Searching Central Difference Convolutional Networks for Face Anti-Spoofing**
**搜索中央差异卷积神经网络以进行人脸反欺诈**

- 论文：https://arxiv.org/abs/2003.04092

- 代码：https://github.com/ZitongYu/CDCN

## 人脸表情识别

**Suppressing Uncertainties for Large-Scale Facial Expression Recognition**
**抑制大规模面部表情识别中的不确定性**

- 论文：https://arxiv.org/abs/2002.10392 

- 代码（即将开源）：https://github.com/kaiwang960112/Self-Cure-Network 

## 人脸转正

**Rotate-and-Render: Unsupervised Photorealistic Face Rotation from Single-View Images**
**旋转和渲染：从单视图图像中无监督地生成真实人脸旋转**

- 论文：https://arxiv.org/abs/2003.08124
- 代码：https://github.com/Hangz-nju-cuhk/Rotate-and-Render

## 人脸3D重建

**AvatarMe: Realistically Renderable 3D Facial Reconstruction "in-the-wild"**
**变量名：真实感人的3D面部重建“野外”**

- 论文：https://arxiv.org/abs/2003.13845
- 数据集：https://github.com/lattas/AvatarMe

**FaceScape: a Large-scale High Quality 3D Face Dataset and Detailed Riggable 3D Face Prediction**
**FaceScape：大规模高质量3D面部数据集和详细可调节3D面部预测**

- 论文：https://arxiv.org/abs/2003.13989
- 代码：https://github.com/zhuhao-nju/facescape

<a name="Human-Pose-Estimation"></a>

# 人体姿态估计(2D/3D)

## 2D人体姿态估计

**TransMoMo: Invariance-Driven Unsupervised Video Motion Retargeting**
**TransMoMo：不变量驱动的无监督视频运动重定位**

- 主页：https://yzhq97.github.io/transmomo/

- 论文：https://arxiv.org/abs/2003.14401
- 代码：https://github.com/yzhq97/transmomo.pytorch

**HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation**
**更高的HRNet：用于底层人体姿态估计的规模感知表示学习**

- 论文：https://arxiv.org/abs/1908.10357
- 代码：https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation

**The Devil is in the Details: Delving into Unbiased Data Processing for Human Pose Estimation**
**细节决定一切：深入挖掘无偏见的人体姿态估计数据处理技术**

- 论文：https://arxiv.org/abs/1911.07524 
- 代码：https://github.com/HuangJunJie2017/UDP-Pose
- 解读：https://zhuanlan.zhihu.com/p/92525039

**Distribution-Aware Coordinate Representation for Human Pose Estimation**
**分布式感知的坐标表示用于人体姿态估计**

- 主页：https://ilovepose.github.io/coco/ 

- 论文：https://arxiv.org/abs/1910.06278 

- 代码：https://github.com/ilovepose/DarkPose 

## 3D人体姿态估计

 **Cascaded Deep Monocular 3D Human Pose Estimation With Evolutionary Training Data**

- 论文：https://arxiv.org/abs/2006.07778
- 代码：https://github.com/Nicholasli1995/EvoSkeleton 

**Fusing Wearable IMUs with Multi-View Images for Human Pose Estimation: A Geometric Approach**
**用于人体姿态估计的可穿戴惯性测量单元与多视角图像的融合：一种几何方法**

- 主页：https://www.zhe-zhang.com/cvpr2020
- 论文：https://arxiv.org/abs/2003.11163

- 代码：https://github.com/CHUNYUWANG/imu-human-pose-pytorch

**Bodies at Rest: 3D Human Pose and Shape Estimation from a Pressure Image using Synthetic Data**
**休息时的身体：使用合成数据从压力图像中估计3D人体姿态和形状**

- 论文下载链接：https://arxiv.org/abs/2004.01166

- 代码：https://github.com/Healthcare-Robotics/bodies-at-rest
- 数据集：https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KOA4ML

**Self-Supervised 3D Human Pose Estimation via Part Guided Novel Image Synthesis**
**自我监督3D人体姿态估计通过部分引导的新图像合成**

- 主页：http://val.cds.iisc.ac.in/pgp-human/
- 论文：https://arxiv.org/abs/2004.04400

**Compressed Volumetric Heatmaps for Multi-Person 3D Pose Estimation**
**压缩体积热图多人体3D姿势估计**

- 论文：https://arxiv.org/abs/2004.00329
- 代码：https://github.com/fabbrimatteo/LoCO

**VIBE: Video Inference for Human Body Pose and Shape Estimation**
**VIBE：用于人体姿势和形状估计的视觉推理**

- 论文：https://arxiv.org/abs/1912.05656 
- 代码：https://github.com/mkocabas/VIBE

**Back to the Future: Joint Aware Temporal Deep Learning 3D Human Pose Estimation**
**回到未来：联合感时深层学习3D人体姿态估计**

- 论文：https://arxiv.org/abs/2002.11251 
- 代码：https://github.com/vnmr/JointVideoPose3D

**Cross-View Tracking for Multi-Human 3D Pose Estimation at over 100 FPS**
**跨视角跟踪多人体3D姿势估计，超过100帧每秒**

- 论文：https://arxiv.org/abs/2003.03972
- 数据集：暂无

<a name="Human-Parsing"></a>

# 人体解析

**Correlating Edge, Pose with Parsing**
**相关边缘、姿态与解析**

- 论文：https://arxiv.org/abs/2005.01431

- 代码：https://github.com/ziwei-zh/CorrPM

<a name="Scene-Text-Detection"></a>

# 场景文本检测

**STEFANN: Scene Text Editor using Font Adaptive Neural Network**
**STEFANN：使用自适应神经网络的字体适应场景文本编辑器**

- 主页：https://prasunroy.github.io/stefann/

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/html/Roy_STEFANN_Scene_Text_Editor_Using_Font_Adaptive_Neural_Network_CVPR_2020_paper.html
- 代码：https://github.com/prasunroy/stefann
- 数据集：https://drive.google.com/open?id=1sEDiX_jORh2X-HSzUnjIyZr-G9LJIw1k

**ContourNet: Taking a Further Step Toward Accurate Arbitrary-Shaped Scene Text Detection**
**ContourNet：在准确任意形状场景文本检测方面取得进一步进展**

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_ContourNet_Taking_a_Further_Step_Toward_Accurate_Arbitrary-Shaped_Scene_Text_CVPR_2020_paper.pdf
- 代码：https://github.com/wangyuxin87/ContourNet 

**UnrealText: Synthesizing Realistic Scene Text Images from the Unreal World**
**UnrealText：从虚幻世界中合成真实场景文本图像**

- 论文：https://arxiv.org/abs/2003.10608
- 代码和数据集：https://github.com/Jyouhou/UnrealText/

**ABCNet: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network**
**ABCNet：实时场景文本检测与自适应贝塞尔曲线网络**

- 论文：https://arxiv.org/abs/2002.10200 
- 代码（即将开源）：https://github.com/Yuliang-Liu/bezier_curve_text_spotting
- 代码（即将开源）：https://github.com/aim-uofa/adet

**Deep Relational Reasoning Graph Network for Arbitrary Shape Text Detection**
**深度关系推理图网络用于任意形状文本检测**

- 论文：https://arxiv.org/abs/2003.07493

- 代码：https://github.com/GXYM/DRRG

<a name="Scene-Text-Recognition"></a>

# 场景文本识别

**SEED: Semantics Enhanced Encoder-Decoder Framework for Scene Text Recognition**
**种子：用于场景文本识别的语义增强编码器-解码器框架**

- 论文：https://arxiv.org/abs/2005.10977
- 代码：https://github.com/Pay20Y/SEED

**UnrealText: Synthesizing Realistic Scene Text Images from the Unreal World**
**UnrealText：从Unreal世界合成真实场景文本图像**

- 论文：https://arxiv.org/abs/2003.10608
- 代码和数据集：https://github.com/Jyouhou/UnrealText/

**ABCNet: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network**
**ABCNet：实时场景文本识别与自适应贝塞尔曲线网络**

- 论文：https://arxiv.org/abs/2002.10200 
- 代码（即将开源）：https://github.com/aim-uofa/adet

**Learn to Augment: Joint Data Augmentation and Network Optimization for Text Recognition**
**学习增强：联合数据增强和网络优化用于文本识别**

- 论文：https://arxiv.org/abs/2003.06606

- 代码：https://github.com/Canjie-Luo/Text-Image-Augmentation

<a name="Feature"></a>

# 特征(点)检测和描述

**SuperGlue: Learning Feature Matching with Graph Neural Networks**
**SuperGlue: 学习利用图神经网络进行特征匹配**

- 论文：https://arxiv.org/abs/1911.11763
- 代码：https://github.com/magicleap/SuperGluePretrainedNetwork

<a name="Super-Resolution"></a>

# 超分辨率

## 图像超分辨率

**Closed-Loop Matters: Dual Regression Networks for Single Image Super-Resolution**
**闭环重要：用于单张图像超分辨率的双回归网络**

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/html/Guo_Closed-Loop_Matters_Dual_Regression_Networks_for_Single_Image_Super-Resolution_CVPR_2020_paper.html
- 代码：https://github.com/guoyongcs/DRN

**Learning Texture Transformer Network for Image Super-Resolution**
**学习用于图像超分辨率的三维纹理转换器网络**

- 论文：https://arxiv.org/abs/2006.04139

- 代码：https://github.com/FuzhiYang/TTSR

**Image Super-Resolution with Cross-Scale Non-Local Attention and Exhaustive Self-Exemplars Mining**
**图像超分辨率与跨尺度非局部关注和全面自样例挖掘**

- 论文：https://arxiv.org/abs/2006.01424
- 代码：https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention

**Structure-Preserving Super Resolution with Gradient Guidance**
**保留结构超分辨率与梯度指导**

- 论文：https://arxiv.org/abs/2003.13081

- 代码：https://github.com/Maclory/SPSR

**Rethinking Data Augmentation for Image Super-resolution: A Comprehensive Analysis and a New Strategy**
**重思图像超分辨率中的数据增强：一个全面的分析和一种新策略**

论文：https://arxiv.org/abs/2004.00448

代码：https://github.com/clovaai/cutblur

## 视频超分辨率

**TDAN: Temporally-Deformable Alignment Network for Video Super-Resolution**
**TDAN：用于视频超分辨率的时间可变形对齐网络**

- 论文：https://arxiv.org/abs/1812.02898
- 代码：https://github.com/YapengTian/TDAN-VSR-CVPR-2020

**Space-Time-Aware Multi-Resolution Video Enhancement**
**空间时间感知的多分辨率视频增强**

- 主页：https://alterzero.github.io/projects/STAR.html
- 论文：http://arxiv.org/abs/2003.13170
- 代码：https://github.com/alterzero/STARnet

**Zooming Slow-Mo: Fast and Accurate One-Stage Space-Time Video Super-Resolution**
**缩放慢动作：快速而准确的一阶段空间-时间视频超分辨率**

- 论文：https://arxiv.org/abs/2002.11616 
- 代码：https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020 

<a name="Model-Compression"></a>

# 模型压缩/剪枝

**DMCP: Differentiable Markov Channel Pruning for Neural Networks**
**DMCP：用于神经网络的 Differentiable Markov Channel 修剪**

- 论文：https://arxiv.org/abs/2005.03354
- 代码：https://github.com/zx55/dmcp

**Forward and Backward Information Retention for Accurate Binary Neural Networks**
**正向和反向信息保留对准确二进制神经网络的影响**

- 论文：https://arxiv.org/abs/1909.10788

- 代码：https://github.com/htqin/IR-Net

**Towards Efficient Model Compression via Learned Global Ranking**
**通过对全局排名进行学习来提高模型压缩效率**

- 论文：https://arxiv.org/abs/1904.12368
- 代码：https://github.com/cmu-enyac/LeGR

**HRank: Filter Pruning using High-Rank Feature Map**
**HRank：使用高秩特征图进行过滤剪枝**

- 论文：http://arxiv.org/abs/2002.10179
- 代码：https://github.com/lmbxmu/HRank 

**GAN Compression: Efficient Architectures for Interactive Conditional GANs**
**GAN压缩：用于交互式条件GAN的高效架构**

- 论文：https://arxiv.org/abs/2003.08936

- 代码：https://github.com/mit-han-lab/gan-compression

**Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression**
**组稀疏性：网络压缩中的滤波器修剪和分解之间的关节**

- 论文：https://arxiv.org/abs/2003.08935

- 代码：https://github.com/ofsoundof/group_sparsity

<a name="Action-Recognition"></a>

# 视频理解/行为识别

**Oops! Predicting Unintentional Action in Video**
**哎呀！预测视频中的无意动作**

- 主页：https://oops.cs.columbia.edu/

- 论文：https://arxiv.org/abs/1911.11206
- 代码：https://github.com/cvlab-columbia/oops
- 数据集：https://oops.cs.columbia.edu/data

**PREDICT & CLUSTER: Unsupervised Skeleton Based Action Recognition**
**预测与聚类：基于无监督骨骼动作识别**

- 论文：https://arxiv.org/abs/1911.12409
- 代码：https://github.com/shlizee/Predict-Cluster 

**Intra- and Inter-Action Understanding via Temporal Action Parsing**
**内联和交互动作理解通过时间动作解析**

- 论文：https://arxiv.org/abs/2005.10229
- 主页和数据集：https://sdolivia.github.io/TAPOS/

**3DV: 3D Dynamic Voxel for Action Recognition in Depth Video**
**3DV：深度视频中的动作识别三维动态体素**

- 论文：https://arxiv.org/abs/2005.05501
- 代码：https://github.com/3huo/3DV-Action

**FineGym: A Hierarchical Video Dataset for Fine-grained Action Understanding**
**细粒度动作理解的无缝视频数据集**

- 主页：https://sdolivia.github.io/FineGym/
- 论文：https://arxiv.org/abs/2004.06704

**TEA: Temporal Excitation and Aggregation for Action Recognition**
**TEA： Temporal Excitation and Aggregation for Action Recognition（时间激励和汇聚用于动作识别）**

- 论文：https://arxiv.org/abs/2004.01398

- 代码：https://github.com/Phoenix1327/tea-action-recognition

**X3D: Expanding Architectures for Efficient Video Recognition**
**X3D：用于高效视频识别的扩展架构**

- 论文：https://arxiv.org/abs/2004.04730

- 代码：https://github.com/facebookresearch/SlowFast

**Temporal Pyramid Network for Action Recognition**
**时空金字塔网络用于动作识别**

- 主页：https://decisionforce.github.io/TPN

- 论文：https://arxiv.org/abs/2004.03548 
- 代码：https://github.com/decisionforce/TPN 

## 基于骨架的动作识别

**Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition**
**解开和统一基于骨架的动作识别图形卷积**

- 论文：https://arxiv.org/abs/2003.14111
- 代码：https://github.com/kenziyuliu/ms-g3d

<a name="Crowd-Counting"></a>

# 人群计数

<a name="Depth-Estimation"></a>

# 深度估计

**BiFuse: Monocular 360◦ Depth Estimation via Bi-Projection Fusion**
**BiFuse：单目360°深度估计通过双投影融合**

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_BiFuse_Monocular_360_Depth_Estimation_via_Bi-Projection_Fusion_CVPR_2020_paper.pdf
- 代码：https://github.com/Yeh-yu-hsuan/BiFuse

**Focus on defocus: bridging the synthetic to real domain gap for depth estimation**
**关注去关注：为深度估计连接合成与真实领域的差距**

- 论文：https://arxiv.org/abs/2005.09623
- 代码：https://github.com/dvl-tum/defocus-net

**Bi3D: Stereo Depth Estimation via Binary Classifications**
**Bi3D：通过二进制分类进行立体深度估计**

- 论文：https://arxiv.org/abs/2005.07274

- 代码：https://github.com/NVlabs/Bi3D

**AANet: Adaptive Aggregation Network for Efficient Stereo Matching**
**AANet：用于高效立体匹配的自适应聚合网络**

- 论文：https://arxiv.org/abs/2004.09548
- 代码：https://github.com/haofeixu/aanet

**Towards Better Generalization: Joint Depth-Pose Learning without PoseNet**
**向更好的泛化：联合深度姿态学习，无需PoseNet**

- 论文：https://github.com/B1ueber2y/TrianFlow

- 代码：https://github.com/B1ueber2y/TrianFlow

## 单目深度估计

**On the uncertainty of self-supervised monocular depth estimation**
**关于自监督单目深度估计的不确定性**

- 论文：https://arxiv.org/abs/2005.06209
- 代码：https://github.com/mattpoggi/mono-uncertainty

**3D Packing for Self-Supervised Monocular Depth Estimation**
**3D 包装用于自监督单目深度估计**

- 论文：https://arxiv.org/abs/1905.02693
- 代码：https://github.com/TRI-ML/packnet-sfm
- Demo视频：https://www.bilibili.com/video/av70562892/

**Domain Decluttering: Simplifying Images to Mitigate Synthetic-Real Domain Shift and Improve Depth Estimation**
**领域去噪：简化图像以减轻合成-真实领域平移并改善深度估计**

- 论文：https://arxiv.org/abs/2002.12114
- 代码：https://github.com/yzhao520/ARC

<a name="6DOF"></a>

# 6D目标姿态估计

 **PVN3D: A Deep Point-wise 3D Keypoints Voting Network for 6DoF Pose Estimation**

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/papers/He_PVN3D_A_Deep_Point-Wise_3D_Keypoints_Voting_Network_for_6DoF_CVPR_2020_paper.pdf
- 代码：https://github.com/ethnhe/PVN3D

**MoreFusion: Multi-object Reasoning for 6D Pose Estimation from Volumetric Fusion**
**MoreFusion：多模态融合体积估计的6D姿态估计**

- 论文：https://arxiv.org/abs/2004.04336
- 代码：https://github.com/wkentaro/morefusion

**EPOS: Estimating 6D Pose of Objects with Symmetries**
**EPOS：使用对称性估计6D物体姿势**

主页：http://cmp.felk.cvut.cz/epos

论文：https://arxiv.org/abs/2004.00605

**G2L-Net: Global to Local Network for Real-time 6D Pose Estimation with Embedding Vector Features**
**G2L-Net：用于实时6D姿态估计的全球到局部网络，具有嵌入向量特征**

- 论文：https://arxiv.org/abs/2003.11089

- 代码：https://github.com/DC1991/G2L_Net

<a name="Hand-Pose"></a>

# 手势估计

**HOPE-Net: A Graph-based Model for Hand-Object Pose Estimation**
**HOPE-Net：基于图的模型，用于手部姿势估计**

- 论文：https://arxiv.org/abs/2004.00060

- 主页：http://vision.sice.indiana.edu/projects/hopenet

**Monocular Real-time Hand Shape and Motion Capture using Multi-modal Data**
**单目实时手势和动作捕捉，利用多模态数据**

- 论文：https://arxiv.org/abs/2003.09572

- 代码：https://github.com/CalciferZh/minimal-hand

<a name="Saliency"></a>

# 显著性检测

**JL-DCF: Joint Learning and Densely-Cooperative Fusion Framework for RGB-D Salient Object Detection**
**JL-DCF: 联合学习和密集协作融合框架 for RGB-D 显著目标检测**

- 论文：https://arxiv.org/abs/2004.08515

- 代码：https://github.com/kerenfu/JLDCF/

**UC-Net: Uncertainty Inspired RGB-D Saliency Detection via Conditional Variational Autoencoders**
**UC-Net：通过条件变分自动编码器在不确定性驱动下的鲁棒RGB-D显著性检测**

- 主页：http://dpfan.net/d3netbenchmark/

- 论文：https://arxiv.org/abs/2004.05763
- 代码：https://github.com/JingZhang617/UCNet

<a name="Denoising"></a>

# 去噪

**A Physics-based Noise Formation Model for Extreme Low-light Raw Denoising**
**一种基于物理的噪声形成模型，用于实现极端低光下的原始去噪**

- 论文：https://arxiv.org/abs/2003.12751

- 代码：https://github.com/Vandermode/NoiseModel

**CycleISP: Real Image Restoration via Improved Data Synthesis**
**循环ISP：通过改进的数据合成实现真实图像恢复**

- 论文：https://arxiv.org/abs/2003.07761

- 代码：https://github.com/swz30/CycleISP

<a name="Deraining"></a>

# 去雨

**Multi-Scale Progressive Fusion Network for Single Image Deraining**
**多尺度渐进融合网络单张图像去噪**

- 论文：https://arxiv.org/abs/2003.10985
- 代码：https://github.com/kuihua/MSPFN

**Detail-recovery Image Deraining via Context Aggregation Networks**
**详细恢复图像去噪通过上下文聚合网络**

- 论文：https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_Detail-recovery_Image_Deraining_via_Context_Aggregation_Networks_CVPR_2020_paper.html
- 代码：https://github.com/Dengsgithub/DRD-Net

<a name="Deblurring"></a>

# 去模糊

## 视频去模糊

**Cascaded Deep Video Deblurring Using Temporal Sharpness Prior**
**级联深度视频去模糊 使用时间锐度优先**

- 主页：https://csbhr.github.io/projects/cdvd-tsp/index.html 
- 论文：https://arxiv.org/abs/2004.02501 
- 代码：https://github.com/csbhr/CDVD-TSP

<a name="Dehazing"></a>

# 去雾

**Domain Adaptation for Image Dehazing**
**图像去雾的领域自适应**

- 论文：https://arxiv.org/abs/2005.04668

- 代码：https://github.com/HUSTSYJ/DA_dahazing

**Multi-Scale Boosted Dehazing Network with Dense Feature Fusion**
**多尺度增强去雾网络与稠密特征融合**

- 论文：https://arxiv.org/abs/2004.13388

- 代码：https://github.com/BookerDeWitt/MSBDN-DFF

<a name="Feature"></a>

# 特征点检测与描述

**ASLFeat: Learning Local Features of Accurate Shape and Localization**
**ASLFeat: 学习准确形状和位置的局部特征**

- 论文：https://arxiv.org/abs/2003.10071

- 代码：https://github.com/lzx551402/aslfeat

<a name="VQA"></a>

# 视觉问答(VQA)

**VC R-CNN：Visual Commonsense R-CNN** 

- 论文：https://arxiv.org/abs/2002.12204
- 代码：https://github.com/Wangt-CN/VC-R-CNN

<a name="VideoQA"></a>

# 视频问答(VideoQA)

**Hierarchical Conditional Relation Networks for Video Question Answering**
**层次条件关系网络 for 视频问答**

- 论文：https://arxiv.org/abs/2002.10698
- 代码：https://github.com/thaolmk54/hcrn-videoqa

<a name="VLN"></a>

# 视觉语言导航

**Towards Learning a Generic Agent for Vision-and-Language Navigation via Pre-training**
**朝着通过预训练学习视觉语言导航的通**

- 论文：https://arxiv.org/abs/2002.10638
- 代码（即将开源）：https://github.com/weituo12321/PREVALENT

<a name="Video-Compression"></a>

# 视频压缩

**Learning for Video Compression with Hierarchical Quality and Recurrent Enhancement**
**学习视频压缩的层次质量和递归增强**

- 论文：https://arxiv.org/abs/2003.01966 
- 代码：https://github.com/RenYang-home/HLVC

<a name="Video-Frame-Interpolation"></a>

# 视频插帧

**AdaCoF: Adaptive Collaboration of Flows for Video Frame Interpolation**
**AdaCoF：用于视频帧插切的流体自适应协作**

- 论文：https://arxiv.org/abs/1907.10244
- 代码：https://github.com/HyeongminLEE/AdaCoF-pytorch

**FeatureFlow: Robust Video Interpolation via Structure-to-Texture Generation**
**特征流：通过结构到纹理生成来实现稳健的视频插值**

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/html/Gui_FeatureFlow_Robust_Video_Interpolation_via_Structure-to-Texture_Generation_CVPR_2020_paper.html

- 代码：https://github.com/CM-BF/FeatureFlow

**Zooming Slow-Mo: Fast and Accurate One-Stage Space-Time Video Super-Resolution**
**缩放慢动作：快速而准确的一阶段时空中视频超分辨率**

- 论文：https://arxiv.org/abs/2002.11616
- 代码：https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020

**Space-Time-Aware Multi-Resolution Video Enhancement**
**空间-时间感知的多分辨率视频增强**

- 主页：https://alterzero.github.io/projects/STAR.html
- 论文：http://arxiv.org/abs/2003.13170
- 代码：https://github.com/alterzero/STARnet

**Scene-Adaptive Video Frame Interpolation via Meta-Learning**
**场景自适应视频帧插值通过元学习**

- 论文：https://arxiv.org/abs/2004.00779
- 代码：https://github.com/myungsub/meta-interpolation

**Softmax Splatting for Video Frame Interpolation**
**软最大值平滑视频帧插值**

- 主页：http://sniklaus.com/papers/softsplat
- 论文：https://arxiv.org/abs/2003.05534
- 代码：https://github.com/sniklaus/softmax-splatting

<a name="Style-Transfer"></a>

# 风格迁移

**Diversified Arbitrary Style Transfer via Deep Feature Perturbation**
**通过对深度特征扰动的扩展，多领域任意风格迁移**

- 论文：https://arxiv.org/abs/1909.08223
- 代码：https://github.com/EndyWon/Deep-Feature-Perturbation

**Collaborative Distillation for Ultra-Resolution Universal Style Transfer**
**合作蒸馏：超分辨率通用风格迁移**

- 论文：https://arxiv.org/abs/2003.08436

- 代码：https://github.com/mingsun-tse/collaborative-distillation

<a name="Lane-Detection"></a>

# 车道线检测

**Inter-Region Affinity Distillation for Road Marking Segmentation**
**区域间亲和度蒸发用于道路分割**

- 论文：https://arxiv.org/abs/2004.05304
- 代码：https://github.com/cardwing/Codes-for-IntRA-KD

<a name="HOI"></a>

# "人-物"交互(HOT)检测

**PPDM: Parallel Point Detection and Matching for Real-time Human-Object Interaction Detection**
**PPDM：并行点检测与匹配实时人体对象交互检测**

- 论文：https://arxiv.org/abs/1912.12898
- 代码：https://github.com/YueLiao/PPDM

**Detailed 2D-3D Joint Representation for Human-Object Interaction**
**详细的人机交互2D-3D关节表示**

- 论文：https://arxiv.org/abs/2004.08154

- 代码：https://github.com/DirtyHarryLYL/DJ-RN

**Cascaded Human-Object Interaction Recognition**
**级联的人机交互识别**

- 论文：https://arxiv.org/abs/2003.04262

- 代码：https://github.com/tfzhou/C-HOI

**VSGNet: Spatial Attention Network for Detecting Human Object Interactions Using Graph Convolutions**
**VSGNet：用于检测人类对象交互的时空注意力网络**

- 论文：https://arxiv.org/abs/2003.05541
- 代码：https://github.com/ASMIftekhar/VSGNet

<a name="TP"></a>

# 轨迹预测

**The Garden of Forking Paths: Towards Multi-Future Trajectory Prediction**
**叉路花园：多未来轨迹预测**

- 论文：https://arxiv.org/abs/1912.06445
- 代码：https://github.com/JunweiLiang/Multiverse
- 数据集：https://next.cs.cmu.edu/multiverse/

**Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction**
**社会-STGCNN：用于人类轨迹预测的社会时空图卷积神经网络**

- 论文：https://arxiv.org/abs/2002.11927 
- 代码：https://github.com/abduallahmohamed/Social-STGCNN 

<a name="Motion-Predication"></a>

# 运动预测

**Collaborative Motion Prediction via Neural Motion Message Passing**
**通过神经运动消息传递实现协作运动预测**

- 论文：https://arxiv.org/abs/2003.06594
- 代码：https://github.com/PhyllisH/NMMP

**MotionNet: Joint Perception and Motion Prediction for Autonomous Driving Based on Bird's Eye View Maps**
**MotionNet：基于鸟瞰图的自动驾驶感知与运动预测**

- 论文：https://arxiv.org/abs/2003.06754

- 代码：https://github.com/pxiangwu/MotionNet

<a name="OF"></a>

# 光流估计

**Learning by Analogy: Reliable Supervision from Transformations for Unsupervised Optical Flow Estimation**
**学习通过类比：从变换中可靠的监督用于无监督光学流估计**

- 论文：https://arxiv.org/abs/2003.13045
- 代码：https://github.com/lliuz/ARFlow 

<a name="IR"></a>

# 图像检索

**Evade Deep Image Retrieval by Stashing Private Images in the Hash Space**
**通过将私人图像存储在哈希空间中来逃避深度图像检索**

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/html/Xiao_Evade_Deep_Image_Retrieval_by_Stashing_Private_Images_in_the_CVPR_2020_paper.html
- 代码：https://github.com/sugarruy/hashstash

<a name="Virtual-Try-On"></a>

# 虚拟试衣

**Towards Photo-Realistic Virtual Try-On by Adaptively Generating↔Preserving Image Content**
**朝着照片真实的虚拟试穿，通过自适应生成和保留图像内容**

- 论文：https://arxiv.org/abs/2003.05863
- 代码：https://github.com/switchablenorms/DeepFashion_Try_On

<a name="HDR"></a>

# HDR

**Single-Image HDR Reconstruction by Learning to Reverse the Camera Pipeline**
**单张图像超分辨率重建通过学习反向相机管道**

- 主页：https://www.cmlab.csie.ntu.edu.tw/~yulunliu/SingleHDR

- 论文下载链接：https://www.cmlab.csie.ntu.edu.tw/~yulunliu/SingleHDR_/00942.pdf

- 代码：https://github.com/alex04072000/SingleHDR

<a name="AE"></a>

# 对抗样本

**Enhancing Cross-Task Black-Box Transferability of Adversarial Examples With Dispersion Reduction**
**增强对抗性样本的跨任务传递性**

- 论文：https://openaccess.thecvf.com/content_CVPR_2020/papers/Lu_Enhancing_Cross-Task_Black-Box_Transferability_of_Adversarial_Examples_With_Dispersion_Reduction_CVPR_2020_paper.pdf
- 代码：https://github.com/erbloo/dr_cvpr20 

**Towards Large yet Imperceptible Adversarial Image Perturbations with Perceptual Color Distance**
**朝着大规模而难以察觉的对抗性图像扰动，感知颜色距离**

- 论文：https://arxiv.org/abs/1911.02466
- 代码：https://github.com/ZhengyuZhao/PerC-Adversarial 

<a name="3D-Reconstructing"></a>

# 三维重建

**Unsupervised Learning of Probably Symmetric Deformable 3D Objects from Images in the Wild**
**来自野外图像中可能对称变形的三维对象的未监督学习**

- **CVPR 2020 Best Paper**
- 主页：https://elliottwu.com/projects/unsup3d/
- 论文：https://arxiv.org/abs/1911.11130
- 代码：https://github.com/elliottwu/unsup3d

**Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization**
**多层像素对齐隐式函数用于高分辨率3D人体数字化**

- 主页：https://shunsukesaito.github.io/PIFuHD/
- 论文：https://arxiv.org/abs/2004.00452
- 代码：https://github.com/facebookresearch/pifuhd

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/papers/Patel_TailorNet_Predicting_Clothing_in_3D_as_a_Function_of_Human_CVPR_2020_paper.pdf
- 代码：https://github.com/chaitanya100100/TailorNet
- 数据集：https://github.com/zycliao/TailorNet_dataset

**Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion**
**隐式函数在三维形状重建和补全特征空间中的作用**

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/papers/Chibane_Implicit_Functions_in_Feature_Space_for_3D_Shape_Reconstruction_and_CVPR_2020_paper.pdf
- 代码：https://github.com/jchibane/if-net

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/papers/Mir_Learning_to_Transfer_Texture_From_Clothing_Images_to_3D_Humans_CVPR_2020_paper.pdf
- 代码：https://github.com/aymenmir1/pix2surf

<a name="DC"></a>

# 深度补全

**Uncertainty-Aware CNNs for Depth Completion: Uncertainty from Beginning to End**
**不确定性感知的卷积神经网络用于深度补全：从始至终的不确定性**

论文：https://arxiv.org/abs/2006.03349

代码：https://github.com/abdo-eldesokey/pncnn

<a name="SSC"></a>

# 语义场景补全

**3D Sketch-aware Semantic Scene Completion via Semi-supervised Structure Prior**
**通过半监督结构先验的3D Sketch-aware语义场景补全**

- 论文：https://arxiv.org/abs/2003.14052
- 代码：https://github.com/charlesCXK/TorchSSC

<a name="Captioning"></a>

# 图像/视频描述

**Syntax-Aware Action Targeting for Video Captioning**
**语法感知动作目标识别用于视频字幕**

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/papers/Zheng_Syntax-Aware_Action_Targeting_for_Video_Captioning_CVPR_2020_paper.pdf
- 代码：https://github.com/SydCaption/SAAT 

<a name="WP"></a>

# 线框解析

**Holistically-Attracted Wireframe Parser**
**整体吸引线框解析器**

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/html/Xue_Holistically-Attracted_Wireframe_Parsing_CVPR_2020_paper.html

- 代码：https://github.com/cherubicXN/hawp

<a name="Datasets"></a>

# 数据集

**OASIS: A Large-Scale Dataset for Single Image 3D in the Wild**
**OASIS：一个大规模的野外单张图像3D数据集**

- 论文：https://arxiv.org/abs/2007.13215
- 数据集：https://oasis.cs.princeton.edu/

**STEFANN: Scene Text Editor using Font Adaptive Neural Network**
**STEFANN：使用自适应神经网络的字体适应场景文本编辑器**

- 主页：https://prasunroy.github.io/stefann/

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/html/Roy_STEFANN_Scene_Text_Editor_Using_Font_Adaptive_Neural_Network_CVPR_2020_paper.html
- 代码：https://github.com/prasunroy/stefann
- 数据集：https://drive.google.com/open?id=1sEDiX_jORh2X-HSzUnjIyZr-G9LJIw1k

**Interactive Object Segmentation with Inside-Outside Guidance**
**交互式物体分割与内部-外部引导**

- 论文下载链接：http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Interactive_Object_Segmentation_With_Inside-Outside_Guidance_CVPR_2020_paper.pdf
- 代码：https://github.com/shiyinzhang/Inside-Outside-Guidance
- 数据集：https://github.com/shiyinzhang/Pixel-ImageNet

**Video Panoptic Segmentation**
**视频全景分割**

- 论文：https://arxiv.org/abs/2006.11339
- 代码：https://github.com/mcahny/vps
- 数据集：https://www.dropbox.com/s/ecem4kq0fdkver4/cityscapes-vps-dataset-1.0.zip?dl=0

**FSS-1000: A 1000-Class Dataset for Few-Shot Segmentation**
**FSS-1000：为少量样本分割提供了一个1000类别的数据集**

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/html/Li_FSS-1000_A_1000-Class_Dataset_for_Few-Shot_Segmentation_CVPR_2020_paper.html

- 代码：https://github.com/HKUSTCV/FSS-1000

- 数据集：https://github.com/HKUSTCV/FSS-1000

**3D-ZeF: A 3D Zebrafish Tracking Benchmark Dataset**
**3D-ZeF：3D斑马鱼追踪基准数据集**

- 主页：https://vap.aau.dk/3d-zef/
- 论文：https://arxiv.org/abs/2006.08466
- 代码：https://bitbucket.org/aauvap/3d-zef/src/master/
- 数据集：https://motchallenge.net/data/3D-ZeF20

**TailorNet: Predicting Clothing in 3D as a Function of Human Pose, Shape and Garment Style**
**TailorNet：预测衣物在3D中的形状和风格，作为人体姿态、形状和衣物风格的函数**

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/papers/Patel_TailorNet_Predicting_Clothing_in_3D_as_a_Function_of_Human_CVPR_2020_paper.pdf
- 代码：https://github.com/chaitanya100100/TailorNet
- 数据集：https://github.com/zycliao/TailorNet_dataset

**Oops! Predicting Unintentional Action in Video**
**哎呀！预测视频中的无意动作**

- 主页：https://oops.cs.columbia.edu/

- 论文：https://arxiv.org/abs/1911.11206
- 代码：https://github.com/cvlab-columbia/oops
- 数据集：https://oops.cs.columbia.edu/data

**The Garden of Forking Paths: Towards Multi-Future Trajectory Prediction**
**叉路花园：迈向多未来轨迹预测**

- 论文：https://arxiv.org/abs/1912.06445
- 代码：https://github.com/JunweiLiang/Multiverse
- 数据集：https://next.cs.cmu.edu/multiverse/

**Open Compound Domain Adaptation**
**开放复合域自适应**

- 主页：https://liuziwei7.github.io/projects/CompoundDomain.html
- 数据集：https://drive.google.com/drive/folders/1_uNTF8RdvhS_sqVTnYx17hEOQpefmE2r?usp=sharing
- 论文：https://arxiv.org/abs/1909.03403
- 代码：https://github.com/zhmiao/OpenCompoundDomainAdaptation-OCDA

**Intra- and Inter-Action Understanding via Temporal Action Parsing**
**内联和交互动作理解通过时间动作解析**

- 论文：https://arxiv.org/abs/2005.10229
- 主页和数据集：https://sdolivia.github.io/TAPOS/

**Dynamic Refinement Network for Oriented and Densely Packed Object Detection**
**用于定向和密集包装对象检测的动态细化网络**

- 论文下载链接：https://arxiv.org/abs/2005.09973

- 代码和数据集：https://github.com/Anymake/DRN_CVPR2020

**COCAS: A Large-Scale Clothes Changing Person Dataset for Re-identification**
**COCAS：一个大规模衣物更换人员数据集，用于重新识别**

- 论文：https://arxiv.org/abs/2005.07862

- 数据集：暂无

**KeypointNet: A Large-scale 3D Keypoint Dataset Aggregated from Numerous Human Annotations**
**KeypointNet：一个从多个人类标注中汇集的大规模3D关键点数据集**

- 论文：https://arxiv.org/abs/2002.12687

- 数据集：https://github.com/qq456cvb/KeypointNet

**MSeg: A Composite Dataset for Multi-domain Semantic Segmentation**
**MSeg: 一个用于多领域语义分割的复合数据集**

- 论文：http://vladlen.info/papers/MSeg.pdf
- 代码：https://github.com/mseg-dataset/mseg-api
- 数据集：https://github.com/mseg-dataset/mseg-semantic

**AvatarMe: Realistically Renderable 3D Facial Reconstruction "in-the-wild"**
**AvatarMe：实际可感的3D面部重建“野外”**

- 论文：https://arxiv.org/abs/2003.13845
- 数据集：https://github.com/lattas/AvatarMe

**Learning to Autofocus**
**学会自聚焦**

- 论文：https://arxiv.org/abs/2004.12260
- 数据集：暂无

**FaceScape: a Large-scale High Quality 3D Face Dataset and Detailed Riggable 3D Face Prediction**
**FaceScape：大规模高质量3D面部数据集和详细可调节3D面部预测**

- 论文：https://arxiv.org/abs/2003.13989
- 代码：https://github.com/zhuhao-nju/facescape

**Bodies at Rest: 3D Human Pose and Shape Estimation from a Pressure Image using Synthetic Data**
**休息着的身体：从压力图像中使用合成数据估算3D人体姿态和形状**

- 论文下载链接：https://arxiv.org/abs/2004.01166

- 代码：https://github.com/Healthcare-Robotics/bodies-at-rest
- 数据集：https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KOA4ML

**FineGym: A Hierarchical Video Dataset for Fine-grained Action Understanding**
**细粒度动作理解的大规模视频数据集**

- 主页：https://sdolivia.github.io/FineGym/
- 论文：https://arxiv.org/abs/2004.06704

**A Local-to-Global Approach to Multi-modal Movie Scene Segmentation**
**一种从局部到全球的多模态电影场景分割方法**

- 主页：https://anyirao.com/projects/SceneSeg.html

- 论文下载链接：https://arxiv.org/abs/2004.02678

- 代码：https://github.com/AnyiRao/SceneSeg

**Deep Homography Estimation for Dynamic Scenes**
**深度仿射估计用于动态场景**

- 论文：https://arxiv.org/abs/2004.02132

- 数据集：https://github.com/lcmhoang/hmg-dynamics

**Assessing Image Quality Issues for Real-World Problems**
**评估真实世界问题中的图像质量问题**

- 主页：https://vizwiz.org/tasks-and-datasets/image-quality-issues/
- 论文：https://arxiv.org/abs/2003.12511

**UnrealText: Synthesizing Realistic Scene Text Images from the Unreal World**
**UnrealText：从Unreal世界中合成真实场景文本图像**

- 论文：https://arxiv.org/abs/2003.10608
- 代码和数据集：https://github.com/Jyouhou/UnrealText/

**PANDA: A Gigapixel-level Human-centric Video Dataset**
**PANDA：一个拥有千万像素级人类中心化视频数据集**

- 论文：https://arxiv.org/abs/2003.04852

- 数据集：http://www.panda-dataset.com/

**IntrA: 3D Intracranial Aneurysm Dataset for Deep Learning**
**IntrA：用于深度学习的颅内动脉瘤3D数据集**

- 论文：https://arxiv.org/abs/2003.02920
- 数据集：https://github.com/intra3d2019/IntrA

**Cross-View Tracking for Multi-Human 3D Pose Estimation at over 100 FPS**
**跨视角跟踪在超过100帧每秒的条件下用于多人体3D姿势估计**

- 论文：https://arxiv.org/abs/2003.03972
- 数据集：暂无

<a name="Others"></a>

# 其他

**CONSAC: Robust Multi-Model Fitting by Conditional Sample Consensus**
**CONSAC：通过条件样本共识的鲁棒多模型拟合**

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/html/Kluger_CONSAC_Robust_Multi-Model_Fitting_by_Conditional_Sample_Consensus_CVPR_2020_paper.html
- 代码：https://github.com/fkluger/consac

**Learning to Learn Single Domain Generalization**
**学习如何学习单一领域的推广**

- 论文：https://arxiv.org/abs/2003.13216
- 代码：https://github.com/joffery/M-ADA

**Open Compound Domain Adaptation**
**开放复合域自适应**

- 主页：https://liuziwei7.github.io/projects/CompoundDomain.html
- 数据集：https://drive.google.com/drive/folders/1_uNTF8RdvhS_sqVTnYx17hEOQpefmE2r?usp=sharing
- 论文：https://arxiv.org/abs/1909.03403
- 代码：https://github.com/zhmiao/OpenCompoundDomainAdaptation-OCDA

**Differentiable Volumetric Rendering: Learning Implicit 3D Representations without 3D Supervision**
**可微体积渲染：无3D监督学习隐式表示的3D表示**

- 论文：http://www.cvlibs.net/publications/Niemeyer2020CVPR.pdf

- 代码：https://github.com/autonomousvision/differentiable_volumetric_rendering

**QEBA: Query-Efficient Boundary-Based Blackbox Attack**
**基于查询效率的边界攻击查询**

- 论文：https://arxiv.org/abs/2005.14137
- 代码：https://github.com/AI-secure/QEBA

**Equalization Loss for Long-Tailed Object Recognition**
**平衡损失，用于识别长尾目标**

- 论文：https://arxiv.org/abs/2003.05176
- 代码：https://github.com/tztztztztz/eql.detectron2

**Instance-aware Image Colorization**
**实例感知图像颜色化**

- 主页：https://ericsujw.github.io/InstColorization/
- 论文：https://arxiv.org/abs/2005.10825
- 代码：https://github.com/ericsujw/InstColorization

**Contextual Residual Aggregation for Ultra High-Resolution Image Inpainting**
**针对超高清图像修复的上下文残差聚合**

- 论文：https://arxiv.org/abs/2005.09704

- 代码：https://github.com/Atlas200dk/sample-imageinpainting-HiFill

**Where am I looking at? Joint Location and Orientation Estimation by Cross-View Matching**
**我在看着什么？跨视图匹配的联合位置和方向估计**

- 论文：https://arxiv.org/abs/2005.03860
- 代码：https://github.com/shiyujiao/cross_view_localization_DSM

**Epipolar Transformers**
**极性变换器**

- 论文：https://arxiv.org/abs/2005.04551

- 代码：https://github.com/yihui-he/epipolar-transformers 

**Bringing Old Photos Back to Life**
**把老照片唤醒**

- 主页：http://raywzy.com/Old_Photo/
- 论文：https://arxiv.org/abs/2004.09484

**MaskFlownet: Asymmetric Feature Matching with Learnable Occlusion Mask**
**maskflownet：可学习遮挡掩膜的不对称特征匹配**

- 论文：https://arxiv.org/abs/2003.10955 

- 代码：https://github.com/microsoft/MaskFlownet 

**Self-Supervised Viewpoint Learning from Image Collections**
**图像集合上的自监督观点学习**

- 论文：https://arxiv.org/abs/2004.01793
- 论文2：https://research.nvidia.com/sites/default/files/pubs/2020-03_Self-Supervised-Viewpoint-Learning/SSV-CVPR2020.pdf 
- 代码：https://github.com/NVlabs/SSV 

**Towards Discriminability and Diversity: Batch Nuclear-norm Maximization under Label Insufficient Situations**
**朝着可区分性和多样性：在标签不足情况下的批归一化核最大值**

- Oral

- 论文：https://arxiv.org/abs/2003.12237 
- 代码：https://github.com/cuishuhao/BNM 

**Towards Learning Structure via Consensus for Face Segmentation and Parsing**
**面向通过共识进行人脸分割和解析的学习结构**

- 论文：https://arxiv.org/abs/1911.00957
- 代码：https://github.com/isi-vista/structure_via_consensus

**Plug-and-Play Algorithms for Large-scale Snapshot Compressive Imaging**
**用于大规模快照压缩成像的插值和播放算法**

- Oral
- 论文：https://arxiv.org/abs/2003.13654

- 代码：https://github.com/liuyang12/PnP-SCI

**Lightweight Photometric Stereo for Facial Details Recovery**
**轻质光度立体恢复面部细节**

- 论文：https://arxiv.org/abs/2003.12307
- 代码：https://github.com/Juyong/FacePSNet

**Footprints and Free Space from a Single Color Image**
**单色图像的足迹和空隙**

- 论文：https://arxiv.org/abs/2004.06376

- 代码：https://github.com/nianticlabs/footprints

**Self-Supervised Monocular Scene Flow Estimation**
**单目场景流估计的自我监督**

- 论文：https://arxiv.org/abs/2004.04143
- 代码：https://github.com/visinf/self-mono-sf

**Quasi-Newton Solver for Robust Non-Rigid Registration**
**准-Newton 求解器：用于稳健非刚性配准**

- 论文：https://arxiv.org/abs/2004.04322
- 代码：https://github.com/Juyong/Fast_RNRR

**A Local-to-Global Approach to Multi-modal Movie Scene Segmentation**
**一种从局部到全球的多模态电影场景分割方法**

- 主页：https://anyirao.com/projects/SceneSeg.html

- 论文下载链接：https://arxiv.org/abs/2004.02678

- 代码：https://github.com/AnyiRao/SceneSeg

**DeepFLASH: An Efficient Network for Learning-based Medical Image Registration**
**DeepFLASH：一种基于学习的医学图像配准高效网络**

- 论文：https://arxiv.org/abs/2004.02097

- 代码：https://github.com/jw4hv/deepflash

**Self-Supervised Scene De-occlusion**
**自监督场景遮挡**

- 主页：https://xiaohangzhan.github.io/projects/deocclusion/
- 论文：https://arxiv.org/abs/2004.02788
- 代码：https://github.com/XiaohangZhan/deocclusion

**Polarized Reflection Removal with Perfect Alignment in the Wild** 

- 主页：https://leichenyang.weebly.com/project-polarized.html
- 代码：https://github.com/ChenyangLEI/CVPR2020-Polarized-Reflection-Removal-with-Perfect-Alignment 

**Background Matting: The World is Your Green Screen**
**背景渲染：世界是你的绿色背景**

- 论文：https://arxiv.org/abs/2004.00626
- 代码：http://github.com/senguptaumd/Background-Matting

**What Deep CNNs Benefit from Global Covariance Pooling: An Optimization Perspective**
**深度卷积神经网络从全局协方差池化中受益：一种优化视角**

- 论文：https://arxiv.org/abs/2003.11241

- 代码：https://github.com/ZhangLi-CS/GCP_Optimization

**Look-into-Object: Self-supervised Structure Modeling for Object Recognition**
**Look-into-Object：自监督结构建模对象识别**

- 论文：暂无
- 代码：https://github.com/JDAI-CV/LIO 

 **Video Object Grounding using Semantic Roles in Language Description**

- 论文：https://arxiv.org/abs/2003.10606
- 代码：https://github.com/TheShadow29/vognet-pytorch 

**Dynamic Hierarchical Mimicking Towards Consistent Optimization Objectives**
**动态层次模仿以实现一致优化目标**

- 论文：https://arxiv.org/abs/2003.10739
- 代码：https://github.com/d-li14/DHM 

**SDFDiff: Differentiable Rendering of Signed Distance Fields for 3D Shape Optimization**
**SDFDiff：用于3D形状优化的高质量符号距离场的可微渲染**

- 论文：http://www.cs.umd.edu/~yuejiang/papers/SDFDiff.pdf
- 代码：https://github.com/YueJiang-nj/CVPR2020-SDFDiff 

**On Translation Invariance in CNNs: Convolutional Layers can Exploit Absolute Spatial Location**
**在CNNs中的翻译不变性：卷积层可以利用绝对空间位置**

- 论文：https://arxiv.org/abs/2003.07064

- 代码：https://github.com/oskyhn/CNNs-Without-Borders

**GhostNet: More Features from Cheap Operations**
**鬼影网络：更多的低价操作功能**

- 论文：https://arxiv.org/abs/1911.11907

- 代码：https://github.com/iamhankai/ghostnet

**AdderNet: Do We Really Need Multiplications in Deep Learning?** 

- 论文：https://arxiv.org/abs/1912.13200 
- 代码：https://github.com/huawei-noah/AdderNet

**Deep Image Harmonization via Domain Verification** 

- 论文：https://arxiv.org/abs/1911.13239 
- 代码：https://github.com/bcmi/Image_Harmonization_Datasets

**Blurry Video Frame Interpolation**
**模糊视频帧插值**

- 论文：https://arxiv.org/abs/2002.12259 
- 代码：https://github.com/laomao0/BIN

**Extremely Dense Point Correspondences using a Learned Feature Descriptor**
**极密点对应使用学习特征描述符**

- 论文：https://arxiv.org/abs/2003.00619 
- 代码：https://github.com/lppllppl920/DenseDescriptorLearning-Pytorch

**Filter Grafting for Deep Neural Networks**
**深度神经网络的过滤融合**

- 论文：https://arxiv.org/abs/2001.05868
- 代码：https://github.com/fxmeng/filter-grafting
- 论文解读：https://www.zhihu.com/question/372070853/answer/1041569335

**Action Segmentation with Joint Self-Supervised Temporal Domain Adaptation**
**联合自监督时空领域自适应动作分割**

- 论文：https://arxiv.org/abs/2003.02824 
- 代码：https://github.com/cmhungsteve/SSTDA

**Detecting Attended Visual Targets in Video**
**检测视频中的出席视觉目标**

- 论文：https://arxiv.org/abs/2003.02501 

- 代码：https://github.com/ejcgt/attention-target-detection

**Deep Image Spatial Transformation for Person Image Generation**
**深度图像空间变换用于人物图像生成**

- 论文：https://arxiv.org/abs/2003.00696 
- 代码：https://github.com/RenYurui/Global-Flow-Local-Attention

 **Rethinking Zero-shot Video Classification: End-to-end Training for Realistic Applications** 

- 论文：https://arxiv.org/abs/2003.01455
- 代码：https://github.com/bbrattoli/ZeroShotVideoClassification

https://github.com/charlesCXK/3D-SketchAware-SSC

https://github.com/Anonymous20192020/Anonymous_CVPR5767

https://github.com/avirambh/ScopeFlow

https://github.com/csbhr/CDVD-TSP

https://github.com/ymcidence/TBH

https://github.com/yaoyao-liu/mnemonics

https://github.com/meder411/Tangent-Images

https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch

https://github.com/sjmoran/deep_local_parametric_filters

https://github.com/charlesCXK/3D-SketchAware-SSC

https://github.com/bermanmaxim/AOWS

https://github.com/dc3ea9f/look-into-object 

<a name="Not-Sure"></a>

# 不确定中没中

**FADNet: A Fast and Accurate Network for Disparity Estimation**
**FADNet：用于差异估计的快速且准确网络**

- 论文：还没出来
- 代码：https://github.com/HKBU-HPML/FADNet

https://github.com/rFID-submit/RandomFID：不确定中没中

https://github.com/JackSyu/AE-MSR：不确定中没中

https://github.com/fastconvnets/cvpr2020：不确定中没中

https://github.com/aimagelab/meshed-memory-transformer：不确定中没中

https://github.com/TWSFar/CRGNet：不确定中没中

https://github.com/CVPR-2020/CDARTS：不确定中没中

https://github.com/anucvml/ddn-cvprw2020：不确定中没中

https://github.com/dl-model-recommend/model-trust：不确定中没中

https://github.com/apratimbhattacharyya18/CVPR-2020-Corr-Prior：不确定中没中

https://github.com/onetcvpr/O-Net：不确定中没中

https://github.com/502463708/Microcalcification_Detection：不确定中没中

https://github.com/anonymous-for-review/cvpr-2020-deep-smoke-machine：不确定中没中

https://github.com/anonymous-for-review/cvpr-2020-smoke-recognition-dataset：不确定中没中

https://github.com/cvpr-nonrigid/dataset：不确定中没中

https://github.com/theFool32/PPBA：不确定中没中

https://github.com/Realtime-Action-Recognition/Realtime-Action-Recognition
