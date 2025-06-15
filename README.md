## Coding_optimized_PoinTramba From scratch!!
The public version is a simple visible structue of my codes , only offers my minds and style . The Final Experiments is still cotinuing...   
该公开版本项目仅展示了本人的思路,代码流程及风格,最终实验尚未结束...(痛苦.jpg)

### PoinTrambaUltra   
PointTrambaUltra : A moderate-depth hybrid point cloud backbone that balances structured local encoding (EdgeConv), sequence modeling (Mamba), and self-attention-based global fusion (Transformer).   
一个平衡结构化局部编码（EdgeConv）、序列建模（Mamba）和基于自注意力的全局融合（Transformer）的中深度混合点云主干。   


### DataSet  
#### ScanObjNN
```
wget http://hkust-vgd.ust.hk/scanobjectnn/h5_files.zip
```

#### ModelNet40
```
wget https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
```

#### ShapeNetPart
```
wget https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
```
Please create a data file and modify the dataset path in configuration files (dataset_configs). 
```
├── data
    ├── ScanObjNN
        ├── h5_files
            ├── main_split
            └── ....
    ├── ModeNet40
        └── modelnet40_ply_hdf5_2048
            ├── ply_data_train0.h5
            └── ....
    └── ShapeNetPart
        └── shapenetcore_partanno_segmentation_benchmark_v0_normal
            ├── 02691156
            └── ....


```

因为 ShapeNetPart原始数据是：
超多文件夹（02691156, 02958343, …）
每个类文件夹里是很多很多 .txt 文件（一个 .txt 就是一个 shape的点云数据）
但官方没有直接提供好 train/test 划分！
所以需要运行脚本：
```bash
scripts/generate_shapenet_split.py
```
遍历所有这些 .txt
随机打乱
按 85%/15% 划分为 train_split.txt 和 test_split.txt
保存到 data/ShapeNetPart/ 目录下

### Loss
在预训练中，一开始使用最最最最简单的loss
但是要达到高水平paper要求，显然不能这样

很多大模型自监督里（比如SimMIM, MaskPoint, Point-BERT...），
他们一般是：

总loss = α × Mask Loss + β × 额外辅助Loss

比如可以引入：

对比学习loss：比如Contrastive Loss，让不同样本更区分开

Consistency Loss：比如输入扰动后的点云，特征提取后仍然一致

组内Token自监督：mask的是group center，但组内每个token本身也自监督预测局部patch

用importance打分作为mask策略（比如更加难的区域优先mask）



Q：PoinTramba自监督预训练的损失是啥？
他们是用的 重建中心点坐标的L2回归损失（MSE Loss）。

具体就是：

取一部分子group（通过mask）不送进网络

要求网络 根据可见的groups推测被mask掉groups的中心点坐标。

损失 = 预测中心点 vs 真实中心点 的 均方误差（MSE Loss）。

总结一下：


论文	自监督预训练损失（mask-based）
PoinTramba	MSE(预测中心点, 真中心点)
Point-MAE	MSE(重建点云 patch)
MaskPoint	MSE(重建mask patch)
Point-BERT	CrossEntropy（预测mask token类别）
所以 PoinTramba确实没有搞复杂的importance-aware mask loss，
他们只是：mask group ➔ 重建 center ➔ 计算MSE loss。

idea:
把 Chamfer Distance 作为 标准版预训练 Loss
（跟 PoinTramba 那套一样，接 extensions 里的 CUDA chamfer）2. 同时保留咱自己设计的 Importance-Aware Masking Loss（创新思路消融）3. 用一个 pretrain.yaml 中的 flag（比如 loss_type: "chamfer" 或 "importance"）来控制。

#### *呃呃但是我还是得说一下这个版本我没有改预训练进去，因为 run model from scratch的效果在当时跑的并不好，包括如果您复现本项目会发现acc/miou并不理想*


### Structure optimized
创新主干架构
    DGCNN+Transformer1(强局部建模"组内")->Mamba(半局部半全局'局部+中程'建模"组间")->Transformer2(全局聚合 + 位置信息补全)  
    ｜ 相关文献:  
    ｜ 1.Perceiver: General Perception with Iterative Attention(2021) - https://arxiv.org/abs/2103.03206 - 其核心结构:Cross-Attention + Mixer + Attention(动机:多模态融合后强化结构信息)  
    ｜ 2.UniFormer: Unifying Convolution and Self-attention for Visual Recognition(2023) - https://arxiv.org/abs/2201.09450 - 其核心结构:Local Transformer + MLP + Global Transformer(动机:局部建模 → 全局融合 → 局部细化)  
    ｜ 3.ViTAEv2: Vision Transformer Advanced by Exploring Inductive Bias for Image Recognition and Beyond(2022) - https://arxiv.org/abs/2202.10108 - 其核心结构:CNN + Transformer + Transformer(动机:先局部建模，再全局聚合)  
    ｜ 4.PointMamba: A Simple State Space Model for Point Cloud Analysis(2024) - https://arxiv.org/abs/2402.10739 - (说明了 Mamba 对于结构化局部特征提取和全局建模更高效、收敛更快;支持结构排序（如 Farthest Point Sampling / Importance Ranking）后送入序列模型;其中关于 结构引导 + Mamba 编码 的描述，与我们目前使用 BIO → Mamba → Transformer 是一脉相承的)  
    ｜ 5.PoinTramba(2024)  
    ｜ 6.PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies(2022) - https://arxiv.org/abs/2206.04670 -(虽然不直接用穿插结构，但强调“轻量级结构 → 深层细化”的有效性)  
    Transformer1 建立基础 group 语义关系 -> 用 BIO 序列送入 Mamba 模块 -> 将 Mamba 输出的embedding送入transformer2 -> 得到的结果与transformer1得到的结果做BiCrossFusion   

### Bicross-Fusion
思路支持文献:Co-Scale Conv-Attentional Image Transformers - Co-Scale Conv-Attentional Image Transformers(2021) - 使用多分支特征交叉融合进行结构增强  
通过交叉注意力机制计算来自 x1 和 x2 的相互依赖关系，并融合为新的特征表示，继承了 x1 的结构信息，又引入了 x2 的全局建模能力。  


### Star
If you find these projects useful, please consider give me a Star ~~~ :D

### Acknowledgement
thank [PoinTramba](https://github.com/xiaoyao3302/PoinTramba),[PointMamba](https://github.com/LMD0311/PointMamba), [PointCloudMamba](https://github.com/SkyworkAI/PointCloudMamba) [PointBERT](https://github.com/lulutang0608/Point-BERT), [Mamba](https://github.com/state-spaces/mamba) and other relevant works for their amazing open-sourced projects!

