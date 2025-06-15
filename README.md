### DataSet
现在版本已经下载好了数据集

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
