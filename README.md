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


### Acknowledgement
thank [PoinTramba](https://github.com/xiaoyao3302/PoinTramba),[PointMamba](https://github.com/LMD0311/PointMamba), [PointCloudMamba](https://github.com/SkyworkAI/PointCloudMamba) [PointBERT](https://github.com/lulutang0608/Point-BERT), [Mamba](https://github.com/state-spaces/mamba) and other relevant works for their amazing open-sourced projects!

