## Coding_optimized_PoinTramba From scratch!!
The public version is a simple visible structue of the code. The Final Experiments is still cotinuing...   

### PoinTrambaUltra   
PointTrambaUltra : A moderate-depth hybrid point cloud backbone that balances structured local encoding (EdgeConv), sequence modeling (Mamba), and self-attention-based global fusion (Transformer).   


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




### Acknowledgement
thank [PoinTramba](https://github.com/xiaoyao3302/PoinTramba),[PointMamba](https://github.com/LMD0311/PointMamba), [PointCloudMamba](https://github.com/SkyworkAI/PointCloudMamba) [PointBERT](https://github.com/lulutang0608/Point-BERT), [Mamba](https://github.com/state-spaces/mamba) and other relevant works for their amazing open-sourced projects!

