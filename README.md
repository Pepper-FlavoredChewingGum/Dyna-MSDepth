# Dyna-MSDepth: Multi-scale self-supervised monocular depth estimation network for
visual simultaneous localization and mapping in dynamic scenes



Qualatative depth estimation results: TUM, BONN, KITTI, DDAD

   ![image-20230521160102935](C:\Users\李英朝\AppData\Roaming\Typora\typora-user-images\image-20230521160102935.png)



## Install
```
conda create -n depth_env python=3.8
conda activate depth_env
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Dataset

We organize the video datasets into the following format for training and testing models:

    Dataset
      -Training
        --Scene0000
          ---*.jpg (list of color images)
          ---cam.txt (3x3 camera intrinsic matrix)
          ---depth (a folder containing ground-truth depth maps, optional for validation)
          ---leres_depth (a folder containing psuedo-depth generated by LeReS, it is required for training SC-DepthV3)
        --Scene0001
        ...
        train.txt (containing training scene names)
        val.txt (containing validation scene names)
      -Testing
        --color (containg testing images)
        --depth (containg ground-truth depths)
        --seg_mask (containing semantic segmentation masks for depth evaluation on dynamic/static regions)

Pre-processed datasets:

[**[kitti, ddad, bonn, tum]**](https://1drv.ms/u/s!AiV6XqkxJHE2mUFwH6FrHGCuh_y6?e=RxOheF) 


## Training

```bash
python train.py --config $CONFIG --dataset_dir $DATASET
```
### No GT depth for validation
Add "--val_mode photo" in the training script or the configure file, which uses the photometric loss for validation. 
```bash
python train.py --config $CONFIG --dataset_dir $DATASET --val_mode photo
```


## Testing (Evaluation on Full Images)

    python test.py --config $CONFIG --dataset_dir $DATASET --ckpt_path $CKPT


## Evaluation on dynamic/static regions

### Inference
```bash
python inference.py --config $YOUR_CONFIG \
--input_dir $TESTING_IMAGE_FOLDER \
--output_dir $RESULTS_FOLDER \
--ckpt_path $YOUR_CKPT \
--save-vis --save-depth
```

### Evaluation
```bash
python eval_depth.py \
--dataset $DATASET_FOLDER \
--pred_depth=$RESULTS_FOLDER \
--gt_depth=$GT_FOLDER \
--seg_mask=$SEG_MASK_FOLDER
```


## References

**Unsupervised Scale-consistent Depth Learning from Video (IJCV 2021)** 

**Unsupervised Scale-consistent Depth and Ego-motion Learning from Monocular Video (NeurIPS 2019)**

**Auto-Rectify Network for Unsupervised Indoor Depth Estimation (TPAMI 2022)** 

**SC-DepthV3: Robust Self-supervised Monocular Depth Estimation for Dynamic Scenes (ArXiv 2022)** 