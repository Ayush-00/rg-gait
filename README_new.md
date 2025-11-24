# ResGait: Residual Correction for Occluded Gait Recognition with Holistic Retention

This is the code for "ResGait: Residual Correction for Occluded Gait Recognition with Holistic Retention".

Our code is based on the OpenGait Repository (https://github.com/ShiqiYu/OpenGait). Please follow the instructions in the repository to set up the environment and the datasets.



### Training

Stage 1: For training the residual models, an occlusion evaluation module (OEM) is required. This is similar to the occlusion detector in MimicGait (https://github.com/Ayush-00/mimicgait). The OEM should be trained and the path specified in 'occ_detector_path' in the config file. 

Stage 2: For training the Gait Signature Extractor (GSE), follow the instructions in the OpenGait repository to train the original backbones. Specifically, use a command like: 
    
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 opengait/main.py --cfgs ./configs/gaitbase/gaitbase_grew.yaml --phase train
```
This will train the Gait Signature Extractor (GSE) with the specified backbone. The config file should be adjusted according to the model/dataset you want to train on.

Stage 3: Finally, to train the Residual models, use the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 opengait/main.py --cfgs ./configs/gaitbase/gaitbase_grew_residual.yaml --phase train
```

Adjust the config file according to the model/dataset you want to train on. We have provided the config files for GREW and Gait3D datsaets with GaitBase, DeepGaitV2 and SwinGait backbones. 
All training and evaluation hyperparameters are specified in the config files.



### Testing
To run testing, replace `--phase train` with `--phase test` in the above commands. The model to be tested can be specified in the config file. Paths are automatically inferred from the model and the dataset name. All saved models are in the 'output/' folder by default.



### Citation
If you find our work useful in your research, please consider citing the following paper:

```
@inproceedings{gupta2025mind,
  title={Mind the Gap: Bridging Occlusion in Gait Recognition via Residual Gap Correction},
  author={Gupta, Ayush and Huang, Siyuan and Chellappa, Rama},
  booktitle={IEEE International Joint Conference on Biometrics (IJCB)},
  year={2025}
}
```