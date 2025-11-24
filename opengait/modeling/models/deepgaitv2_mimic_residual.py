import torch
import torch.nn as nn

import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, conv1x1, conv3x3, BasicBlock2D, BasicBlockP3D, BasicBlock3D, Occ_Detector_Amount

from einops import rearrange

blocks_map = {
    '2d': BasicBlock2D, 
    'p3d': BasicBlockP3D, 
    '3d': BasicBlock3D
}

class DeepGaitV2_Mimic_Component(nn.Module):

    def __init__(self, model_cfg):
        super(DeepGaitV2_Mimic_Component, self).__init__()
        self.build_network(model_cfg)

    def build_network(self, model_cfg):
        mode = model_cfg['Backbone']['mode']
        assert mode in blocks_map.keys()
        block = blocks_map[mode]

        in_channels = model_cfg['Backbone']['in_channels']
        layers      = model_cfg['Backbone']['layers']
        channels    = model_cfg['Backbone']['channels']

        if mode == '3d': 
            strides = [
                [1, 1], 
                [1, 2, 2], 
                [1, 2, 2], 
                [1, 1, 1]
            ]
        else: 
            strides = [
                [1, 1], 
                [2, 2], 
                [2, 2], 
                [1, 1]
            ]

        self.inplanes = channels[0]
        self.layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(in_channels, self.inplanes, 1), 
            nn.BatchNorm2d(self.inplanes), 
            nn.ReLU(inplace=True)
        ))
        self.layer1 = SetBlockWrapper(self.make_layer(BasicBlock2D, channels[0], strides[0], blocks_num=layers[0], mode=mode))

        self.layer2 = self.make_layer(block, channels[1], strides[1], blocks_num=layers[1], mode=mode)
        self.layer3 = self.make_layer(block, channels[2], strides[2], blocks_num=layers[2], mode=mode)
        self.layer4 = self.make_layer(block, channels[3], strides[3], blocks_num=layers[3], mode=mode)

        if mode == '2d': 
            self.layer2 = SetBlockWrapper(self.layer2)
            self.layer3 = SetBlockWrapper(self.layer3)
            self.layer4 = SetBlockWrapper(self.layer4)

        self.FCs = SeparateFCs(16, channels[3], channels[2])
        #self.BNNecks = SeparateBNNecks(16, channels[2], class_num=model_cfg['SeparateBNNecks']['class_num'])

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

    def make_layer(self, block, planes, stride, blocks_num, mode='2d'):

        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            if mode == '3d':
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=stride, padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            elif mode == '2d':
                downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride=stride), nn.BatchNorm2d(planes * block.expansion))
            elif mode == 'p3d':
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=[1, *stride], padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            else:
                raise TypeError('xxx')
        else:
            downsample = lambda x: x

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        s = [1, 1] if mode in ['2d', 'p3d'] else [1, 1, 1]
        for i in range(1, blocks_num):
            layers.append(
                    block(self.inplanes, planes, stride=s)
            )
        return nn.Sequential(*layers)

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs

        sils = ipts[0].unsqueeze(1)
        #assert sils.size(-1) in [44, 88]

        del ipts
        out0 = self.layer0(sils)
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3) # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = self.TP(out4, seqL, options={"dim": 2})[0]  # [n, c, h, w]

        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        #embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]

        #embed = embed_1

        # retval = {
        #     'training_feat': {
        #         'triplet': {'embeddings': embed_1, 'labels': labs},
        #         'softmax': {'logits': logits, 'labels': labs}
        #     },
        #     'visual_summary': {
        #         'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w'),
        #     },
        #     'inference_feat': {
        #         'embeddings': embed
        #     }
        # }

        return embed_1

class DeepGaitV2_Mimic_Residual(BaseModel):
    def build_network(self, model_cfg):
        self.full_component = DeepGaitV2_Mimic_Component(model_cfg)
        self.mimic_component = DeepGaitV2_Mimic_Component(model_cfg)
        self.BNNecks = SeparateBNNecks(16, model_cfg['Backbone']['channels'][2], class_num=model_cfg['SeparateBNNecks']['class_num'])
            
        self.full_component.requires_grad_(False)

        if 'detector_type' in model_cfg.keys() and model_cfg['detector_type'] == 'deep':
            #self.occ_detector = Occ_Detector_Deep(parts_num = model_cfg['OccMixerFCs']['parts_num'])
            raise NotImplementedError('Deep Occ Detector not implemented yet for residual mimic model')
        else:
            self.occ_detector = Occ_Detector_Amount(parts_num = model_cfg['OccMixerFCs']['parts_num'])

        self.occ_detector.requires_grad_(False)
        
        self.occ_mixer_fc = SeparateFCs(**model_cfg['OccMixerFCs'])

    def init_parameters(self):
        super().init_parameters()
        if 'mimic_cfg' in self.cfgs['model_cfg']:
            model_path = os.path.join(
                './output',
                self.cfgs['data_cfg']['dataset_name'], 
                self.cfgs['model_cfg']['mimic_cfg']['teacher_model_name'], 
                self.cfgs['model_cfg']['mimic_cfg']['teacher_save_name'], 'checkpoints', 
                f"{self.cfgs['model_cfg']['mimic_cfg']['teacher_save_name']}-{str(self.cfgs['model_cfg']['mimic_cfg']['teacher_model_iter']).zfill(5)}.pt")
            
            device_rank = torch.distributed.get_rank()
            device=torch.device("cuda", device_rank)
            teacher_model = torch.load(model_path, map_location=device)
            if torch.distributed.get_rank() == 0:
                print(f"\nLoaded teacher model from {model_path}!\n")
            
            full_dict = {}
            BNNeck_dict = {}
            for k,v in teacher_model['model'].items():
                if k.split('.')[0] == 'BNNecks':
                    # remove the first BNNeck from the key
                    k = '.'.join(k.split('.')[1:])
                    BNNeck_dict[k] = v
                else:
                    full_dict[k] = v
            self.full_component.load_state_dict(full_dict)
            self.BNNecks.load_state_dict(BNNeck_dict)
            
            if 'init_mimic' in self.cfgs['model_cfg']['mimic_cfg']:
                if self.cfgs['model_cfg']['mimic_cfg']['init_mimic']:
                    self.mimic_component.load_state_dict(full_dict)
                    
                    if torch.distributed.get_rank() == 0:
                        print(f"\nLoaded Mimic model from {model_path}!\n")
            
            
            ################# Load occ detector #################
            
            
            #Removing the module from the state dict
            model_cfg = self.cfgs['model_cfg']
            saved_weights = torch.load(model_cfg['occ_detector_path'],  map_location=device)
            new_dict = {}
            for k, v in saved_weights.items():
                if k.startswith('module.'):
                    new_dict[k[7:]] = v
                else:
                    new_dict[k] = v
            
            #new_dict doesnt have the module. prefix. Now, remove unwanted layers
            filter_dict = {}
            for k, v in new_dict.items():
                if k.split('.')[0] in ['fc2', 'classification_head']: #, 'layer5']:
                    #print("Removing: {}".format(k))
                    continue    # These are not needed
                else: 
                    filter_dict[k] = v
            
            
            self.occ_detector.load_state_dict(filter_dict)
            
            if torch.distributed.get_rank() == 0:
                print("\nLoaded Occ Detector from: {}\n".format(model_cfg['occ_detector_path']))
            
            #Uncomment
            #self.visible_component.load_state_dict(visible_dict)
            # self.full_component.load_state_dict(invisible_dict)
            # self.fcs_unified.load_state_dict(fcs_dict)
            #self.BNNecks_unified.load_state_dict(bnnecks_dict)
        else:
            raise ValueError('The mimic_cfg must be specified in model_cfg for full_mimic training')

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs

        #sils = ipts[0].unsqueeze(1)

        visible_sils = ipts[0] #[:,:,:,:,0]
        #full_sils = ipts[0][:,:,:,:,1]
        
        #assert sils.size(-1) in [44, 88]

        del ipts

        sils = visible_sils.unsqueeze(1)

        with torch.no_grad():
            occ_embed = self.occ_detector(sils, seqL)       #[n, occ_dim, p]
        
        visible_inputs = [[visible_sils], labs, typs, vies, seqL]
        #full_inputs = [[full_sils], labs, typs, vies, seqL]

        with torch.no_grad():
            occ_embed, occ_amts = self.occ_detector(sils, seqL) #[n, occ_dim, p], [n]
            occ_amts = occ_amts.unsqueeze(-1).unsqueeze(-1)  #[n, 1, 1]
        mimic_embed = self.mimic_component(visible_inputs)  #(n, c, p)
        mimic_occ = torch.cat([mimic_embed, occ_embed], dim=1)  #(n, c+occ_dim, p)
        mimic_occ_aware = self.occ_mixer_fc(mimic_occ)  #(n, c, p)

        #if self.training:
        with torch.no_grad():
            full_embed = self.full_component(visible_inputs)
            
        embed = occ_amts*mimic_occ_aware + full_embed  #(n, c, p)  # TODO: Experiment: Simple weighted addition??
        
        
        embed_2, logits = self.BNNecks(embed)  # [n, c, p]
        
        inference_embed=embed 

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils,'n c s h w -> (n s) c h w')
            },
            'inference_feat': {
                'embeddings': inference_embed
            }
        }
        
        return retval

        

