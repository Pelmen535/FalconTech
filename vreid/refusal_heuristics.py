"""Global, label-free match-evidence gates for refusal experiments."""
import numpy as np
from .raw_refusal import RAW_NAMES as RAW_TAIL_NAMES
from .general_refusal import FEATURE_NAMES

# The frozen raw module names its local18 + channel45 tail. The actual
# raw93 matrix begins with the relative30 features; preserve that old file.
RAW_NAMES=FEATURE_NAMES+RAW_TAIL_NAMES

CHANNELS=('DINO_CLS','DINO_PATCH_MEAN','CNN_RAW')

def candidate_rules():
    rules=[{'name':'unconditional-control','kind':'all'}]
    for channels,votes in [([CHANNELS[0],CHANNELS[2]],2),([CHANNELS[1],CHANNELS[2]],2),(list(CHANNELS),2),(list(CHANNELS),3)]:
        for z in [None,2.0]:
            rules.append({'name':'+'.join(channels)+f'-votes{votes}-z{z}','kind':'agreement','channels':channels,'votes':votes,'z_min':z})
    for z in [1.5,2.5,3.5]:
        for gap in [0.,.1,.3]:
            rules.append({'name':f'CLS-CNN-z{z}-gap{gap}','kind':'margin','channels':[CHANNELS[0],CHANNELS[2]],'z_min':z,'gap_min':gap})
    for percentile in [.98,.995]:
        for z in [2.,3.]:
            rules.append({'name':f'CLS-CNN-percentile{percentile}-z{z}','kind':'percentile','channels':[CHANNELS[0],CHANNELS[2]],'percentile':percentile,'z_min':z})
    return rules

def gate(features,rule):
    x=np.asarray(features,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=len(RAW_NAMES) or not np.isfinite(x).all():
        raise ValueError('Expected finite raw93 match features')
    if rule['kind']=='all':return np.ones(len(x),dtype=bool)
    channels=rule['channels']
    if not channels or any(c not in CHANNELS for c in channels):raise ValueError('Invalid evidence channels')
    def fields(field):return x[:,[RAW_NAMES.index(c+'_'+field) for c in channels]]
    if rule['kind']=='agreement':
        supported=fields('selected_percentile')>=1-1e-7
        if rule['z_min'] is not None:supported&=fields('selected_z')>=rule['z_min']
        return supported.sum(1)>=rule['votes']
    if rule['kind']=='margin':return (fields('selected_z').min(1)>=rule['z_min'])&(fields('gap2_z').min(1)>=rule['gap_min'])
    if rule['kind']=='percentile':return (fields('selected_percentile').min(1)>=rule['percentile'])&(fields('selected_z').min(1)>=rule['z_min'])
    raise ValueError('Unknown match-evidence gate')
