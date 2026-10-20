"""October FP32 CPU encoders: same parameters, separate numeric cache identity."""
from pathlib import Path
import json,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def load_dino(path):
    import torch
    from vreid.train import ReIDModel
    # Numeric NPZ only; source tensor hashes are verified during lossless export,
    # and the complete artifact SHA is verified by the runtime manifest.
    import zipfile
    with zipfile.ZipFile(path) as archive:
        metadata=json.loads(archive.read('metadata.json'))
    if metadata['dim']!=2048 or metadata['img_size']!=384 or metadata['model_name']!='vit_large_patch16_dinov3.lvd1689m':
        raise ValueError('Unexpected CPU backbone architecture')
    model=ReIDModel(metadata['model_name'],metadata['img_size'],pretrained=False)
    with np.load(path,allow_pickle=False) as arrays:
        for section,module in [('backbone',model.backbone),('bnneck',model.bnneck)]:
            # Copy one decoded tensor into its final storage at a time. Keeping
            # a full reconstructed state duplicates the entire FP32 backbone.
            targets=module.state_dict()
            with torch.no_grad():
                for name,target in targets.items():
                    key=section+'/'+name;layout=metadata['tensor_layout'][key]
                    encoded=arrays[key]
                    destination=target.detach().numpy()
                    if destination.shape!=tuple(layout['shape']) or destination.dtype!=np.dtype(layout['dtype']) or not destination.flags.c_contiguous:
                        raise ValueError('CPU parameter shape/dtype/layout mismatch: '+key)
                    if encoded.dtype!=np.uint8 or encoded.shape!=(destination.dtype.itemsize,destination.size):
                        raise ValueError('CPU byte-shuffle payload mismatch: '+key)
                    # CPU inference-only loader: decode byte planes directly
                    # into the tensor's final contiguous storage. No float
                    # conversion, temporary full state or extra tensor copy.
                    octets=destination.reshape(-1).view(np.uint8).reshape(-1,destination.dtype.itemsize)
                    np.copyto(octets,encoded.T,casting='no')
                    del destination,octets,encoded
    model.eval()
    return model,metadata


class CPUFeatures:
    def __init__(self,root,spec,logs,batch_size=1,live_check=None):
        import torch
        from .october_manifest import artifact
        from .october_cnn import CNN
        torch.set_num_threads(2)
        self.torch=torch;self.batch_size=batch_size;self.live_check=live_check
        self.dino,metadata=load_dino(artifact(root,spec['cpu_environment']['checkpoint']))
        self.mean=np.asarray(metadata['mean'],np.float32)[None,None,:]
        self.std=np.asarray(metadata['std'],np.float32)[None,None,:]
        checkpoint=torch.load(Path(root)/'var/three-directions-20261009/specialist-cnn.pt',map_location='cpu',weights_only=True)
        self.cnn=CNN().eval();self.cnn.load_state_dict(checkpoint['model'],strict=True)
        self.pool=None;self.workers=None;self.set_workers(4)

    def set_workers(self,requested):
        workers=min(int(requested),4)
        if workers<0:raise ValueError('Preprocessing workers must be nonnegative')
        if workers!=self.workers:
            if self.pool:self.pool.shutdown(wait=True)
            self.pool=ThreadPoolExecutor(max_workers=workers) if workers else None
            self.workers=workers

    def prepare_dual(self,row):
        from PIL import Image
        from .october_images import crop_image
        crop=crop_image(row)
        dino=np.asarray(crop.resize((384,384),Image.Resampling.BICUBIC),np.float32)
        dino=((dino/np.float32(255)-self.mean)/self.std).transpose(2,0,1).copy()
        cnn=np.asarray(crop.resize((256,256),Image.Resampling.BICUBIC),np.float32).transpose(2,0,1).copy()
        return self.torch.from_numpy(dino),self.torch.from_numpy(cnn)

    def extract(self,rows):
        if not rows:raise ValueError('Cannot encode an empty observation set')
        outputs=[];start=time.perf_counter();torch=self.torch
        # CPU batches are capped for bounded activation memory; the requested
        # size is recorded separately. This never falls back to the v1.4 model.
        size=min(self.batch_size,4)
        with torch.inference_mode():
            for off in range(0,len(rows),size):
                if self.live_check:self.live_check()
                part=rows[off:off+size]
                pairs=list(self.pool.map(self.prepare_dual,part)) if self.pool else [self.prepare_dual(r) for r in part]
                x=torch.stack([p[0] for p in pairs]);cnn=torch.stack([p[1] for p in pairs])
                _,value=self.dino.features(x)
                raw=torch.nn.functional.normalize(value.float(),dim=1).numpy()
                specialist=self.cnn(cnn).numpy().astype(np.float16)
                outputs.append((raw,specialist))
        raw=np.concatenate([r[0] for r in outputs]);cnn=np.concatenate([r[1] for r in outputs])
        if not np.isfinite(raw).all() or not np.isfinite(cnn).all():raise ValueError('Nonfinite CPU output')
        elapsed=time.perf_counter()-start
        return {'keys':np.array([r['key'] for r in rows]),'emb':raw,'specialist':cnn},{'count':len(rows),'seconds':elapsed,'FPS':len(rows)/elapsed,'scope':'actual FP32 CPU October DINO+CNN, JPEG/crop/preprocess/L2','requested_batch_size':self.batch_size,'effective_batch_size':size}

    def close(self):
        if self.pool:self.pool.shutdown(wait=True);self.pool=None
        self.dino=None;self.cnn=None
