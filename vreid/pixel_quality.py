"""Deterministic per-image crop quality; not source/camera/ID routing."""
import numpy as np
from PIL import Image
NAMES=['luma_mean','luma_std','luma_p05','luma_p10','luma_p25','luma_p50','luma_p75','luma_p90','luma_p95','dark03','dark08','bright95','range90','gradient_x','gradient_y','gradient_std','laplace_var','R_mean','G_mean','B_mean','R_std','G_std','B_std','saturation_mean','saturation_std','saturation_p90','center_luma','border_luma','center_border_abs','edge_dark_fraction','log_crop_pixels','crop_aspect','crop_relative_area','bbox_present']
PHOTOMETRIC=30

def quality(image,geometry=(0.,0.,0.,0.)):
 im=image.convert('RGB').resize((96,96),Image.Resampling.BICUBIC);a=np.asarray(im,dtype=np.float32)/255.;y=.299*a[:,:,0]+.587*a[:,:,1]+.114*a[:,:,2];q=np.quantile(y,[.05,.1,.25,.5,.75,.9,.95]);gx=np.diff(y,axis=1);gy=np.diff(y,axis=0);lap=y[1:-1,:-2]+y[1:-1,2:]+y[:-2,1:-1]+y[2:,1:-1]-4*y[1:-1,1:-1];hi=a.max(2);lo=a.min(2);sat=(hi-lo)/np.maximum(hi,1e-6);center=y[24:72,24:72];border=np.r_[y[:8].ravel(),y[-8:].ravel(),y[:, :8].ravel(),y[:,-8:].ravel()]
 x=np.r_[y.mean(),y.std(),q,(y<.03).mean(),(y<.08).mean(),(y>.95).mean(),q[-1]-q[0],np.abs(gx).mean(),np.abs(gy).mean(),np.r_[gx.ravel(),gy.ravel()].std(),lap.var(),a.mean((0,1)),a.std((0,1)),sat.mean(),sat.std(),np.quantile(sat,.9),center.mean(),border.mean(),abs(center.mean()-border.mean()),(border<.08).mean(),geometry].astype(np.float32);assert x.shape==(len(NAMES),) and np.isfinite(x).all();return x
