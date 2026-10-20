"""Full1024D token matching, no fitted PCA and no camera/source/vehicleID inputs."""

def describe(a,b):
 import torch
 from torch.nn import functional as F
 a=F.normalize(a.float(),dim=-1,eps=1e-8);b=F.normalize(b.float(),dim=-1,eps=1e-8);s=torch.bmm(a,b.transpose(1,2));n=s.shape[0];qm,qi=s.max(2);gm,gi=s.max(1);maxima=torch.cat([qm,gm],1).sort(1).values;qt=torch.tensor([0,.1,.25,.5,.75,.9,.95,.99,1],device=s.device);dist=torch.quantile(s.reshape(n,-1),qt,dim=1).T;d=s.diagonal(dim1=1,dim2=2);flip=torch.arange(36,device=s.device).reshape(6,6).flip(1).flatten();f=s[:,torch.arange(36,device=s.device),flip];diagonal=torch.stack([d.mean(1),d.std(1,correction=0),d.amin(1),d.amax(1),f.mean(1),f.std(1,correction=0),f.amin(1),f.amax(1)],1)
 topq=s.topk(2,dim=2).values;topg=s.topk(2,dim=1).values;margins=torch.cat([topq[:,:,0]-topq[:,:,1],topg[:,0,:]-topg[:,1,:]],1);marginstats=torch.cat([torch.quantile(margins,torch.tensor([0,.25,.5,.75,1.],device=s.device),dim=1).T,margins.mean(1)[:,None],margins.std(1,correction=0)[:,None]],1);back=gi.gather(1,qi);mutual=back==torch.arange(36,device=s.device)[None,:];num=mutual.sum(1);mv=torch.where(mutual,qm,0);mutualstats=torch.stack([num.float()/36,mv.sum(1)/num.clamp_min(1),(mv.square().sum(1)/num.clamp_min(1)).sqrt(),mv.amax(1)],1)
 def quad(x):
  grid=x.reshape(n,6,6,-1);return F.normalize(torch.stack([grid[:,i:i+3,j:j+3].mean((1,2)) for i,j in [(0,0),(0,3),(3,0),(3,3)]],1),dim=-1,eps=1e-8)
 quads=torch.bmm(quad(a),quad(b).transpose(1,2)).reshape(n,16).sort(1).values;globalcos=(F.normalize(a.mean(1),dim=1)*F.normalize(b.mean(1),dim=1)).sum(1,keepdim=True)
 mass=torch.exp((s-s.amax((1,2),keepdim=True))/.05)
 for _ in range(8):mass=mass/torch.sqrt((mass.sum(2,keepdim=True)*mass.sum(1,keepdim=True)).clamp_min(1e-16))
 mass=mass/mass.sum((1,2),keepdim=True).clamp_min(1e-16);balanced=torch.stack([(mass*s).sum((1,2)),-(mass*mass.clamp_min(1e-20).log()).sum((1,2)),mass.amax((1,2))],1);x=torch.cat([maxima,dist,diagonal,marginstats,mutualstats,quads,globalcos,balanced],1);assert x.shape==(n,120);return x

def features(a,b):
 import torch
 raw=describe(a,b);ca=a.float()-a.float().mean(1,keepdim=True);cb=b.float()-b.float().mean(1,keepdim=True);center=describe(ca,cb);v=torch.stack([ca.norm(dim=-1).mean(1),cb.norm(dim=-1).mean(1),ca.norm(dim=-1).std(1,correction=0),cb.norm(dim=-1).std(1,correction=0)],1);out=torch.cat([raw,center,v],1);assert out.shape[1]==244 and torch.isfinite(out).all();return out
