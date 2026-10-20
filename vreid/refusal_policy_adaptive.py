"""Gallery-supported visual calibration, preserving all prior accepted answers."""
from pathlib import Path
import hashlib,json
from types import SimpleNamespace
import joblib
import numpy as np
from .refusal_policy_expanded import ExpandedRefusalPolicy as _Expanded
from .refusal_policy_bagged import BaggedRefusalPolicy,digest

PROFILE='adaptive-calibration-v1'


def verified_selection(root):
    root=Path(root);selected=json.loads((root/'var/refusal-generalization-20261018/runtime-selection.json').read_text(encoding='utf-8'))
    if (selected['profile']!=PROFILE or selected['base_profile']!='expanded-source-v1'
            or selected['query_style_min']!=.8 or selected['gallery_style_min']!=.8
            or selected['gallery_size_cap']!=128):
        raise ValueError('Adaptive calibration identity mismatch')
    if selected['status'] not in ['DEVELOPMENT_PASSED_PENDING_CONFIRMATION_NATIVE','READY_LOCAL_WINDOWS_WITH_TRADEOFF']:
        raise ValueError('Adaptive calibration not qualified for validation')
    for name,expected in selected['files'].items():
        if digest(root/name)!=expected:raise ValueError('Adaptive calibration checksum mismatch: '+name)
    return selected


def project(features,projections):
    emb=np.asarray(features['emb'],dtype=np.float32)
    cnn=np.asarray(features['specialist'],dtype=np.float32)
    if emb.ndim!=2 or emb.shape[1]!=2048 or cnn.shape!=(len(emb),2048):
        raise ValueError('Expected DINO2048/CNN2048 raw features')
    out=[]
    for a,pc in zip([emb[:,:1024],emb[:,1024:],cnn],projections):
        if not np.isfinite(a).all():raise ValueError('Nonfinite style features')
        a=a/np.linalg.norm(a,axis=1,keepdims=True).clip(1e-8)
        out.append(np.clip(pc.transform(a)/np.sqrt(pc.explained_variance_.clip(1e-7)),-8,8).astype(np.float32))
    return np.concatenate(out,axis=1)


class AdaptiveRefusalPolicy:
    def __init__(self,baseline,style_model,projections,density,thresholds,enabled):
        self.baseline=baseline;self.style=style_model;self.projections=projections;self.density=density
        self.classes=style_model.classes_.tolist();self.thresholds=dict(thresholds);self.enabled=set(enabled)
        if (len(projections)!=3 or style_model.n_features_in_!=96
                or self.classes!=density['classes'] or set(self.classes)!=set(thresholds)
                or density['size_cap']!=128 or density['mean'].shape!=(len(self.classes),96)
                or density['precision'].shape!=(len(self.classes),96,96)
                or density['limits'].shape!=(len(self.classes),)):
            raise ValueError('Adaptive model dimensions/classes mismatch')
        if not self.enabled.issubset(self.classes):raise ValueError('Invalid enabled style')
        for name,t in self.thresholds.items():
            if not (0<t<1 or t==1.0000001):raise ValueError('Invalid adaptive threshold')
        self._gallery_key=None;self._gallery_value=None

    @classmethod
    def load(cls,root):
        root=Path(root);s=verified_selection(root)
        return cls(_Expanded.load(root),joblib.load(root/s['style_model']),
            joblib.load(root/s['projections']),joblib.load(root/s['density']),s['thresholds'],s['enabled_styles'])

    def gallery(self,features):
        # Content hashing catches both replacement and in-place changes without
        # storing query data or a second copy of the raw gallery.
        h=hashlib.sha256()
        for name in ['emb','specialist']:
            a=np.ascontiguousarray(features[name]);h.update(str((a.shape,a.dtype.str)).encode());h.update(memoryview(a).cast('B'))
        key=h.digest()
        if key==self._gallery_key:return self._gallery_value
        z=project(features,self.projections);probability=self.style.predict_proba(z).mean(0)
        i=int(np.argmax(probability));name=self.classes[i]
        delta=z.mean(0,dtype=np.float64)-self.density['mean'][i]
        distance=float((delta@self.density['precision'][i]@delta)*min(len(z),128))
        supported=(name in self.enabled and self.thresholds[name]<1
            and probability[i]>=.8 and distance<=self.density['limits'][i])
        value=dict(index=i,style=name,supported=bool(supported),probability=float(probability[i]),
                   distance=distance,limit=float(self.density['limits'][i]))
        self._gallery_key,self._gallery_value=key,value
        return value

    def predict(self,result,legacy_confidence,boundary):
        previous=self.baseline.predict(result,legacy_confidence,boundary)
        rows=np.flatnonzero(~previous['accepted'])
        if not len(rows):return previous
        g=self.gallery(result['_gallery_features'])
        if not g['supported']:return previous
        q={k:np.asarray(v)[rows] for k,v in result['_query_features'].items()}
        probability=self.style.predict_proba(project(q,self.projections))
        valid=(np.argmax(probability,axis=1)==g['index'])&(probability.max(1)>=.8)
        score=np.zeros(len(previous['accepted']),dtype=float)
        score[rows]=np.where(valid,previous['expanded_score'][rows],0.)
        proxy=SimpleNamespace(threshold=self.thresholds[g['style']])
        return BaggedRefusalPolicy.supplement(proxy,previous,score,np.zeros(len(score),bool),boundary)
