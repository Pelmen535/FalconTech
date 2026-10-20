"""Measured ordered counterfactual regions of the actual October raw similarity.

Uses the active encoder and fixed full-gallery transforms. It does not explain
nonlinear reranking/refusal, use another model, or rescale values to force closure.
"""
import numpy as np
from .october_images import crop_image,restoration_path
from .explain import heatmap_png,top_regions


def explain_pair(runtime,message):
    math=runtime.math
    if message['gallery_signature']!=runtime.gallery_signature:raise ValueError('Gallery must be registered before explanation')
    ids=math.gallery['keys'].tolist()
    if message['gallery_id'] not in ids:raise ValueError('Explanation observation not registered')
    index=ids.index(message['gallery_id']);grid=4
    records=[{'key':'__explain_query__','path':message['query_path'],'bbox':message['query_bbox']},
             {'key':message['gallery_id'],'path':message['candidate_path'],'bbox':message['candidate_bbox']}]
    frame_keys={str(k).split('#')[0] for k in ids}
    while records[0]['key'] in frame_keys:records[0]['key']+='_'
    query,_=runtime.extract(records[:1],1)
    result=math.route(query);target=float(result['raw'][0,index])
    query_vectors=[]
    for cnn,(state,graph) in zip((False,True),math.branches):
        mapped,_=math.mapped(query,cnn)
        query_vectors.append(math.adapt_query(mapped,state)['emb'][0])
    native=runtime.extractor();previous=native.batch_size

    def encoded(crop,count):
        native.batch_size=count
        rows=[{'key':str(i),'_crop':crop} for i in range(count)]
        values,_=native.extract(rows)
        return values

    def score(values,side,position=0):
        if side=='query':
            packet={k:values[k][position:position+1] for k in ('emb','specialist')};packet['keys']=np.array(['__explain_query__'])
            similarities=[]
            for cnn,(state,graph) in zip((False,True),math.branches):
                mapped,_=math.mapped(packet,cnn);adapted=math.adapt_query(mapped,state)['emb'][0]
                similarities.append((state[2]['emb']@adapted)[index])
        else:
            packet={k:v.copy() for k,v in math.gallery.items()}
            for k in ('emb','specialist'):packet[k][index]=values[k][position]
            similarities=[]
            for cnn,(state,graph),q in zip((False,True),math.branches,query_vectors):
                mapped,_=math.mapped(packet,cnn);adapted=math.adapt_query(mapped,state)['emb']
                similarities.append((adapted@q)[index])
        # Match the actual float32 ensemble operation order.
        return float((.5*np.asarray([similarities[0]],np.float32)+.5*np.asarray([similarities[1]],np.float32))[0])

    sides={}
    try:
        for name,record in zip(('query','candidate'),records):
            crop=crop_image(record);regime=None;endpoint_error=None
            # Original gallery rows may have been produced in a different TRT
            # batch. Verify the original score before attributing any regions.
            regimes=(1,4,2,3) if runtime.device=='cpu' else (1,16,8,32)
            for count in regimes:
                values=encoded(crop,count)
                matches=[(abs(score(values,name,i)-target),i) for i in range(count)]
                error,position=min(matches)
                if error<=1e-6:regime=(count,position);endpoint_error=error;break
            if regime is None:
                raise ValueError('Cannot reproduce selected raw similarity within 1e-6; no heatmap published')
            count,position=regime;values=[]
            for image in restoration_path(crop,grid):values.append(score(encoded(image,count),name,position))
            contributions=np.diff(np.asarray(values,np.float64)).reshape(grid,grid)
            constant=float(values[0]);total=float(contributions.sum());error=abs(constant+total-target)
            if error>1e-6:raise ValueError('Counterfactual reconstruction exceeds 1e-6; no heatmap published')
            display=crop.copy();display.thumbnail((960,960))
            sides[name]={'png':heatmap_png(display,contributions),'display_size':list(display.size),'grid':[grid,grid],
                         'constant':constant,'patch_sum':total,'reconstruction_error':error,
                         'contributions':contributions.tolist(),'top_regions':top_regions(contributions,message.get('regions',3)),
                         'encoder_batch_size':count,'encoder_position':position,'endpoint_error':endpoint_error,
                         'baseline':'RGB127 crop','restoration_order':'row-major, 16 regions'}
        return {'cosine':target,'threshold':math.boundary,'threshold_scale':'calibrated_acceptance_score',
                'method':'ordered_counterfactual_regions_actual_october_raw_similarity',
                'explained_score':'mean cosine of gallery-adapted DINO and CNN branches',
                'scope':'Fixed full-gallery raw similarity only; does not decompose nonlinear reranking or refusal. Ordered path is not Shapley attribution.',
                'gallery_generation':math.generation,'sides':sides}
    finally:native.batch_size=previous
