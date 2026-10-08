"""Read pinned compact supervision, never image/feature values or oracle IDs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch


SCHEMA='sam_surface_motion_regions_v1'
FIELDS=('region_group','region_different','prediction_region')


class IdentityRegionAnnotations:
    def __init__(self,path,expected_sha256):
        file=Path(path);raw=file.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected_sha256:
            raise ValueError('region annotation manifest identity differs')
        doc=json.loads(raw)
        if doc.get('schema')!=SCHEMA or doc.get('complete') is not True or doc.get('supervision_only') is not True:
            raise ValueError('unversioned or incomplete region annotations')
        self.records={}
        for row in doc['records']:
            key=(row['source_split'],*row['source_frames'])
            if key in self.records:raise ValueError('duplicate annotated source clock')
            self.records[key]=row

    def read(self,split,indices,raw_hashes):
        key=(split,*indices)
        if key not in self.records:
            raise ValueError('missing supervision for declared exposure; do not silently skip the sample: '+str(key))
        row=self.records[key]
        if list(raw_hashes)!=row['raw_sha256']:
            raise ValueError('region supervision belongs to other raw sensor bytes')
        path=Path(row['labels_path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['labels_sha256']:
            raise ValueError('region annotation changed')
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!=set(FIELDS)|{'source_frames'}:
                raise ValueError('annotation payload must contain only declared labels and clocks')
            if not np.array_equal(z['source_frames'],indices):
                raise ValueError('annotation source clock differs')
            result={name:torch.from_numpy(z[name]) for name in FIELDS}
        n=1024
        if result['region_group'].shape!=(2,n) or result['region_group'].dtype!=torch.long:
            raise ValueError('region groups lost the full 32x32 source sampler')
        if result['region_different'].shape!=(2,n,n) or result['region_different'].dtype!=torch.bool:
            raise ValueError('region negatives must retain every sampled region edge')
        if result['prediction_region'].shape!=(2,2,n) or result['prediction_region'].dtype!=torch.long:
            raise ValueError('prediction strata differ from the four original source directions')
        return {'identity_'+name:value for name,value in result.items()}
