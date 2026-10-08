"""Training-only correspondence labels with explicit source clocks/charts."""
from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass(frozen=True)
class IdentityCorrespondence:
    cross_source: Tensor  # B,C,N,2 endpoint coordinates, source camera c
    cross_target: Tensor  # target camera is 1-c, no fabricated common XY chart
    cross_valid: Tensor
    temporal_source: Tensor  # past-to-current in camera c
    temporal_target: Tensor
    temporal_valid: Tensor
    source_frames: Tensor  # B,2 absolute source rows: past/current
    camera_names: tuple[str,...] = ('top','wrist')
    region_group: Tensor | None = None  # B,C,N, -1 unknown, current chart only
    region_different: Tensor | None = None  # B,C,G,G, independently witnessed
    prediction_region: Tensor | None = None  # B,kind,C,N, target-current strata

    def validate(self,*,batch,device):
        if self.camera_names!=('top','wrist') or self.source_frames.shape!=(batch,2):
            raise ValueError('identity labels lost their calibrated camera/source clock')
        if self.source_frames.dtype!=torch.long or self.source_frames.device!=device:
            raise ValueError('identity source clock must be int64 on the label device')
        if bool((self.source_frames[:,1]<self.source_frames[:,0]).any()):
            raise ValueError('identity temporal labels cannot reverse the observed source clock')
        for prefix in ('cross','temporal'):
            source=getattr(self,prefix+'_source');target=getattr(self,prefix+'_target');valid=getattr(self,prefix+'_valid')
            if source.ndim!=4 or source.shape[:2]!=(batch,2) or source.shape[-1]!=2 or target.shape!=source.shape or valid.shape!=source.shape[:-1]:
                raise ValueError('identity labels require full pair/camera/coordinate axes')
            if source.dtype!=torch.float32 or target.dtype!=torch.float32 or valid.dtype!=torch.bool:
                raise ValueError('identity coordinates require FP32 and independent Boolean support')
            for value in (source,target,valid):
                if value.requires_grad or value.device!=device:raise ValueError('identity labels must be detached on the label device')
            for value in (source,target):
                safe=torch.where(valid[...,None],value,0.)
                if not bool(torch.isfinite(safe).all()) or bool((safe.abs()>1+1e-6).any()):
                    raise ValueError('supported correspondence escaped its endpoint image chart')
        if bool((self.temporal_valid.any((1,2))&(self.source_frames[:,1]==self.source_frames[:,0])).any()):
            raise ValueError('padded duplicate frame cannot manufacture temporal evidence')
        values=(self.region_group,self.region_different,self.prediction_region)
        if any(v is not None for v in values):
            if not all(v is not None for v in values):raise ValueError('incomplete region supervision')
            n=self.cross_source.shape[2]
            if self.region_group.shape!=(batch,2,n) or self.region_different.shape!=(batch,2,n,n) or self.prediction_region.shape!=(batch,2,2,n):
                raise ValueError('region labels lost source/camera/direction axes')
            if self.region_group.dtype!=torch.long or self.region_different.dtype!=torch.bool or self.prediction_region.dtype!=torch.long:
                raise ValueError('region evidence requires integer groups and Boolean edges')
            for value in values:
                if value.requires_grad or value.device!=device:raise ValueError('region evidence must be detached on label device')
            if bool(((self.region_group < -1)|(self.region_group >= n)).any()) or bool((self.prediction_region < -1).any()):
                raise ValueError('invalid region ID')
            if not torch.equal(self.region_different,self.region_different.transpose(-1,-2)) or bool(self.region_different.diagonal(dim1=-2,dim2=-1).any()):
                raise ValueError('region negatives must be symmetric, irreflexive evidence')
