import unittest
import torch
from clearvla.mainline.model.task_execution import TaskConditionedTargetBinder


class FullTokenBindingTests(unittest.TestCase):
    def test_mask_before_projection_and_reduction(self):
        torch.manual_seed(911)
        binder=TaskConditionedTargetBinder(16,4)
        task=torch.randn(2,7,16);mask=torch.zeros(2,7,dtype=torch.bool);mask[0,:3]=True
        objects=torch.randn(2,4,2,16);supported=torch.ones(2,4,dtype=torch.bool)
        views=torch.ones(2,4,2,dtype=torch.bool);mass=torch.rand(2,4,2);history=torch.randn(2,16)
        def run(t):return binder(t,objects,supported,history=history,view_support=views,view_mass=mass,task_mask=mask)
        before=run(task)
        poisoned=torch.where(mask[...,None],task,float('nan'))
        after=run(poisoned)
        torch.testing.assert_close(before.mass,after.mass,rtol=0,atol=0)
        assert after.mass[1].count_nonzero()==0
        after.mass[0,0].backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in binder.parameters())

    def test_camera_permutation_and_weighted_read(self):
        torch.manual_seed(191)
        binder=TaskConditionedTargetBinder(16,4)
        task=torch.randn(2,7,16);objects=torch.randn(2,4,2,16)
        supported=torch.ones(2,4,dtype=torch.bool);views=torch.ones(2,4,2,dtype=torch.bool)
        mass=torch.rand(2,4,2);history=torch.randn(2,16);mask=torch.ones(2,7,dtype=torch.bool)
        a=binder(task,objects,supported,history=history,view_support=views,view_mass=mass,task_mask=mask)
        b=binder(task,objects.flip(2),supported,history=history,view_support=views.flip(2),view_mass=mass.flip(2),task_mask=mask)
        torch.testing.assert_close(a.mass,b.mass)
        only_first=torch.zeros_like(mass);only_first[...,0]=1
        c=binder(task,objects,supported,history=history,view_support=views,view_mass=only_first,task_mask=mask)
        d=binder(task,objects[:,:,0],supported,history=history,task_mask=mask)
        torch.testing.assert_close(c.mass,d.mass,atol=1e-7,rtol=1e-6)


if __name__=='__main__':unittest.main()
