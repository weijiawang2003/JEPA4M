import numpy as np
import torch
from jepa4math.cube import CubeEnv
from jepa4math.data import generate_walks
from jepa4math.models import CubeModel,sigreg

def test_all_generated_transitions_replay_and_no_inverse_bug():
    env=CubeEnv(3);data=generate_walks(3,256,8,71)
    assert set(data["action"][:256])==set(range(12))
    for s,a,nxt in zip(data["state"],data["action"],data["next_state"]):
        assert np.array_equal(s[env.permutations[a]],nxt)
    a=data["action"].reshape(8,256)
    assert not np.any(a[1:]==(a[:-1]^1))
    assert np.all(data["walk_step"].reshape(8,256)[:,0]==np.arange(1,9))

def test_sigreg_discourages_collapse_and_backpropagates_both_branches():
    torch.manual_seed(1)
    collapsed=torch.zeros(512,32,requires_grad=True)
    gaussian=torch.randn(512,32,requires_grad=True)
    assert sigreg(collapsed,128)>10*sigreg(gaussian,128)
    model=CubeModel(2,64,32,1)
    env=CubeEnv(2)
    s=torch.from_numpy(np.stack([env.scramble(5,i)[0] for i in range(32)]))
    z=model.encoder(s);zn=model.encoder(s[:,env.permutations[0].copy()])
    z.retain_grad();zn.retain_grad()
    loss=(model.predictor(z,torch.zeros(32,dtype=torch.long))-zn).square().mean()+.1*(sigreg(z)+sigreg(zn))
    loss.backward()
    assert z.grad.norm()>0 and zn.grad.norm()>0
    assert all(torch.isfinite(p.grad).all() for p in model.encoder.parameters())
