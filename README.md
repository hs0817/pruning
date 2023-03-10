# pruning
**Describe the bug**:

When I used the torch.chunk function to divide features into channel dimensions, the following errors occurred :

    [2023-03-10 12:11:16] start to speedup the model 
    [2023-03-10 12:11:16] infer module masks...
    [2023-03-10 12:11:16] Update mask for .aten::size.122
    [2023-03-10 12:11:16] Update mask for .aten::size.124
    [2023-03-10 12:11:16] Update mask for .aten::size.126
    [2023-03-10 12:11:16] Update mask for .aten::size.127
    [2023-03-10 12:11:16] Update mask for .aten::Int.123
    [2023-03-10 12:11:16] Update mask for .aten::Int.125
    [2023-03-10 12:11:16] Update mask for .aten::remainder.128
    [2023-03-10 12:11:16] Update mask for .aten::remainder.132
    [2023-03-10 12:11:16] Update mask for .aten::rsub.129
    [2023-03-10 12:11:16] Update mask for .aten::rsub.133
    [2023-03-10 12:11:16] Update mask for .aten::remainder.130
    [2023-03-10 12:11:16] Update mask for .aten::remainder.134
    [2023-03-10 12:11:16] Update mask for .aten::Int.131
    [2023-03-10 12:11:16] Update mask for .aten::Int.135
    [2023-03-10 12:11:16] Update mask for .aten::constant_pad_nd.136
    [2023-03-10 12:11:16] Update mask for intro
    [2023-03-10 12:11:16] Update mask for encoders.0.0.norm
    [2023-03-10 12:11:16] Update mask for encoders.0.0.conv1
    [2023-03-10 12:11:16] Update mask for encoders.0.0.conv2
    [2023-03-10 12:11:16] Update mask for encoders.0.0.aten::chunk.146
    [2023-03-10 12:11:16] Update mask for encoders.0.0.prim::ListUnpack.147
    [2023-03-10 12:11:16] Update mask for encoders.0.0.att1.0
    Traceback (most recent call last):
      File "/media/hs/Data/issue_5416/prune.py", line 73, in <module>
        ModelSpeedup(model, (torch.rand(1, 3, 64, 64).to(device)), masks).speedup_model()
      File "/home/hs/anaconda3/envs/nni/lib/python3.8/site-packages/nni/compression/pytorch/speedup/compressor.py", line 547, in speedup_model
        self.infer_modules_masks()
      File "/home/hs/anaconda3/envs/nni/lib/python3.8/site-packages/nni/compression/pytorch/speedup/compressor.py", line 384, in infer_modules_masks
        self.update_direct_sparsity(curnode)
      File "/home/hs/anaconda3/envs/nni/lib/python3.8/site-packages/nni/compression/pytorch/speedup/compressor.py", line 245, in update_direct_sparsity
        _auto_infer = AutoMaskInference(
      File "/home/hs/anaconda3/envs/nni/lib/python3.8/site-packages/nni/compression/pytorch/speedup/infer_mask.py", line 83, in __init__
        self.output = self.module(*dummy_input)
      File "/home/hs/anaconda3/envs/nni/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
        return forward_call(*input, **kwargs)
    TypeError: forward() missing 1 required positional argument: 'input'

I use the following operations to replace torch.chunk, but after I prune, the channel dimensions of e, f, and g will be different, resulting in problems in subsequent addition and multiplication operations.

    def forward(self, x):
        x1 = self.norm(x)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        
        # e, f, g = x1.chunk(3, dim=1)
        
        dim = (x1.shape[1]) // 3
        e = x1[:, 0:dim, :, :]
        f = x1[:, dim: dim * 2, :, :]
        g = x1[:, dim * 2: dim * 3, :, :]
        
        e = self.att1(e)
        f = self.att2(f)
        ef = e + f
        ef = self.sigmoid(ef)
        efg = ef * g
        efg = self.conv3(efg)
        out = efg + x
        return out

the following errors occurred :

    Traceback (most recent call last):
      File "/media/hs/Data/issue_5416/prune.py", line 77, in <module>
        flops, params2, _ = count_flops_params(model, dummy_input)
      File "/home/hs/anaconda3/envs/nni/lib/python3.8/site-packages/nni/compression/pytorch/utils/counter.py", line 395, in count_flops_params
        model(*x)
      File "/home/hs/anaconda3/envs/nni/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
        return forward_call(*input, **kwargs)
      File "/media/hs/Data/issue_5416/model.py", line 122, in forward
        x = encoder(x)
      File "/home/hs/anaconda3/envs/nni/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
        return forward_call(*input, **kwargs)
      File "/home/hs/anaconda3/envs/nni/lib/python3.8/site-packages/torch/nn/modules/container.py", line 141, in forward
        input = module(input)
      File "/home/hs/anaconda3/envs/nni/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
        return forward_call(*input, **kwargs)
      File "/media/hs/Data/issue_5416/model.py", line 42, in forward
        e = self.att1(e)
      File "/home/hs/anaconda3/envs/nni/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
        return forward_call(*input, **kwargs)
      File "/home/hs/anaconda3/envs/nni/lib/python3.8/site-packages/torch/nn/modules/container.py", line 141, in forward
        input = module(input)
      File "/home/hs/anaconda3/envs/nni/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1120, in _call_impl
        result = forward_call(*input, **kwargs)
      File "/home/hs/anaconda3/envs/nni/lib/python3.8/site-packages/torch/nn/modules/conv.py", line 446, in forward
        return self._conv_forward(input, self.weight, self.bias)
      File "/home/hs/anaconda3/envs/nni/lib/python3.8/site-packages/torch/nn/modules/conv.py", line 442, in _conv_forward
        return F.conv2d(input, weight, bias, self.stride,
    RuntimeError: Given groups=1, weight of size [8, 24, 1, 1], expected input[1, 16, 64, 64] to have 24 channels, but got 16 channels instead

    Process finished with exit code 1



**Environment**:
- NNI version: 2.10
- Training service (local|remote|pai|aml|etc): local
- Python version: 3.8.16
- PyTorch version: 1.10.1
- Cpu or cuda version: cuda10.2+cudnn7.6.5

