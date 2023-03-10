import torch
import yaml

from collections import OrderedDict
from model import FNAFNet as Net
from nni.compression.pytorch.pruning import L1NormPruner, L2NormPruner, FPGMPruner
from nni.compression.pytorch.speedup import ModelSpeedup
from nni.compression.pytorch.utils import count_flops_params
from yaml import CLoader as Loader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
yaml_file = './net.yml'
x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
s = x['network_g'].pop('type')
model = Net(**x['network_g'])
model_statedict = torch.load("./Model.pth")
try:
    model.load_state_dict(model_statedict['params'])
except:
    try:
        model.load_state_dict(model_statedict["state_dict"])
    except:
        state_dict = model_statedict["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
# print(model)
model.cuda()
model.eval()

config_list = [{
    'sparsity_per_layer': 0.5,
    # 'op_types': ['Linear', 'Conv2d'],
    'op_names': [
                 'encoders.0.0.conv1',
                 'encoders.0.0.conv2',
                 # 'encoders.0.0.att1.0',
                 # 'encoders.0.0.att1.2',
                 # 'encoders.0.0.att2.1',
                 # 'encoders.0.0.att2.3',
                 # 'encoders.0.0.conv3',
    ]
},
    {
    'exclude': True,
    'op_names': ['ending', 'intro']
    }
]

dummy_input = torch.rand(1, 3, 64, 64).to(device)
pruner = L1NormPruner(model, config_list, mode='dependency_aware', dummy_input=dummy_input)
# pruner = L2NormPruner(model, config_list, mode='dependency_aware', dummy_input=dummy_input)
# pruner = FPGMPruner(model, config_list, mode='dependency_aware', dummy_input=dummy_input)
# pruner = L1NormPruner(model, config_list)

# compress the model and generate the masks
_, masks = pruner.compress()

# show the masks sparsity
for name, mask in masks.items():
    print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))

# need to unwrap the model, if the model is wrapped before speedup
pruner._unwrap_model()

print("----------------before pruning-----------")
flops, params1, _ = count_flops_params(model, dummy_input)
print(f"FLOPs: {flops}, params: {params1}")

ModelSpeedup(model, (torch.rand(1, 3, 64, 64).to(device)), masks).speedup_model()
print(model)

print("----------------after pruning-----------")
flops, params2, _ = count_flops_params(model, dummy_input)
print(f"FLOPs: {flops}, params: {params2}")
print('prune params:', params1-params2)
print('prune percentage:', (params1-params2)/params1)


