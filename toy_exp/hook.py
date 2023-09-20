import torch
import timm

def hook(m, input_g, output_g):
    input_g_ = input_g[1] * 2
    return (input_g[0], input_g_, input_g[2])

model = timm.create_model('resnet18')
for name, module in model.named_children():
    if 'conv' in name:
        module.register_full_backward_hook(hook)

input_tensor = torch.randn(size=(1, 3, 224, 224))
label = torch.zeros(size=(1,))
label = label.long()

loss = torch.nn.CrossEntropyLoss()
output = model(input_tensor)
cost = loss(output, label)
cost.backward()



