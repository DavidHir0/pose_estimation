import torch

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

def some_function():
    model.eval()
    input_data = torch.randn(1, 3, 224, 224)
    output = model(input_data)
    print(output)


some_function()