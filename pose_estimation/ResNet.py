import torch

print("test")
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
print("test2")
def some_function():
    model.eval()
    input_data = torch.randn(1, 3, 224, 224)
    output = model(input_data)
    print(output)


some_function()