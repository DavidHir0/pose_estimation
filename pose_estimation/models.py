import torch
from torch import nn
from torch.hub import load

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
resnet = load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)


        print(self.resnet[-1][0].conv2)
        self.resnet[-1][0].downsample[0].stride = (1, 1)
        self.resnet[-1][0].conv2.stride = (1, 1)
        print(self.resnet[-1][0].conv2)
        # Add a deconvolutional layer
        self.deconv = nn.ConvTranspose2d(
            in_channels=2048,
            out_channels=24,
            kernel_size=4,
            stride=2,
            padding=1
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.deconv(x)
        return x

# Create an instance of your model
model = MyModel()
print(model)

# Assuming 'random_input' and other code remain unchanged

# Set the model to evaluation mode
model.eval()

random_input = torch.randn(1, 3, 800, 800)
# Perform a forward pass with the random input
with torch.no_grad():
    output = model(random_input)

# Print the dimensions of the output tensor
print("Output shape:", output.shape)

#print(model)