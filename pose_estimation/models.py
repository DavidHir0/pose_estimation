import torch
from torch import nn
from torch.hub import load
from pose_estimation.data_pp.utils import make_kp_maps

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
resnet = load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)

class KpDetector(nn.Module):
    def __init__(self):
        super(KpDetector, self).__init__()
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Change stride of last Block
        self.resnet[-1][0].downsample[0].stride = (1, 1)
        self.resnet[-1][0].conv2.stride = (1, 1)

        # Use Dilated Convolution for all 3x3 Convolutional Layers in Last Block
        for block in self.resnet[-1]:
            for layer in block.children():
                if isinstance(layer, nn.Conv2d) and layer.kernel_size == (3, 3):
                    layer.dilation = (2, 2)

        # Add a deconvolutional layer
        self.deconv = nn.ConvTranspose2d(
            in_channels=2048,
            out_channels=20,
            kernel_size=4,
            stride=2,
            padding=1
        )


    def forward(self, x):
        x = self.resnet(x)
        x = self.deconv(x)
        return x

    # TRAIN KEYPOINT DETECTOR
    def train_kp_model(self, device, train_loader, optimizer, criterion, scheduler=None, num_epochs=10):
        self.to(device)
        self.train()

        for epoch in range(num_epochs):
            running_loss = 0.0

            with tqdm(train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
                
                for inputs, labels in tepoch:
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()

                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)

                    tepoch.set_postfix(loss=loss.item())

                epoch_loss = running_loss / len(train_loader.dataset)

                if scheduler:
                    scheduler.step()

                tepoch.set_postfix(loss=f"Epoch Loss: {epoch_loss:.4f}")
        print('Training complete.')

    # SAVE MODEL
    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)


model = KpDetector()
print(model.resnet[-1])
print(model.resnet[-1][2].conv2.dilation)