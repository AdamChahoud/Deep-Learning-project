import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), #padding cuz overlapping filter conv
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*8*8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    
def build_model(cfg):
    if cfg["model"]["name"] == "small_cnn":
        return SmallCNN(cfg["model"]["num_classes"])
    
    raise ValueError(f"Unknown model: {cfg["model"]["name"]}")