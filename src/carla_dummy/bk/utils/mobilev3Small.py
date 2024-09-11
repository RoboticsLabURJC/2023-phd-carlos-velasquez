import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MobileV3Small(nn.Module):
    def __init__(self):
        super(MobileV3Small, self).__init__()
        self.encoder = models.mobilenet_v3_small(pretrained=True).features
        self.aspp = ASPP()
        self.decoder = Decoder()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.aspp(x)
        x = self.decoder(x)
        return x

class ASPP(nn.Module):
    def __init__(self):
        super(ASPP, self).__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=3, stride=1, padding=6, dilation=6),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=3, stride=1, padding=12, dilation=12),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=3, stride=1, padding=18, dilation=18),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(576, 256, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(1280, 256, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=1, stride=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        return x

def load_model(model_path, device):
    model = MobileV3Small().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    return model

class LaneDetector:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(model_path, self.device)
        self.model.eval()

    def detect_lanes(self, image):
        image = image.to(self.device)
        with torch.no_grad():
            output = self.model(image)
        return output

if __name__ == "__main__":
    lane_detector = LaneDetector(model_path="/home/canveo/Documents/carla_laneddetection/lane_detection.pth")
