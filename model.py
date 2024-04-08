import torch
import torch.nn as nn

from torchvision.models.mobilenetv3 import mobilenet_v3_small, MobileNet_V3_Small_Weights

from typing import Optional

class CloudClassifier(nn.Module):

    def __init__(self, num_classes=11, dropout=0.2):
        super().__init__()
        # use mobilenet_v3 as the base model
        base_model = mobilenet_v3_small()
        self.features = base_model.features
        self.avgpool = base_model.avgpool

        # build a different classifier for CCSN dataset
        lastconv_output_channels = base_model.classifier[0].in_features
        last_channel = base_model.classifier[0].out_features
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_model(weights: Optional[MobileNet_V3_Small_Weights], **kwargs) -> CloudClassifier:
    model = CloudClassifier(**kwargs)
    # load weights only for the feature extractor
    state_dict = {k: v for k,v in weights.get_state_dict(progress=True).items() if k.startswith('features')}
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    for missing_key in missing_keys:
        assert missing_key.startswith('classifier'), f"Missing key: {missing_key}"
    assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"
    return model
