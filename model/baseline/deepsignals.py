from torch import nn
import timm

from model.baseline.cnn_lstm_lib import ConvLSTM


class DeepsignalsBaseline(nn.Module):
    def __init__(self, num_ind_classes=4):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                dilation=(1, 2),
                padding="same",
            ),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                dilation=(1, 2),
                padding="same",
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                dilation=(1, 2),
                padding="same",
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=1,
                kernel_size=3,
                dilation=(1, 2),
                padding="same",
            ),
        )
        self.image_features = timm.create_model(
            "vgg16", pretrained=True, features_only=True
        )

        self.cnn_lstm = ConvLSTM(512, 256, (7, 7), 3, batch_first=True)

        self.indicator_head = nn.Sequential(
            nn.LayerNorm(12544), nn.Linear(12544, num_ind_classes)
        )
        self.rear_head = nn.Sequential(nn.LayerNorm(12544), nn.Linear(12544, 3))
        self.heading_head = nn.Sequential(nn.LayerNorm(12544), nn.Linear(12544, 4))

    def forward(self, images, headings):
        b, i, c, h, w = images.shape

        attn = self.attention(images.view(b * i, c, h, w))
        image_attn = images * attn.view(b, i, 1, h, w)

        image_features = self.image_features(image_attn.view(b * i, c, h, w))
        image_features = image_features[5].view(b, i, 512, 7, 7)  # take 512x7x7

        _, last_states = self.cnn_lstm(image_features)
        feature_encoding = last_states[0][0].flatten(start_dim=1)

        return (
            self.indicator_head(feature_encoding),
            self.rear_head(feature_encoding),
            self.heading_head(feature_encoding),
        )
