import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.model(x)

class DualDiscriminatorModule(nn.Module):
    def __init__(self, feature_dim):
        super(DualDiscriminatorModule, self).__init__()
        self.feature_dim = feature_dim
        self.discriminator_t = Discriminator(self.feature_dim)
        self.discriminator_v = Discriminator(self.feature_dim)
        self.mapping_t_to_v = nn.Linear(self.feature_dim, self.feature_dim)
        self.mapping_v_to_t = nn.Linear(self.feature_dim, self.feature_dim)

    def forward(self, features_t, features_v):
        # Mapping textual to visual and vice versa
        mapped_t_to_v = self.mapping_t_to_v(features_t)
        mapped_v_to_t = self.mapping_v_to_t(features_v)

        # Discriminator outputs
        real_t = self.discriminator_t(features_t)
        real_v = self.discriminator_v(features_v)
        fake_t = self.discriminator_t(mapped_v_to_t)
        fake_v = self.discriminator_v(mapped_t_to_v)

        # Adversarial losses
        adv_loss_t = torch.mean(F.relu(1.0 - real_t) + F.relu(1.0 + fake_t))
        adv_loss_v = torch.mean(F.relu(1.0 - real_v) + F.relu(1.0 + fake_v))

        # Cycle-consistency loss
        cycle_loss_t = F.l1_loss(self.mapping_v_to_t(mapped_t_to_v), features_t)
        cycle_loss_v = F.l1_loss(self.mapping_t_to_v(mapped_v_to_t), features_v)
        cycle_loss = cycle_loss_t + cycle_loss_v

        # Total loss
        total_loss = adv_loss_t + adv_loss_v + cycle_loss

        return total_loss, adv_loss_t, adv_loss_v, cycle_loss

