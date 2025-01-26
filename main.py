import torch
import torch.nn as nn
import torch.optim as optim
from data_process.dataset import DataLoader, prepare_data
from data_process.unimodal_graph import TextualGraphEncoder, VisualGraphEncoder
from model.multimodal_fusion import MultiModalGraphEncoder
from model.contrastive_local import LocalContrastiveLearning
from model.contrastive_global import GlobalContrastiveLearning
from model.dual_discriminator import DualDiscriminatorModule



class HGCALModel(nn.Module):
    def __init__(self):
        super(HGCALModel, self).__init__()
        self.text_encoder = TextualGraphEncoder()
        self.visual_encoder = VisualGraphEncoder()
        self.multimodal_encoder = MultiModalGraphEncoder()
        self.local_contrastive = LocalContrastiveLearning()
        self.global_contrastive = GlobalContrastiveLearning()
        self.dual_discriminator = DualDiscriminatorModule()
        self.readout = DualDiscriminatorModule.ReadoutFunction()

    def forward(self, text_data, visual_data):
        text_features = self.text_encoder(text_data)
        visual_features = self.visual_encoder(visual_data)
        multimodal_features = self.multimodal_encoder(text_features, visual_features)

        local_loss = self.local_contrastive(multimodal_features)
        global_loss = self.global_contrastive(multimodal_features)
        discriminator_loss = self.dual_discriminator(multimodal_features)

        graph_representation = self.readout(multimodal_features)

        return graph_representation, local_loss, global_loss, discriminator_loss


def train(model, data_loader, optimizer, criterion, lambda_cl, beta_adv, gamma_ddl):
    model.train()
    total_loss = 0
    for text_data, visual_data, labels in data_loader:
        fixed_text, matched_image, unmatched_image = prepare_data(text_data, visual_data, labels)

        optimizer.zero_grad()
        representation, local_loss, global_loss, discriminator_loss = model(fixed_text, matched_image)

        classification_output = torch.sigmoid(representation)
        classification_loss = criterion(classification_output, labels)

        # Calculating total loss with hyperparameters
        total_loss = classification_loss + lambda_cl * local_loss + beta_adv * global_loss + gamma_ddl * discriminator_loss
        total_loss.backward()
        optimizer.step()

        print(f'Training Loss: {total_loss.item()}')

    return total_loss


def main():
    lambda_cl = 0.5  # Weight for contrastive loss
    beta_adv = 0.3  # Weight for adversarial loss
    gamma_ddl = 0.2  # Weight for dual discriminator loss

    model = HGCALModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # Assuming DataLoader is defined properly in dataset.py
    data_loader = DataLoader()

    epochs = 10
    for epoch in range(epochs):
        loss = train(model, data_loader, optimizer, criterion, lambda_cl, beta_adv, gamma_ddl)
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


if __name__ == "__main__":
    main()
