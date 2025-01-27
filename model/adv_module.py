import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the Invariant Projection Layer (shared subspace mapping)
class InvariantProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(InvariantProjectionLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Project the features to a shared subspace
        return self.fc(x)


# Define Discriminator D0 (Binary Classifier for modality-invariant feature)
class Discriminator0(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator0, self).__init__()
        self.fc = nn.Linear(input_dim, 1)  # 1 output for binary classification

    def forward(self, x):
        # Output the probability of the input coming from modality t (1) or modality v (0)
        return torch.sigmoid(self.fc(x))


# Define Discriminator D1 (To blend features from both modalities)
class Discriminator1(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator1, self).__init__()
        self.fc = nn.Linear(input_dim, 1)  # Binary output for classification

    def forward(self, x):
        # Output probability
        return torch.sigmoid(self.fc(x))


# Define the ACIL module with adversarial and cycle consistency loss
class AdversarialCycleConsistentInvariantLearning(nn.Module):
    def __init__(self, input_dim, shared_dim, cycle_loss_weight=0.1):
        super(AdversarialCycleConsistentInvariantLearning, self).__init__()

        # Initialize the invariant projection layer (maps modality-specific features to shared space)
        self.invariant_projection = InvariantProjectionLayer(input_dim, shared_dim)

        # Initialize the two discriminators
        self.D_0 = Discriminator0(shared_dim)  # First discriminator for modality-invariant features
        self.D_1 = Discriminator1(shared_dim)  # Second discriminator for optimal feature blending

        # Cycle consistency loss weight
        self.cycle_loss_weight = cycle_loss_weight

    def forward(self, text_embedding, image_embedding, cycle_loss=False):
        # Project text and image embeddings into shared space
        projected_text = self.invariant_projection(text_embedding)
        projected_image = self.invariant_projection(image_embedding)

        # Get the modality-invariant weights using D_0
        w_t = 1 - self.D_0(projected_text)  # Weight for text modality (1 - D_0)
        w_v = 1 - self.D_0(projected_image)  # Weight for image modality (1 - D_0)

        # Compute the adversarial loss for D_1 (blending features)
        adv_loss = self.adversarial_loss(w_t, w_v, projected_text, projected_image)

        # If cycle consistency is included, apply cycle loss (optional)
        if cycle_loss:
            cycle_loss_value = self.cycle_loss_weight * self.compute_cycle_loss(text_embedding, image_embedding)
            return adv_loss + cycle_loss_value

        return adv_loss

    def adversarial_loss(self, w_t, w_v, projected_text, projected_image):
        """
        Computes the adversarial loss by blending features from the text and image modalities
        with respect to the optimal adversarial discriminator D1.
        The loss involves maximizing the modality-invariant alignment while reducing modality
        bias using weighted adversarial loss.
        """
        prob_text = self.D_1(projected_text)
        prob_image = self.D_1(projected_image)

        # Weighted adversarial loss for text modality
        loss_text = w_t * torch.log(prob_text + 1e-8)  # Adding epsilon for numerical stability
        loss_image = w_v * torch.log(1 - prob_image + 1e-8)  # Adding epsilon for numerical stability

        # Combine the losses
        loss = -(loss_text + loss_image).mean()  # Minimize the adversarial loss
        return loss

    def compute_cycle_loss(self, text_embedding, image_embedding):
        """
        Computes cycle consistency loss (ensures that both the modality-specific features and the
        projected invariant representations can be reconstructed back).
        Here, a simple MSE (Mean Squared Error) loss is used for the cycle consistency.
        """
        cycle_loss_value = F.mse_loss(text_embedding, image_embedding)
        return cycle_loss_value

    def jsd_loss(self, w_t, w_v, pe_t, pe_v):
        """
        Computes Jensen-Shannon Divergence (JSD) loss to measure the divergence between
        two distributions and align modality-invariant features.
        """
        # Compute the JSD between the weighted text and image distributions
        p_t = F.softmax(pe_t, dim=-1)
        p_v = F.softmax(pe_v, dim=-1)

        # Apply the weightings
        weighted_p_t = w_t * p_t
        weighted_p_v = w_v * p_v

        # Compute the Jensen-Shannon Divergence
        kl_t = torch.sum(weighted_p_t * torch.log(weighted_p_t / (weighted_p_t + weighted_p_v + 1e-8) + 1e-8), dim=-1)
        kl_v = torch.sum(weighted_p_v * torch.log(weighted_p_v / (weighted_p_t + weighted_p_v + 1e-8) + 1e-8), dim=-1)

        jsd_loss = 0.5 * (kl_t + kl_v).mean()
        return jsd_loss
