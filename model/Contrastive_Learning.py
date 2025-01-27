import torch
import torch.nn.functional as F
from model.aggregator import MultimodalGraphReadout

class InterGraphContrastiveLearning:
    def __init__(self, tau=0.07, aggregator=None):
        self.tau = tau
        self.aggregator = aggregator if aggregator else MultimodalGraphReadout(64, 64, 64)

    def inter_graph_contrastive_loss(self, text_graph_rep, image_graph_rep, batch_size):
        """
        Calculate the Inter-Graph Contrastive Loss to align text and image representations.

        Args:
        - text_graph_rep (Tensor): Textual graph representation (Batch x Features)
        - image_graph_rep (Tensor): Visual graph representation (Batch x Features)
        - batch_size (int): Number of samples in the batch

        Returns:
        - inter_loss (Tensor): Inter-Graph Contrastive Loss
        """
        # Aggregate text and image graph representations
        text_rep, _ = self.aggregator(text_graph_rep)
        image_rep, _ = self.aggregator(image_graph_rep)

        # Normalize representations
        text_rep = F.normalize(text_rep, p=2, dim=-1)
        image_rep = F.normalize(image_rep, p=2, dim=-1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(text_rep, image_rep.T) / self.tau

        # Positive samples (diagonal of the similarity matrix)
        positive_samples = torch.eye(batch_size).to(text_rep.device)

        # Logits for contrastive loss
        logits = similarity_matrix - similarity_matrix.max(dim=-1, keepdim=True)[0]  # Stability
        exp_logits = torch.exp(logits)
        positive_logits = exp_logits * positive_samples
        sum_exp_logits = exp_logits.sum(dim=-1, keepdim=True)

        # Contrastive loss
        inter_loss = -torch.log(positive_logits.sum() / sum_exp_logits.sum())

        return inter_loss


class IntraGraphContrastiveLearning:
    def __init__(self, tau=0.07):
        self.tau = tau

    def intra_graph_contrastive_loss(self, graph_rep, negative_graphs, batch_size):
        """
        Calculate the Intra-Graph Contrastive Loss within each modality (text or image).

        Args:
        - graph_rep (Tensor): Graph representation after cross-modal fusion (Batch x Features)
        - negative_graphs (List[Tensor]): List of negative samples (each a batch x features)
        - batch_size (int): Number of samples in the batch

        Returns:
        - intra_loss (Tensor): Intra-Graph Contrastive Loss
        """
        # Normalize graph representations
        graph_rep = F.normalize(graph_rep, p=2, dim=-1)
        negative_graphs = [F.normalize(neg, p=2, dim=-1) for neg in negative_graphs]

        # Calculate cosine similarities
        positive_sim = torch.matmul(graph_rep, graph_rep.T) / self.tau  # Similarity between positive pairs

        # Similarity with negative samples
        negative_sims = [
            torch.matmul(graph_rep, neg.T) / self.tau for neg in negative_graphs
        ]

        # Calculate the InfoNCE loss for positive and negative pairs
        positive_samples = torch.eye(batch_size).to(graph_rep.device)
        exp_positive_sim = torch.exp(positive_sim)
        exp_negative_sims = [torch.exp(neg_sim) for neg_sim in negative_sims]

        intra_loss = -torch.log(
            exp_positive_sim / (torch.sum(exp_positive_sim) + sum(torch.sum(neg_sim) for neg_sim in exp_negative_sims))
        )

        return intra_loss
