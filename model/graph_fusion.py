import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphFusionModule(nn.Module):
    def __init__(self, m_dim, aggregator, readout_t, readout_v):
        """
        Args:
            m_dim (int): The dimensionality of the final fused representation.
            aggregator (nn.Module): The aggregation function used to combine node embeddings.
            readout_t (nn.Module): The readout function for the textual modality.
            readout_v (nn.Module): The readout function for the visual modality.
        """
        super(GraphFusionModule, self).__init__()

        self.aggregator = aggregator  # e.g., AttentionAggregator, MeanAggregator, or MaxAggregator
        self.readout_t = readout_t  # Textual readout
        self.readout_v = readout_v  # Visual readout
        self.project_m = nn.Linear(m_dim, m_dim)  # Projecting to a common feature space

        # Additional layers for visual reasoning features
        self.vrf_project_t = nn.Linear(m_dim, m_dim)  # Project VRF for text modality
        self.vrf_project_v = nn.Linear(m_dim, m_dim)  # Project VRF for visual modality

    def forward(self, text_graph_repr, visual_graph_repr, vrf_t, vrf_v, mask):
        """
        Forward pass to fuse the textual and visual graph representations with VRF.

        Args:
            text_graph_repr (Tensor): Textual graph representations (Batch x Features)
            visual_graph_repr (Tensor): Visual graph representations (Batch x Features)
            vrf_t (Tensor): Textual visual reasoning features (Batch x Features)
            vrf_v (Tensor): Visual visual reasoning features (Batch x Features)
            mask (Tensor): A mask to specify valid positions in the graph for aggregation

        Returns:
            Tensor: Fused representation
        """
        # Aggregate textual graph representations
        text_aggregated, _ = self.readout_t(text_graph_repr, mask)  # Textual graph readout

        # Aggregate visual graph representations
        visual_aggregated, _ = self.readout_v(visual_graph_repr, mask)  # Visual graph readout

        # Optionally, apply visual reasoning features (VRF) for text and visual modalities
        text_with_vrf = text_aggregated + self.vrf_project_t(vrf_t)  # Adding VRF to text representation
        visual_with_vrf = visual_aggregated + self.vrf_project_v(vrf_v)  # Adding VRF to visual representation

        # Concatenate the aggregated textual and visual features along with VRF features
        fused_repr = torch.cat([text_with_vrf, visual_with_vrf], dim=-1)  # Concatenate text and visual

        # Optionally, apply an aggregator to fuse the features
        fused_repr = self.aggregator(fused_repr)  # Aggregator applied on the concatenated features

        # Project the fused representation into the desired dimensional space
        final_fused_repr = F.relu(self.project_m(fused_repr))

        return final_fused_repr

