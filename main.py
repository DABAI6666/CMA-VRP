import torch
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
from utils.data import load_data, DataIterator
from mymodel import SimilarityModule, DetectionModule
from clip import CLIP
from model.aggregator import MaxAggregator, AttentionAggregator
from model.adv_module import AdversarialCycleConsistentInvariantLearning  # Import the new module
from model.graph_fusion import GraphFusionModule  # Assuming the graph fusion module is imported here
from utils.syntactic_utils import build_positionizer, build_dependencyizer, build_image_relizer, build_Part_of_Speechlizer
import pickle
import math

# Configs
DEVICE = "cuda:0"
NUM_WORKER = 1
BATCH_SIZE = 64
LR = 1e-3
L2 = 0  # 1e-5
NUM_EPOCH = 50
seed = 825
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_vrf(knowledge_base="data/vrf/vrf_base.pkl"):
    knowledge = pickle.load(open(knowledge_base, 'rb'))
    return knowledge

def prepare_data(batch, dependency_tokenizer, position_tokenizer, rel_tokenizer, knowledge):
    # Unpack batch data
    bert_tokens, lengths, token_masks, sens_lens, token_ranges, token_dependency_masks, \
    token_syntactic_position, token_edge_data, token_frequency_graph, pospeech_tokens, image_rel_matrix, \
    image_rel_mask, image_feature, entity_tags, tags = batch

    task_id = 0  # Example task_id, you might have specific logic to get this.
    task_caption = "some_caption"  # You need to fetch this depending on your dataset/task
    knw_values, knw_keys = knowledge.get(task_caption, (None, None))

    return bert_tokens, image_feature, token_dependency_masks, token_syntactic_position, image_rel_matrix, image_rel_mask, knw_values

def train():
    # ---  Load Config ---
    device = torch.device(DEVICE)
    num_workers = NUM_WORKER
    batch_size = BATCH_SIZE
    lr = LR
    l2 = L2
    num_epoch = NUM_EPOCH

    # --- Load VRF Data ---
    knowledge = load_vrf()

    position_tokenizer = build_positionizer("../unified_tags_datasets/no_none_unified_tags_txt/")
    dependency_tokenizer = build_dependencyizer("../unified_tags_datasets/no_none_unified_tags_txt/")
    rel_tokenizer = build_image_relizer("../unified_tags_datasets/no_none_image_path/")
    pospeech_tokenizer = build_Part_of_Speechlizer("../unified_tags_datasets/no_none_unified_tags_txt/")

    # --- Load Data ---
    dataset_dir = '../twitter/'

    train_set = load_data(
        f"{dataset_dir}/train.json",
        f"{dataset_dir}/train_image/",
        position_tokenizer,
        dependency_tokenizer,
        rel_tokenizer,
        pospeech_tokenizer,
    )

    test_set = load_data(
        f"{dataset_dir}/test.json",
        f"{dataset_dir}/test_image/",
        position_tokenizer,
        dependency_tokenizer,
        rel_tokenizer,
        pospeech_tokenizer,
    )

    # DataLoader setup
    train_loader = DataIterator(train_set)
    test_loader = DataIterator(test_set)

    # --- Build Model & Trainer ---
    similarity_module = SimilarityModule()
    similarity_module.to(device)
    detection_module = DetectionModule()
    detection_module.to(device)
    clip_module = CLIP(64)
    clip_module.to(device)
    loss_func_similarity = torch.nn.CosineEmbeddingLoss(margin=0.2)
    loss_func_clip = torch.nn.CrossEntropyLoss()
    loss_func_detection = torch.nn.CrossEntropyLoss()
    loss_func_skl = torch.nn.KLDivLoss(reduction='batchmean')
    optim_task_similarity = torch.optim.Adam(
        similarity_module.parameters(), lr=lr, weight_decay=l2
    )  # also called task1
    optimizer_task_clip = torch.optim.AdamW(
        clip_module.parameters(), lr=0.001, weight_decay=5e-4)
    optim_task_detection = torch.optim.Adam(
        detection_module.parameters(), lr=lr, weight_decay=l2
    )  # also called task2

    # --- Initialize Contrastive Learning ---
    from model.Contrastive_Learning import IntraGraphContrastiveLearning, InterGraphContrastiveLearning
    inter_cl = InterGraphContrastiveLearning(tau=0.07)
    intra_cl = IntraGraphContrastiveLearning(tau=0.07)

    # --- Initialize Graph Fusion ---
    aggregator = MaxAggregator()  # Using MaxAggregator for fusion, can change this to other types
    graph_fusion = GraphFusionModule(m_dim=128, aggregator=aggregator, readout_t=AttentionAggregator(hidden_size=64), readout_v=AttentionAggregator(hidden_size=64))

    # --- Initialize Invariant Feature Extractor ---
    adv_module = AdversarialCycleConsistentInvariantLearning(input_dim=128, shared_dim=64)

    # --- Model Training ---
    best_acc = 0
    step = 0
    for epoch in range(num_epoch):
        similarity_module.train()
        clip_module.train()
        corrects_pre_similarity = 0
        corrects_pre_detection = 0
        loss_similarity_total = 0
        loss_clip_total = 0
        loss_detection_total = 0
        similarity_count = 0
        detection_count = 0

        for i, batch in tqdm(enumerate(train_loader)):
            bert_tokens, image_feature, token_dependency_masks, token_syntactic_position, image_rel_matrix, image_rel_mask, knw_values = prepare_data(
                batch, dependency_tokenizer, position_tokenizer, rel_tokenizer, knowledge
            )

            bert_tokens = bert_tokens.to(device)
            image_feature = image_feature.to(device)
            token_dependency_masks = token_dependency_masks.to(device)
            token_syntactic_position = token_syntactic_position.to(device)
            image_rel_matrix = image_rel_matrix.to(device)
            image_rel_mask = image_rel_mask.to(device)

            # --- TASK1 Similarity ---
            text_aligned_match, image_aligned_match, pred_similarity_match = similarity_module(bert_tokens, image_feature)
            text_aligned_unmatch, image_aligned_unmatch, pred_similarity_unmatch = similarity_module(bert_tokens, image_feature.roll(shifts=3, dims=0))

            similarity_pred = torch.cat([pred_similarity_match.argmax(1), pred_similarity_unmatch.argmax(1)], dim=0)
            similarity_label_0 = torch.cat([torch.ones(pred_similarity_match.shape[0]), torch.zeros(pred_similarity_unmatch.shape[0])], dim=0).to(device)
            similarity_label_1 = torch.cat([torch.ones(pred_similarity_match.shape[0]), -1 * torch.ones(pred_similarity_unmatch.shape[0])], dim=0).to(device)

            text_aligned_4_task1 = torch.cat([text_aligned_match, text_aligned_unmatch], dim=0)
            image_aligned_4_task1 = torch.cat([image_aligned_match, image_aligned_unmatch], dim=0)
            loss_similarity = loss_func_similarity(text_aligned_4_task1, image_aligned_4_task1, similarity_label_1)

            optim_task_similarity.zero_grad()
            loss_similarity.backward()
            optim_task_similarity.step()

            corrects_pre_similarity += similarity_pred.eq(similarity_label_0).sum().item()

            # --- Inter-Graph Contrastive Loss ---
            inter_loss = inter_cl.inter_graph_contrastive_loss(text_aligned_4_task1, image_aligned_4_task1, batch_size)

            # --- Intra-Graph Contrastive Loss ---
            intra_loss = intra_cl.intra_graph_contrastive_loss(image_aligned_4_task1, [text_aligned_4_task1], batch_size)

            # Calculate contrastive loss
            cl_loss = inter_loss + intra_loss

            # --- Graph Fusion ---
            fused_repr = graph_fusion(text_aligned_4_task1, image_aligned_4_task1, mask=torch.ones(batch_size, 128).to(device))

            # --- Invariant Feature Extraction ---
            adv_loss = adv_module(text_aligned_4_task1, image_aligned_4_task1)

            # Combine graph fusion and invariant features
            combined_repr = torch.cat([fused_repr, adv_loss], dim=-1)

            # --- TASK2 Detection ---
            pre_detection, attention_score, skl_score = detection_module(bert_tokens, image_feature, combined_repr, combined_repr)
            loss_detection = loss_func_detection(pre_detection, torch.tensor([1])) + 0.5 * loss_func_skl(attention_score, skl_score)  # Placeholder for actual labels

            # Apply final total loss
            lambda_cl = 0.5
            beta_adv = 0.2
            total_loss = loss_detection + lambda_cl * cl_loss + beta_adv * loss_detection

            optim_task_detection.zero_grad()
            total_loss.backward()
            optim_task_detection.step()

            pre_label_detection = pre_detection.argmax(1)
            corrects_pre_detection += pre_label_detection.sum().item()

            loss_detection_total += loss_detection.item()
            detection_count += bert_tokens.shape[0]

        acc_detection_train = corrects_pre_detection / detection_count

        # --- Test ---
        acc_detection_test, loss_detection_test, cm_detection, cr_detection = test(clip_module, detection_module, test_loader)

        if acc_detection_test > best_acc:
            best_acc = acc_detection_test
            torch.save(detection_module.state_dict(), "best_detection_model.pth")

        print(f"EPOCH = {epoch + 1}, acc_detection_train = {acc_detection_train}, acc_detection_test = {acc_detection_test}, best_acc = {best_acc}")
        print(f"Confusion Matrix: {cm_detection}")
        print(f"Classification Report: {cr_detection}")

def test(clip_module, detection_module, test_loader, dependency_tokenizer, position_tokenizer, rel_tokenizer, knowledge):
    clip_module.eval()
    detection_module.eval()
    device = torch.device(DEVICE)
    loss_func_detection = torch.nn.CrossEntropyLoss()

    detection_count = 0
    loss_detection_total = 0
    detection_pre_label_all = []
    detection_label_all = []

    with torch.no_grad():
        for batch in test_loader:
            # Prepare data using the provided tokenizers and knowledge
            bert_tokens, image_feature, token_dependency_masks, token_syntactic_position, image_rel_matrix, image_rel_mask, knw_values = prepare_data(
                batch, dependency_tokenizer, position_tokenizer, rel_tokenizer, knowledge
            )

            bert_tokens = bert_tokens.to(device)
            image_feature = image_feature.to(device)
            token_dependency_masks = token_dependency_masks.to(device)
            token_syntactic_position = token_syntactic_position.to(device)
            image_rel_matrix = image_rel_matrix.to(device)
            image_rel_mask = image_rel_mask.to(device)

            # Forward pass through the clip module
            image_aligned, text_aligned = clip_module(image_feature, bert_tokens)
            logits = torch.matmul(image_aligned, text_aligned.T) * math.exp(0.07)
            labels = torch.arange(bert_tokens.size(0)).to(device)

            # Forward pass through the detection module
            pre_detection, attention_score, skl_score = detection_module(bert_tokens, image_feature, text_aligned, image_aligned)
            loss_detection = loss_func_detection(pre_detection, torch.tensor([1])) 
            pre_label_detection = pre_detection.argmax(1)

            # Collect results for evaluation
            loss_detection_total += loss_detection.item()
            detection_count += bert_tokens.shape[0]

            detection_pre_label_all.append(pre_label_detection.detach().cpu().numpy())
            detection_label_all.append(torch.tensor([1]).detach().cpu().numpy())

        # Calculate final metrics
        loss_detection_test = loss_detection_total / detection_count
        detection_pre_label_all = np.concatenate(detection_pre_label_all, 0)
        detection_label_all = np.concatenate(detection_label_all, 0)

        acc_detection_test = accuracy_score(detection_pre_label_all, detection_label_all)
        cm_detection = confusion_matrix(detection_pre_label_all, detection_label_all)
        cr_detection = classification_report(detection_pre_label_all, detection_label_all, target_names=['Real News', 'Fake News'], digits=3)

    return acc_detection_test, loss_detection_test, cm_detection, cr_detection


def main():
    setup_seed(seed)
    train()

if __name__ == "__main__":
    main()
