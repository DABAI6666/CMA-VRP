import torch
import os
import json
import pickle
from transformers import BertTokenizer
from unified_tags_datasets.dependency_graph import tokenize as text_sample_tokenize
import numpy as np


class Instance(object):
    def __init__(self, tokenizer, sentence_pack, dependency_mask_pack, edge_data_pack,
                 syntax_position_pack, frequency_graph_pack, pospeech_data_pack, image_relation_pack,
                 image_feature_pack,
                 position_tokenizer, dependency_tokenizer, rel_tokenizer, pospeech_tokenizer, args):

        # 处理图像关系和标签
        bbox_labels, rel_labels_text = image_relation_pack["bbox_labels"], image_relation_pack["rel_labels"]
        rel_labels = []
        for rel_text in rel_labels_text:
            label_s = rel_text.split("_")
            label_1, label_2 = label_s[0], label_s[-2]
            relation_word = label_s[1].split("=>")[1].strip()
            rel_labels.append([int(label_1), str(relation_word), int(label_2[-1])])

        # 文本数据处理
        self.sentence = text_sample_tokenize(" ".join(sentence_pack["token"]))
        self.tokens = self.sentence.split()
        self.token_range = []
        image_feature_pack = torch.tensor(image_feature_pack)
        self.image_feature_len, self.image_feature_dim = image_feature_pack.shape
        self.sen_length = len(self.tokens)
        self.bert_tokens = tokenizer.encode(self.sentence)
        self.length = len(self.bert_tokens)

        # 最大序列长度设置
        max_sequence_len = max(args.max_sequence_len, self.length)

        # 初始化各种图形和特征数据
        self.dependency_mask_seq = torch.zeros(max_sequence_len, max_sequence_len).long()
        self.edge_data_seq = torch.zeros(max_sequence_len, max_sequence_len).long()
        self.syntax_position_seq = torch.zeros(max_sequence_len, max_sequence_len).long()
        self.frequency_graph = torch.zeros(max_sequence_len, max_sequence_len).long()

        # 获取依赖关系、位置、图像关系等图形数据
        position_matrix = position_tokenizer.position_to_index(syntax_position_pack)
        dependency_edge = dependency_tokenizer.dependency_to_index(edge_data_pack, dependency_mask_pack)
        image_rel_matrix, image_mask = rel_tokenizer.relation_to_graph(self.image_feature_len, rel_labels)
        pospeech_data_pack = pospeech_tokenizer.pospeech_to_index(pospeech_data_pack)

        # 填充各类数据
        self.dependency_mask_seq[0:self.length, 0:self.length] = torch.from_numpy(dependency_mask_pack).long()
        self.syntax_position_seq[0:self.length, 0:self.length] = torch.from_numpy(position_matrix).long()
        self.edge_data_seq[0:self.length, 0:self.length] = torch.from_numpy(dependency_edge).long()
        self.frequency_graph[0:self.length, 0:self.length] = torch.from_numpy(frequency_graph_pack).long()
        self.image_rel_matrix = torch.from_numpy(image_rel_matrix).long()
        self.image_rel_mask = torch.from_numpy(image_mask).long()
        self.image_feature = image_feature_pack
        self.bert_tokens_padding = torch.zeros(max_sequence_len).long()
        self.pospeech_padding = torch.zeros(max_sequence_len).long()

        self.entity_tags = torch.zeros(max_sequence_len).long()
        self.tags = torch.zeros(max_sequence_len, max_sequence_len).long()
        self.mask = torch.zeros(max_sequence_len)

        # 填充pospeech数据
        self.pospeech_padding[0:self.length] = torch.tensor(pospeech_data_pack).long()

        # 处理token range
        token_start = 1
        for i, w in enumerate(self.tokens):
            token_list.append(tokenizer.tokenize(w))
            token_end = token_start + len(tokenizer.encode(w, add_special_tokens=False))
            self.token_range.append([token_start, token_end - 1])
            token_start = token_end

        # 设置标签
        self.entity_tags[self.length:] = -1
        self.entity_tags[0] = -1
        self.entity_tags[self.length - 1] = -1
        self.tags[:, :] = -1
        for i in range(1, self.length - 1):
            for j in range(i, self.length - 1):
                self.tags[i][j] = 0

        # 设置实体标签和关系标签
        for index, triple in enumerate(sentence_pack['label_list']):
            triple = triple[0]
            begin_entity_infor = triple['beg_ent']
            end_entity_infor = triple['sec_ent']
            relation_tags = int(unified2id[triple['relation'].lower()])

            # Begin entity
            begin_entity_tags = int(unified2id[begin_entity_infor["tags"].lower()])
            begin_entity_span = begin_entity_infor["pos"]
            end_entity_tags = int(unified2id[end_entity_infor["tags"].lower()])
            end_entity_span = end_entity_infor["pos"]

            # 设置标签
            l, r = begin_entity_span[0], begin_entity_span[1] - 1
            start = self.token_range[l][0]
            end = self.token_range[r][1]
            for i in range(start, end + 1):
                for j in range(i, end + 1):
                    self.tags[i][j] = begin_entity_tags
            for i in range(l, r + 1):
                set_tag = begin_entity_tags
                al, ar = self.token_range[i]
                self.entity_tags[al] = set_tag
                self.entity_tags[al + 1:ar + 1] = -1
                self.tags[al + 1:ar + 1, :] = -1
                self.tags[:, al + 1:ar + 1] = -1

            # End entity
            l, r = end_entity_span[0], end_entity_span[1] - 1
            start = self.token_range[l][0]
            end = self.token_range[r][1]
            for i in range(start, end + 1):
                for j in range(i, end + 1):
                    self.tags[i][j] = end_entity_tags
            for i in range(l, r + 1):
                set_tag = end_entity_tags
                pl, pr = self.token_range[i]
                self.entity_tags[pl] = set_tag
                self.entity_tags[pl + 1:pr + 1] = -1
                self.tags[pl + 1:pr + 1, :] = -1
                self.tags[:, pl + 1:pr + 1] = -1

            al, ar = begin_entity_span[0], begin_entity_span[1] - 1
            pl, pr = end_entity_span[0], end_entity_span[1] - 1
            for i in range(al, ar + 1):
                for j in range(pl, pr + 1):
                    sal, sar = self.token_range[i]
                    spl, spr = self.token_range[j]
                    self.tags[sal:sar + 1, spl:spr + 1] = -1
                    if i > j:
                        self.tags[spl][sal] = relation_tags
                    else:
                        self.tags[sal][spl] = relation_tags


class DataIterator(object):
    def __init__(self, instances, args):
        self.instances = instances
        self.args = args
        self.batch_count = math.ceil(len(instances) / args.batch_size)

    def get_batch(self, index):
        sentence_ids = []
        sentences = []
        sens_lens = []
        token_ranges = []
        bert_tokens = []
        lengths = []
        token_masks = []
        entity_tags = []
        tags = []
        token_dependency_masks = []
        token_syntactic_position = []
        token_edge_datas = []
        token_frequency_graph = []
        image_feature = []
        image_rel_matrix = []
        image_rel_mask = []
        pospeech = []

        for i in range(index * self.args.batch_size,
                       min((index + 1) * self.args.batch_size, len(self.instances))):
            sentences.append(self.instances[i].sentence)
            sens_lens.append(self.instances[i].sen_length)
            token_ranges.append(self.instances[i].token_range)

            bert_tokens.append(self.instances[i].bert_tokens_padding)
            lengths.append(self.instances[i].length)
            token_masks.append(self.instances[i].mask)
            token_edge_datas.append(self.instances[i].edge_data_seq)
            token_dependency_masks.append(self.instances[i].dependency_mask_seq)
            token_syntactic_position.append(self.instances[i].syntax_position_seq)
            token_frequency_graph.append(self.instances[i].frequency_graph)
            pospeech.append(self.instances[i].pospeech_padding)
            image_rel_matrix.append(self.instances[i].image_rel_matrix)
            image_rel_mask.append(self.instances[i].image_rel_mask)
            image_feature.append(self.instances[i].image_feature)
            entity_tags.append(self.instances[i].entity_tags)
            tags.append(self.instances[i].tags)

        # 转换成tensor
        bert_tokens = torch.stack(bert_tokens).to(self.args.device)
        lengths = torch.tensor(lengths).to(self.args.device)
        token_masks = torch.stack(token_masks).to(self.args.device)
        token_dependency_masks = torch.stack(token_dependency_masks).to(self.args.device)
        token_syntactic_position = torch.stack(token_syntactic_position).to(self.args.device)
        token_edge_data = torch.stack(token_edge_datas).to(self.args.device)
        token_frequency_graph = torch.stack(token_frequency_graph).to(self.args.device)
        pospeech_tokens = torch.stack(pospeech).to(self.args.device)
        image_rel_matrix = torch.stack(image_rel_matrix).to(self.args.device)
        image_rel_mask = torch.stack(image_rel_mask).to(self.args.device)
        image_feature = torch.stack(image_feature).to(self.args.device)

        entity_tags = torch.stack(entity_tags).to(self.args.device)
        tags = torch.stack(tags).to(self.args.device)

        return bert_tokens, lengths, token_masks, sens_lens, token_ranges, token_dependency_masks, \
               token_syntactic_position, token_edge_data, token_frequency_graph, pospeech_tokens, \
               image_rel_matrix, image_rel_mask, image_feature, entity_tags, tags


def load_data(text_path, image_path, position_tokenizer, dependency_tokenizer, rel_tokenizer, pospeech_tokenizer, args):
    # 读取数据
    fout_undir_file = os.path.join(f"{text_path}undirbert.dependency_graph")
    syntax_position_file = os.path.join(f"{text_path}bert.syntaxPosition")
    dependency_type_file = os.path.join(f"{text_path}bert.dependency")
    frequency_file = os.path.join(f"{text_path}frequency.graph")
    pospeech_flie = os.path.join(f"{text_path}bert.pos_sequence")

    image_relation_file = os.path.join(f"{image_path}/new_prediction.pickle")
    image_feature_file = os.path.join(f"{image_path}/new_output_feature.pickle")

    text_file = open(text_path, "r")
    dependency_undir = open(fout_undir_file, 'rb')
    edge_type = open(dependency_type_file, "rb")
    syntax_position = open(syntax_position_file, "rb")
    frequency_file = open(frequency_file, "rb")
    pospeech_flie = open(pospeech_flie, "rb")
    image_relation_file = open(image_relation_file, "rb")
    image_feature_file = open(image_feature_file, "rb")

    # 读取文件内容
    sentence_packs = json.load(text_file)
    dependency_mask_data = pickle.load(dependency_undir)
    edge_data = pickle.load(edge_type)
    syntax_position_data = pickle.load(syntax_position)
    frequency_graph_data = pickle.load(frequency_file)
    pospeech_data = pickle.load(pospeech_flie)
    image_infor_data = pickle.load(image_relation_file)
    image_feature_data = pickle.load(image_feature_file)

    # 关闭文件
    dependency_undir.close()
    edge_type.close()
    syntax_position.close()
    text_file.close()
    pospeech_flie.close()

    # 创建实例列表
    instances = []
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_path)

    # 构建实例
    for i, sentence_pack in enumerate(sentence_packs):
        instances.append(Instance(tokenizer, sentence_pack, dependency_mask_data[i], edge_data[i],
                                  syntax_position_data[i], frequency_graph_data[i], pospeech_data[i],
                                  image_infor_data[i], image_feature_data[i], position_tokenizer,
                                  dependency_tokenizer, rel_tokenizer, pospeech_tokenizer, args))

    return instances
