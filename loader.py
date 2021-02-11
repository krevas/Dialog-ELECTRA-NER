import os
import copy
import json
import logging

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset

from tokenizer import tokenize

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    """

    def __init__(self, guid, words, labels):
        self.guid = guid
        self.words = words
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_ids = label_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def convert_input_file_to_tensor_dataset(lines,
                                         args,
                                         tokenizer,
                                         pad_token_label_id,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []

    for words in lines:
        tokens = []
        slot_label_mask = []
        for word in words:
            tokens.extend([word[0]])
            slot_label_mask.extend([0])

        special_tokens_count = 2
        if len(tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[: (args.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[:(args.max_seq_len - special_tokens_count)]
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [pad_token_label_id]

        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = args.max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_slot_label_mask.append(slot_label_mask)

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask)

    return dataset

def ner_convert_examples_to_features(
        args,
        examples,
        tokenizer,
        label_lst,
        max_seq_length,
        task,
        pad_token_label_id=-100,
):
    label_map = {label: i for i, label in enumerate(label_lst)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example {} of {}".format(ex_index, len(examples)))

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = word[0]
            tokens.extend([word_tokens])
            label_ids.extend([label_map[label]])

        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]

        tokens += [tokenizer.sep_token]
        label_ids += [pad_token_label_id]

        tokens = [tokenizer.cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids

        token_type_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length
        label_ids += [pad_token_label_id] * padding_length
        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s " % " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label_ids=label_ids)
        )
    return features


class NerProcessor(object):
    """Processor for the Naver NER data set """

    def __init__(self, args, tokenizer):
        self.args = args
        self.tag_list = set()
        self.tag_list.add('O')
        self.tokenizer = tokenizer
        self.tag_map = {'ORG':'OG','DAT':'DT','PER':'PS','LOC':'LC','TIM':'TI'}

    def get_labels(self):
        return sorted(list(self.tag_list))

    @classmethod
    def _read_file(cls, input_files):
        dataset = []
        for input_file in input_files:
            with open(input_file) as f:
                data = json.load(f)
                dataset.extend(data['document'])
        return dataset

    def _create_examples(self, dataset, set_type):
        examples = []
        row_cnt = 0
        for row in dataset:
            if row['sentence'].strip() and len(row['entity']) != 0:
                token_list = tokenize(self.tokenizer, row['sentence'])
                bio_list = ['O' for _ in token_list]
                entity_check = False
                for entity in row['entity']:
                    tag = entity['tag']
                    if tag in self.tag_map.keys():
                        tag = self.tag_map.get(tag)
                    if tag not in ['DT','LC','OG','PS','TI']:continue
                    entity_in = False
                    for idx, token in enumerate(token_list):
                        if token[2] == entity['start']:
                            bio_list[idx] = 'B-'+tag
                            self.tag_list.add('B-'+tag)
                            entity_check = True
                            entity_in = True
                        elif entity_in and token[2] >= entity['end']:
                            break
                        elif entity_in:
                            bio_list[idx] = 'I-'+tag
                            self.tag_list.add('I-'+tag)
                
                assert len(token_list) == len(bio_list)
                
                if entity_check:
                    guid = "%s-%s" % (set_type, row_cnt)
                    examples.append(InputExample(guid=guid, words=token_list, labels=bio_list))
                    row_cnt += 1
        
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file
        logger.info("LOOKING AT {}".format(file_to_read))
        return self._create_examples(self._read_file(file_to_read), mode)


def ner_load_and_cache_examples(args, tokenizer, mode):
    processor = NerProcessor(args, tokenizer)
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            str(args.task),
            str(args.max_seq_len),
            mode
        )
    )
    cached_label_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_label".format(
            str(args.task),
            str(args.max_seq_len),
            mode
        )
    )
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        labels = [label.strip() for label in open(cached_label_file)]
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise ValueError("For mode, only train, dev, test is avaiable")

        pad_token_label_id = CrossEntropyLoss().ignore_index
        features = ner_convert_examples_to_features(
            args,
            examples,
            tokenizer,
            processor.get_labels(),
            max_seq_length=args.max_seq_len,
            task=args.task,
            pad_token_label_id=pad_token_label_id
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
        labels = processor.get_labels()
        with open(cached_label_file,'w') as writer:
            for label in labels:
                writer.write(label+'\n')
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)
    
    return (dataset, labels)

