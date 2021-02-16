import os
import random
import logging
import argparse

import streamlit as st
import numpy as np
import kss
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoModelForTokenClassification

import SessionState
from loader import convert_input_file_to_tensor_dataset
from tokenizer import ElectraTokenizerOffset, tokenize
from streamlit_util import create_explainer, produce_text_display
from utils import token_check

logger = logging.getLogger(__name__)

MODEL_PATH = './model'
LABEL_PATH = './label/label.txt'
NO_CUDA = False
BATCH_SIZE = 32

def get_device():
    return "cuda" if torch.cuda.is_available() and not NO_CUDA else "cpu"

@st.cache(allow_output_mutation=True)
def get_sample():
    sample = [line.strip() for line in open('./sample.txt')]
    return sample

@st.cache(allow_output_mutation=True)
def get_args():
    return torch.load(os.path.join(MODEL_PATH, 'training_args.bin'))

@st.cache(allow_output_mutation=True)
def load_model(device):
    if not os.path.exists(MODEL_PATH):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        tokenizer = ElectraTokenizerOffset.from_pretrained(MODEL_PATH, do_lower_case=False)
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model, tokenizer

def predict(text):

    args = get_args()
    device = get_device()
    model, tokenizer = load_model(device)
    label_lst = [label.strip() for label in open(LABEL_PATH)]

    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index

    if len(text) > 128:
        raw_lines = kss.split_sentences(text)
        lines = [tokenize(tokenizer, line) for line in raw_lines]
    else:
        line = text.strip()
        lines, raw_lines = [tokenize(tokenizer, line)], [line]
    dataset = convert_input_file_to_tensor_dataset(lines, args, tokenizer, pad_token_label_id)
    
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE)

    all_slot_label_mask = None
    preds = None

    for batch in data_loader:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": None}
            if args.model_type != "distilkobert":
                inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                all_slot_label_mask = batch[3].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=2)
    slot_label_map = {i: label for i, label in enumerate(label_lst)}
    preds_list = [[] for _ in range(preds.shape[0])]
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                preds_list[i].append(slot_label_map[preds[i][j]])


    entity_list = []
    for line, tokens, preds in zip(raw_lines, lines, preds_list):
        flag = False
        prev_word = None
        entity = []
        for word, pred in zip(tokens, preds):
            if flag and 'B-' in pred:
                end = word[2]
                check = token_check(prev_word[0],prev_word[1],tag)
                if check[0]:
                    end = end - check[1]
                entity.append((start,end,tag))
                start = word[2]
                tag = pred[2::]
            elif flag and pred =='O':
                end = word[2]
                check = token_check(prev_word[0],prev_word[1],tag)
                if check[0]:
                    end = end - check[1]
                entity.append((start,end,tag))
                flag = False
            elif 'B-' in pred:
                start = word[2]
                flag = True
                tag = pred[2::]
            prev_word = word
        if flag:
            end = len(tokens)
            entity.append((start,end,tag))
        entity_list.append(entity)
    return entity_list, raw_lines

if __name__ == "__main__":
    color_dict = {
          'PS': ["#17becf", "#9edae5"], # blues
          'LC': ["#9467bd", "#c5b0d5"], # purples
          'OG': ["#74c476", "#c7e9c0"], # greens
          'DT': ["#e377c2", "#f7b6d2"], # pinks
          'TI': ["#e3c477", "#f7b6d2"] # pinks
    }
    ent_dict = {
          "인명": "PS",
          "장소": "LC",
          "기관": "OG",
          "날짜": "DT",
          "시간": "TI"
    }
    st.set_page_config(page_title='Dialog-ELECTRA NER', page_icon=':fire:')
    st.title("Korean Named Entity Recognition")
    st.text("")
    st.subheader('NER Model Description')
    st.markdown("""- 대화체 언어모델인 Dialog-ELECTRA를 fine-tuning하였습니다.
                   \n- Example 버튼을 눌러 샘플 텍스트를 변경할 수 있습니다.""")
    explainer = create_explainer(color_dict, ent_dict)
    st.markdown(explainer, unsafe_allow_html=True)

    sample_list = get_sample()
    
    user_prompt = "What text do you want to predict on?"

    text = st.empty()
    session_state = SessionState.get(name='', user_input="")
    
    if session_state.user_input == "":
        user_input = text.text_area(user_prompt, height=150)
        session_state.user_input = user_input
    else:
        user_input = text.text_area(user_prompt, session_state.user_input, height=150)
        session_state.user_input = user_input

    col1, col2 = st.beta_columns([0.15, 1])
    
    if col1.button('Example'):
        default_input = random.choice(sample_list)
        user_input = text.text_area(user_prompt, default_input, height=150)
        session_state.user_input = user_input
    
    if col2.button("Analysis"):
        entity, lines = predict(session_state.user_input)
        st.subheader("Prediction Result")
        display = produce_text_display(lines, entity, color_dict)
        
        st.markdown(display, unsafe_allow_html=True)
