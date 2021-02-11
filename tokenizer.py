from typing import Tuple, List

import numpy as np
from transformers import ElectraTokenizer


class ElectraTokenizerOffset(ElectraTokenizer):
    def _whitespace_tokenize(self, text):
        """Runs basic whitespace cleaning and splitting on a piece of text."""
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens

    def _tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                for sub_token in self._tokenize_word(token):
                    split_tokens.append(sub_token)
        else:
            split_tokens = self._tokenize_word(text)
        return split_tokens

    def _tokenize_word(self, text, max_input_chars_per_word=100):
        output_tokens = []
        for token in self._whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > max_input_chars_per_word:
                output_tokens.append(token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                if sub_tokens:
                    output_tokens.extend(sub_tokens)

                if start > 0:
                    tok = '##' + token[start:]
                else:
                    tok = token[start:]
                output_tokens.append(tok)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def tokenize(tokenizer: ElectraTokenizerOffset, sentence: str) -> List[Tuple[str, int, int, int]]:
    tok_list = tokenizer.tokenize(sentence)
    res_list = []
    len_sent = len(sentence)
    begin = 0
    for idx, raw_tok in enumerate(tok_list):
        tok = ''.join(raw_tok.split('##'))
        while begin < len_sent and sentence[begin].isspace():
            begin += 1

        end = begin + len(tok)
        if end >= len_sent or sentence[end].isspace():
            sp = 1
        else:
            sp = 0
        res_list.append((raw_tok, sp, begin, end))
        begin = end
    return res_list


def get_convert_map(sentence: str, tok_list: List[Tuple[str, int, int, int]]) -> 'np.ndarray[np.int]':
    id_map = np.zeros(len(sentence) + 1, np.int)
    end_ids = [t[3] for t in tok_list]
    for id in end_ids:
        if id > len(sentence):
            print('error')
    id_map[end_ids] = 1
    id_map = np.cumsum(id_map)
    return id_map
