#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import pos_tag
from collections import defaultdict
import re
import string

import numpy as np
from tqdm import tqdm

import gensim
import transformers

from typing import List


def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 


def get_candidates(lemma, pos, get_lexemes=False) -> List[str]:
    # Part 1
    candidates = []
    for synset in wn.synsets(lemma, pos=pos): # iterate through all synsets
        lexemes = synset.lemmas() # lexemes in synset contain possible candidates
        for lexeme in lexemes:
            lemma_candidate = lexeme.name().replace('_', ' ').lower() # get name of lemma, replace "_" with " ", and lowercase
            if lemma_candidate != lemma: # only add to candidate set if lemmas is different from input lemma
                if not get_lexemes:
                    candidates.append(lemma_candidate)
                else:
                    candidates.append(lexeme)

    if not get_lexemes:
        return list(set(candidates))
    else:
        return candidates


def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'


def wn_frequency_predictor(context : Context) -> str:
    lemma = context.lemma
    pos = context.pos
    candidates = get_candidates(lemma, pos, get_lexemes=True)

    candidate_freqs = defaultdict(int)
    for l in candidates:
        candidate_freqs[l.name().replace('_', ' ').lower()] += l.count()
    return sorted(candidate_freqs.items(), reverse=True, key=lambda i: i[1])[0][0]


def wn_simple_lesk_predictor(context : Context) -> str:
    lemma = context.lemma
    pos = context.pos

    lemma_context_tokenized = normalize_tokenize(context.left_context).union(normalize_tokenize(context.right_context))

    synsets_overlaps = []

    for synset in wn.synsets(lemma, pos=pos):
        if len(synset.lemmas()) == 1 and synset.lemmas()[0].name().lower().replace('_', '') == lemma: # skip synset if it only has 1 element with the target lemma
            continue
        examples = synset.examples()
        definition = synset.definition()

        if examples:
            examples_tokenized = set.union(*[normalize_tokenize(example) for example in examples])
        else:
            examples_tokenized = set()

        definition_tokenized = normalize_tokenize(definition)
        hypernym_examples_definitions_tokenized = get_examples_definitions_of_hypernyms_as_tokens(synset)
        joined_set = examples_tokenized.union(definition_tokenized).union(hypernym_examples_definitions_tokenized)

        overlap_len = get_overlap_len(lemma_context_tokenized, joined_set)
        synsets_overlaps.append((synset, overlap_len))

    synsets_overlaps = sorted(synsets_overlaps, key=lambda x: x[1], reverse=True) # sort in descending order by overlap count

    if len(synsets_overlaps) == 1: # only one synset of the target lemma
        return get_most_freq_lexeme_in_synset(synsets_overlaps[0][0], lemma_filter=lemma)

    else:
        synsets_overlaps_with_max_overlap = [pair for pair in synsets_overlaps if pair[1] == synsets_overlaps[0][1]] # get all synsets-overlap pairs with largest overlap
        if len(synsets_overlaps_with_max_overlap) == 1:
            return get_most_freq_lexeme_in_synset(synsets_overlaps_with_max_overlap[0][0], lemma_filter=lemma)
        else: # tie between synsets
            most_freq_lexemes_and_counts = [get_most_freq_lexeme_in_synset(synset, lemma_filter=lemma, get_count=True) for synset, overlap in synsets_overlaps_with_max_overlap]
            return sorted(most_freq_lexemes_and_counts, reverse=True, key=lambda i: i[1])[0][0]


def get_most_freq_lexeme_in_synset(synset, lemma_filter=None, get_count=False):
    lexeme_freqs = defaultdict(int)
    for l in synset.lemmas():
        l_name = l.name().replace('_', ' ').lower()
        if l_name == lemma_filter: continue  # don't count lemmas if they are the same as target lemma
        lexeme_freqs[l_name] = l.count()

    if get_count:
        sorted_lexeme_freqs = sorted(lexeme_freqs.items(), reverse=True, key=lambda i: i[1])
        return sorted_lexeme_freqs[0][0], sorted_lexeme_freqs[0][1]

    else:
        return sorted(lexeme_freqs.items(), reverse=True, key=lambda i: i[1])[0][0]


def get_overlap_len(s1: set, s2: set):
    return len(s1.intersection(s2))


def get_examples_definitions_of_hypernyms_as_tokens(synset):
    hypernym_synsets = synset.hypernyms()
    tokens = set()

    for synset in hypernym_synsets:
        examples = synset.examples()
        definition = synset.definition()

        if examples:
            examples_tokenized = set.union(*[normalize_tokenize(example) if example else set() for example in examples])
        else:
            examples_tokenized = set()

        definition_tokenized = normalize_tokenize(definition)

        tokens = tokens.union(examples_tokenized.union(definition_tokenized))

    return tokens


def get_max_length_contexts(contexts):
    return max([len(context.left_context) + len(context.right_context) + 1 for context in contexts])


def normalize_tokenize(s):
    if type(s) == list:
      s = " ".join(s)

    tokens = set()
    s_orig_raw_tokens = tokenize(s)

    for word in s_orig_raw_tokens:
        word = word.lower() # make word lowercase
        word = re.sub(r'\d+', '', word) # remove numbers
        word = word.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
        if word not in set(stopwords.words('english')) and word != '':
            tokens.add(word)

    return tokens


def write_predictions(contexts, preds, file_path=None, print_stdout=False, predictor=None):
    if print_stdout:
        if not predictor: raise Exception('Must provide the name of the predictor used to generate the predictions if printing.')
        print(f'Predictions using {predictor} predictor:')
        for context, pred in zip(contexts, preds):
            print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, pred))
        print()
    if file_path:
        with open(file_path, 'w') as f:
            for context, pred in zip(contexts, preds):
                f.write(f'{context.lemma}.{context.pos} {context.cid} :: {pred}\n')


class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self, context : Context) -> str:
        lemma = context.lemma
        pos = context.pos

        synonyms = get_candidates(lemma, pos=pos)
        similarities = {}
        for synonym in synonyms:
            try:
                similarities[synonym] = self.model.similarity(lemma, synonym)
            except KeyError: # if synonym is not in the embedding index
                continue

        return max(similarities, key=similarities.get)


class BertPredictor(object):
    def __init__(self, pretrained='distilbert-base-uncased'):
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained(pretrained)
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained(pretrained)

    def _create_sentence_w_mask_token_from_context(self, left_context_toks, right_context_toks):
        return f"{' '.join(left_context_toks)} {self.tokenizer.mask_token} {' '.join(right_context_toks)}"

    def _get_input_representation(self, sentence):
        inp_toks = self.tokenizer.encode(sentence)
        inp_rep = np.array(inp_toks).reshape((1, -1))
        return inp_rep

    def _get_index_of_target(self, encoded_toks):
        return list(encoded_toks).index(self.tokenizer.mask_token_id)

    def _get_best_words_from_pred_scores(self, pred_scores):
        words = np.array(list(reversed(np.argsort(pred_scores))))
        return self.tokenizer.convert_ids_to_tokens(words)

    def _get_best_sub_in_wordnet(self, bert_pred_words, wn_candidates):
        for word in bert_pred_words:
            if word in wn_candidates:
                return word
        return None

    def predict(self, context : Context) -> str:
        lemma = context.lemma
        pos = context.pos
        synonyms = get_candidates(lemma, pos=pos)

        inp_sentence = self._create_sentence_w_mask_token_from_context(context.left_context, context.right_context)
        inp_rep = self._get_input_representation(inp_sentence)
        index_of_target = self._get_index_of_target(inp_rep[0])

        pred_scores = self.model.predict(inp_rep, verbose=0)[0][0] # get predictions of first (only) batch
        best_subs_for_target = self._get_best_words_from_pred_scores(pred_scores[index_of_target])
        #print(best_subs_for_target)

        best_sub_bert = self._get_best_sub_in_wordnet(best_subs_for_target, synonyms)
        if best_sub_bert:
            return best_sub_bert
        return wn_frequency_predictor(context)


class CustomNewPredictor(BertPredictor):
    def __init__(self, w2v_filename, pretrained='distilbert-base-uncased'):
        super().__init__(pretrained=pretrained)
        self.w2v = Word2VecSubst(w2v_filename)

    def _get_input_representation(self, sentence, padding_max_len):
        return self.tokenizer.encode_plus(sentence, padding='max_length', max_length=padding_max_len, return_tensors='np')

    def _get_index_of_target(self, inp):
        return list(inp['input_ids'][0]).index(self.tokenizer.mask_token_id)

    def _get_similarities_of_words_to_target_word(self, synonyms, target):
        similarities = {}
        for synonym in synonyms:
            if synonym == target: continue
            try:
                similarities[synonym] = self.w2v.model.similarity(synonym, target)
            except KeyError:  # if synonym is not in the embedding index
                continue
        return similarities

    def _nltk2wn(self, nltk_pos):
        nltk_wn_mapper = {'NN': wn.NOUN, 'JJ': wn.ADJ,
                      'VB': wn.VERB, 'RB': wn.ADV}
        try:
            return nltk_wn_mapper[nltk_pos]
        except KeyError:
            return None

    def _get_best_words_w_pos_tag_filter(self, words_tags, wn_pos_tag_filter):
        words = []
        for (word, tag) in words_tags:
            if self._nltk2wn(tag) == wn_pos_tag_filter:
                words.append(word)
        return words

    def _get_best_words_from_pred_scores(self, pred_scores, limit=None, pos_tag_filter=None, wn_synonyms=None):
        words = super()._get_best_words_from_pred_scores(pred_scores)
        if wn_synonyms:
            words += list(wn_synonyms)
        if pos_tag_filter:
            pos_tags = pos_tag(words)
            words = self._get_best_words_w_pos_tag_filter(pos_tags, pos_tag_filter)
        if limit:
            return words[:limit]
        return words

    def _get_closest_synonym(self, synonyms, target):
        similarities = self._get_similarities_of_words_to_target_word(synonyms, target)
        return sorted(similarities, key=similarities.get)[-1]

    def predict(self, context: Context, max_len_contexts=None) -> str:
        lemma = context.lemma
        pos = context.pos
        wn_candidates = get_candidates(lemma, pos)

        inp_sentence = self._create_sentence_w_mask_token_from_context(context.left_context, context.right_context)

        if max_len_contexts: # option to create padded inp representation so all inputs have equal length
            inp_rep = self._get_input_representation(inp_sentence, max_len_contexts)
            index_of_target = self._get_index_of_target(inp_rep)
            pred_scores = self.model(**inp_rep)[0][0] # get predictions of first (only) batch
        else:
            inp_rep = super()._get_input_representation(inp_sentence)
            index_of_target = super()._get_index_of_target(inp_rep[0])
            pred_scores = self.model.predict(inp_rep, verbose=0)[0][0]  # get predictions of first (only) batch

        best_subs_for_target = self._get_best_words_from_pred_scores(pred_scores[index_of_target], pos_tag_filter=pos, limit=None, wn_synonyms=wn_candidates)# list of words from bert and wordnet candidates; only gets words w/ same POS as target word
        best_sub = self._get_closest_synonym(best_subs_for_target, lemma) # get best substitution by getting synonym with highest similarity score using word2vec

        return best_sub


if __name__=="__main__":
    # At submission time, this program should run your best predictor (part 6).

    # Instantiate word2vec predictor
    W2VMODEL_FILENAME = 'word2vec_embeddings.gz'
    #word2vec_predictor = Word2VecSubst(W2VMODEL_FILENAME)

    # Instantiate bert predictor
    bert_predictor = BertPredictor()

    # Instantiate Part 6 new predictor
    #custom_predictor = CustomNewPredictor(W2VMODEL_FILENAME, pretrained='distilbert-base-uncased') # if resources is not an issue, use pretrained='bert-base-uncased' for a bigger model

    # Get contexts
    contexts = list(read_lexsub_xml(sys.argv[1]))
    max_len_contexts = get_max_length_contexts(contexts)

    smurf_preds = []
    freq_preds = []
    lesk_preds = []
    word2vec_preds = []
    bert_preds = []
    custom_preds = []

    for context in tqdm(contexts, total=len(contexts)):
        #smurf_preds.append(smurf_predictor(context))
        #freq_preds.append(wn_frequency_predictor(context))
        #lesk_preds.append(wn_simple_lesk_predictor(context))
        #word2vec_preds.append(word2vec_predictor.predict_nearest(context))
        bert_preds.append(bert_predictor.predict(context))
        #custom_preds.append(custom_predictor.predict(context, max_len_contexts=None))

    #write_predictions(contexts, smurf_preds, file_path='./smurf.predict', print_stdout=True, predictor='smurf')
    #write_predictions(contexts, freq_preds, file_path='./frequency.predict', print_stdout=True, predictor='frequency')
    #write_predictions(contexts, lesk_preds, file_path='./lesk.predict', print_stdout=True, predictor='lesk')
    #write_predictions(contexts, word2vec_preds, file_path='./word2vec.predict', print_stdout=True, predictor='word2vec')
    write_predictions(contexts, bert_preds, file_path='./bert.predict', print_stdout=True, predictor='bert')
    #write_predictions(contexts, custom_preds, file_path='./custom.predict', print_stdout=True, predictor='custom')