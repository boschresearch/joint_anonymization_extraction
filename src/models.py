# Stacked model for joint anonymization/NER
# Copyright (c) 2020 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# This source code is derived from flairNLP Project V 0.4.5
#   (https://github.com/flairNLP/flair/releases/tag/v0.4.5)
# Copyright (c) 2018 Zalando SE, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import torch.nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import flair.nn
import torch

from flair.data import Dictionary, Sentence, Token, Label
from flair.datasets import DataLoader
from flair.embeddings import TokenEmbeddings
from flair.file_utils import cached_path
from flair.models.sequence_tagger_model import *

from typing import List, Tuple, Union, Dict

from src import gumbel_softmax

def mask_from_on_hot(x, o):
    m = x[:,o]
    return 1-m


class EmbeddingModule(flair.nn.Model):
    def __init__(
        self,
        embeddings: TokenEmbeddings,
        dropout: float = 0.0,
        word_dropout: float = 0.05,
        locked_dropout: float = 0.5,
    ):
        """
        Initializes a EmbeddingModule
        :param embeddings: word embeddings used in tagger
        :param dropout: dropout probability
        :param word_dropout: word dropout probability
        :param locked_dropout: locked dropout probability
        """

        super(EmbeddingModule, self).__init__()

        self.embeddings = embeddings
        self.use_dropout: float = dropout
        self.use_word_dropout: float = word_dropout
        self.use_locked_dropout: float = locked_dropout

        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)
        if word_dropout > 0.0:
            self.word_dropout = flair.nn.WordDropout(word_dropout)
        if locked_dropout > 0.0:
            self.locked_dropout = flair.nn.LockedDropout(locked_dropout)

        rnn_input_dim: int = self.embeddings.embedding_length
        self.relearn_embeddings: bool = True
        if self.relearn_embeddings:
            self.embedding2nn = torch.nn.Linear(rnn_input_dim, rnn_input_dim)
    
    def forward(self, sentences: List[Sentence]):
        self.zero_grad()

        self.embeddings.embed(sentences)

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)

        # initialize zero-padded word embeddings tensor
        sentence_tensor = torch.zeros(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.embeddings.embedding_length,
            ],
            dtype=torch.float,
            device=flair.device,
        )

        for s_id, sentence in enumerate(sentences):
            # fill values with word embeddings
            sentence_tensor[s_id][: len(sentence)] = torch.cat(
                [token.get_embedding().unsqueeze(0) for token in sentence], 0
            )
        sentence_tensor = sentence_tensor.transpose_(0, 1)
        
        if self.use_dropout > 0.0:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_word_dropout > 0.0:
            sentence_tensor = self.word_dropout(sentence_tensor)
        if self.use_locked_dropout > 0.0:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        if self.relearn_embeddings:
            sentence_tensor = self.embedding2nn(sentence_tensor)
        return sentence_tensor
    
    
class RecurrentModule(flair.nn.Model):
    def __init__(
        self,
        rnn_input_dim: int,
        hidden_size: int,
        rnn_layers: int = 1,
        train_initial_hidden_state: bool = False,
    ):
        """
        Initializes a RecurrentModule
        :param rnn_input_dim: embedding length that goes into RNN
        :param hidden_size: number of hidden states in RNN
        :param rnn_layers: number of RNN layers
        :param train_initial_hidden_state: if True, trains initial hidden state of RNN
        """

        super(RecurrentModule, self).__init__()

        # initialize the network architecture
        self.nlayers: int = rnn_layers

        self.train_initial_hidden_state = train_initial_hidden_state
        self.bidirectional = True
        self.rnn_type = "LSTM"
            
        num_directions = 2 if self.bidirectional else 1
        self.rnn = getattr(torch.nn, self.rnn_type)(
            rnn_input_dim,
            hidden_size,
            num_layers=self.nlayers,
            dropout=0.0 if self.nlayers == 1 else 0.5,
            bidirectional=True,
        )
                
        # Create initial hidden state and initialize it
        if self.train_initial_hidden_state:
            self.hs_initializer = torch.nn.init.xavier_normal_
              
            self.lstm_init_h = Parameter(
                torch.randn(self.nlayers * num_directions, self.hidden_size),
                requires_grad=True,
            )
            
            self.lstm_init_c = Parameter(
                torch.randn(self.nlayers * num_directions, self.hidden_size),
                requires_grad=True,
            )
                
    def forward(self, sentence_tensor, lengths):
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            sentence_tensor, lengths, enforce_sorted=False
        )

        # if initial hidden state is trainable, use this state
        if self.train_initial_hidden_state:
            initial_hidden_state = [
                self.lstm_init_h.unsqueeze(1).repeat(1, len(sentences), 1),
                self.lstm_init_c.unsqueeze(1).repeat(1, len(sentences), 1),
            ]
            rnn_output, hidden = self.rnn(packed, initial_hidden_state)
        else:
            rnn_output, hidden = self.rnn(packed)

        sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            rnn_output, batch_first=True
        )
        return sentence_tensor



class StackedSequenceTagger(flair.models.sequence_tagger_model.SequenceTagger):
    def __init__(
        self,
        hidden_size: int,
        embeddings: TokenEmbeddings,
        ce_tag_dictionary: Dictionary,
        anon_tag_dictionary: Dictionary,
        ce_tag_type: str,
        anon_tag_type: str,
        use_crf: bool = True,
        use_rnn: bool = True,
        rnn_layers: int = 1,
        dropout: float = 0.0,
        word_dropout: float = 0.05,
        locked_dropout: float = 0.5,
        rnn_type: str = "LSTM",
        pickle_module: str = "pickle",
        beta: float = 1.0,
        loss_weights: Dict[str, float] = None,
        hidden_layer_after_lstm: int = 128
    ):
        """
        Initializes a StackedSequenceTagger
        :param hidden_size: number of hidden states in RNN
        :param embeddings: word embeddings used in tagger
        :param tag_dictionary: dictionary of tags you want to predict
        :param tag_type: string identifier for tag type
        :param use_crf: if True use CRF decoder, else project directly to tag space
        :param use_rnn: if True use RNN layer, otherwise use word embeddings directly
        :param rnn_layers: number of RNN layers
        :param dropout: dropout probability
        :param word_dropout: word dropout probability
        :param locked_dropout: locked dropout probability
        :param train_initial_hidden_state: if True, trains initial hidden state of RNN
        :param beta: Parameter for F-beta score for evaluation and training annealing
        :param loss_weights: Dictionary of weights for classes (tags) for the loss function
        (if any tag's weight is unspecified it will default to 1.0)

        """
        super(flair.models.sequence_tagger_model.SequenceTagger, self).__init__()

        self.use_rnn = use_rnn
        self.hidden_size = hidden_size
        self.use_crf: bool = use_crf
        self.rnn_layers: int = rnn_layers

        self.trained_epochs: int = 0

        self.embeddings = embeddings
        self.embed_model = EmbeddingModule(
            embeddings=embeddings,
            dropout=dropout,
            word_dropout=word_dropout,
            locked_dropout=locked_dropout,
        )

        # set the dictionaries
        self.ce_tag_dictionary = ce_tag_dictionary
        self.anon_tag_dictionary = anon_tag_dictionary
        self.ce_tagset_size: int = len(ce_tag_dictionary)
        self.anon_tagset_size: int = len(anon_tag_dictionary)
        self.ce_tag_type = ce_tag_type
        self.anon_tag_type = anon_tag_type
        self.o_label = anon_tag_dictionary.get_idx_for_item("O")

        self.beta = beta

        self.weight_dict = loss_weights
        # Initialize the weight tensor
        if loss_weights is not None:
            n_classes = len(self.tag_dictionary)
            weight_list = [1. for i in range(n_classes)]
            for i, tag in enumerate(self.tag_dictionary.get_items()):
                if tag in loss_weights.keys():
                    weight_list[i] = loss_weights[tag]
            self.loss_weights = torch.FloatTensor(weight_list).to(flair.device)
        else:
            self.loss_weights = None

        # initialize the network architecture
        self.nlayers: int = rnn_layers
        self.hidden_word = None

        self.pickle_module = pickle_module

        rnn_input_dim: int = self.embeddings.embedding_length
        
        self.bidirectional = True
        self.rnn_type = "LSTM"
        self.hidden_layer_after_lstm = hidden_layer_after_lstm
        
        num_directions = 2 if self.bidirectional else 1
        
        self.anon_rnn = RecurrentModule(
            rnn_input_dim=rnn_input_dim,
            hidden_size=hidden_size,
            rnn_layers=rnn_layers,
        )
                
        self.ce_rnn = RecurrentModule(
            rnn_input_dim=rnn_input_dim,
            hidden_size=hidden_size,
            rnn_layers=rnn_layers,
        )
                
        if self.hidden_layer_after_lstm > 0:
            self.ce_hidden_layer = torch.nn.Linear(hidden_size * num_directions, hidden_layer_after_lstm)
            self.ce_relu = torch.nn.ReLU()
            self.ce_linear = torch.nn.Linear(hidden_layer_after_lstm, len(self.ce_tag_dictionary))
            
            self.anon_hidden_layer = torch.nn.Linear(hidden_size * num_directions, hidden_layer_after_lstm)
            self.anon_relu = torch.nn.ReLU()
            self.anon_linear = torch.nn.Linear(hidden_layer_after_lstm, len(self.anon_tag_dictionary))
            
        else:
            # final linear map to tag space
            self.anon_linear = torch.nn.Linear(hidden_size * num_directions, len(anon_tag_dictionary))
            self.ce_linear = torch.nn.Linear(hidden_size * num_directions, len(ce_tag_dictionary))
            
        # embedding for anonymization labels
        self.anon_embedding = torch.nn.Embedding(len(anon_tag_dictionary), self.embeddings.embedding_length)

        if self.use_crf:
            self.ce_transitions = torch.nn.Parameter(
                torch.randn(self.ce_tagset_size, self.ce_tagset_size)
            )
            self.ce_transitions.detach()[
                self.ce_tag_dictionary.get_idx_for_item(START_TAG), :
            ] = -10000
            self.ce_transitions.detach()[
                :, self.ce_tag_dictionary.get_idx_for_item(STOP_TAG)
            ] = -10000
            
            self.anon_transitions = torch.nn.Parameter(
                torch.randn(self.anon_tagset_size, self.anon_tagset_size)
            )
            self.anon_transitions.detach()[
                self.anon_tag_dictionary.get_idx_for_item(START_TAG), :
            ] = -10000
            self.anon_transitions.detach()[
                :, self.anon_tag_dictionary.get_idx_for_item(STOP_TAG)
            ] = -10000

        self.set_output_to_ce()
        self.to(flair.device)
        
        
    def set_output_to_anon(self):
        """ Sets the StackedSequenceTagger to output ANON labels """
        self.output_anon = True
        self.transitions = self.anon_transitions
        self.tag_dictionary = self.anon_tag_dictionary
        self.tagset_size = self.anon_tagset_size
        self.tag_type = self.anon_tag_type
        
    def set_output_to_ce(self):
        """ Sets the StackedSequenceTagger to output CE labels """
        self.output_anon = False
        self.transitions = self.ce_transitions
        self.tag_dictionary = self.ce_tag_dictionary
        self.tagset_size = self.ce_tagset_size
        self.tag_type = self.ce_tag_type
        
    def set_output(self, name):
        if name.lower() == 'anon':
            self.set_output_to_anon()
        elif name.lower() == 'ce':
            self.set_output_to_ce()
        
    
    def forward(self, sentences: List[Sentence]):
        self.zero_grad()
        
        embedded_sentence_tensor = self.embed_model(sentences)
        num_sents, len_sents, len_emb = embedded_sentence_tensor.shape
        
        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        sentence_tensor = self.anon_rnn(embedded_sentence_tensor, lengths)
        
        if self.embed_model.use_dropout > 0.0:
            sentence_tensor = self.embed_model.dropout(sentence_tensor)
        if self.embed_model.use_locked_dropout > 0.0:
            sentence_tensor = self.embed_model.locked_dropout(sentence_tensor)
            
        if self.hidden_layer_after_lstm > 0:
            sentence_tensor = self.anon_hidden_layer(sentence_tensor)
            sentence_tensor = self.anon_relu(sentence_tensor)
                
        features = self.anon_linear(sentence_tensor)
        if self.output_anon:
            return features
        
        _, _, len_features = features.shape
        features = gumbel_softmax(features, 0.5)
        features = features.view(len_sents * num_sents, len_features)
        
        # Generate anonymization mask and labels
        mask = mask_from_on_hot(features, self.o_label)
        mask = mask.unsqueeze(1).repeat(1, len_emb)
        
        features = argmax_batch(features)
        label_embeds = self.anon_embedding(features) 
        embedded_sentence_tensor = embedded_sentence_tensor.view(len_sents * num_sents, len_emb)
        
        # Apply mask embedding layer
        sentence_tensor = mask * label_embeds
        sentence_tensor += (1.0-mask) * embedded_sentence_tensor
        sentence_tensor = sentence_tensor.view(num_sents, len_sents, len_emb)
        
        # Retrieve CE labels
        sentence_tensor = self.ce_rnn(sentence_tensor, lengths)
        
        if self.embed_model.use_dropout > 0.0:
            sentence_tensor = self.embed_model.dropout(sentence_tensor)
        if self.embed_model.use_locked_dropout > 0.0:
            sentence_tensor = self.embed_model.locked_dropout(sentence_tensor)
            
        if self.hidden_layer_after_lstm > 0:
            sentence_tensor = self.ce_hidden_layer(sentence_tensor)
            sentence_tensor = self.ce_relu(sentence_tensor)
        
        features = self.ce_linear(sentence_tensor)

        return features

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "embeddings": self.embeddings,
            "hidden_size": self.hidden_size,
            "ce_tag_dictionary": self.ce_tag_dictionary,
            "anon_tag_dictionary": self.anon_tag_dictionary,
            "ce_tag_type": self.ce_tag_type,
            "anon_tag_type": self.anon_tag_type,
            "use_crf": self.use_crf,
            "use_rnn": self.use_rnn,
            "rnn_layers": self.rnn_layers,
            "use_word_dropout": self.embed_model.use_word_dropout,
            "use_locked_dropout": self.embed_model.use_locked_dropout,
            "use_hidden_layer_after_lstm": self.hidden_layer_after_lstm,
            "beta": self.beta,
            "weight_dict": self.weight_dict,
        }
        return model_state

    def _init_model_with_state_dict(state):
        use_dropout = 0.0 if not "use_dropout" in state.keys() else state["use_dropout"]
        use_word_dropout = (
            0.0 if not "use_word_dropout" in state.keys() else state["use_word_dropout"]
        )
        use_locked_dropout = (
            0.0
            if not "use_locked_dropout" in state.keys()
            else state["use_locked_dropout"]
        )
        hidden_layer_after_lstm = (
            128
            if not "hidden_layer_after_lstm" in state.keys()
            else state["hidden_layer_after_lstm"]
        )
        beta = 1.0 if "beta" not in state.keys() else state["beta"]
        weights = None if "weight_dict" not in state.keys() else state["weight_dict"]
        
        model = StackedSequenceTagger(
            hidden_size=state["hidden_size"],
            embeddings=state["embeddings"],
            ce_tag_dictionary=state["ce_tag_dictionary"],
            anon_tag_dictionary=state["anon_tag_dictionary"],
            ce_tag_type=state["ce_tag_type"],
            anon_tag_type=state["anon_tag_type"],
            use_crf=state["use_crf"],
            use_rnn=state["use_rnn"],
            rnn_layers=state["rnn_layers"],
            dropout=use_dropout,
            word_dropout=use_word_dropout,
            locked_dropout=use_locked_dropout,
            beta=beta,
            loss_weights=weights,
            hidden_layer_after_lstm=hidden_layer_after_lstm
        )
        try:
            model.set_output_to_ce()
            model.load_state_dict(state["state_dict"])
        except RuntimeError:
            model.set_output_to_anon()
            model.load_state_dict(state["state_dict"])
        return model



class MultitaskSequenceTagger(flair.models.sequence_tagger_model.SequenceTagger):
    def __init__(
        self,
        hidden_size: int,
        embeddings: TokenEmbeddings,
        ce_tag_dictionary: Dictionary,
        anon_tag_dictionary: Dictionary,
        ce_tag_type: str,
        anon_tag_type: str,
        use_crf: bool = True,
        use_rnn: bool = True,
        rnn_layers: int = 1,
        dropout: float = 0.0,
        word_dropout: float = 0.05,
        locked_dropout: float = 0.5,
        rnn_type: str = "LSTM",
        pickle_module: str = "pickle",
        beta: float = 1.0,
        loss_weights: Dict[str, float] = None,
        hidden_layer_after_lstm: int = 128
    ):
        """
        Initializes a MultitaskSequenceTagger
        :param hidden_size: number of hidden states in RNN
        :param embeddings: word embeddings used in tagger
        :param tag_dictionary: dictionary of tags you want to predict
        :param tag_type: string identifier for tag type
        :param use_crf: if True use CRF decoder, else project directly to tag space
        :param use_rnn: if True use RNN layer, otherwise use word embeddings directly
        :param rnn_layers: number of RNN layers
        :param dropout: dropout probability
        :param word_dropout: word dropout probability
        :param locked_dropout: locked dropout probability
        :param train_initial_hidden_state: if True, trains initial hidden state of RNN
        :param beta: Parameter for F-beta score for evaluation and training annealing
        :param loss_weights: Dictionary of weights for classes (tags) for the loss function
        (if any tag's weight is unspecified it will default to 1.0)

        """
        super(flair.models.sequence_tagger_model.SequenceTagger, self).__init__()

        self.use_rnn = use_rnn
        self.hidden_size = hidden_size
        self.use_crf: bool = use_crf
        self.rnn_layers: int = rnn_layers

        self.trained_epochs: int = 0

        self.embeddings = embeddings
        self.embed_model = EmbeddingModule(
            embeddings=embeddings,
            dropout=dropout,
            word_dropout=word_dropout,
            locked_dropout=locked_dropout,
        )

        # set the dictionaries
        self.ce_tag_dictionary = ce_tag_dictionary
        self.anon_tag_dictionary = anon_tag_dictionary
        self.ce_tagset_size: int = len(ce_tag_dictionary)
        self.anon_tagset_size: int = len(anon_tag_dictionary)
        self.ce_tag_type = ce_tag_type
        self.anon_tag_type = anon_tag_type
        self.o_label = anon_tag_dictionary.get_idx_for_item("O")

        self.beta = beta

        self.weight_dict = loss_weights
        # Initialize the weight tensor
        if loss_weights is not None:
            n_classes = len(self.tag_dictionary)
            weight_list = [1. for i in range(n_classes)]
            for i, tag in enumerate(self.tag_dictionary.get_items()):
                if tag in loss_weights.keys():
                    weight_list[i] = loss_weights[tag]
            self.loss_weights = torch.FloatTensor(weight_list).to(flair.device)
        else:
            self.loss_weights = None

        # initialize the network architecture
        self.nlayers: int = rnn_layers
        self.hidden_word = None

        self.pickle_module = pickle_module

        rnn_input_dim: int = self.embeddings.embedding_length
        
        self.bidirectional = True
        self.rnn_type = "LSTM"
        self.hidden_layer_after_lstm = hidden_layer_after_lstm
        
        num_directions = 2 if self.bidirectional else 1
        
        self.shared_rnn = RecurrentModule(
            rnn_input_dim=rnn_input_dim,
            hidden_size=hidden_size,
            rnn_layers=rnn_layers,
        )
                
        if self.hidden_layer_after_lstm > 0:
            self.ce_hidden_layer = torch.nn.Linear(hidden_size * num_directions, hidden_layer_after_lstm)
            self.ce_relu = torch.nn.ReLU()
            self.ce_linear = torch.nn.Linear(hidden_layer_after_lstm, len(self.ce_tag_dictionary))
            
            self.anon_hidden_layer = torch.nn.Linear(hidden_size * num_directions, hidden_layer_after_lstm)
            self.anon_relu = torch.nn.ReLU()
            self.anon_linear = torch.nn.Linear(hidden_layer_after_lstm, len(self.anon_tag_dictionary))
            
        else:
            # final linear map to tag space
            self.anon_linear = torch.nn.Linear(hidden_size * num_directions, len(anon_tag_dictionary))
            self.ce_linear = torch.nn.Linear(hidden_size * num_directions, len(ce_tag_dictionary))

        if self.use_crf:
            self.ce_transitions = torch.nn.Parameter(
                torch.randn(self.ce_tagset_size, self.ce_tagset_size)
            )
            self.ce_transitions.detach()[
                self.ce_tag_dictionary.get_idx_for_item(START_TAG), :
            ] = -10000
            self.ce_transitions.detach()[
                :, self.ce_tag_dictionary.get_idx_for_item(STOP_TAG)
            ] = -10000
            
            self.anon_transitions = torch.nn.Parameter(
                torch.randn(self.anon_tagset_size, self.anon_tagset_size)
            )
            self.anon_transitions.detach()[
                self.anon_tag_dictionary.get_idx_for_item(START_TAG), :
            ] = -10000
            self.anon_transitions.detach()[
                :, self.anon_tag_dictionary.get_idx_for_item(STOP_TAG)
            ] = -10000

        self.set_output_to_ce()
        self.to(flair.device)
        
        
    def set_output_to_anon(self):
        """ Sets the StackedSequenceTagger to output ANON labels """
        self.output_anon = True
        self.transitions = self.anon_transitions
        self.tag_dictionary = self.anon_tag_dictionary
        self.tagset_size = self.anon_tagset_size
        self.tag_type = self.anon_tag_type
        
    def set_output_to_ce(self):
        """ Sets the StackedSequenceTagger to output CE labels """
        self.output_anon = False
        self.transitions = self.ce_transitions
        self.tag_dictionary = self.ce_tag_dictionary
        self.tagset_size = self.ce_tagset_size
        self.tag_type = self.ce_tag_type
        
    def set_output(self, name):
        if name.lower() == 'anon':
            self.set_output_to_anon()
        elif name.lower() == 'ce':
            self.set_output_to_ce()
    
    def forward(self, sentences: List[Sentence]):
        self.zero_grad()
        
        embedded_sentence_tensor = self.embed_model(sentences)
        num_sents, len_sents, len_emb = embedded_sentence_tensor.shape
        
        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        sentence_tensor = self.shared_rnn(embedded_sentence_tensor, lengths)
        
        if self.embed_model.use_dropout > 0.0:
            sentence_tensor = self.embed_model.dropout(sentence_tensor)
        if self.embed_model.use_locked_dropout > 0.0:
            sentence_tensor = self.embed_model.locked_dropout(sentence_tensor)
            
        if self.output_anon: # output ANON labels
            if self.hidden_layer_after_lstm > 0:
                sentence_tensor = self.anon_hidden_layer(sentence_tensor)
                sentence_tensor = self.anon_relu(sentence_tensor)
            features = self.anon_linear(sentence_tensor)
            return features
        else: # output CE labels
            if self.hidden_layer_after_lstm > 0:
                sentence_tensor = self.ce_hidden_layer(sentence_tensor)
                sentence_tensor = self.ce_relu(sentence_tensor)
            features = self.ce_linear(sentence_tensor)
            return features

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "embeddings": self.embeddings,
            "hidden_size": self.hidden_size,
            "ce_tag_dictionary": self.ce_tag_dictionary,
            "anon_tag_dictionary": self.anon_tag_dictionary,
            "ce_tag_type": self.ce_tag_type,
            "anon_tag_type": self.anon_tag_type,
            "use_crf": self.use_crf,
            "use_rnn": self.use_rnn,
            "rnn_layers": self.rnn_layers,
            "use_word_dropout": self.embed_model.use_word_dropout,
            "use_locked_dropout": self.embed_model.use_locked_dropout,
            "use_hidden_layer_after_lstm": self.hidden_layer_after_lstm,
            "beta": self.beta,
            "weight_dict": self.weight_dict,
        }
        return model_state

    def _init_model_with_state_dict(state):
        use_dropout = 0.0 if not "use_dropout" in state.keys() else state["use_dropout"]
        use_word_dropout = (
            0.0 if not "use_word_dropout" in state.keys() else state["use_word_dropout"]
        )
        use_locked_dropout = (
            0.0
            if not "use_locked_dropout" in state.keys()
            else state["use_locked_dropout"]
        )
        hidden_layer_after_lstm = (
            128
            if not "hidden_layer_after_lstm" in state.keys()
            else state["hidden_layer_after_lstm"]
        )
        beta = 1.0 if "beta" not in state.keys() else state["beta"]
        weights = None if "weight_dict" not in state.keys() else state["weight_dict"]
        
        model = MultitaskSequenceTagger(
            hidden_size=state["hidden_size"],
            embeddings=state["embeddings"],
            ce_tag_dictionary=state["ce_tag_dictionary"],
            anon_tag_dictionary=state["anon_tag_dictionary"],
            ce_tag_type=state["ce_tag_type"],
            anon_tag_type=state["anon_tag_type"],
            use_crf=state["use_crf"],
            use_rnn=state["use_rnn"],
            rnn_layers=state["rnn_layers"],
            dropout=use_dropout,
            word_dropout=use_word_dropout,
            locked_dropout=use_locked_dropout,
            beta=beta,
            loss_weights=weights,
            hidden_layer_after_lstm=hidden_layer_after_lstm
        )
        try:
            model.set_output_to_ce()
            model.load_state_dict(state["state_dict"])
        except RuntimeError:
            model.set_output_to_anon()
            model.load_state_dict(state["state_dict"])
        return model