# BytePairEmbeddings that can average vectors over a single word
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

import numpy as np
import torch
from typing import List
from pathlib import Path

from bpemb import BPEmb

import flair
from flair.data import Sentence
from flair.embeddings import TokenEmbeddings

class CustomBytePairEmbeddings(TokenEmbeddings):
    def __init__(
        self,
        language: str,
        dim: int = 300,
        syllables: int = 100000,
        cache_dir=Path(flair.cache_root) / "embeddings",
        emb_method='avg',
    ):
        """
        Initializes BP embeddings. Constructor downloads required files if not there.
        """

        self.name: str = f"bpe-{language}-{syllables}-{dim}"
        self.static_embeddings = True
        self.embedder = BPEmb(lang=language, vs=syllables, dim=dim, cache_dir=cache_dir)

        self.emb_method = emb_method.lower()
        if self.emb_method in ['avg', 'first', 'last']:
            self.__embedding_length: int = self.embedder.emb.vector_size
        else:
            self.__embedding_length: int = self.embedder.emb.vector_size * 2
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):

                if "field" not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value

                if word.strip() == "":
                    # empty words get no embedding
                    token.set_embedding(
                        self.name, torch.zeros(self.embedding_length, dtype=torch.float)
                    )
                else:
                    # all other words get embedded
                    embeddings = self.embedder.embed(word.lower())
                    if self.emb_method == 'first':
                        embedding = embeddings[0]
                    elif self.emb_method == 'last':
                        embedding = embeddings[-1]
                    elif self.emb_method == 'avg':
                        embedding = np.average(embeddings, axis=0)
                    else:
                        embedding = np.concatenate((embeddings[0], embeddings[-1]))
                    token.set_embedding(
                        self.name, torch.tensor(embedding, dtype=torch.float)
                    )

        return sentences

    def __str__(self):
        return self.name

    def extra_repr(self):
        return "model={}".format(self.name)
