# Trainer base class for stacked/multitask trainer
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
    
import logging
from flair.training_utils import log_line
from pathlib import Path
from typing import List, Union
import time
import sys

import datetime

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.sgd import SGD
from torch.utils.data.dataset import ConcatDataset

import flair
import flair.nn
from flair.data import MultiCorpus, Corpus
from flair.datasets import DataLoader
from flair.optim import ExpAnnealLR
from flair.training_utils import (
    init_output_file,
    WeightExtractor,
    log_line,
    add_file_handler,
    Result,
    store_embeddings,
)

log = logging.getLogger("flair")

class BatchWiseMultiTaskTrainer:
    
    def __init__(self, trainer_list, base_path_list, learning_rate_list, pretrain_list=None):
        self.trainer_list = trainer_list
        self.base_path_list = base_path_list
        self.learning_rate_list = learning_rate_list
        self.pretrain_list = pretrain_list
        
    def train(self, max_epochs: int = 100, **kwargs):
        log.info(f"Prepare data loader")
        for trainer, base_path, lr in zip(self.trainer_list, self.base_path_list, self.learning_rate_list):
            trainer.max_epochs = max_epochs
            trainer.prepare_data(base_path, lr, **kwargs)
            
        if self.pretrain_list is not None:
            for i, epochs_pretrain in enumerate(self.pretrain_list):
                log_line(log)
                log.info(f"Pretrain Trainer {i} for {epochs_pretrain} epochs")
                
                for epoch in range(0, epochs_pretrain):
                    trainer = self.trainer_list[i]
                    trainer.cur_epoch = epoch
                    trainer.prepare_epoch()
                    
                    for batch_no, batch in enumerate(trainer.batch_loader):
                        trainer.train_batch(batch_no, batch)
                    trainer.eval_after_epoch()
          
        log_line(log)
        log.info(f"Start multi-task training")
        for epoch in range(0, max_epochs):
            quit = False
            for trainer in self.trainer_list:
                trainer.cur_epoch = epoch
                trainer.prepare_epoch()
            
                if trainer.learning_rate < trainer.min_learning_rate:
                    quit = True
            if quit:
                log.info('Quitting')
                break
            
            num_batches = min([len(t.batch_loader) for t in self.trainer_list])
            iters = [iter(t.batch_loader) for t in self.trainer_list]
            batch_no = 0
            
            while batch_no < num_batches:
                for i, trainer in enumerate(self.trainer_list):
                    batch = next(iters[i])
                    trainer.train_batch(batch_no, batch)
                batch_no += 1
                
            for trainer in self.trainer_list:
                trainer.eval_after_epoch()
                
        for trainer in self.trainer_list:
            trainer.make_final_predictions()
            


class SingleTaskTrainer:
    def __init__(
        self,
        model: flair.nn.Model,
        corpus: Corpus,
        model_mode: str,
        optimizer: torch.optim.Optimizer = SGD,
        epoch: int = 0,
        optimizer_state: dict = None,
        scheduler_state: dict = None,
    ):
        """
        Initialize a model trainer
        :param model: The model that you want to train. The model should inherit from flair.nn.Model
        :param corpus: The dataset used to train the model, should be of type Corpus
        :param optimizer: The optimizer to use (typically SGD or Adam)
        :param epoch: The starting epoch (normally 0 but could be higher if you continue training model)
        :param optimizer_state: Optimizer state (necessary if continue training from checkpoint)
        :param scheduler_state: Scheduler state (necessary if continue training from checkpoint)
        """
        self.model: flair.nn.Model = model
        self.corpus: Corpus = corpus
        self.optimizer_type: torch.optim.Optimizer = optimizer
        self.epoch: int = epoch
        self.max_epochs: int = -1
        self.scheduler_state: dict = scheduler_state
        self.optimizer_state: dict = optimizer_state
        self.display_name = model_mode
        self.model_mode = model_mode.lower().strip()
            
            
    def train(self, max_epochs: int = 100, **kwargs):
        self.model.set_output(self.model_mode)
        self.max_epochs = max_epochs
        self.prepare_data(**kwargs)
        for epoch in range(0 + self.epoch, max_epochs + self.epoch):
            self.cur_epoch = epoch
            self.prepare_epoch()
            
            if self.learning_rate < self.min_learning_rate:
                log_line(log)
                log.info("Quitting Training as one Model finished")
                log_line(log)
                break
            
            for batch_no, batch in enumerate(self.batch_loader):
                self.train_batch(batch_no, batch)
                
            self.eval_after_epoch()
        self.make_final_predictions()

        
    def prepare_data(
        self,
        base_path: Union[Path, str],
        learning_rate: float = 0.1,
        mini_batch_size: int = 32,
        eval_mini_batch_size: int = None,
        anneal_factor: float = 0.5,
        patience: int = 3,
        min_learning_rate: float = 0.0001,
        train_with_dev: bool = False,
        monitor_train: bool = False,
        monitor_test: bool = False,
        embedding_storage_mode: str = "cpu",
        checkpoint: bool = False,
        save_final_model: bool = True,
        anneal_with_restarts: bool = False,
        shuffle: bool = True,
        param_selection_mode: bool = False,
        num_workers: int = 6,
        **kwargs,
    ) -> dict:
        """
        Trains any class that implements the flair.nn.Model interface.
        :param base_path: Main path to which all output during training is logged and models are saved
        :param learning_rate: Initial learning rate
        :param mini_batch_size: Size of mini-batches during training
        :param eval_mini_batch_size: Size of mini-batches during evaluation
        :param max_epochs: Maximum number of epochs to train. Terminates training if this number is surpassed.
        :param anneal_factor: The factor by which the learning rate is annealed
        :param patience: Patience is the number of epochs with no improvement the Trainer waits
         until annealing the learning rate
        :param min_learning_rate: If the learning rate falls below this threshold, training terminates
        :param train_with_dev: If True, training is performed using both train+dev data
        :param monitor_train: If True, training data is evaluated at end of each epoch
        :param monitor_test: If True, test data is evaluated at end of each epoch
        :param embedding_storage_mode: One of 'none' (all embeddings are deleted and freshly recomputed),
        'cpu' (embeddings are stored on CPU) or 'gpu' (embeddings are stored on GPU)
        :param checkpoint: If True, a full checkpoint is saved at end of each epoch
        :param save_final_model: If True, final model is saved
        :param anneal_with_restarts: If True, the last best model is restored when annealing the learning rate
        :param shuffle: If True, data is shuffled during training
        :param param_selection_mode: If True, testing is performed against dev data. Use this mode when doing
        parameter selection.
        :param num_workers: Number of workers in your data loader.
        :param sampler: You can pass a data sampler here for special sampling of data.
        :param kwargs: Other arguments for the Optimizer
        :return:
        """
        
        self.shuffle = shuffle
        self.embedding_storage_mode = embedding_storage_mode
        self.checkpoint = checkpoint
        self.save_final_model = save_final_model
        self.anneal_with_restarts = anneal_with_restarts
        self.num_workers = num_workers
        
        self.mini_batch_size = mini_batch_size
        if eval_mini_batch_size is None:
            self.eval_mini_batch_size = mini_batch_size
        else:
            self.eval_mini_batch_size = eval_mini_batch_size

        # cast string to Path
        if type(base_path) is str:
            self.base_path = Path(base_path)

        self.log_handler = add_file_handler(log, self.base_path / "training.log")

        if self.display_name is not None:
            log_line(log)
            log.info(f'Model: {self.display_name}')
        log_line(log)
        log.info(f'Model: "{self.model}"')
        log_line(log)
        log.info(f'Corpus: "{self.corpus}"')
        log_line(log)
        log.info("Parameters:")
        log.info(f' - learning_rate: "{learning_rate}"')
        log.info(f' - mini_batch_size: "{mini_batch_size}"')
        log.info(f' - patience: "{patience}"')
        log.info(f' - anneal_factor: "{anneal_factor}"')
        log.info(f' - max_epochs: "{self.max_epochs}"')
        log.info(f' - shuffle: "{shuffle}"')
        log.info(f' - train_with_dev: "{train_with_dev}"')
        log_line(log)
        log.info(f'Model training base path: "{base_path}"')
        log_line(log)
        log.info(f"Device: {flair.device}")
        log_line(log)
        log.info(f"Embedding storage mode: {embedding_storage_mode}")

        # determine what splits (train, dev, test) to evaluate and log
        self.monitor_train = monitor_train
        self.monitor_test = monitor_test
        self.param_selection_mode = param_selection_mode
        self.train_with_dev = train_with_dev
            
        self.log_train = True if self.monitor_train else False
        self.log_test = (
            True
            if (not self.param_selection_mode and self.corpus.test and self.monitor_test)
            else False
        )
        self.log_dev = True if not self.train_with_dev else False

        # prepare loss logging file and set up header
        self.loss_txt = init_output_file(self.base_path, "loss.tsv")

        self.weight_extractor = WeightExtractor(self.base_path)

        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.previous_learning_rate = learning_rate
        self.optimizer: torch.optim.Optimizer = self.optimizer_type(
            self.model.parameters(), lr=self.learning_rate, **kwargs
        )
        if self.optimizer_state is not None:
            optimizer.load_state_dict(self.optimizer_state)

        # minimize training loss if training with dev data, else maximize dev score
        self.anneal_mode = "min" if self.train_with_dev else "max"

        self.anneal_factor = anneal_factor
        self.patience = patience
        self.scheduler: ReduceLROnPlateau = ReduceLROnPlateau(
            self.optimizer,
            factor=self.anneal_factor,
            patience=self.patience,
            mode=self.anneal_mode,
            verbose=True,
        )

        if self.scheduler_state is not None:
            self.scheduler.load_state_dict(self.scheduler_state)

        self.train_data = self.corpus.train

        # if training also uses dev data, include in training set
        if self.train_with_dev:
            self.train_data = ConcatDataset([self.corpus.train, self.corpus.dev])

        self.dev_score_history = []
        self.dev_loss_history = []
        self.train_loss_history = []
        
    def prepare_epoch(self):
        log_line(log)
        
        for group in self.optimizer.param_groups:
            self.learning_rate = group["lr"]

        # reload last best model if annealing with restarts is enabled
        if (
            self.learning_rate != self.previous_learning_rate
            and self.anneal_with_restarts
            and (base_path / "best-model.pt").exists()
        ):
            log.info("resetting to best model")
            self.model.load(base_path / "best-model.pt")

        self.previous_learning_rate = self.learning_rate

        # stop training if learning rate becomes too small
        if self.learning_rate < self.min_learning_rate:
            log_line(log)
            log.info("learning rate too small - quitting training!")
            log_line(log)
            return

        self.batch_loader = DataLoader(
            self.train_data,
            batch_size=self.mini_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

        self.model.train()
        self.model.set_output(self.model_mode)
            
        self.train_loss: float = 0
        self.seen_batches = 0
        self.total_number_of_batches = len(self.batch_loader)

        self.modulo = max(1, int(self.total_number_of_batches / 10))
        self.batch_time = 0
        
        
    def train_batch(self, batch_no, batch):
        # process mini-batches
        start_time = time.time()
        
        self.model.set_output(self.model_mode)
        loss = self.model.forward_loss(batch)

        self.optimizer.zero_grad()
        # Backward
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        self.seen_batches += 1
        self.train_loss += loss.item()

        # depending on memory mode, embeddings are moved to CPU, GPU or deleted
        store_embeddings(batch, self.embedding_storage_mode)

        self.batch_time += time.time() - start_time
        if batch_no % self.modulo == 0:
            model_string = f'model: {self.display_name} - ' if self.display_name is not None else ''
            log.info(
                f"{model_string}epoch {self.cur_epoch + 1} - iter {batch_no}/{self.total_number_of_batches} - loss "
                f"{self.train_loss / self.seen_batches:.8f} - samples/sec: {self.mini_batch_size * self.modulo / self.batch_time:.2f}"
            )
            self.batch_time = 0
            iteration = self.cur_epoch * self.total_number_of_batches + batch_no
            if not self.param_selection_mode:
                self.weight_extractor.extract_weights(
                    self.model.state_dict(), iteration
                )

    def eval_after_epoch(self):
        self.model.set_output(self.model_mode)
        self.train_loss /= self.seen_batches
        self.model.eval()
        
        model_string = f'MODEL: {self.display_name} - ' if self.display_name is not None else ''
        log_line(log)
        log.info(
            f"{model_string}EPOCH {self.cur_epoch + 1} done: loss {self.train_loss:.4f} - lr {self.learning_rate:.4f}"
        )
        
        # anneal against train loss if training with dev, otherwise anneal against dev score
        current_score = self.train_loss

        # evaluate on train / dev / test split depending on training settings
        result_line: str = ""

        if self.log_train:
            train_eval_result, train_loss = self.model.evaluate(
                DataLoader(
                    self.corpus.train,
                    batch_size=self.eval_mini_batch_size,
                    num_workers=self.num_workers,
                ),
                embedding_storage_mode=self.embedding_storage_mode,
            )
            result_line += f"\t{train_eval_result.log_line}"

            # depending on memory mode, embeddings are moved to CPU, GPU or deleted
            store_embeddings(self.corpus.train, self.embedding_storage_mode)

        if self.log_dev:
            dev_eval_result, dev_loss = self.model.evaluate(
                DataLoader(
                    self.corpus.dev,
                    batch_size=self.eval_mini_batch_size,
                    num_workers=self.num_workers,
                ),
                embedding_storage_mode=self.embedding_storage_mode,
            )
            result_line += f"\t{dev_loss}\t{dev_eval_result.log_line}"
            log.info(
                f"DEV : loss {dev_loss} - score {dev_eval_result.main_score}"
            )
            
            # calculate scores using dev data if available
            # append dev score to score history
            self.dev_score_history.append(dev_eval_result.main_score)
            self.dev_loss_history.append(dev_loss)
            
            current_score = dev_eval_result.main_score
            
            # depending on memory mode, embeddings are moved to CPU, GPU or deleted
            store_embeddings(self.corpus.dev, self.embedding_storage_mode)
            
            
        if self.log_test:
            test_eval_result, test_loss = self.model.evaluate(
                DataLoader(
                    self.corpus.test,
                    batch_size=self.eval_mini_batch_size,
                    num_workers=self.num_workers,
                ),
                self.base_path / "test.tsv",
                embedding_storage_mode=self.embedding_storage_mode,
            )
            result_line += f"\t{test_loss}\t{test_eval_result.log_line}"
            log.info(
                f"TEST : loss {test_loss} - score {test_eval_result.main_score}"
            )
            
            # depending on memory mode, embeddings are moved to CPU, GPU or deleted
            store_embeddings(self.corpus.test, self.embedding_storage_mode)

        # determine learning rate annealing through scheduler
        self.scheduler.step(current_score)
        
        self.train_loss_history.append(self.train_loss)

        # determine bad epoch number
        try:
            bad_epochs = self.scheduler.num_bad_epochs
        except:
            bad_epochs = 0
        for group in self.optimizer.param_groups:
            new_learning_rate = group["lr"]
        if new_learning_rate != self.previous_learning_rate:
            bad_epochs = self.patience + 1

        # log bad epochs
        log.info(f"BAD EPOCHS (no improvement): {bad_epochs}")

        # output log file
        with open(self.loss_txt, "a") as f:
            
            # make headers on first epoch
            if self.cur_epoch == 0:
                f.write(
                    f"EPOCH\tTIMESTAMP\tBAD_EPOCHS\tLEARNING_RATE\tTRAIN_LOSS"
                )
                
                if self.log_train:
                    f.write(
                        "\tTRAIN_"
                        + "\tTRAIN_".join(
                            train_eval_result.log_header.split("\t")
                        )
                    )
                if self.log_dev:
                    f.write(
                        "\tDEV_LOSS\tDEV_"
                        + "\tDEV_".join(dev_eval_result.log_header.split("\t"))
                    )
                if self.log_test:
                    f.write(
                        "\tTEST_LOSS\tTEST_"
                        + "\tTEST_".join(
                            test_eval_result.log_header.split("\t")
                        )
                    )

            f.write(
                f"\n{self.cur_epoch}\t{datetime.datetime.now():%H:%M:%S}\t{bad_epochs}\t{self.learning_rate:.4f}\t{self.train_loss}"
            )
            f.write(result_line)

            # if checkpoint is enable, save model at each epoch
            if self.checkpoint and not self.param_selection_mode:
                self.model.save_checkpoint(
                    self.base_path / "checkpoint.pt",
                    self.optimizer.state_dict(),
                    self.scheduler.state_dict(),
                    self.cur_epoch + 1,
                    train_loss,
                )

            # if we use dev data, remember best model based on dev evaluation score
            if (
                not self.train_with_dev
                and not self.param_selection_mode
                and current_score == self.scheduler.best
            ):
                self.model.save(self.base_path / "best-model.pt")
               
            
    def make_final_predictions(self):
        self.model.set_output(self.model_mode)
        # if we do not use dev data for model selection, save final model
        if self.save_final_model and not self.param_selection_mode:
            self.model.save(self.base_path / "final-model.pt")
            
        # test best model if test data is present
        if self.corpus.test:
            final_score = self.final_test(self.base_path, self.eval_mini_batch_size, self.num_workers)
        else:
            final_score = 0
            log.info("Test data not provided setting final score to 0")

        log.removeHandler(self.log_handler)

        return {
            "test_score": final_score,
            "dev_score_history": self.dev_score_history,
            "train_loss_history": self.train_loss_history,
            "dev_loss_history": self.dev_loss_history,
        }

    def final_test(
        self, base_path: Path, eval_mini_batch_size: int, num_workers: int = 8
    ):

        self.model.set_output(self.model_mode)
        log_line(log)
        log.info("Testing using best model ...")

        self.model.eval()

        if (base_path / "best-model.pt").exists():
            self.model = self.model.load(base_path / "best-model.pt")

        test_results, test_loss = self.model.evaluate(
            DataLoader(
                self.corpus.test,
                batch_size=eval_mini_batch_size,
                num_workers=num_workers,
            ),
            out_path=base_path / "test.tsv",
            embedding_storage_mode="none",
        )

        test_results: Result = test_results
        log.info(test_results.log_line)
        log.info(test_results.detailed_results)
        log_line(log)

        # get and return the final test score of best model
        final_score = test_results.main_score

        return final_score

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint, corpus: Corpus, optimizer: torch.optim.Optimizer = SGD
    ):
        return ModelTrainer(
            checkpoint["model"],
            corpus,
            optimizer,
            epoch=checkpoint["epoch"],
            optimizer_state=checkpoint["optimizer_state_dict"],
            scheduler_state=checkpoint["scheduler_state_dict"],
        )

    def find_learning_rate(
        self,
        base_path: Union[Path, str],
        file_name: str = "learning_rate.tsv",
        start_learning_rate: float = 1e-7,
        end_learning_rate: float = 10,
        iterations: int = 100,
        mini_batch_size: int = 32,
        stop_early: bool = True,
        smoothing_factor: float = 0.98,
        **kwargs,
    ) -> Path:
        best_loss = None
        moving_avg_loss = 0

        # cast string to Path
        self.model.set_output(self.model_mode)
        if type(base_path) is str:
            base_path = Path(base_path)
        learning_rate_tsv = init_output_file(base_path, file_name)

        with open(learning_rate_tsv, "a") as f:
            f.write("ITERATION\tTIMESTAMP\tLEARNING_RATE\tTRAIN_LOSS\n")

        optimizer = self.optimizer(
            self.model.parameters(), lr=start_learning_rate, **kwargs
        )

        train_data = self.corpus.train

        batch_loader = DataLoader(train_data, batch_size=mini_batch_size, shuffle=True)

        scheduler = ExpAnnealLR(optimizer, end_learning_rate, iterations)

        model_state = self.model.state_dict()
        model_device = next(self.model.parameters()).device
        self.model.train()

        for itr, batch in enumerate(batch_loader):
            loss = self.model.forward_loss(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            optimizer.step()
            scheduler.step(1)
            learning_rate = scheduler.get_lr()[0]

            loss_item = loss.item()
            if itr == 0:
                best_loss = loss_item
            else:
                if smoothing_factor > 0:
                    moving_avg_loss = (
                        smoothing_factor * moving_avg_loss
                        + (1 - smoothing_factor) * loss_item
                    )
                    loss_item = moving_avg_loss / (1 - smoothing_factor ** (itr + 1))
                if loss_item < best_loss:
                    best_loss = loss

            if stop_early and (loss_item > 4 * best_loss or torch.isnan(loss)):
                log_line(log)
                log.info("loss diverged - stopping early!")
                break

            if itr > iterations:
                break

            with open(str(learning_rate_tsv), "a") as f:
                f.write(
                    f"{itr}\t{datetime.datetime.now():%H:%M:%S}\t{learning_rate}\t{loss_item}\n"
                )

        self.model.load_state_dict(model_state)
        self.model.to(model_device)

        log_line(log)
        log.info(f"learning rate finder finished - plot {learning_rate_tsv}")
        log_line(log)

        return Path(learning_rate_tsv)
