"""Utilities related to interpretation of model results.

This borrows ideas heavily from fastai which provides interpreter classes
for different kinds of models.
"""
from dataclasses import dataclass, field
import math

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import pytorch_lightning as pl
import torch

import tqdm
from typing import List, Any, Optional, Tuple, Union

__all__ = [
    "Interpretation",
    "InterpretationItem",
]


@dataclass
class InterpretationItem:
    """Provides tools to introspect model performance on a single item."""

    target: Any
    predicted_target: Any
    loss: float
    index: int
    parent_dataloader: DataLoader

    @property
    def dataset(self):
        """Fetches the original dataset used to train and containing this item.

        We need to unwrap the dataset in case we are actually dealing
        with a Subset. We should obtain an indexed Dataset at the end
        of the day, and we will know this is the case because we use
        the sentinel attribute `is_indexed` to mark this.

        This may fail sometimes, but this is better than returning junk
        data which is what happens if we get a shuffled view over the
        dataset.
        """
        dset = self.parent_dataloader.dataset

        if isinstance(dset, Subset):
            dset = dset.dataset

        assert dset.is_indexed == True
        return dset

    def show(self, input_formatter, target_formatter, ax=None, pullback=True):
        """Plots this item onto the provided axes. See also the `show` method of `Interpretation`."""
        if ax is None:
            _, ax = plt.subplots()

        dset = self.dataset
        with dset.no_transforms():
            x = dset[self.index][0]

        if input_formatter is not None:
            input_formatter.show(x, ax)

        ax.set_title(
            "Item {index}; loss={loss:.3f}\n".format(index=self.index, loss=float(self.loss))
        )

        if target_formatter is not None:
            if hasattr(target_formatter, "context"):
                target_formatter.context = dict(is_ground_truth=True)

            target = self.decodes_target(self.target) if pullback else self.target
            target_formatter.show(target, ax)

            if hasattr(target_formatter, "context"):
                target_formatter.context = dict(is_ground_truth=False)

            predicted = (
                self.decodes_target(self.predicted_target) if pullback else self.predicted_target
            )
            target_formatter.show(predicted, ax)

    def decodes_target(self, value: Any) -> Any:
        """Pulls the predicted target backwards through the transformation stack.

        Pullback continues until an irreversible transform is met in order
        to be able to plot targets and predictions in a natural space.
        """
        tfm = self.dataset.transforms
        if hasattr(tfm, "decodes_target"):
            return tfm.decodes_target(value)

        return value


@dataclass
class Interpretation:
    """Provides utilities to interpret predictions of a model.

    Importantly, this is not intended to provide any model introspection
    tools.
    """

    model: pl.LightningModule
    train_dataloader: DataLoader
    val_dataloaders: DataLoader

    train: bool = True
    val_index: int = 0

    train_items: List[InterpretationItem] = field(init=False, repr=False)
    val_item_lists: List[List[InterpretationItem]] = field(init=False, repr=False)

    @property
    def items(self) -> List[InterpretationItem]:
        """All of the ``InterpretationItem`` instances inside this instance."""
        if self.train:
            return self.train_items

        return self.val_item_lists[self.val_index]

    def top_losses(self, ascending=False) -> List[InterpretationItem]:
        """Orders the items by loss."""
        key = lambda item: item.loss if ascending else -item.loss
        return sorted(self.items, key=key)

    def show(
        self,
        n_items: Optional[Union[int, Tuple[int, int]]] = 9,
        items: Optional[List[InterpretationItem]] = None,
        input_formatter=None,
        target_formatter=None,
    ) -> None:
        """Plots a subset of the interpreted items.

        For each item, we "plot" its data, its label, and model performance characteristics
        on this item.

        For example, on an image classification task this might mean to plot the image,
        the images class name as a label above it, the predicted class, and the numerical loss.
        """
        layout = None

        if items is None:
            if isinstance(n_items, (tuple, list)):
                layout = n_items
            else:
                n_rows = int(math.ceil(n_items ** 0.5))
                layout = (n_rows, n_rows)

            items = self.top_losses()[:n_items]
        else:
            n_items = len(items)
            n_rows = int(math.ceil(n_items ** 0.5))
            layout = (n_rows, n_rows)

        _, axes = plt.subplots(*layout, figsize=(layout[0] * 3, layout[1] * 4))

        items_with_nones = list(items) + [None] * (np.product(layout) - n_items)
        for item, ax in zip(items_with_nones, axes.ravel()):
            if item is None:
                ax.axis("off")
            else:
                item.show(input_formatter, target_formatter, ax)

        plt.tight_layout()

    @classmethod
    def from_trainer(cls, trainer: pl.Trainer):
        """Builds an interpreter from an instance of a `pytorch_lightning.Trainer`."""
        return cls(trainer.model, trainer.train_dataloader, trainer.val_dataloaders)

    def dataloader_to_item_list(self, dataloader: DataLoader) -> List[InterpretationItem]:
        """Converts a data loader into a list of interpretation items corresponding to the data samples."""
        items = []

        for batch in tqdm.tqdm(dataloader.iter_all()):
            x, y, indices = batch
            with torch.no_grad():
                y_hat = self.model(x).cpu()
                y_hats = torch.unbind(y_hat, axis=0)
                ys = torch.unbind(y, axis=0)

                losses = [self.model.criterion(yi_hat, yi) for yi_hat, yi in zip(y_hats, ys)]

            for (yi, yi_hat, loss, index) in zip(ys, y_hats, losses, torch.unbind(indices, axis=0)):
                items.append(
                    InterpretationItem(
                        torch.squeeze(yi),
                        torch.squeeze(yi_hat),
                        torch.squeeze(loss),
                        int(index),
                        dataloader,
                    )
                )

        return items

    def __post_init__(self):
        """Populates train_items and val_item_lists.

        This is done by iterating through the dataloaders and pushing data through the models.
        """
        self.train_items = self.dataloader_to_item_list(self.train_dataloader)
        self.val_item_lists = [self.dataloader_to_item_list(dl) for dl in self.val_dataloaders]
