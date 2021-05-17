import pytorch_lightning as pl
from torch import nn, optim
import torch.nn.functional as F

__all__ = ["BaselineRegression", "LinearRegression"]


class LinearRegression(pl.LightningModule):
    input_dimensions = 200 * 200
    output_dimensions = 1

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(self.input_dimensions, self.output_dimensions)
        self.criterion = F.mse_loss

    def forward(self, x):
        flat_x = x.view(x.size(0), -1)
        return self.linear(flat_x)

    def training_step(self, batch, batch_index):
        x, y = batch
        return self.criterion(self(x), y)

    def validation_step(self, batch, batch_index):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=3e-3)


class BaselineRegression(pl.LightningModule):
    input_dimensions = 200 * 200
    output_dimensions = 1

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(self.input_dimensions, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, self.output_dimensions)
        self.criterion = F.mse_loss

    def forward(self, x):
        flat_x = x.view(x.size(0), -1)
        h1 = F.relu(self.l1(flat_x))
        h2 = F.relu(self.l2(h1))
        output = self.l3(h2)
        return output

    def training_step(self, batch, batch_index):
        x, y = batch
        return self.criterion(self(x).squeeze(), y)

    def validation_step(self, batch, batch_index):
        x, y = batch
        loss = self.criterion(self(x).squeeze(), y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=3e-3)
