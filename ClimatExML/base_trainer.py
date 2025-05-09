from abc import ABC, abstractmethod
import lightning as pl

class BaseSuperResolutionTrainer(pl.LightningModule, ABC):
    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass