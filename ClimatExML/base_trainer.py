from abc import ABC, abstractmethod
import lightning as pl

class BaseSuperResolutionTrainer(pl.LightningModule, ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass