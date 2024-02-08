from transformers import PreTrainedModel
from .model import TransformerModel

class scGPT_ForPretraining(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.scGPT_backbone = TransformerModel(config)
    
    def forward(self):
        raise NotImplementedError


class scGPT_ForSequenceClassification(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.scGPT_backbone = TransformerModel(config)
    
    def forward(self):
        raise NotImplementedError