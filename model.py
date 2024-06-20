from transformers import BartForConditionalGeneration, BartTokenizer
import torch.nn as nn


class TextSummaryModel(nn.Module):
    def __init__(self, pretrained_model_name_or_path):
        super(TextSummaryModel, self).__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.bart.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return {"loss": outputs.loss, "logits": outputs.logits}
