from transformers import AutoTokenizer, AutoModel, BertPreTrainedModel, BertModel

# class DprEncoder():
#     def __init__(self, model_name, config):
#         super(DprEncoder, self).__init__(config)

#         self.bert = AutoModel.from_pretrained(model_name=model_name, config=config)
#         self.init_weights()

#     def forward(self, input_ids, attention_mask=None, token_type_ids=None):
#         outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         pooled_output = outputs[1] # cls 에 해당되는 embedding
#         return pooled_output


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config=config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1] # cls 에 해당되는 embedding
        return pooled_output