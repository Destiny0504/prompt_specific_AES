import torch
import torch.nn as nn
import transformers


# class EFL_score_classifier(nn.Module):
#     def __init__(
#         self, model_name: str = "google-bert/bert-base-uncased", drop: float = 0.0, EFL_score: int = 20
#     ):

#         super(EFL_score_classifier, self).__init__()

#         self.pretrained_model = transformers.BertModel.from_pretrained(model_name)

#         # EFL score classifier
#         self.scorer = nn.Sequential(
#             nn.Linear(in_features=768, out_features=768),
#             nn.Dropout(p = drop, inplace=False), 
#             nn.Linear(in_features=768, out_features=EFL_score),
#         )

#         self.prompt_classifier = nn.Sequential(
#             nn.Linear(in_features=768, out_features=8)
#         )

#     def forward(self, batch_x):

#         model_predict = self.pretrained_model(**batch_x)

#         prompt_feature = self.prompt_classifier(model_predict.last_hidden_state[:, 1, :])

#         cls_feature = model_predict.last_hidden_state[:, 0, :]
        
#         score = self.scorer(cls_feature)

#         return score, cls_feature, prompt_feature
    
#     def forward_with_positive(self, batch_x):

#         model_predict = self.pretrained_model(**batch_x)
#         model_predict_positive = self.pretrained_model(**batch_x)

#         prompt_feature = self.prompt_classifier(model_predict.last_hidden_state[:, 1, :])

#         cls_feature = model_predict.last_hidden_state[:, 0, :]
#         cls_feature_positive = model_predict_positive.last_hidden_state[:, 0, :]
        
#         score = self.scorer(cls_feature)
#         score_positive = self.scorer(cls_feature_positive)

#         return score, score_positive, cls_feature, cls_feature_positive, prompt_feature


class EFL_scorer(nn.Module):
    def __init__(
        self, model_name: str = "google-bert/bert-base-uncased", drop: float = 0.0, token_num = 0
    ):

        super(EFL_scorer, self).__init__()

        self.pretrained_model = transformers.BertModel.from_pretrained(model_name)
        self.pretrained_model.resize_token_embeddings(token_num)

        # EFL score classifier
        self.scorer = nn.Sequential(
            nn.Linear(in_features=768, out_features=768),
            nn.Dropout(p = drop, inplace=False), 
            nn.Linear(in_features=768, out_features=1),
        )

        self.prompt_classifier = nn.Sequential(
            nn.Linear(in_features=768, out_features=8)
        )

    def forward(self, batch_x):

        
        model_predict = self.pretrained_model(**batch_x)

        prompt_feature = self.prompt_classifier(model_predict.last_hidden_state[:, 1, :])
        cls_feature = model_predict.last_hidden_state[:, 0, :]
        
        # score = torch.sigmoid(self.scorer(cls_feature))
        score = self.scorer(cls_feature)

        return score, cls_feature, prompt_feature


# class EFL_scorer_convolution(nn.Module):
#     def __init__(
#         self, model_name: str = "google-bert/bert-base-uncased", drop: float = 0.0
#     ):

#         super(EFL_scorer, self).__init__()

#         self.pretrained_model = transformers.BertModel.from_pretrained(model_name)

#         # EFL score classifier
#         self.scorer = nn.Sequential(
#             nn.Linear(in_features=768, out_features=768),
#             nn.Dropout(p = drop, inplace=False), 
#             nn.Linear(in_features=768, out_features=1),
#         )

#         self.prompt_classifier = nn.Sequential(
#             nn.Linear(in_features=768, out_features=8)
#         )

#     def forward(self, batch_x):

        
#         model_predict = self.pretrained_model(**batch_x)

#         prompt_feature = self.prompt_classifier(model_predict.last_hidden_state[:, 1, :])
#         cls_feature = model_predict.last_hidden_state[:, 0, :]
        
#         # score = torch.sigmoid(self.scorer(cls_feature))
#         score = self.scorer(cls_feature)

#         return score, cls_feature, 

class R2Bert(nn.Module):
    def __init__(
        self, model_name: str = "google-bert/bert-base-uncased", drop: float = 0.0):
        super(R2Bert, self).__init__()
        self.pretrained_model = transformers.BertModel.from_pretrained(model_name)

        # AES scorer
        self.scorer = nn.Sequential(
            nn.Linear(in_features=768, out_features=768),
            nn.Dropout(p = drop, inplace=False), 
            nn.Linear(in_features=768, out_features=1),
        )

    def forward(self, batch_x):
        model_predict = self.pretrained_model(**batch_x)
        
        cls_feature = model_predict.last_hidden_state[:, 0, :]
        
        score = torch.sigmoid(self.scorer(cls_feature))

        return score