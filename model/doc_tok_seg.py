import os
import utils
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm
from transformers import BertConfig, CONFIG_NAME, BertTokenizer, BertPreTrainedModel, BertConfig, BertModel

asap_ranges = {
    1: (2.0, 12.0),
    2: (1.0, 6.0),
    3: (0.0, 3.0),
    4: (0.0, 3.0),
    5: (0.0, 4.0),
    6: (0.0, 4.0),
    7: (0.0, 30.0),
    8: (0.0, 60.0),
    9: (0.5, 9.0),
    10: (1.0, 24.0),
}


asap_essay_lengths = {
    1: 649,
    2: 704,
    3: 219,
    4: 203,
    5: 258,
    6: 289,
    7: 371,
    8: 1077,
    9: 415,
    10: 1024,
    11: 252
}

class DocumentBertScoringModel():
    def __init__(self, mode, args=None):
        if args is not None:
            self.args = vars(args)
        self.bert_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
        config = BertConfig.from_pretrained("google-bert/bert-base-uncased")
        self.config = config
        self.prompt = int(args.prompt_id)
        chunk_sizes_str = self.args['chunk_sizes']
        self.chunk_sizes = []
        self.bert_batch_sizes = []
        if "0" != chunk_sizes_str:
            for chunk_size_str in chunk_sizes_str.split("_"):
                chunk_size = int(chunk_size_str)
                self.chunk_sizes.append(chunk_size)
                bert_batch_size = int(asap_essay_lengths[self.prompt] / chunk_size) + 1
                self.bert_batch_sizes.append(bert_batch_size)
        bert_batch_size_str = ",".join([str(item) for item in self.bert_batch_sizes])

        print("prompt:%d, asap_essay_length:%d" % (self.prompt, asap_essay_lengths[self.prompt]))
        print("chunk_sizes_str:%s, bert_batch_size_str:%s" % (chunk_sizes_str, bert_batch_size_str))

        if mode == "test":
            self.bert_regression_by_word_document = DocumentBertCombineWordDocumentLinear.from_pretrained(
                "google-bert/bert-base-uncased",
                config=config
            )
            self.bert_regression_by_chunk = DocumentBertSentenceChunkAttentionLSTM.from_pretrained(
                "google-bert/bert-base-uncased",
                config=config)
        elif mode == "train":
            self.bert_regression_by_word_document = DocumentBertCombineWordDocumentLinear.from_pretrained(
                "google-bert/bert-base-uncased",
                config=config
            )
            self.bert_regression_by_chunk = DocumentBertSentenceChunkAttentionLSTM.from_pretrained(
                "google-bert/bert-base-uncased",
                config=config)

    def train(self, data):
        torch.autograd.set_detect_anomaly(True)
        step_count = 0
        correct_output = None
        writer = SummaryWriter(self.args['exp_name'])
        score_loss_fn = nn.MSELoss(reduction='mean')
        word_document_optimizer = utils.load_optimizer(
            lr=self.args['lr'],
            model_param=self.bert_regression_by_word_document.named_parameters(),
            weight_decay=5e-3,
        )
        chunk_optimizer = utils.load_optimizer(
            lr=self.args['lr'],
            model_param=self.bert_regression_by_chunk.named_parameters(),
            weight_decay=5e-3,
        )

        if isinstance(data, tuple) and len(data) == 2:
            document_representations_word_document, document_sequence_lengths_word_document = utils.encode_documents(
                data[0], self.bert_tokenizer, max_input_length=512)
            document_representations_chunk_list, document_sequence_lengths_chunk_list = [], []
            for i in range(len(self.chunk_sizes)):
                document_representations_chunk, document_sequence_lengths_chunk = utils.encode_documents(
                    data[0],
                    self.bert_tokenizer,
                    max_input_length=self.chunk_sizes[i])
                document_representations_chunk_list.append(document_representations_chunk)
                document_sequence_lengths_chunk_list.append(document_sequence_lengths_chunk)
            
            correct_output = torch.FloatTensor([(label - asap_ranges[self.prompt][0]) / (asap_ranges[self.prompt][1] - asap_ranges[self.prompt][0]) for label in data[1]])

        self.bert_regression_by_word_document.to(device=self.args['device'])
        self.bert_regression_by_word_document.train()
        self.bert_regression_by_chunk.to(device=self.args['device'])
        self.bert_regression_by_chunk.init_lstm_params()
        self.bert_regression_by_chunk.train()
        
        epoches = tqdm(range(self.args['epoch']))
        for epoch in epoches:
            loss = 0.0
            for i in range(0, document_representations_word_document.shape[0], self.args['batch_size']):
                target = correct_output[i:i + self.args['batch_size']].squeeze().to(self.args['device'])

                batch_document_tensors_word_document = document_representations_word_document[i:i + self.args['batch_size']].to(device=self.args['device'])
                batch_predictions_word_document = self.bert_regression_by_word_document(batch_document_tensors_word_document, device=self.args['device'])
                batch_predictions_word_document = torch.squeeze(batch_predictions_word_document)

                batch_predictions_word_chunk_sentence_doc = batch_predictions_word_document
                for chunk_index in range(len(self.chunk_sizes)):
                    batch_document_tensors_chunk = document_representations_chunk_list[chunk_index][i:i + self.args['batch_size']].to(
                        device=self.args['device'])
                    batch_predictions_chunk = self.bert_regression_by_chunk(
                        batch_document_tensors_chunk,
                        device=self.args['device'],
                        bert_batch_size=self.bert_batch_sizes[chunk_index]
                    )
                    batch_predictions_chunk = torch.squeeze(batch_predictions_chunk)
                    batch_predictions_word_chunk_sentence_doc = torch.add(batch_predictions_word_chunk_sentence_doc, batch_predictions_chunk)
                    # print(batch_predictions_word_chunk_sentence_doc)

                loss = score_loss_fn(batch_predictions_word_chunk_sentence_doc.squeeze(), target)
                
                loss.backward()
                word_document_optimizer.step()
                word_document_optimizer.zero_grad()
                chunk_optimizer.step()
                chunk_optimizer.zero_grad()
                step_count += 1
                if step_count % self.args['save_step'] == 0:
                    epoches.set_description(
                        f"Epoch :{epoch} " + f"Steps :{step_count}" + f" loss :{round(loss.item(), 4)}"
                    )
                    writer.add_scalar("training_loss", loss, step_count)
                    if epoch >= 75:
                        with open(f"{self.args['exp_name']}/chunk/step_{step_count}.model", "wb") as f:
                            torch.save(self.bert_regression_by_chunk.state_dict(), f)
                        with open(f"{self.args['exp_name']}/word_document/step_{step_count}.model", "wb") as f:
                            torch.save(self.bert_regression_by_word_document.state_dict(), f)
                        # self.bert_regression_by_chunk.save_pretrained(self.args['exp_name'] + "/chunk", from_pt=True)
                        # self.bert_regression_by_word_document.save_pretrained(self.args['exp_name'] + "/word_document", from_pt=True) 

    def predict_for_regress(self, data):
        writer = SummaryWriter(self.args['exp_name'] + '/result')
        for ckpt in range(
            int(self.args['start_checkpoint']), int(self.args['end_checkpoint']) + 1, int(self.args['save_step'])
        ):
            self.bert_regression_by_word_document.load_state_dict(
                torch.load(
                    f"{self.args['exp_name']}/word_document/step_{ckpt}.model",
                    map_location="cpu",
                )
            )
            self.bert_regression_by_chunk.load_state_dict(
                torch.load(
                    f"{self.args['exp_name']}/chunk/step_{ckpt}.model",
                    map_location="cpu",
                )
            )
            correct_output = None
            if isinstance(data, tuple) and len(data) == 2:
                document_representations_word_document, document_sequence_lengths_word_document = utils.encode_documents(
                    data[0], self.bert_tokenizer, max_input_length=512)
                document_representations_chunk_list, document_sequence_lengths_chunk_list = [], []
                for i in range(len(self.chunk_sizes)):
                    document_representations_chunk, document_sequence_lengths_chunk = utils.encode_documents(
                        data[0],
                        self.bert_tokenizer,
                        max_input_length=self.chunk_sizes[i])
                    document_representations_chunk_list.append(document_representations_chunk)
                    document_sequence_lengths_chunk_list.append(document_sequence_lengths_chunk)
                correct_output = torch.FloatTensor(data[1])

            self.bert_regression_by_word_document.to(device=self.args['device'])
            self.bert_regression_by_chunk.to(device=self.args['device'])

            self.bert_regression_by_word_document.eval()
            self.bert_regression_by_chunk.eval()

            with torch.no_grad():
                predictions = torch.empty((document_representations_word_document.shape[0]))
                for i in range(0, document_representations_word_document.shape[0], self.args['batch_size']):
                    batch_document_tensors_word_document = document_representations_word_document[i:i + self.args['batch_size']].to(device=self.args['device'])
                    batch_predictions_word_document = self.bert_regression_by_word_document(batch_document_tensors_word_document, device=self.args['device'])
                    batch_predictions_word_document = torch.squeeze(batch_predictions_word_document)

                    batch_predictions_word_chunk_sentence_doc = batch_predictions_word_document
                    for chunk_index in range(len(self.chunk_sizes)):
                        batch_document_tensors_chunk = document_representations_chunk_list[chunk_index][i:i + self.args['batch_size']].to(
                            device=self.args['device'])
                        batch_predictions_chunk = self.bert_regression_by_chunk(
                            batch_document_tensors_chunk,
                            device=self.args['device'],
                            bert_batch_size=self.bert_batch_sizes[chunk_index]
                        )
                        batch_predictions_chunk = torch.squeeze(batch_predictions_chunk)
                        batch_predictions_word_chunk_sentence_doc = torch.add(batch_predictions_word_chunk_sentence_doc, batch_predictions_chunk)
                    predictions[i:i + self.args['batch_size']] = batch_predictions_word_chunk_sentence_doc
            assert correct_output.shape == predictions.shape

            prediction_scores = []
            label_scores = []
            predictions = predictions.cpu().numpy()
            correct_output = correct_output.cpu().numpy()
            outfile = open(os.path.join(self.args['exp_name'], self.args['result_file']), "a")
            for index, item in enumerate(predictions):
                prediction_scores.append(utils.fix_score(item, int(self.args['test_id'])))
                label_scores.append(correct_output[index])
                outfile.write("%f\t%f\n" % (label_scores[-1], prediction_scores[-1]))
            
            label_scores = [int(score) for score in label_scores]
            prediction_scores = [int(score) for score in prediction_scores]

            min_score = min(min(prediction_scores), min(label_scores))
            max_score = max(max(prediction_scores), max(label_scores)) + 1
            test_eva_res = cohen_kappa_score(prediction_scores, label_scores, labels=[i for i in range(min_score, max_score)], weights="quadratic")
            
            print(f"step : {ckpt} qwk:", float(test_eva_res))
            outfile.write("step : %d\t  qwk : %f\n" % (ckpt,  float(test_eva_res)))
            outfile.close()
            writer.add_scalar(f"QWK prompt {self.args['test_id']}", test_eva_res, ckpt)
        return float(test_eva_res)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(7)


class DocumentBertSentenceChunkAttentionLSTM(BertPreTrainedModel):
    def __init__(self, bert_model_config: BertConfig):
        super(DocumentBertSentenceChunkAttentionLSTM, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        self.dropout = nn.Dropout(p=0.1)
        self.lstm = nn.LSTM(bert_model_config.hidden_size,bert_model_config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(bert_model_config.hidden_size, 1)
        )
        self.w_omega = nn.Parameter(torch.Tensor(bert_model_config.hidden_size, bert_model_config.hidden_size))
        self.b_omega = nn.Parameter(torch.Tensor(1, bert_model_config.hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(bert_model_config.hidden_size, 1))

    def forward(self, document_batch: torch.Tensor, device='cpu', bert_batch_size=0):
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                        min(document_batch.shape[1],
                                            bert_batch_size),
                                        self.bert.config.hidden_size), dtype=torch.float, device=device)
        for doc_id in range(document_batch.shape[0]):
            bert_output[doc_id][:bert_batch_size] = self.dropout(self.bert(document_batch[doc_id][:bert_batch_size,0],
                                                                           token_type_ids=document_batch[doc_id][:bert_batch_size, 1],
                                                                           attention_mask=document_batch[doc_id][:bert_batch_size, 2])[1])
        assert not torch.isnan(bert_output).any()
        output, (_, _) = self.lstm(bert_output.permute(1, 0, 2))
        assert not torch.isnan(output[0, :, :]).any()
        output = output.permute(1, 0, 2)
        # (batch_size, seq_len, num_hiddens)
        attention_w = torch.tanh(torch.matmul(output, self.w_omega) + self.b_omega)
        attention_u = torch.matmul(attention_w, self.u_omega)  # (batch_size, seq_len, 1)
        attention_score = nn.functional.softmax(attention_u, dim=1)  # (batch_size, seq_len, 1)
        attention_hidden = output * attention_score  # (batch_size, seq_len, num_hiddens)
        attention_hidden = torch.sum(attention_hidden, dim=1)  # 加权求和 (batch_size, num_hiddens)
        prediction = self.mlp(attention_hidden)
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction
    
    def init_lstm_params(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
        # print("LSTM weight_ih_l0 std:", self.lstm.weight_ih_l0.std())
        # print("LSTM weight_hh_l0 std:", self.lstm.weight_hh_l0.std())
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        nn.init.uniform_(self.b_omega, -0.1, 0.1)
        self.mlp.apply(init_weights)

class DocumentBertCombineWordDocumentLinear(BertPreTrainedModel):
    def __init__(self, bert_model_config: BertConfig):
        super(DocumentBertCombineWordDocumentLinear, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        self.bert_batch_size = 1
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)

        self.mlp = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size * 2, 1)
        )
        self.mlp.apply(init_weights)

    def forward(self, document_batch: torch.Tensor, device='cpu'):
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                        min(document_batch.shape[1], self.bert_batch_size),
                                        self.bert.config.hidden_size * 2),
                                  dtype=torch.float, device=device)
        for doc_id in range(document_batch.shape[0]):
            all_bert_output_info = self.bert(document_batch[doc_id][:self.bert_batch_size,0],
                                             token_type_ids=document_batch[doc_id][:self.bert_batch_size, 1],
                                             attention_mask=document_batch[doc_id][:self.bert_batch_size, 2])
            bert_token_max = torch.max(all_bert_output_info[0], 1)
            bert_output[doc_id][:self.bert_batch_size] = torch.cat((bert_token_max.values, all_bert_output_info[1]), 1)

        prediction = self.mlp(bert_output.view(bert_output.shape[0], -1))
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction
    

