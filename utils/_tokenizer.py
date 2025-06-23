import torch, math
import transformers


def load_tokenizer(model_name: str = "microsoft/deberta-v3-base"):
    return transformers.BertTokenizer.from_pretrained(
        model_name,
        pad_token="[PAD]",
        additional_special_tokens=[
            "[Prompt 1]",
            "[Prompt 2]",
            "[Prompt 3]",
            "[Prompt 4]",
            "[Prompt 5]",
            "[Prompt 6]",
            "[Prompt 7]",
            "[Prompt 8]",
        ],
    )
def load_baseline_tokenizer(model_name: str = "google-bert/bert-base-uncased"):
    return transformers.BertTokenizer.from_pretrained(
        model_name,
        pad_token="[PAD]"
    )

def encode_documents(documents: list, tokenizer: transformers.BertTokenizer, max_input_length):
    tokenized_documents = [tokenizer.tokenize(document) for document in documents]
    max_sequences_per_document = math.ceil(max(len(x)/(max_input_length-2) for x in tokenized_documents))
    output = torch.zeros(size=(len(documents), max_sequences_per_document, 3, max_input_length), dtype=torch.long)
    document_seq_lengths = []
    for doc_index, tokenized_document in enumerate(tokenized_documents):
        max_seq_index = 0
        for seq_index, i in enumerate(range(0, len(tokenized_document), (max_input_length-2))):
            raw_tokens = tokenized_document[i:i+(max_input_length-2)]
            tokens = []
            input_type_ids = []
            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in raw_tokens:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_masks = [1] * len(input_ids)

            while len(input_ids) < max_input_length:
                input_ids.append(0)
                input_type_ids.append(0)
                attention_masks.append(0)
            output[doc_index][seq_index] = torch.cat((torch.LongTensor(input_ids).unsqueeze(0),
                                                      torch.LongTensor(input_type_ids).unsqueeze(0),
                                                      torch.LongTensor(attention_masks).unsqueeze(0)),
                                                     dim=0)
            max_seq_index = seq_index
        document_seq_lengths.append(max_seq_index+1)
    return output, torch.LongTensor(document_seq_lengths)

