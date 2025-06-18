import transformers


def load_tokenizer(model_name: str = "microsoft/deberta-v3-base"):
    return transformers.DebertaV2Tokenizer.from_pretrained(
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