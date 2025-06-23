import argparse, json, os, torch
from utils import load_asap_data, set_seed
from model import DocumentBertScoringModel


def _initialize_arguments(p: argparse.ArgumentParser):
    p.add_argument('--bert_model_path', help='bert_model_path')
    p.add_argument('--efl_encode', action='store_true', help='is continue training')
    p.add_argument('--lr', help='learning rate', type=float)
    p.add_argument('--dropout', help='dropout', type=float)
    p.add_argument('--epoch', help='epoch', type=int)
    p.add_argument('--seed', help='seed', type=int)
    p.add_argument('--batch_size', help='batch_size', type=int)
    p.add_argument('--save_step', help='save step', type=int)
    p.add_argument('--bert_batch_size', help='bert_batch_size', type=int)
    p.add_argument('--cuda', action='store_true', help='use gpu or not')
    p.add_argument('--device')
    p.add_argument('--exp_name', help='exp name')
    p.add_argument('--dataset_path', help='train data file')
    p.add_argument('--data_dir', help='data directory to store asap experiment data')
    p.add_argument('--data_sample_rate', help='data_sample_rate', type=float)
    p.add_argument('--prompt_id', help='prompt_id')
    p.add_argument('--fold', help='fold')
    p.add_argument('--chunk_sizes', help='chunk_sizes', type=str)
    p.add_argument('--result_file', help='pred result file path', type=str)

    args = p.parse_args()
    args.exp_name = "./exp/doc_tok_seg/%s_%s" % (args.exp_name, args.prompt_id)
    # args.bert_model_path = args.exp_name

    if torch.cuda.is_available() and args.cuda:
        args.device = 'cuda:0'
    else:
        args.dev = 'cpu'
    return args


if __name__ == "__main__":
    # initialize arguments
    p = argparse.ArgumentParser()
    args = _initialize_arguments(p)
    print(args)
    try:
        os.makedirs(args.exp_name + "/chunk")
        os.makedirs(args.exp_name + "/word_document")
        with open(f"{args.exp_name}/hyperparameters.json", "w") as f:
            f.write(json.dumps(args.__dict__))
    except:
        with open(f"{args.exp_name}/hyperparameters.json", "w") as f:
            f.write(json.dumps(args.__dict__))
    # load data
    train = load_asap_data(args.dataset_path, "train", args.prompt_id)
    set_seed(args.seed)

    train_documents, train_labels = [], []
    for text, label in train:
        train_documents.append(text)
        train_labels.append(label)

    print("sample number:", len(train_documents))
    print("label number:", len(train_labels))

    model = DocumentBertScoringModel(mode="train", args=args)
    model.train((train_documents, train_labels))