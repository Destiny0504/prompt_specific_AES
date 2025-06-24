import argparse, os, torch
from utils import load_asap_data, set_seed
from model import DocumentBertScoringModel


def _initialize_arguments(p: argparse.ArgumentParser):
    p.add_argument('--bert_model_path', help='bert_model_path')
    p.add_argument('--efl_encode', action='store_true', help='is continue training')
    p.add_argument('--dropout', help='dropout', type=float)
    p.add_argument('--batch_size', help='batch_size', type=int)
    p.add_argument('--lr', help='learning rate', type=float)
    p.add_argument('--seed', help='seed', type=int)
    p.add_argument('--bert_batch_size', help='bert_batch_size', type=int)
    p.add_argument('--cuda', action='store_true', help='use gpu or not')
    p.add_argument('--device')
    p.add_argument('--model_directory', help='model_directory')
    p.add_argument('--test_id', help='test data file')
    p.add_argument('--data_dir', help='data directory to store asap experiment data')
    p.add_argument('--data_sample_rate', help='data_sample_rate', type=float)
    p.add_argument('--prompt_id', help='prompt_id')
    p.add_argument('--fold', help='fold')
    p.add_argument('--exp_name', help='exp name')
    p.add_argument('--chunk_sizes', help='chunk_sizes', type=str)
    p.add_argument('--result_file', help='pred result file path', type=str)
    p.add_argument(
        "--start_checkpoint", default=0, help="start checkpoint", type=int
    )
    p.add_argument("--end_checkpoint", default=0, help="last checkpoint", type=int)
    p.add_argument('--save_step', help='save step', type=int)
    p.add_argument('--dataset_path', help='train data file')
    args = p.parse_args()
    args.exp_name = "./exp/doc_tok_seg/%s_%s" % (args.exp_name, args.prompt_id)
    ckpt_list = [int(file_name[5:-6]) for file_name in os.listdir(args.exp_name + '/chunk') if 'model' in file_name]
    args.start_checkpoint = min(ckpt_list)
    args.end_checkpoint = max(ckpt_list)
    if torch.cuda.is_available() and args.cuda:
        args.device = 'cuda:0'
    else:
        args.dev = 'cpu'
    return args


if __name__ == "__main__":
    # initialize arguments
    parser = argparse.ArgumentParser()
    args = _initialize_arguments(parser)

    # load data
    test = load_asap_data(args.dataset_path, "test", args.test_id)
    set_seed(args.seed)

    test_documents, test_labels = [], []
    for text, label in test:
        test_documents.append(text)
        test_labels.append(label)

    print("sample number:", len(test_documents))
    print("label number:", len(test_labels))

    model = DocumentBertScoringModel(mode="test", args=args)
    model.predict_for_regress((test_documents, test_labels))