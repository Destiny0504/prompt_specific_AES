import argparse
import os
import time

import torch
import transformers
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score

import utils
from dataset import AsapDatasetForBaseline
from model import R2Bert

def preprocess(
    data, tokenizer: transformers.PreTrainedTokenizerFast
):
    tokenized_str = tokenizer(
        [x[0] for x in data],
        truncation=True,
        max_length=512,
        padding="longest",
        return_tensors="pt",
    )
    
    # The reason we need to minus one is because of [CLS] token.
    return tokenized_str, torch.FloatTensor([float(x[1]) for x in data])

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, help="batch size", type=int)
    parser.add_argument("--prompt_id", default=1, help="prompt id", type=int)
    parser.add_argument(
        "--gpu", default=0, help="which gpu do you want to use", type=int
    )
    parser.add_argument(
        "--start_checkpoint", default=0, help="start checkpoint", type=int
    )
    parser.add_argument("--end_checkpoint", default=0, help="last checkpoint", type=int)
    parser.add_argument(
        "--save_step", default=0, help="each step between checkpoint", type=int
    )
    parser.add_argument(
        "--exp_name", default=None, help="name your experiment", type=str
    )
    parser.add_argument(
        "--model_name",
        default="bert-base-chinese",
        help="Select the model from hugging face",
        type=str,
    )
    parser.add_argument("--dataset_path", default="./data/learning-agency-lab-automated-essay-scoring-2/test.csv", help="trainingset's path", type=str)
    return parser.parse_args()


def main(args):
    # DEVICE = "cuda:" + str(args.gpu)
    EXP_FILE = args.exp_name
    DEVICE = 'cpu'
    print(DEVICE)
    MODEL_NAME = args.model_name

    tokenizer = utils.load_baseline_tokenizer(MODEL_NAME)
    model = R2Bert(model_name=MODEL_NAME)

    writer = SummaryWriter(f"./exp/result/{EXP_FILE}")

    for ckpt in range(
        args.start_checkpoint, args.end_checkpoint + 1, args.save_step
    ):
        all_target = list()
        all_predict = list()
        model = utils.load_model(
            model=model,
            exp_name=EXP_FILE,
            checkpoint=ckpt,
        )
        model = model.to(DEVICE)
        model.eval()
        try:
            os.makedirs("./exp/result/" + EXP_FILE)
        except:
            pass
        dataset = AsapDatasetForBaseline(args.dataset_path, args.prompt_id, train=False)
        test_loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda x: preprocess(
                data=x, tokenizer=tokenizer
            ),
        )

        test_data = tqdm(test_loader)
        
        with torch.no_grad():
            for data, target in test_data:
                data = data.to(DEVICE)
                # target = target.to(DEVICE)
                # target = (target - dataset.label_min[args.dataset_path[-4:-3]]) / dataset.label_scaler[args.dataset_path[-4:-3]]
                score  = model(data)
                # print(torch.abs((score.squeeze() - target)))
                score = score.squeeze().tolist()
                target = target.tolist()
                all_predict += score
                all_target += target
        # print(all_predict)
        all_predict = [round(predict * dataset.label_scaler[str(args.prompt_id)] + dataset.label_min[str(args.prompt_id)]) for predict in all_predict]
        all_target = [int(target) for target in all_target]
        print(all_predict)
        print(all_target)
        # min_score = dataset.label_min[args.dataset_path[-4:-3]]
        min_score = min(min(all_predict), min(all_target))
        # range_ = dataset.label_scaler[args.dataset_path[-4:-3]] + 1
        max_score = max(max(all_predict), max(all_target)) + 1

        # score = cohen_kappa_score(all_predict, all_target, labels=[i for i in range(min_score, min_score + range_)], weights="quadratic")
        score = cohen_kappa_score(all_predict, all_target, labels=[i for i in range(min_score, max_score)], weights="quadratic")
        print(score)
        writer.add_scalar(f"QWK prompt {args.prompt_id}", score, ckpt)
if __name__ == "__main__":
    main(args=get_args())
    