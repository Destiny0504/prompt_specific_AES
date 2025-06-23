import argparse
import json
import logging
import os

import torch
import torch.nn as nn
import transformers
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from dataset import AsapMixedDatasetForRegression
from model import EFL_scorer


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
    return tokenized_str, torch.FloatTensor([float(x[1]) for x in data]), torch.LongTensor([float(x[2]) for x in data])


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--lr", default=2e-5, help="learning rate", type=float)
    parser.add_argument("--epoch", default=40, help="epoch", type=int)
    parser.add_argument("--seed", default=2873, help="random seed", type=int)
    parser.add_argument("--batch_size", default=32, help="batch size", type=int)
    parser.add_argument(
        "--weight_decay",
        default=1e-2,
        help="This is the value of optimizer AdamW's weight decay",
        type=float,
    )
    parser.add_argument(
        "--smooth_factor",
        default=0.1,
        help="This is the label smoothing factor for loss function",
        type=float,
    )
    parser.add_argument(
        "--tau",
        default=0.1,
        help="Tempature for contrastive learning.",
        type=float,
    )
    parser.add_argument(
        "--gpu", default=0, help="Select the model for training.", type=int
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Select the model from hugging face.",
        type=str,
    )
    parser.add_argument(
        "--exp_name", default="cws_model", help="Name your experiment.", type=str
    )
    parser.add_argument(
        "--start_log", default=5000, help="The step that start logging", type=int
    )
    parser.add_argument(
        "--accumulation_step", default=2, help="Gradient accumulation step", type=int
    )
    parser.add_argument("--save_step", default=2000, help="Saving intervals", type=int)
    parser.add_argument("--dataset_path", default="./data/learning-agency-lab-automated-essay-scoring-2/train.csv", help="trainingset's path", type=str)
    parser.add_argument("--supervise_contrastive_on_cls", action="store_true", default=False)
    
    return parser.parse_args()


def main(args):
    """The whole training process.

    Args:
        args (argparse.Namespace): The arguments we used in our work, such as \
            hyperparameters and model. 
    """

    EPOCH = args.epoch
    # DEVICE = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    DEVICE = f"cuda:{args.gpu}"
    # DEVICE = "cpu"
    LOG_STEP = args.save_step
    SAVE_STEP = args.save_step
    MODEL_NAME = args.model_name
    EXP_FILE = args.exp_name
    logging.basicConfig(filename="example.log", level=logging.DEBUG)
    assert MODEL_NAME is not None

    print(f"Using {DEVICE}")
    utils.set_seed(args.seed)

    tokenizer = utils.load_tokenizer(MODEL_NAME)

    dataset = AsapMixedDatasetForRegression(args.dataset_path)

    try:
        os.makedirs(f"./exp/{EXP_FILE}")
        os.makedirs(f"./exp/log/{EXP_FILE}")
        with open(f"./exp/{EXP_FILE}/hyperparameters.json", "w") as f:
            f.write(json.dumps(args.__dict__))
    except:
        with open(f"./exp/{EXP_FILE}/hyperparameters.json", "w") as f:
            f.write(json.dumps(args.__dict__))

    count = 1

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: preprocess(
            data=x, tokenizer=tokenizer
        ),
    )

    model = EFL_scorer(
        model_name=MODEL_NAME,
        drop=args.dropout,
        token_num=len(tokenizer)
    )
    model = model.to(DEVICE)
    model.train()
    print("In training mode" if model.training else "In testing mode")

    # declare loss_fn for training
    dataset_label_loss_fn = nn.CrossEntropyLoss()
    score_loss_fn = nn.MSELoss(reduction='mean')
    # score_loss_fn = nn.L1Loss()

    writer = SummaryWriter(f"./exp/log/{EXP_FILE}")

    # declare optimizer
    optimizer = utils.load_optimizer(
        lr=args.lr,
        model_param=model.named_parameters(),
        weight_decay=args.weight_decay,
    )

    # declare scheduler
    scheduler = utils.load_scheduler(
        optimizer=optimizer,
        num_warmup_steps=(
            len(dataset) * EPOCH // args.accumulation_step // args.batch_size // 10
        )
        + 1,
        num_training_steps=(
            len(dataset) * EPOCH // args.accumulation_step // args.batch_size
        )
        + 1,
    )

    for epoch in range(EPOCH):
        train_data = tqdm(train_loader)
        loss = 0.0

        for data, target, dataset_label in train_data:
            target = target.to(DEVICE)
            data = data.to(DEVICE)
            dataset_label = dataset_label.to(DEVICE)

            # Pass the data through the whole model
            score, cls_feature, prompt_feature = model(data)

            loss = score_loss_fn(score.squeeze(), target.squeeze())
            # additional loss
            # loss += dataset_label_loss_fn(prompt_feature.view(-1, 8), dataset_label.view(-1))
            loss += utils.margin_loss(score.squeeze(), target.squeeze())

            if args.supervise_contrastive_on_cls:
                        
            # loss = score_loss_fn(cls_output.view(-1), target.view(-1))
                supervised_contrative_loss = utils.supervised_contrative_loss(cls_feature, dataset_label.view(-1), tau=args.tau)

                loss += supervised_contrative_loss
            # Loss should be divided by args.accumulation_step, because every iter only contribute
            # loss / args.accumulation_step 's loss while we are stimulating the original
            # batch size's loss.
            loss = loss / args.accumulation_step

            # Calculate the gradient (default create_graph is False)
            loss.backward()

            # If the count % args.accumulation_step == 0, the we update the model.
            if count % args.accumulation_step == 0:

                # Update the model's parameter
                optimizer.step()

                # Refresh the gradient
                optimizer.zero_grad()
                scheduler.step()

            if count % LOG_STEP == 0:
                train_acc = torch.mean(torch.abs((score.squeeze() - target)))

                writer.add_scalar("training_acc", train_acc, count)
                writer.add_scalar("training_loss", loss, count)
                train_data.set_description(
                    f"Epoch :{epoch}" + f" Acc : {train_acc}" + f" loss :{round(loss.item(), 4)}"
                )
                logging.info(f"Training Loss : {loss}")

                # Save the model.
                if count % SAVE_STEP == 0 and count >= args.start_log:
                    with open(f"./exp/{EXP_FILE}/step_{count}.model", "wb") as f:
                        torch.save(model.state_dict(), f)
            count += 1

if __name__ == "__main__":
    main(args=get_args())