from utils import (
    set_app_args_parser,
    check_app_args,
    set_device,
    read_dataset,
    convert_data_to_features,
    get_tensor_dataset,
)

from config import (
    TOKENIZER_CACHE_DIR,
    MODEL_CACHE_DIR,
    CHECK_POINTS_DIR,
    MODEL_CONFIG_NAME,
    MODEL_WEIGHTS_NAME,
    EVALUATION_CACHE_DIR,
)

from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from datetime import date
import os
import glob
import random
import numpy as np
import torch
from tqdm import tqdm, trange
import evaluate

import logging
logging.basicConfig(
        format="%(message)s",
        # format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_optimizer(model_, learning_rate, optimizer_warmup_steps, n_train_optimization_steps):
    # Prepare optimizer
    param_optimizer = list(model_.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if "pooler" not in n[0]]

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        # optimizer_grouped_parameters,
        model_.parameters(),
        lr=learning_rate,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=optimizer_warmup_steps,
        num_training_steps=n_train_optimization_steps,
    )

    return optimizer, scheduler


def get_data(data_file, arguments, tokenizer_, is_training):
    dataset = read_dataset(
        data_file=data_file,
        is_training=is_training,
    )
    features = convert_data_to_features(
        dataset=dataset,
        tokenizer=tokenizer_,
        max_seq_length=arguments.max_seq_length,
        max_query_length=arguments.max_query_length,
        max_answer_length=arguments.max_answer_length,
        is_training=is_training,
    )

    if is_training:
        tensor_data = get_tensor_dataset(features, is_training)
        sampler = RandomSampler(tensor_data)
        batch_size = arguments.train_batch_size
        data_loader = DataLoader(
            tensor_data, sampler=sampler, batch_size=batch_size
        )
        n_train_optimization_steps = (
                int(
                    len(dataset)
                    / batch_size
                    / arguments.gradient_accumulation_steps
                )
                * arguments.num_train_epochs
        ) if is_training else None
        return data_loader, n_train_optimization_steps
    else:
        data_tensors = []
        for feature in tqdm(features):
            data_tensors.append(
                [
                    torch.tensor([feature.input_ids], dtype=torch.long),
                    torch.tensor([feature.input_mask], dtype=torch.long),
                    torch.tensor([feature.segment_ids], dtype=torch.long),
                ]
            )
        true_answers = []
        for data in dataset:
            true_answers.append([data.answer_text])
        return data_tensors, true_answers


def save_model(tokenizer_, model_, epoch):
    folder_path = f"{CHECK_POINTS_DIR}{date.today()}/epoch_{epoch}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save a trained model, configuration and tokenizer
    model_to_save = (
        model_.module if hasattr(model_, "module") else model_
    )  # Only save the model it-self

    torch.save(model_to_save.state_dict(), f"{folder_path}/{MODEL_WEIGHTS_NAME}")
    model_to_save.config.to_json_file(f"{folder_path}/{MODEL_CONFIG_NAME}")
    tokenizer_.save_vocabulary(f"{CHECK_POINTS_DIR}{date.today()}")


def train(arguments, tokenizer_, model_, device_, latest_epoch_no_):
    logger.info("\n\n**** START TRAINING *****")
    logger.info("Getting training data...")
    train_dataloader, n_train_optimization_steps = get_data(
        data_file=arguments.train_file,
        arguments=arguments,
        tokenizer_=tokenizer_,
        is_training=True,
    )
    logger.info("Setting model and optimizer...")
    model_.train()
    optimizer, scheduler = set_optimizer(
        model_,
        arguments.learning_rate,
        arguments.optimizer_warmup_steps,
        n_train_optimization_steps,
    )

    loss = 0
    total_loss = 0
    global_step = 0
    # accumulating_batch_count = 0
    pbar = tqdm(train_dataloader, disable=True)
    print(f"pbar : {len(pbar)}")
    max_epoch = int(arguments.num_train_epochs) + latest_epoch_no_ + 1
    for epoch in range(latest_epoch_no_ + 1, max_epoch):
        logger.info(f"Training epoch: {epoch}")
        for step, batch in enumerate(pbar):
            batch = tuple(t.to(device_) for t in batch)
            (
                input_ids,
                input_mask,
                segment_ids,
            ) = batch
            outputs = model_(input_ids, labels=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()
            pbar.update(1)
            if step % 10 == 0:
                pbar.set_description(desc=f"Average loss:{np.mean(total_loss)}")
                total_loss = 0
            if (step + 1) % arguments.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model_.zero_grad()
                global_step += 1
        logger.info(f"Loss: {loss}")
        if epoch % 10 == 0 or epoch == max_epoch - 1:
            save_model(tokenizer_, model_, epoch)
    model_.to(device_)


# def get_additional_info(input_ids, device_):
#     n_dim = list(input_ids.shape)[0]
#     additional_sequence_ids = []
#     additional_input_masks = []
#     for i in range(n_dim):
#         additional_sequence_ids.append([2])
#         additional_input_masks.append([1])
#     additional_sequence_ids = torch.tensor(additional_sequence_ids)
#     additional_sequence_ids = additional_sequence_ids.to(device_)
#     additional_input_masks = torch.tensor(additional_input_masks)
#     additional_input_masks = additional_input_masks.to(device_)
#     return [additional_sequence_ids, additional_input_masks]


def predict(arguments, tokenizer_, model_, device_, temperature=1, top_p=0.9):
    logger.info("**** START PREDICTION *****")
    logger.info("Getting prediction data...")
    data_tensors, true_answers = get_data(
        data_file=arguments.predict_file,
        arguments=arguments,
        tokenizer_=tokenizer_,
        is_training=False,
    )
    model_.eval()
    generated_list = []
    generated_answers = []
    filter_value = -float("Inf")

    additional_sequence_ids = torch.tensor([[2]])
    additional_sequence_ids = additional_sequence_ids.to(device_)
    additional_input_masks = torch.tensor([[1]])
    additional_input_masks = additional_input_masks.to(device_)

    logger.info("Generating prediction ...")
    with torch.no_grad():
        for data in data_tensors:
            input_ids, input_mask, segment_ids = data
            input_ids = input_ids.to(device_)
            input_mask = input_mask.to(device_)
            segment_ids = segment_ids.to(device_)

            entry_finished = False
            answer = None
            for i in range(arguments.max_answer_length):
                outputs = model_(input_ids, labels=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
                loss, logits = outputs[:2]
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                input_ids = torch.cat((input_ids, next_token), dim=1)
                segment_ids = torch.cat((segment_ids, additional_sequence_ids), dim=1)
                input_mask = torch.cat((input_mask, additional_input_masks), dim=1)

                if next_token in tokenizer_.encode("[EOD]"):
                    entry_finished = True
                answer = torch.cat((answer, next_token), dim=1) if answer != None else next_token

                if entry_finished:
                    output_list = list(input_ids.cpu().squeeze().numpy())
                    output_text = tokenizer_.decode(output_list)
                    generated_list.append(output_text)
                    break
            if not entry_finished:
                output_list = list(input_ids.cpu().squeeze().numpy())
                output_text = f"{tokenizer_.decode(output_list)} [EOD]"
                generated_list.append(output_text)

            answer = tokenizer_.decode(list(answer.cpu().squeeze().numpy()))
            answer = answer[:-6] if answer.endswith("[EOD]") else answer
            generated_answers.append(answer.strip())
    return generated_list, generated_answers, true_answers


def measure_evaluation(predictions, references):
    bleu = evaluate.load("bleu", cache_dir=EVALUATION_CACHE_DIR)
    meteor = evaluate.load("meteor", cache_dir=EVALUATION_CACHE_DIR)
    rouge = evaluate.load("rouge", cache_dir=EVALUATION_CACHE_DIR)

    result_bleu_1 = bleu.compute(predictions=predictions, references=references, max_order=1)
    result_bleu_4 = bleu.compute(predictions=predictions, references=references, max_order=4)
    result_meteor = meteor.compute(predictions=predictions, references=references)
    result_rouge = rouge.compute(predictions=predictions, references=references)
    print("\n")
    print(f"bleu-1: {result_bleu_1}")
    print(f"bleu-4: {result_bleu_4}")
    print(f"meteor: {result_meteor}")
    print(f"rouge: {result_rouge}")


def load_model(arguments):
    if arguments.local_model_folder is not None:
        model_ = GPT2LMHeadModel.from_pretrained(arguments.local_model_folder)
        tokenizer_ = GPT2Tokenizer.from_pretrained(arguments.local_tokenizer_folder)
        latest_epoch_no = arguments.latest_epoch_no
    else:
        list_dates = [x[1] for x in os.walk(CHECK_POINTS_DIR)][0]
        latest_date = max(list_dates)
        list_epochs = [x[0] for x in os.walk(f"{CHECK_POINTS_DIR}{latest_date}")]
        latest_epoch = max(list_epochs)
        model_ = GPT2LMHeadModel.from_pretrained(latest_epoch)
        tokenizer_ = GPT2Tokenizer.from_pretrained(f"{CHECK_POINTS_DIR}{latest_date}")
        if str(latest_date) == str(date.today()):
            latest_epoch_no = int(latest_epoch.split("_")[-1])
        else:
            latest_epoch_no = 0
        logger.info(f"Loaded model from: {latest_epoch}")
        logger.info(f"Loaded tokenizer from: {CHECK_POINTS_DIR}{latest_date}")
    return model_, tokenizer_, latest_epoch_no


def main():
    logger.info("Checking application arguments ...")
    args = set_app_args_parser().parse_args()
    check_app_args(args)
    logger.info(f"\nValidated arguments: {args}")

    logger.info("\nSetting device ...")
    device, n_gpu = set_device(args.local_rank, args.no_cuda)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    latest_epoch_no = 0
    if args.load_local_model:
        logger.info("\nLoading latest local model...")
        model, tokenizer, latest_epoch_no = load_model(args)
    else:
        logger.info("\nLoading model from HuggingFace...")
        model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=MODEL_CACHE_DIR)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=TOKENIZER_CACHE_DIR)

    logger.info("\nAdding new special tokens to tokenizer vocabulary...")
    tokenizer.add_tokens(["[BOD]", "[SEP]", "[EOD]"])
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    if args.do_train:
        train(args, tokenizer, model, device, latest_epoch_no)
    if args.do_predict:
        generated_list, generated_answers, true_answers = predict(args, tokenizer, model, device)
        print("\n")
        print(generated_answers)
        print("\n")
        print(true_answers)
        measure_evaluation(generated_answers, true_answers)
        # for idx in range(len(generated_list)):
        #     print("\n")
        #     logger.info(generated_list[idx])
        #     logger.info(generated_answers[idx])
        #     logger.info(true_answers[idx])


if __name__ == '__main__':
    main()
