from data import NarrativeQADataset, NarrativeQAInputFeatures
from torch.utils.data import TensorDataset
import os
import argparse
import torch
import json
from tqdm import tqdm

from config import CHECK_POINTS_DIR

import logging
logger = logging.getLogger(__name__)

WHITE_SPACE_CODES = [" ", "\t", "\r", "\n", ""]


def set_app_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="data in json format for training",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="data in json format for evaluation",
    )
    parser.add_argument(
        "--max_seq_length",
        default=1024,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--max_query_length",
        default=128,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_predict", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--train_batch_size",
        default=4,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--predict_batch_size",
        default=1,
        type=int,
        help="Total batch size for predictions.",
    )
    parser.add_argument(
        "--optimizer_warmup_steps",
        default=200,
        type=int,
        help="The initial warm up steps for Adam.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=100,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
        "of training.",
    )
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json "
        "output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=3,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )
    parser.add_argument(
        "--load_local_model",
        action="store_true",
        help="If yes, load model from local.",
    )

    return parser


def check_app_args(arguments):
    if not arguments.do_train and not arguments.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")
    if arguments.do_train and not arguments.train_file:
        raise ValueError("If `do_train` is True, then `train_file` must be specified.")
    if arguments.do_predict and not arguments.predict_file:
        raise ValueError("If `do_predict` is True, then `predict_file` must be specified.")
    if arguments.load_local_model:
        if not (os.path.exists(CHECK_POINTS_DIR) or os.listdir(CHECK_POINTS_DIR)):
            raise ValueError("There is no model from local.")


def set_device(local_rank, is_no_cuda):
    if local_rank == -1 or is_no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not is_no_cuda else "cpu"
        )
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}".format(
            device, n_gpu, bool(local_rank != -1)
        )
    )
    return device, n_gpu


# Accumulated batch size (since GPT2 is so big)
def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None


def read_dataset(data_file, is_training):
    with open(data_file, "r", encoding="utf-8") as reader:
        input_data = json.load(reader)["data"]

    result = []
    for entry in input_data:
        for para in entry["paragraphs"]:
            context = para["context"]
            context_tokens = [word for word in context.split(" ") if word not in WHITE_SPACE_CODES]
            context_text = " ".join(context_tokens)
            for qa in para["qas"]:
                qa_id = qa['id']
                question_text = qa["question"]
                if not question_text.endswith("?"):
                    question_text = f"{question_text}?"
                answer_text = qa["answers"] if is_training else None
                result.append(
                    NarrativeQADataset(
                        qa_id=qa_id,
                        question_text=question_text,
                        answer_text=answer_text,
                        context_text=context_text,
                        context_tokens=context_tokens,
                    )
                )
        break
    return result


def tokenize_data(data, tokenizer, max_seq_length, max_query_length, is_training):
    is_data_truncated = 0

    query_tokens = tokenizer.tokenize(data.question_text)
    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]
        is_data_truncated = 1

    if is_training:
        answer_tokens = tokenizer.tokenize(data.answer_text)
        # The -5 accounts for [CLS], [SEP], [SEP], [SEP] and [SEP]
        max_context_length = max_seq_length - len(query_tokens) - len(answer_tokens) - 5
    else:
        answer_tokens = None
        # The -5 accounts for [CLS], [SEP], [SEP]
        max_context_length = max_seq_length - len(query_tokens) - 3

    context_tokens = tokenizer.tokenize(data.context_text)
    if len(context_tokens) > max_context_length:
        context_tokens = context_tokens[:max_context_length - 1]
        is_data_truncated = 1

    return context_tokens, query_tokens, answer_tokens, is_data_truncated


def set_input_tokens(context_tokens, query_tokens, answer_tokens, is_training):
    tokens = []
    segment_ids = []
    current_segment_id = 0
    tokens.append("[BOD]")
    segment_ids.append(current_segment_id)

    for token in context_tokens:
        tokens.append(token)
        segment_ids.append(current_segment_id)
    tokens.append("[SEP]")
    segment_ids.append(current_segment_id)
    current_segment_id = current_segment_id + 1

    for token in query_tokens:
        tokens.append(token)
        segment_ids.append(current_segment_id)

    if answer_tokens:
        tokens.append("[SEP]")
        segment_ids.append(current_segment_id)
        current_segment_id = current_segment_id + 1
        for token in answer_tokens:
            tokens.append(token)
            segment_ids.append(current_segment_id)

    if is_training:
        tokens.append("[EOD]")
        segment_ids.append(current_segment_id)
    else:
        tokens.append("[SEP]")
        segment_ids.append(current_segment_id)

    return tokens, segment_ids


def add_padding(input_ids, input_mask, segment_ids, max_seq_length, is_training):
    if is_training:
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids


def convert_data_to_features(dataset, tokenizer, max_seq_length, max_query_length, is_training):
    result = []
    n_truncated_data = 0

    for (data_index, data) in enumerate(tqdm(dataset)):
        context_tokens, query_tokens, answer_tokens, is_data_truncated = tokenize_data(
            data, tokenizer, max_seq_length, max_query_length, is_training
        )
        tokens, segment_ids = set_input_tokens(context_tokens, query_tokens, answer_tokens, is_training)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        input_ids, input_mask, segment_ids = add_padding(input_ids, input_mask, segment_ids, max_seq_length, is_training)
        result.append(
            NarrativeQAInputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
            )
        )
        n_truncated_data += is_data_truncated

    if n_truncated_data > 0:
        logger.warning(f"No of truncated data: {n_truncated_data}")
    return result


def get_tensor_dataset(features, is_training):
    if is_training:
        all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )

        return TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
        )
    else:
        return TensorDataset(
            torch.tensor([features.input_ids], dtype=torch.long),
            torch.tensor([features.input_mask], dtype=torch.long),
            torch.tensor([features.segment_ids], dtype=torch.long),
        )

