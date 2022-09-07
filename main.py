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
    MODEL_WEIGHTS_NAME
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

import logging
logging.basicConfig(
        format="%(message)s",
        # format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_optimizer(model_, learning_rate, optimizer_warmup_steps, warmup_proportion, n_train_optimization_steps):
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


def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None


def get_data(data_file, arguments, tokenizer_, is_training, is_predict, context_predict=None, question_predict=None):
    dataset = read_dataset(
        data_file=data_file,
        is_training=is_training,
        is_predict=is_predict,
        context_predict=context_predict,
        question_predict=question_predict
    )
    features = convert_data_to_features(
        dataset=dataset,
        tokenizer=tokenizer_,
        max_seq_length=arguments.max_seq_length,
        max_query_length=arguments.max_query_length,
        is_training=is_training,
    )
    tensor_data = get_tensor_dataset(features)
    sampler = RandomSampler(tensor_data) if is_training else SequentialSampler(tensor_data)
    batch_size = arguments.train_batch_size if is_training else arguments.predict_batch_size
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
        is_predict=False,
    )
    logger.info("Setting model and optimizer...")
    model_.train()
    optimizer, scheduler = set_optimizer(
        model_,
        arguments.learning_rate,
        arguments.optimizer_warmup_steps,
        arguments.warmup_proportion,
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


def predict(arguments, tokenizer_, model_, device_, temperature=0.9, top_p=0.8):
    # context = "At the period in which i commence this history there resided in this mansion an elderly spinster of rank named the Honourable Miss Delmar sister of the late Lord de Versely and aunt to the present earl and an Honourable Captain Delmar who was the second son of the deceased nobleman. This property belonged to the Honourable Miss Delmar and was at her entire disposal upon her decease. At the period in which i commence this history there resided in this mansion an elderly spinster of rank named the Honourable Miss Delmar sister of the late Lord de Versely and aunt to the present earl and an Honourable Captain Delmar who was the second son of the deceased nobleman. This property belonged to the Honourable Miss Delmar and was at her entire disposal upon her decease. At the period in which i commence this history there resided in this mansion an elderly spinster of rank named the Honourable Miss Delmar sister of the late Lord de Versely and aunt to the present earl and an Honourable Captain Delmar who was the second son of the deceased nobleman. This property belonged to the Honourable Miss Delmar and was at her entire disposal upon her decease. As soon as the tailor had gone Miss Medea asked me if i would not like to take another run in the garden. i knew that she wished to speak to her father and therefore had a pleasure in disappointing her. i therefore replied that i had been there nearly the whole day and did not wish to go out any more. Never mind whether you wish it or not i wish you to go replied Miss Medea tartly. Medea how can you be so rude. As soon as the tailor had gone Miss Medea asked me if i would not like to take another run in the garden. i knew that she wished to speak to her father and therefore had a pleasure in disappointing her. i therefore replied that i had been there nearly the whole day and did not wish to go out any more. Never mind whether you wish it or not i wish you to go replied Miss Medea tartly. Medea how can you be so rude. As soon as the tailor had gone Miss Medea asked me if i would not like to take another run in the garden. i knew that she wished to speak to her father and therefore had a pleasure in disappointing her. i therefore replied that i had been there nearly the whole day and did not wish to go out any more. Never mind whether you wish it or not i wish you to go replied Miss Medea tartly. Medea how can you be so rude. Know then that when you were last at Madeline Hall i was sent for to draw up the will of the Honourable Miss Delmar and i then discovered that the will which had been made in favour of Lord de Versely to whom Miss Delmar had left everything was by his express desire to be altered in your favour and at the same time the secret of your birth was confided to me. You will see therefore that Lord de Versely did not neglect your interests. The Honourable Miss Delmar having had such a long innings then gave it up because she was out of breath. She reads a great deal and is therefore only a customer to the library. Ladies who are fond of reading are seldom fond of working. Good morning Miss Evans said Captain Bridgeman you come for more food for the mind i presume. Miss Evans gave a bob and turned to my mother. Have you anything new Mrs Keene. i have brought back the three volumes of Godolphin. Yes miss i have some books down to day. Mercy on me how very like. exclaimed Miss Culpepper looking at me and then at her father. Would not you like to go into the garden little boy. continued she there through the passage out of the door you ca not miss it."
    # question = "Who is Miss Delmer"

    context = "amorphus. philautia. asotus. moria. hedon. cos. anaides. gelaia. morphides. prosaites. morus. cupid. mutes. phronesis thauma time. scene gargaphie. induction. the stage. after the second sounding. enter three of the children struggling. child. Pray you away why fellows. Gods so what do you mean. child. Marry that you shall not speak the prologue sir. child. ana. Good play but 'tis too rough and boisterous. amo. i will second it with a stroke easier wherein i will prove his language. a charge. ana. This is filthy and grave now. hed. o 'tis cool and wary play. We must not disgrace our own camerade too much. amo. Jonson never again produced so fresh and lovable a feminine personage as Rachel although in other respects The Case is Altered is not a conspicuous play and save for the satirising of Antony Munday in the person of Antonio Balladino and Gabriel Harvey as well is perhaps the least characteristic of the comedies of Jonson. Every Man in His Humour probably first acted late in the summer of and at the Curtain is commonly regarded as an epoch making play and this view is not unjustified. The play is admirably written and each character is vividly conceived and with a firm touch based on observation of the men of the London of the day. Jonson was neither in this his first great comedy nor in any other play that he wrote a supine classicist urging that English drama return to a slavish adherence to classical conditions. All this points to an association with Henslowe of some duration as no mere tyro would be thus paid in advance upon mere promise. From allusions in Dekker's play Satiromastix it appears that Jonson like Shakespeare began life as an actor and that he ambled in a leather pitch by a play wagon taking at one time the part of Hieronimo in Kyd's famous play The Spanish Tragedy. By the beginning of Jonson though still in needy circumstances had begun to receive recognition. child. Tut fear not child this will never distaste a true sense be not out and good enough. i would thou hadst some sugar candied to sweeten thy mouth. the third sounding. prologue. If gracious silence sweet attention Quick sight and quicker apprehension The lights of judgment's throne shine any where Our doubtful author hopes this is their sphere And therefore opens he himself to those To other weaker beams his labours close As loth to prostitute their virgin strain To every vulgar and adulterate brain. i thought at first he would have plaid the ignorant critic with everything along as he had gone i expected some such device. child. o you shall see me do that rarely lend me thy cloak. child. Soft sir you will speak my prologue in it. child. No would i might never stir then. child. Lend it him lend it him. child. Well you have sworn. gives him the cloak. child. i have. are. You tell us wonders Crites. cri. This is nothing. There stands a neophite glazing of his face Pruning his clothes perfuming of his hair Against his idol enters and repeats Like an unperfect prologue at third music His part of speeches and confederate jests In passion to himself. And i prefer another now far before him a million at least. pha. Who might that be guardian. mor. Marry fair charge Anaides. pha. Anaides. you talk'd of a tune Philautia there's one speaks in a key like the opening of some justice's gate or a postboy's horn as if his voice feared an arrest for some ill words it should give and were loth to come forth. phi. Ay and he has a very imperfect face. pha. Volpone was laid as to scene in Venice. Whether because of the success of Eastward Hoe or for other reasons the other three comedies declare in the words of the prologue to The Alchemist. Our scene is London 'cause we would make known No country's mirth is better than our own."
    question = "Who normally delivers the opening prologue in the play"

    logger.info("**** START PREDICTION *****")
    predict_dataloader, _ = get_data(
        data_file="",
        arguments=arguments,
        tokenizer_=tokenizer_,
        is_training=False,
        is_predict=True,
        context_predict=context,
        question_predict=question
    )
    # model_.eval()
    generated_num = 0
    generated_list = []
    space_tensor = torch.tensor(tokenizer_.encode(" ")).unsqueeze(0)
    space_tensor = space_tensor.to(device_)

    for input_ids, input_mask, segment_ids in tqdm(
            predict_dataloader, desc="Evaluating", disable=False
    ):
        input_ids = input_ids.to(device_)
        input_mask = input_mask.to(device_)
        segment_ids = segment_ids.to(device_)
        filter_value = -float("Inf")
        generated = input_ids
        with torch.no_grad():
            entry_finished = False
            for i in range(arguments.max_answer_length):
                print(f"Generating word number {i} ")
                outputs = model_(input_ids, labels=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
                loss, logits = outputs[:2]
                # print("\n")
                # print(logits)
                # print("\n")

                # softmax_logits = torch.softmax(logits[0, -1], dim=0)
                # print(softmax_logits)
                # print("\n")
                # next_token_id = choose_from_top(softmax_logits.to(device_).numpy(), n=20)
                # print(next_token_id)
                # print("\n")
                # cur_ids = torch.cat([input_ids, torch.ones((1, 1)).long().to(device_) * next_token_id], dim=1)
                # print(cur_ids)
                # print("\n")
                # output_list = list(cur_ids.squeeze().to(device_).numpy())
                # output_text = tokenizer_.decode(output_list)
                # print(output_list)
                # print("\n")
                # print(output_text)
                # print("\n")

                # print(logits[:, -1, :])
                # print("\n")
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                # print(logits)
                # print("\n")
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                # print(sorted_logits)
                # print(len(sorted_logits[0].item()))
                # print("\n")
                # print(sorted_indices)
                # print("\n")
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # print(cumulative_probs)
                # print(len(cumulative_probs[0].item()))
                # print("\n")
                sorted_indices_to_remove = cumulative_probs > top_p
                # print(sorted_indices_to_remove)
                # print("\n")
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                # print(sorted_indices_to_remove)
                # print("\n")
                sorted_indices_to_remove[..., 0] = 0
                # print(sorted_indices_to_remove)
                # print("\n")

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                # print(indices_to_remove)
                # print("\n")
                logits[:, indices_to_remove] = filter_value
                # print(logits)
                # print("\n")

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                print(next_token)
                text = torch.cat((next_token, space_tensor), dim=1)
                text = list(text.cpu().squeeze().numpy())
                print(tokenizer_.decode(text))
                print("\n")
                generated = torch.cat((generated, next_token, space_tensor), dim=1)

                if next_token in tokenizer_.encode("[EOD]"):
                    entry_finished = True

                if entry_finished:
                    generated_num = generated_num + 1
                    output_list = list(generated.cpu().squeeze().numpy())
                    output_text = tokenizer_.decode(output_list)
                    generated_list.append(output_text)
                    break
            if not entry_finished:
                output_list = list(generated.cpu().squeeze().numpy())
                output_text = f"{tokenizer_.decode(output_list)}[EOD]"
                generated_list.append(output_text)
    return generated_list


def load_model():
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
        model, tokenizer, latest_epoch_no = load_model()
    else:
        logger.info("\nLoading model from HuggingFace...")
        model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=MODEL_CACHE_DIR)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=TOKENIZER_CACHE_DIR)

    logger.info("\nAdding new special tokens to tokenizer vocabulary...")
    tokenizer.add_tokens(["[BOD]", "[SEP]", "[EOD]"])
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    if args.do_train:
        train(args, tokenizer, model, device, latest_epoch_no)
    if args.do_predict:
        generated_list = predict(args, tokenizer, model, device)
        for prediction in generated_list:
            logger.info(prediction)


if __name__ == '__main__':
    main()
