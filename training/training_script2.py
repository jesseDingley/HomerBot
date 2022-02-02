"""Training script.

This file fine-tunes DialoGPT from raw csv data and pushes the model to the Hugging Face hub.

DialoGPT is a SOTA large-scale pretrained dialogue response generation model for multiturn conversations.
DialoGPT was pre-trained on the task of Causal Language Modeling (CLM). 
Causal language modeling is the task of predicting the token following a sequence of tokens. 
In this situation, the model only attends to the left context (tokens on the left of the mask). 
Such a training is particularly interesting for generation tasks. 
As DialoGPT was pre-trained with CLM we'll also fine-tune it with CLM.

Before running on Google Cloud DL VM w/ Pytorch, it's important to make sure:

    - all packages are installed:
        $ pip install transformers
        $ pip install wandb

    - Git LFS (large file storage) is installed:
	    $ curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
	    $ sudo apt install git-lfs
	    $ git lfs install

    - you're logged into huggingface to get access token for pushing model:
	    $ huggingface-cli login

    - model repo is cloned
	    $ git clone https://huggingface.co/jesseD/worst_model_ever

Typical usage:
    $ python training_script2.py --dialogpt "large" --train "output_preprocessing_homer_7_concat.csv" --logger "wandb" --run_id 12
"""




"""Parse arguments."""
from argparse import ArgumentParser
if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--dialogpt", type=str, required = True, dest="dialogpt", choices = ["small", "medium", "large"], help = "DialoGPT model size.")
    parser.add_argument("--train", type=str, required = True, dest="data_path", help = "data file for training (train + val)")
    parser.add_argument("--logger", type = str, required = False, default = "none", dest = "logger", choices = ["wandb", "none"], help = "choose from wandb logging or no logging.")
    parser.add_argument("--run_id", type = int, required=False, default= 0, dest="run_nb", help="Run id if logger is wandb")
    args = parser.parse_args()




"""Imports."""
print("Importing libraries...")

# for loading model and tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

# for dataset preparation
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling

# for training
from transformers import Trainer, TrainingArguments

# various
import os 

# for gpu settings
from numba import cuda

# for W&B logging
import wandb

# for calculating perplexity
import math

print("Done.\n")




"""Setup."""

# check which device we're running on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:")
print(device)
print()




"""Constants / hyperparameters."""

# pretrained model
MODEL = "microsoft/DialoGPT-" + args.dialogpt #"microsoft/DialoGPT-small"

# our fine-tuned dialogpt model's name
SAVED_MODEL_NAME = "homer-dialogpt"

# hub model name
HUB_NAME = "worst_model_ever"

# where to send logging results to (WandB, mlflow, etc...)
THIRD_PARTY_LOGGER = args.logger

# training + validation data path
DATA_PATH = args.data_path #"output_preprocessing_homer_7_concat.csv"

# train to validation ratio
# 0.8 means 80% of data is training data
TRAIN_VAL_RATIO = 0.8

# freeze ratio i.e. what percentage of layers in DialoGPT do we freeze ?
FREEZE_RATIO = 0 

# The output directory where the model predictions and checkpoints will be written.
OUTPUT_DIR = "./homer-dialogpt-chkpts"

# number of training epochs
NUM_EPOCHS = 3

# training / validation batch sizes 
TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 16

# set to True to load best model after training
LOAD_BEST_MODEL = False

# number of warmup steps
NUM_WARMUP_STEPS = 0

# weight decay
WEIGHT_DECAY = 0

# maximum sequence length
MAX_LENGTH = 768

# The checkpoint save strategy to adopt during training. Possible values are: "no" / "epochs" / "steps"
# NOTE: SAVE_STRATEGY must equal EVALUATION_STRATEGY when LOAD_BEST_MODEL = True
SAVE_STRATEGY = "epoch"

# evaluation strategy to adopt during training.
EVALUATION_STRATEGY = SAVE_STRATEGY if LOAD_BEST_MODEL else "steps"

# The LOGGING strategy to adopt during training.
LOGGING_STRATEGY = "steps"

# HOW MANY LAYERS TO FREEZE DURING FINE-TUNING
if args.dialogpt == "small":
    LAYERS_TO_FREEZE = int(FREEZE_RATIO * 12)
elif args.dialogpt == "medium":
    LAYERS_TO_FREEZE = int(FREEZE_RATIO * 24)
else:
    LAYERS_TO_FREEZE = int(FREEZE_RATIO * 36) 

# enable wandb logging
if THIRD_PARTY_LOGGER == "wandb":
    os.system("wandb login")
    os.system("WANDB_PROJECT=dialogpt-homer")

# run name
RUN_NAME = "run-" + args.dialogpt + "-" + "-".join(args.data_path.split("_")[2:]) + "-" + args.run_id
print(f"run name: {RUN_NAME}")
print()




"""Load tokenizer and model."""
print("Loading tokenizer and model...")

# load DialoGPT tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL, return_special_tokens_mask=True)

# set the padding token as the end of text token "<|endoftext|>"
# tokenizer.pad_token = tokenizer.eos_token

# add special token for padding 
tokenizer.add_special_tokens({'pad_token': '<|pad|>'})



# load DialoGPT model
model = AutoModelForCausalLM.from_pretrained(MODEL)
model.resize_token_embeddings(len(tokenizer))

# set it to training model
model.train()

print("Done.\n")




"""Load dataset from csv files.

Our dataset is a dialogue dataframe. 
Each row of a dialogue dataframe is dialogue session consisting of several contexts and a response. 
Each context and response (so each df cell) is a whole turn by someone.
"""
print("Loading dataset from csv files...")

df = pd.read_csv(DATA_PATH)
df = df.fillna(" ")

split_index = int(TRAIN_VAL_RATIO * len(df))
train_df = df.iloc[:split_index]
val_df = df.iloc[split_index:]

print("Done.\n")




"""Convert Dialogue datasets to Pytorch Dataset objects.

We need to construct Pytorch datasets in order to use the Hugging Face trainer. 
We construct a dataset from an entire dialogue dataframe.
"""
print("Encoding datasets...")
print()

class DialogueDataset(Dataset):
    """Conversational Dataset.

    Conversational dataset where each item is a string of a section of dialogue.

    Attributes:
        batch_encoding (list): list of {"input ids": [145,2547,...], 
                                        "attention mask": [1,1,...]} for each dialogue instance. 
    """

    def __init__(self, dialogue_df, max_length):
        """Initializes dataset attributes.
        
        Args:
            dialogue_df (pandas.core.DataFrame): dialogue dataframe where each row is a dialogue session.
                                                 dialogue session consists of several contexts + response.
            max_length (int): padding length
        """
        
        # convert dialogue df to a list of strings (dialogue sessions)
        # ex: dialogue_sessions[0] = "some context 2. some conext 1, then a response hihi."
        dialogue_sessions = [" ".join(dialogue_session) for dialogue_session in dialogue_df.values.tolist()]
        
        # tokenize dialogue sessions
        self.encoded_dialogue_sessions = []
        print()
        print(f"encoding {len(dialogue_sessions)} dialogue sessions...")
        for i, dialogue_session in enumerate(dialogue_sessions):
            if i % 2000 == 0:
                print(f"encoded {i}/{len(dialogue_sessions)} dialogue sessions.")
            encoded_dialogue_session = tokenizer(dialogue_session + tokenizer.eos_token, padding="max_length", max_length = max_length, truncation = True, return_tensors = "pt")
            self.encoded_dialogue_sessions.append(encoded_dialogue_session)
        
    def __getitem__(self, idx):
        """Returns the item of index idx."""
        return self.encoded_dialogue_sessions[idx]

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.encoded_dialogue_sessions)




print("encoding training set...")
train_ds = DialogueDataset(train_df, max_length = MAX_LENGTH)

print()
print("encoding validation set...")
val_ds = DialogueDataset(val_df, max_length = MAX_LENGTH)

print("Done.\n")




"""Empty GPU cache"""

torch.cuda.empty_cache()




"""Define training parameters / arguments."""

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, # The output directory where the model predictions and checkpoints will be written.
    overwrite_output_dir = True, # overwrite the content of the output directory
    num_train_epochs = NUM_EPOCHS, # number of training epochs
    per_device_train_batch_size = TRAIN_BATCH_SIZE, # batch size for training
    per_device_eval_batch_size = VAL_BATCH_SIZE,  # batch size for evaluation
    load_best_model_at_end = LOAD_BEST_MODEL, # Whether or not to load the best model found during training at the end of training.
    save_strategy = SAVE_STRATEGY, # The checkpoint save strategy to adopt during training
    evaluation_strategy = SAVE_STRATEGY, # same but for eval
    warmup_steps=NUM_WARMUP_STEPS,# number of warmup steps for learning rate scheduler
    logging_strategy = LOGGING_STRATEGY,# the logging strategy to adopt during training
    report_to = THIRD_PARTY_LOGGER, # third party logger (wand, mlflow, none, ...)
    run_name = RUN_NAME # name of run for wandb
)




"""Define data collator.

The data collator will mask tokens for us for the CLM task.
"""

data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer,
    mlm = False # we're doing clm (causal language modeling) and not mlm (masked language modeling)
)




"""Define trainer."""

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)





"""Freeze lower layers of DialoGPT."""

to_find = [f"h.{i}" for i in range(LAYERS_TO_FREEZE)] # ["h.0", "h.1", ...]
for name, param in model.named_parameters():
    if any(nb in name for nb in to_find): # i.e. this layer needs to be freezed
        param.requires_grad = False




"""Actually train the model."""
print("Training started...")

trainer.train()

print("Done.\n")




"""Conclude wandb logging."""

if THIRD_PARTY_LOGGER == "wandb":
    wandb.finish()




"""Evaluate model with perplexity.

perplexity is a measurement of how well a probability distribution or probability model predicts a sample.
A low perplexity indicates the probability distribution is good at predicting the sample.
If we have a perplexity of 100, it means that whenever the model is trying to guess the next word 
it is as confused as if it had to pick between 100 words.
"""
print("Evaluating model...")

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f} \n")




"""Push model + tokenizer to hub."""
print("pushing model and tokenizer to hub...")

model.push_to_hub(HUB_NAME)
tokenizer.push_to_hub(HUB_NAME)

print("Done.\n")




"""Save model to disk."""
print(f"Saving model to ./{SAVED_MODEL_NAME}")

save_dir = os.path.join(".",SAVED_MODEL_NAME)
os.system("rm -r " + save_dir)

model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

# Save trained model and tokenizer using `save_pretrained()`.
model_to_save.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print("Done.\n")