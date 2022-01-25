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

    - Git LFS (large file storage) is installed:
	    $ curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
	    $ sudo apt install git-lfs
	    $ git lfs install

    - you're logged into huggingface to get access token for pushing model:
	    $ huggingface-cli login

    - model repo is cloned
	    $ git clone https://huggingface.co/jesseD/worst_model_ever

Typical usage:
    $ python trainining_script.py
"""



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

# for argument parsing
from argparse import ArgumentParser

print("Done.\n")




"""Parse arguments."""
if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--dialogpt", type=str, required = True, dest="pretrained", choices = ["small", "medium", "large"] help = "DialoGPT model size.")
    args = parser.parse_args()




"""Setup."""

# # disable WandB logging
# os.environ["WANDB_DISABLED"] = "true"

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
THIRD_PARTY_LOGGER = "none"

# train to validation ratio
# 0.8 means 80% of data is training data
TRAIN_VAL_RATIO = 0.8

# freeze ratio i.e. what percentage of layers in DialoGPT do we freeze ?
FREEZE_RATIO = 2/3 


# The output directory where the model predictions and checkpoints will be written.
OUTPUT_DIR = "./homer-dialogpt-chkpts"

# number of training epochs
NUM_EPOCHS = 2

# training / validation batch sizes 
TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 8

# set to True to load best model after training
LOAD_BEST_MODEL = True

# number of warmup steps
NUM_WARMUP_STEPS = 500

# weight decay
WEIGHT_DECAY = 0.01

# The checkpoint save strategy to adopt during training. Possible values are: "no" / "epochs" / "steps"
SAVE_STRATEGY = "steps"
EVALUATION_STRATEGY = "steps"

# HOW MANY LAYERS TO FREEZE DURING FINE-TUNING
if args.dialogpt == "small":
    LAYERS_TO_FREEZE = int(FREEZE_RATIO * 12)
elif args.dialogpt == "medium":
    LAYERS_TO_FREEZE = int(FREEZE_RATIO * 24)
else:
    LAYERS_TO_FREEZE = int(FREEZE_RATIO * 36) 




"""Load tokenizer and model."""
print("Loading tokenizer and model...")

# load DialoGPT tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# set the padding token as the end of text token "<|endoftext|>"
tokenizer.pad_token = tokenizer.eos_token

# load DialoGPT model
model = AutoModelForCausalLM.from_pretrained(MODEL)

# set it to training model
model.train()

print("Done.\n")




"""Load dataset from csv files.

Our dataset is a dialogue dataframe. 
Each row of a dialogue dataframe is dialogue session consisting of several contexts and a response. 
Each context and response (so each df cell) is a whole turn by someone.
"""
print("Loading dataset from csv files...")

# df = pd.DataFrame([["context2","context1","rep"],["more context2","more context1","another rep"]] * 100, columns = ["c2","c1","r"])
df = pd.read_csv("output_preprocessing_homer_7_concat.csv")
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
        encoded_dialogue_sessions = []
        print()
        print(f"encoding {len(dialogue_sessions)} dialogue sessions...")
        for i, dialogue_session in enumerate(dialogue_sessions):
            if i % 2000 == 0:
                print(f"encoded {i}/{len(dialogue_sessions)} dialogue sessions.")
            encoded_dialogue_session = tokenizer(dialogue_session, padding="max_length", max_length = max_length, truncation = True, return_tensors = "pt")
            encoded_dialogue_sessions.append(encoded_dialogue_session)
        #encoded_dialogue_sessions = tokenizer(dialogue_sessions, padding="max_length", max_length = max_length, truncation = True, return_tensors = "pt")
        
        # return input ids and attention mask
        self.batch_encoding = encoded_dialogue_sessions
        #self.input_ids = encoded_dialogue_sessions["input_ids"]
        #self.attention_masks = encoded_dialogue_sessions["attention_mask"]

    def __getitem__(self, idx):
        #return self.input_ids[idx], self.attention_masks[idx] 
        return self.batch_encoding[idx]

    def __len__(self):
        return len(self.batch_encoding) #self.batch_encoding["input_ids"].shape[0]

print("encoding training set...")
train_ds = DialogueDataset(train_df, max_length = 768)

print()
print("encoding validation set...")
val_ds = DialogueDataset(val_df, max_length = 768)

print("Done.\n")




"""Empty GPU cache"""

#cuda.select_device(0)
#cuda.close()
#cuda.select_device(0)
torch.cuda.empty_cache()




"""Define training parameters / arguments."""

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, # The output directory where the model predictions and checkpoints will be written.
    overwrite_output_dir=True, # overwrite the content of the output directory
    num_train_epochs = NUM_EPOCHS, # number of training epochs
    per_device_train_batch_size=TRAIN_BATCH_SIZE, # batch size for training
    per_device_eval_batch_size=VAL_BATCH_SIZE,  # batch size for evaluation
    load_best_model_at_end = LOAD_BEST_MODEL, # Whether or not to load the best model found during training at the end of training.
    save_strategy = SAVE_STRATEGY, # The checkpoint save strategy to adopt during training
    evaluation_strategy = EVALUATION_STRATEGY,
    warmup_steps=NUM_WARMUP_STEPS,# number of warmup steps for learning rate scheduler
    weight_decay = WEIGHT_DECAY, # weight decay
    report_to = THIRD_PARTY_LOGGER
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


quit()

trainer.train()

print("Done.\n")




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
