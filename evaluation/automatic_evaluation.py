"""File for automatically evaluating HomerBot.

Calculates:
    - BLEU-2 and BLEU-4: 
        (Determines quality of predicted replies compared to groud truth replies)
    - Entropy: 
        (Determines lexical diversity of the bot's generated replies)
"""




"""Imports."""
print("Importing libraries...")

# for loading model and tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

# for calculating BLEU scores
from nltk.translate.bleu_score import sentence_bleu

# for calculating entropy
import math





"""Constants."""

# model name
MODEL = "jesseD/worst_model_ever"

# number of turns to run the evaluation on
# (a turn is a turn from user then bot )
# so TURNS = 5 means 10 lines of dialogue.
TURNS = 5




"""Load model and tokenizer."""
print("Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)




"""Metric functions."""

def BLEU(ref, hyp, order = 4):
    """Calculates BLEU-n score.
    
    Args:
        ref (string): reference string.
        hpy (string): hypothesis string;
        order (Optional[int]): maximum n-gram order to consider when calculating score (generally 2 or 4).
        
    Returns:
        float: BLEU-n score
    """
    # calculate weights
    # i.e. if order == 4 => weights = (0.25,0.25,0.25,0.25)
    weights = tuple([(1/order)] * order)
    
    # calculate BLEU score and return
    return sentence_bleu([ref.split()], hyp.split(), weights = weights)




def entropy_ideal(length):
    """Calculates the ideal Shannon entropy of a string with given length."""

    prob = 1.0 / length
    return -1.0 * length * prob * math.log(prob) / math.log(2.0)




def entropy(string):
    """Calculates the Shannon entropy of a string.
    
    Args:
        string (str): string to calculate entropy of.

    Returns:
        float 
    """

    # get probability of chars in string
    prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]

    # calculate the entropy
    return - sum([ p * math.log(p) / math.log(2.0) for p in prob ])




"""Run dialogue and calculate metrics."""
print("Running dialogue...")

dialogue = [("Hi Homer!", "Hello!"), ("How are you?", "Good thanks."), ("Do you want a beer?", "Yes please.")]
#         #[(other,       homer),    (other,          homer),          (other,                 homer        )]

# limit dialogue to a certain number of turns
dialogue_splitted = dialogue[:TURNS]

# init lists of generated replies (hpy) and ground truth replies (ref)
generated_responses = []
ground_truth_responses = []

# run dialogue
for turn in range(1,len(dialogue_splitted)+1):

    print(f"turn {turn} / {len(dialogue_splitted)} turns")

    # Dialogue history: dialogue up to turn number "turn" excluding homers latest response
    dialogue_history = " ".join(list(sum(dialogue_splitted[:turn], ()))[:-1])

    # tokenize dialogue history to create bot input ids
    dialogue_hist_ids = tokenizer.encode(dialogue_history + tokenizer.eos_token, return_tensors="pt") 

    # generate a response while limiting the total chat history to 1000 tokens, 
    # this code returns chat history ids + reponse ids in one tensor. 
    history_and_response_ids = model.generate(
        dialogue_hist_ids, 
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,  
        no_repeat_ngram_size=3,       
        do_sample=True, 
        top_k=100, 
        top_p=0.7,
        temperature = 0.8
    )

    # get generated response
    response_ids = history_and_response_ids[:, dialogue_hist_ids.shape[-1]:][0]

    # decode response
    generated_response = tokenizer.decode(response_ids, skip_special_tokens=True)

    # get ground truth response
    ground_truth_response = dialogue_splitted[turn-1][1]

    generated_responses.append(generated_response)
    ground_truth_responses.append(ground_truth_response)


# flatten respone lists to strings
generated_responses_string = " ".join(generated_responses)
ground_truth_responses_string = " ".join(ground_truth_responses)




"""Calculate BLEU + entropy."""
# maybe even compare with a user imitating homer (i.e. replace model by a human quoi, ideally dan castellannata)
print(f"BLEU-2: {BLEU(ref = ground_truth_responses_string, hyp = generated_responses_string, order=2)}")
print(f"BLEU-4: {BLEU(ref = ground_truth_responses_string, hyp = generated_responses_string, order=4)}")
print(f"Entropy: {entropy(generated_responses_string)}")
print(f"Ideal entropy: {entropy_ideal(len(generated_responses_string))}")
# print("Dialo-GPT entropy: ")