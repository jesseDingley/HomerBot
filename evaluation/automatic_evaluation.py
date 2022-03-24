"""File for automatically evaluating HomerBot.

Calculates:
    - BLEU-2 and BLEU-4: 
        (Determines quality of predicted replies compared to groud truth replies)
    - Entropy: 
        (Determines lexical diversity of the bot's generated replies)
    - Cosine similarity:
        (Determines proximity beetwen the truth and predicted replies)
    - Jaccard similarity:
        (Determines lexical similarity beetwen the truth and predicted replies)
"""




"""Imports."""
print("Importing libraries...")

# for loading model and tokenizer
from crypt import METHOD_BLOWFISH
from transformers import AutoModelForCausalLM, AutoTokenizer

# for calculating BLEU scores
from nltk.translate.bleu_score import sentence_bleu

# for calculating entropy
import math

# for calculating cosine similarity
from transformers import BertTokenizer, BertModel
from torch.nn import CosineSimilarity

# for calculating Jaccard similarity
from nltk.stem import WordNetLemmatizer

# for dialogue
import pandas as pd
from typing import List, Tuple


"""Constants."""

# models names
MODEL = "DingleyMaillotUrgell/homer-bot"
BERT = "bert-base-uncased"


# number of turns to run the evaluation on
# (a turn is a turn from user then bot )
# so TURNS = 5 means 10 lines of dialogue.
TURNS = 5




"""Load model, tokenizer and lemmatizer."""
print("Loading model, tokenizer and lemmatizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)

tokenizer_bert = BertTokenizer.from_pretrained(BERT)
model_bert = BertModel.from_pretrained(BERT)

lemmatizer = WordNetLemmatizer()


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

def doc2vect(response_string: str):
    """Convert document to a vector

    Args:
        response_string (str): document

    Returns:
        _type_: document convert into tensor
    """
    inputs = tokenizer_bert(response_string, return_tensors="pt")
    outputs = model_bert(**inputs)
    return outputs.last_hidden_state[:,0,:]

def cosine_similarity(ref: str, hyp: str)->float:
    """Calculates cosine similarity

    Args:
        ref (str): reference string.
        hyp (str): hypothesis string.

    Returns:
        float: cosine simalrity
    """
    # convert documents to vectors
    ref_embedding = doc2vect(ref)
    hyp_embedding = doc2vect(hyp)
    
    # calcul cosine similarity beetwen the vectors
    cos = CosineSimilarity(dim=1, eps=1e-6)
    output = cos(ref_embedding, hyp_embedding)
    return output[0]

def doc2words(response_string: str)->List[str]:
    """Convert a string to a list of words lemmatized.

    Args:
        response_string (str): document.

    Returns:
        List[str]: document lemmatized and tokenized.
    """
    response_string = response_string.lower()
    response_token = tokenizer_bert.tokenize(response_string)
    output = [lemmatizer.lemmatize(word) for word in response_token]
    return output

def jaccard_similarity(ref: str, hyp: str)->float:
    """Calculates Jaccard similarity

    Args:
        ref (str): reference string.
        hyp (str): hypothesis string.

    Returns:
        float: Jaccard simalrity
    """
    # List the unique words in a document
    words_ref = set(doc2words(ref))
    words_hyp = set(doc2words(hyp))
    
    # Find the intersection of words list of ref & hyp
    intersection = words_ref.intersection(words_hyp)

    # Find the union of words list of ref & hyp
    union = words_ref.union(words_hyp)
        
    # Calculate Jaccard similarity score 
    # using length of intersection set divided by length of union set
    return float(len(intersection)) / len(union)

"""Parse arguments."""
from argparse import ArgumentParser
if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--test", type=str, required=True, dest="data_path", help = "data file for testing")
    parser.add_argument("--generation_methods", type=str, default='all', choices=['all','topk','greedy','beam'], dest="generation_methods", help="The name of the generation method")
    parser.add_argument("--metrics", type=str, default='all', choices=['all','bleu-2','bleu-4','entropy','ideal entropy','cosine similarity', 'jaccard similarity'], dest="metrics", help="The name of the metrics")
    args = parser.parse_args()
    
DATA_PATH = args.data_path
METHODS = args.generation_methods
METRICS = args.metrics


"""Run dialogue and calculate metrics."""
print("Running dialogue...")

dialogue_df: pd.DataFrame = pd.read_csv(filepath_or_buffer=DATA_PATH, header=0)
dialogue: List[Tuple[str,str]]  = [tuple(x) for x in dialogue_df.values.tolist()]

#dialogue = [("Hi Homer!", "Hello!"), ("How are you?", "Good thanks."), ("Do you want a beer?", "Yes please.")]
#         #[(other,       homer),    (other,          homer),          (other,                 homer        )]

# limit dialogue to a certain number of turns
dialogue_splitted = dialogue[:TURNS]

def run_model(generation_method):
    """Runs model with specified generation method.

    Args:
        generation_method (str): Generation method. One of "greedy", "beam", "topk"

    Returns:
        tuple: (generated reponses, groud truth reponses)
    """
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
        if generation_method == "topk":
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
        elif generation_method == "beam":
            history_and_response_ids = model.generate(
                dialogue_hist_ids,
                max_length = 1000,
                pad_token_id = tokenizer.eos_token_id,
                no_repeat_ngram_size = 3,
                num_beams = 10
            )
        elif generation_method == "greedy":
            history_and_response_ids = model.generate(
                dialogue_hist_ids,
                max_length = 1000,
                pad_token_id = tokenizer.eos_token_id,
                no_repeat_ngram_size = 3
            )
        else:
            raise ValueError("Generation method not recognized.")

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

    return generated_responses_string, ground_truth_responses_string




if METHODS == "all" or METHODS == "topk": 
    TOPK_generated_responses_string, TOPK_ground_truth_responses_string = run_model("topk")
    """Calculate BLEU + entropy."""
    print()
    print("TOPK")
    print()
    if METRICS == "all" or METRICS == "bleu-2": print(f"BLEU-2: {BLEU(ref = TOPK_ground_truth_responses_string, hyp = TOPK_generated_responses_string, order=2)}")
    if METRICS == "all" or METRICS == "bleu-4": print(f"BLEU-4: {BLEU(ref = TOPK_ground_truth_responses_string, hyp = TOPK_generated_responses_string, order=4)}")
    if METRICS == "all" or METRICS == "entropy": print(f"Entropy: {entropy(TOPK_generated_responses_string)}")
    if METRICS == "all" or METRICS == "ideal entropy": print(f"Ideal entropy: {entropy_ideal(len(TOPK_generated_responses_string))}")
    if METRICS == "all" or METRICS == "cosine similarity": print(f"Cosine similarity: {cosine_similarity(ref = TOPK_ground_truth_responses_string, hyp = TOPK_generated_responses_string)}")
    if METRICS == "all" or METRICS == "jaccard similarity": print(f"Jaccard similarity: {jaccard_similarity(ref = TOPK_ground_truth_responses_string, hyp = TOPK_generated_responses_string)}")
    print()
    print()
    
if METHODS == "all" or METHODS == "beam": 
    BEAM_generated_responses_string, BEAM_ground_truth_responses_string = run_model("beam")
    print()
    print("BEAM")
    print()
    if METRICS == "all" or METRICS == "bleu-2": print(f"BLEU-2: {BLEU(ref = BEAM_ground_truth_responses_string, hyp = BEAM_generated_responses_string, order=2)}")
    if METRICS == "all" or METRICS == "bleu-4": print(f"BLEU-4: {BLEU(ref = BEAM_ground_truth_responses_string, hyp = BEAM_generated_responses_string, order=4)}")
    if METRICS == "all" or METRICS == "entropy": print(f"Entropy: {entropy(BEAM_generated_responses_string)}")
    if METRICS == "all" or METRICS == "ideal entropy": print(f"Ideal entropy: {entropy_ideal(len(BEAM_generated_responses_string))}")
    if METRICS == "all" or METRICS == "cosine similarity": print(f"Cosine similarity: {cosine_similarity(ref = BEAM_ground_truth_responses_string, hyp = BEAM_generated_responses_string)}")
    if METRICS == "all" or METRICS == "jaccard similarity": print(f"Jaccard similarity: {jaccard_similarity(ref = BEAM_ground_truth_responses_string, hyp = BEAM_generated_responses_string)}")
    print()
    print()

if METHODS == "all" or METHODS == "greedy": 
    GREEDY_generated_responses_string, GREEDY_ground_truth_responses_string = run_model("greedy")
    print()
    print("GREEDY")
    print()
    if METRICS == "all" or METRICS == "bleu-2": print(f"BLEU-2: {BLEU(ref = GREEDY_ground_truth_responses_string, hyp = GREEDY_generated_responses_string, order=2)}")
    if METRICS == "all" or METRICS == "bleu-4": print(f"BLEU-4: {BLEU(ref = GREEDY_ground_truth_responses_string, hyp = GREEDY_generated_responses_string, order=4)}")
    if METRICS == "all" or METRICS == "entropy": print(f"Entropy: {entropy(GREEDY_generated_responses_string)}")
    if METRICS == "all" or METRICS == "ideal entropy": print(f"Ideal entropy: {entropy_ideal(len(GREEDY_generated_responses_string))}")
    if METRICS == "all" or METRICS == "cosine similarity": print(f"Cosine similarity: {cosine_similarity(ref = GREEDY_ground_truth_responses_string, hyp = GREEDY_generated_responses_string)}")
    if METRICS == "all" or METRICS == "jaccard similarity": print(f"Jaccard similarity: {jaccard_similarity(ref = GREEDY_ground_truth_responses_string, hyp = GREEDY_generated_responses_string)}")
    print()
    print()
