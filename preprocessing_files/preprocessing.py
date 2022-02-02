# librairies
import numpy as np
import pandas as pd
from argparse import ArgumentParser

# utils file
import preprocessing_utils as pu


# argument type to make sure that --nb_contexts > 0
def restriction_nb_contexts(x):
    x = int(x)
    if x < 0:
        raise argparse.ArgumentTypeError("nb_contexts must be greater than 0")
    return x

# argument type to transform --concatenation in boolean
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected for --concatenation.')

if __name__ == '__main__':
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument("--chosen_character", type=str, required=True, choices=['homer','marge','bart','lisa'], dest="chosen_character", help="The name of the character chosen.")
    parser.add_argument("--nb_contexts", type=restriction_nb_contexts, required=True, dest="nb_contexts", help="Number of contexts.")
    parser.add_argument("--concatenation", type=str2bool, required=True, dest="concatenation", help="Set to True if you want to concatenate words said consecutively by the chosen character.")

    # argument key to real name value
    dict_character = {"homer": "Homer Simpson", "bart": "Bart Simpson", "lisa": "Lisa Simpson", "marge": "Marge Simpson"}

    # save arguments
    args = parser.parse_args()

    # constants
    CHOSEN_CHARACTER = dict_character[args.chosen_character]
    NB_CONTEXTS = args.nb_contexts
    CONCATENATION = args.concatenation




# save the original dataset into pandas dataframe
df = pd.read_csv("simpsons_dataset.csv")

# delete the incomplete lines from the dataframe
df = pu.delete_incomplete_lines(df)

# save the dataframe in two lists : one for the character name and the other for his words
char_list, words_list = pu.dataframe_to_lists(df)

# if CONCATENATION, then concatene words said by same character in one scene
if CONCATENATION:
    char_list, words_list = pu.concatene_words_consecutive_character_by_scene(char_list,words_list)

# create number of contexts wanted from each response of the chosen character
contexts_list = pu.create_contexts_for_chosen_character(char_list,words_list,NB_CONTEXTS,CHOSEN_CHARACTER)

contexts_list = pu.remove_empty_context(contexts_list)

# save contexts in a dictionnary
dic = {}
for i, line in enumerate(contexts_list):
    dic[i] = line

# create columns name for the output csv file
columns_list = pu.create_columns(NB_CONTEXTS)

# create final dataframe from contexts dictionnary
df_final = pd.DataFrame.from_dict(dic, orient="index", columns=columns_list)

# export the final dataframe to csv format
if CONCATENATION:
    df_final.to_csv("output_preprocessing_"+args.chosen_character+"_"+str(NB_CONTEXTS)+"_concat.csv",index=False)
else:
    df_final.to_csv("output_preprocessing_"+args.chosen_character+"_"+str(NB_CONTEXTS)+"_no-concat.csv",index=False)
