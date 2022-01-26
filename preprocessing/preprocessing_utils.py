""" Preprocessing file to transforms raw data into exploitable one."""




import numpy as np
import pandas as pd




def delete_incomplete_lines(df):
    """ Deletes the incomplete lines from the dataframe.

    ex:   "raw_character_text" | "spoken_words"
          "Lisa Simpson"       | ""
    or
          "raw_character_text" | "spoken_words"
          ""                   | "Hey, Lisa"

    Args:
        df (pandas.DataFrame): Dataframe concerned.

    Returns:
        pandas.DataFrame: Dataframe without incomplete lines.

    """
    df = df.drop(df[df["raw_character_text"].notna() ^ df["spoken_words"].notna()].index)
    df = df.reset_index(drop=True)

    return df





def dataframe_to_lists(df):
    """ Save the dataframe in two lists : one for the characters and one for words.

    ex: raw_character_text_list[1] = "Lisa Simpson"
        spoken_words_list[1] = "Where is Mr. Bergstrom?"

    Args:
        df (pandas.DataFrame): Dataframe concerned.

    Returns:
        tuple containing
        raw_character_text_list (list): contains character names.
        spoken_words_list (list): contains character words.

    """
    raw_character_text_list = []
    spoken_words_list = []
    for row in df.itertuples():
        try:
            np.isnan(row.raw_character_text) and np.isnan(row.spoken_words)
        except:
            raw_character_text_list.append(row.raw_character_text)
            spoken_words_list.append(row.spoken_words)
        else:
            raw_character_text_list.append("")
            spoken_words_list.append("")

    return raw_character_text_list, spoken_words_list




def text_concatenation_consecutive_char(char_list,words_list,i):
    """ Concatenates text of consecutive char from index i.

    Args:
        char_list (list): contains character names.
        words_list (list): contains character words.
        i (int): index where concatenation starts.

    Returns:
        tuple containing
        result (string): text concatenation.
        nb_lines (int): number of consecutive responses said by same character.

    """
    result = words_list[i]
    nb_lines = 1
    while char_list[i] == char_list[i+1]:
        result += (" "+words_list[i+1])
        i += 1
        nb_lines += 1
        if i == len(char_list)-1:
            break
    return result,nb_lines




def concatene_words_consecutive_character_by_scene(raw_character_text_list,spoken_words_list):
    """ Concatenates words said by each consecutive character in one scene.

    ex: raw_character_text_list[36] = "Bart Simpson" and spoken_words_list[36] = "Hey!"
        raw_character_text_list[37] = "Bart Simpson" and spoken_words_list[37] = "What do you eat?"

        will become:

        concatene_raw_character_list[36] = "Bart Simpson" and spoken_words_list[36] = "Hey! What do you eat?"

    Args:
        raw_character_text_list (list): contains character names.
        spoken_words_list (list): contains character words.

    Returns:
        tuple containing
        concatene_raw_character_list (list): contains concatenated character names.
        concatene_spoken_words_list (list): contains concatenated character words.

    """
    concatene_raw_character_list = []
    concatene_spoken_words_list = []
    i = 0
    while (i < len(raw_character_text_list)-1):
        if raw_character_text_list[i] == raw_character_text_list[i+1]:
            concatene_raw_character_list.append(raw_character_text_list[i])
            text_concatene, nb_lines = text_concatenation_consecutive_char(raw_character_text_list,spoken_words_list,i)
            concatene_spoken_words_list.append(text_concatene)
            i += nb_lines
        else:
            concatene_raw_character_list.append(raw_character_text_list[i])
            concatene_spoken_words_list.append(spoken_words_list[i])
            i += 1

    return concatene_raw_character_list, concatene_spoken_words_list




def create_contexts_for_chosen_character(raw_character_text_list,spoken_words_list,nb_context,chosen_character):
    """ Creates list of contexts for each reponses of the chosen character.

    ex: raw_character_text_list[97] = "" and spoken_words_list[97] = ""
        raw_character_text_list[98] = "Bart Simpson" and spoken_words_list[98] = "Haha!"
        raw_character_text_list[99] = "Lisa Simpson" and spoken_words_list[98] = "Where is my saxophone?"
        raw_character_text_list[100] = "Homer Simpson" and spoken_words_list[100] = "Bart! Where is it?"

        if "Homer Simpson" is the chosen character and we are working with 7 contexts we will get:

        context7 = "" | context6 = "" | context5 = "" | context4 = "" | context3 = "" | context2 = "Haha!" | context1 = "Where is my saxophone?" | response = "Bart! Where is it?"

    Args:
        raw_character_text_list (list): contains character names.
        spoken_words_list (list): contains character words.
        nb_context (int): number of contexts before each reponses.
        chosen_character (string): chosen character to work on.

    Returns:
        list: contains contexts for each responses of the chosen character.

    ex (for one response):
        contexts_list = [[c7,c6,c5,c4,c3,c2,c1,response], []]
        contexts_list = [["","","","","","Haha!","Where is my saxophone?","Bart! Where is it?"]]

    """
    contexts_list = []
    for i in range(len(raw_character_text_list)-1,-1,-1):
        context_one_line_list = []
        if raw_character_text_list[i] == chosen_character:
            context_one_line_list.append(spoken_words_list[i])
            previous_index = i-1
            while ((raw_character_text_list[previous_index] != "") and (len(context_one_line_list) < nb_context+1)):
                context_one_line_list.append(spoken_words_list[previous_index])
                previous_index -= 1
            context_one_line_list.extend([""]*(nb_context+1-len(context_one_line_list)))
            context_one_line_list.reverse()

            contexts_list.append(context_one_line_list)

    contexts_list.reverse()

    return contexts_list




def create_columns(nb_context):
    """ Creates column names for the output csv file.

    Args:
        nb_context (int): number of contexts before each reponses.

    Returns:
        list: contains column names for the output csv file.

    ex (with nb_context = 7):
        columns_list = ["ctxt7","ctxt6","ctxt5","ctxt4","ctxt3","ctxt2","ctxt1","response"]

    """
    columns_list = []
    for i in range(nb_context,0,-1):
        columns_list.append("ctxt"+str(i))
    columns_list.append("response")

    return columns_list
