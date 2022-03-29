""" Preprocessing file to visualize data."""




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def show_number_responses_per_char(char_list):
    """ Show the number of responses from the Simpsons family.

    Args:
        char_list (list): contains character names.

    """
    simpsons_family_char_list = list(filter(lambda a: ((a == "Homer Simpson") or (a == "Marge Simpson") or (a == "Lisa Simpson") or (a == "Bart Simpson")) , char_list))
    unique_family_char, count_responses = np.unique(simpsons_family_char_list, return_counts=True)
    print(unique_family_char)
    print(count_responses)
    print(sum(count_responses))
    plt.rcParams.update({'font.weight': "bold"})
    plt.rcParams["axes.labelweight"] = "bold"
    plt.hist(simpsons_family_char_list, rwidth=2, color="#f7cc37", edgecolor='black', linewidth=1.2)
    plt.ylabel("Nombre de répliques")
    plt.show()




def show_homer_responses_length(char_list,words_list):
    """ Show the length of Homer's responses.

    Args:
        char_list (list): contains character names.
        words_list (list): contains character words.

    """
    homer_words = []
    for i, char in enumerate(char_list):
        if char == "Homer Simpson":
            homer_words.append(words_list[i])
    homer_words_length = [len(x) for x in homer_words]
    homer_words_length.sort()
    plt.plot(homer_words_length,color="#f7cc37",linewidth=4)
    plt.xlabel("Nombre de répliques")
    plt.ylabel("Nombre de caractères par répliques")
    plt.show()
