import re
import nltk
import math
import numpy as np

from nltk.corpus import cmudict
from scipy.interpolate import interp1d
from statistics import mean, mode, StatisticsError

d = cmudict.dict()

extra_abbreviations = ['inc', 'i.e', 'e.g']
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentence_tokenizer._params.abbrev_types.update(extra_abbreviations)



def nsyl(word):
    """
    Function to calculate the number of syllables in a word.

    Uses the CMU Pronunciation dictionary and lexical stress
    to determine a syllable count.

    If the word is not present in the CMU dictionary, the
    function falls back to a sub function to estimate the
    number of syllables in a word.

    :param word: string of a single word (can be hyphenated)
    :return: integer value of the number of syllables
    """

    if word.isalpha():
        try:
            syl_list = []
            phoneme_list = d[word.lower()]
            for phoneme_set in phoneme_list:
                syl_count = []
                for phoneme in phoneme_set:
                    if phoneme[-1].isdigit():
                        syl_count.append(phoneme)
                syl_list.append(len(syl_count))
            if word == 'US':
                return 2
            elif word == 'us':
                return 1
            elif word == 'separate':
                return 2
            elif len(syl_list) > 1:
                # val = syl_list[0]
                # return val
                try:
                    val = mode(syl_list)
                    return val
                except StatisticsError:
                    val = int(round(mean(syl_list), 0))
                    return val
            else:
                return syl_list[0]
        except KeyError:
            # if word not found in cmudict
            if word.lower() == "facebook":
                return 2
            elif word.lower() == "thefacebook":
                return 3
            elif word.lower() == "tumblr":
                return 2
            else:
                return syllables(word)
    elif '-' in word:
        # if the word contains anything other than a-Z or a '-'
        if re.search("[^-a-zA-Z]", word):
            return 1
        # if the word is a-Z followed by a '-' followed by a-Z
        elif re.search("[a-zA-Z]+\-[a-zA-Z]+", word):
            split_word = word.split('-')
            syl_count = 0
            for word in split_word:
                word_syllables = nsyl(word)
                syl_count = syl_count + word_syllables
            return syl_count
        # if the word is a-Z followed by a '-' and nothing else
        elif re.search("[a-zA-Z]+\-\B", word):
            word = word.strip('-')
            return nsyl(word)
        else:
            return 1
    else:
        return 1


def syllables(word):
    """
    Fallback method for estimating the number of syllables in
    a word.

    An improvement built upon the solution suggested at:
    www.stackoverflow.com/questions/14541303/count-the-number-of-syllables-in-a-word

    :param word: string of a single word (can be hyphenated)
    :return: integer value of the number of syllables
    """
    count = 0
    vowels = 'aeiouy'  # Taking advantage of Python treating strings as lists of characters
    word = word.lower()
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le'):
        count += 1
    if word.endswith('sm'):
        count += 1
    if word.endswith('thm'):
        count += 1
    if count == 0:
        count += 1
    return count


def get_words_list(input_text):
    """
    Function to return a list of words from a text.

    :param input_text: string of text to be tokenised
    :return: array of tokenised words
    """
    # REGEX is used to clear up instances of URLs and hyphenated words.
    # URLs are counted as a single word to prevent disproportionately
    # affecting the readability measures - the idea being that URLS are
    # not read by users as regular text is.
    words_list = [word for word in (nltk.word_tokenize(input_text)) if
                  (word.isalnum() or re.search("'[a-zA-z]+", word)
                   or re.search("[a-zA-Z]+\.[a-zA-Z]+\.[a-zA-Z]+(\/[a-zA-Z0-9]*)", word)
                   or re.search("\d+\.\d+", word)
                   or (("-" in word) and re.search("[a-zA-Z0-9]+", word)))]

    return words_list


def get_sentence_list(input_text):
    """
    Function to return a list of sentences from a text.

    :param input_text: string of text to be tokenised
    :return: array of tokenised sentences
    """
    # REGEX ensures a sentence contains an alphabetic character.
    # This removes punctuation only sentences and instances where
    # "1. Introduction" would count as two sentences.
    sentence_list = [sentence for sentence in (nltk.sent_tokenize(input_text)) if
                     (re.search("[a-zA-Z]", sentence))]

    return sentence_list


def fres(input_text):
    """
    Function to calculate the Flesch Reading Ease Score (FRES)
    for a given text.

    :param input_text: string of text to calculate the FRES of
    :return: float of the FRES value
    """
    words_list = get_words_list(input_text)

    total_words = len(words_list)
    sentence_list = get_sentence_list(input_text)
    total_sentences = len(sentence_list)

    total_syllables = 0
    for word in words_list:
        total_syllables = total_syllables + nsyl(word)

    avg_sentence_len = total_words / total_sentences
    avg_syllables = total_syllables / total_words

    return 206.835 - (1.015 * avg_sentence_len) - (84.6 * avg_syllables)


def ari(input_text):
    """
    Function to calculate the Automated Readability Index (ARI)
    of a text.

    :param input_text: string of text to calculate the ARI of
    :return: float of the ARI value
    """
    words_list = get_words_list(input_text)

    total_words = len(words_list)
    sentence_list = get_sentence_list(input_text)
    total_sentences = len(sentence_list)

    total_chars = 0
    for word in words_list:
        total_chars = total_chars + len(word)

    avg_sentence_len = total_words / total_sentences
    avg_chars = total_chars / total_words

    return (4.71 * avg_chars) + (0.5 * avg_sentence_len) - 21.43


def smog(input_text):
    """
    Function to calculate the Simple Measure of Goobldegook (SMOG)
    value of a text.

    :param input_text: string of text to calculate the SMOG value of
    :return: integer of the SMOG value
    """
    # wordsList = [word for word in (nltk.word_tokenize(inputText)) if word.isalpha() or ("-" in word)]
    sentence_list = get_sentence_list(input_text)
    total_sentences = len(sentence_list)

    start = sentence_list[0:9]
    middle = sentence_list[math.floor(((total_sentences / 2) - 5)):round(((total_sentences / 2) - 5))]
    end = sentence_list[(len(sentence_list) - 10):len(sentence_list)]
    selection = " ".join(start) + " ".join(middle) + " ".join(end)
    words_list = get_words_list(selection)

    difficult_words = 0
    for word in words_list:
        if nsyl(word) >= 3:
            difficult_words += 1

    nearest_square = round(math.sqrt(difficult_words))

    return nearest_square + 3


def gfi(input_text):
    """
    Function to calculate the Gunning Fog Index (GFI)
    of a text.

    :param input_text: string of text to calculate the GFI of
    :return: float of the GFI value
    """
    words_list = get_words_list(input_text)
    total_words = len(words_list)
    sentence_list = get_sentence_list(input_text)
    total_sentences = len(sentence_list)

    tagged_words_list = nltk.pos_tag(words_list)

    complex_words = 0

    for item in tagged_words_list:
        current_word = item[0]
        current_tag = item[1]
        word_syllables = nsyl(current_word)
        if nsyl(current_word) >= 3 and ("NNP" not in current_tag) and "-" not in current_word and (
                nsyl(current_word) == 3 and "VB" in current_tag and (
                current_word.endswith('es') or current_word.endswith('ed') or current_word.endswith('ing'))
        ) is False:
            complex_words += 1

    avg_sentence_len = total_words / total_sentences
    percentage_complex_words = complex_words / total_words

    return (avg_sentence_len + percentage_complex_words) * 0.4


def get_complex_words_count(input_text):
    """
    Function to return the number of 'complex' words, where
    a complex word is any word with 3 or more syllables.

    :param input_text: string of text to be analysed
    :return: integer of the number of complex words
    """
    words_list = get_words_list(input_text)
    count = 0

    for word in words_list:
        if nsyl(word) >= 3:
            count += 1

    return count


def get_word_count(input_text):
    """
    Function to return the number of words in a text.

    :param input_text: string of text to be analysed
    :return: integer of the number of words
    """
    words_list = get_words_list(input_text)

    return len(words_list)


def get_sentence_count(input_text):
    """
    Function to return the number of sentences in a text.

    :param input_text: string of text to be analysed
    :return: integer of the number of words
    """
    sentences = get_sentence_list(input_text)

    return len(sentences)


def convert_fres(fres_value):
    """
    Function to calculate the converted age from a
    Flesch Reading Ease Score (FRES) value.

    :param fres_value: FRES value to convert
    :return: float of converted age value
    """
    value_out = ""

    x = [10, 11, 12, 13, 15, 18, 24]
    y = [100, 90, 80, 70, 60, 50, 30]
    ipol = interp1d(y, x)

    if fres_value < 30:
        value_out = "24"
    elif fres_value > 100:
        value_out = "10"
    else:
        value_out = ipol(fres_value)

    return value_out


def convert_ari(ari_value):
    """
    Function to calculate the converted age from a
    Automated Readability Index (ARI) value.

    :param ari_value: ARI value to convert
    :return: float of converted age value
    """
    value_out = ""

    x = [5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24]
    y = list(range(1, 15))
    ipol = interp1d(y, x)

    if ari_value > 14:
        value_out = "24"
    elif ari_value < 1:
        value_out = "5"
    else:
        value_out = ipol(ari_value)

    return value_out


def convert_gfi(gfi_value):
    """
    Function to calculate the converted age from a
    Gunning Fog Index (GFI) value.

    :param gfi_value: GFI value to convert
    :return: float of converted age value
    """
    value_out = ""

    x = [11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 23, 24]
    y = list(range(6, 18))
    ipol = interp1d(y, x)

    if gfi_value > 17:
        value_out = "24"
    elif gfi_value < 6:
        value_out = "11"
    else:
        value_out = ipol(gfi_value)

    return value_out
