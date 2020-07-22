# Readability Analysis Toolkit

_A set of python based readability analysis tools, developed a part of my final year undergraduate project_

My final year undergraduate project focussed around the readability of privacy policies used by social media companies. I created a set of tools to calculate various readability measures and any additional functions needed to calculate these. The functions were developed with web-based privacy policies in mind but should be adaptable or provide a decent guideline of readability for most texts.

## Readability Measures/Tools Included

- Syllable counter
- Flesch Reading Ease Score (FRES)
- Automated Readability Index (ARI)
- Simple Measure of Gobbledegook (SMOG)
- Gunning Fog Index (GFI)
- Age conversions (FRES, ARI and GFI)

## Syllable measuring accuracy

At the time of developing these functions there didn't appear to be any particularly established syllable counting measures or syllable counting tests. To gauge the effectiveness of the solutions, I developed some tests comparing the results produced by the functions I developed and lists of known syllables.
The most exhaustive test compared the function output against a set of words formed from combining the list of [Google's Top 10,000 English words](https://github.com/first20hours/google-10000-english/blob/master/google-10000-english.txt) and the [Moby Hyphenator II Hyphenation List](http://onlinebooks.library.upenn.edu/webbin/gutbook/lookup?num=3204) - this resulted in a 96.55% accuracy.

## Dependencies

These readability tools use Natural Language Toolkit (NLTK) for some base tokenisation and SciPy for 1-D interpolation.
The functions also make use of the regular expressions, math and statistics packages.
