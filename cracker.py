import json

import pandas as pd
import numpy as np
from collections import Counter
import re
from text_stats import TextStats

from itertools import combinations

letters = 'abcdefghijklmnopqrstuvwxyz'

class Cracker:
    def __init__(self, enc_file, words_cnt_file='Data/words_cnt.csv', letters_file='Data/letters.csv'):
        """
        Read the encrypted text and calculate stats on it:
        :param enc_file: the text file with the encrypted text
        """
        self.enc_text = open(enc_file, 'r').read()
        self.text_stats = TextStats()
        self.words_df = pd.read_csv(words_cnt_file, index_col=0)
        self.letter_stats = pd.read_csv(letters_file, index_col=0)
        self._get_words()

        self.known_enc_dict = {}  # The letters we already found

    def _get_words(self):
        """
        Split the enc text to words and store in self.enc_words:
        :return:
        """
        # Remove punctuation from self.enc_text:
        self.clean_enc_text = re.sub(r'[^\w\s]', '', self.enc_text)
        self.enc_words_list = self.clean_enc_text.split()

        # Now count the words and letters:
        self.enc_stats_df, self.enc_words_df = self.text_stats.calc_text_stats(self.enc_words_list)
        # Add a dict for the enc_words by count:
        self.enc_words = self.enc_words_df.cnt.to_dict()

    def _calc_similarity(self):
        """
        Calculate teh similarity between the encrypted and plain letters. Similarity is the sum of squares of diffs
        of the stats on all letters, first letter and last letter percentages.
        :return:
        """
        similarity = pd.DataFrame(np.zeros((len(letters), len(letters))), index=list(letters), columns=list(letters))
        for plain_l in letters:
            for enc_l in letters:
                similarity.loc[plain_l, enc_l] = sum((self.enc_stats_df.loc[enc_l] - self.letter_stats.loc[plain_l]) ** 2)
        self.similarity = similarity

    def get_top_candidates(self, l, n_top=3, n_mult=3, letters_left=None):
        """
        Get the top candidates for a letter by their stats
        :param l: The letter to look for
        :param n_top: number of top options to consider
        :param n_mult: The score multiplier - we only consider options where the letter similarity in n_mult * highest similarity
            This limits the number of options we consider, as this removes unlikely results.
        :param letters_left: If not None, this is the list of letters we can still assign
        :return: The top letter options, sorted
        """
        if letters_left is None:
            sorted_ord = self.similarity.loc[l, :].sort_values()
        else:
            sorted_ord = self.similarity.loc[l, letters_left].sort_values()

        # only include results up to 3 times less likely than the top result, but no less than 3 results:
        res = sorted_ord[sorted_ord < sorted_ord.iloc[0] * n_mult].head(n_top).index.values
        if len(res) < 3:
            res = sorted_ord.head(3).index.values
        return res

    def score_option(self, enc_dict):
        """
        enc_dict is the encryption dict. Encrypt the top words, check how many appear in the original encrypted text
        The score is the number of words found with the encryption, times the likelihood of those words
        :param enc_dict: A dict {plain_letter: enc_letter}
        :return: The score
        """
        enc_trans = str.maketrans(enc_dict)
        top_words_enc = {w.translate(enc_trans) for w in self.top_words}
        n_found = sum([self.enc_words.get(t, 0) for t in top_words_enc])

        # Check that the words we found are according to the stats - score on stats diff:
        found_top_words = sum([self.top_word_stats[w] for w in self.top_words if w.translate(enc_trans) in self.enc_words])

        # Check the percentage of letters with these letters that we have do appear in the top words:
        enc_l_set = set(enc_dict.values())
        enc_words_with_letters = {w for w in self.enc_words if all(l in enc_l_set for l in w)}
        if len(enc_words_with_letters) == 0:
            # No words found. This is not the right solution...
            return 0
        found_enc_words = {w for w in top_words_enc if w in enc_words_with_letters}
        found_enc_ratio = sum(self.enc_words[l] for l in found_enc_words) / sum(self.enc_words[l] for l in enc_words_with_letters)

        return n_found * found_top_words * found_enc_ratio

    def get_top_words(self, top_set):
        """
        Get the top words in English that have only the letters in top_set
        :param top_set: A set of letters we can use
        :return:
        """
        # Get the top 5000 words, then keep only the words made up entirely of the top letters:
        top_words = self.words_df.head(5000).index.astype(str)
        top_words = top_words[top_words.map(lambda x: all(l in top_set for l in x))].values
        self.top_words = top_words
        top_words_df = self.words_df.loc[self.words_df.index.isin(top_words), :]
        print(f"We have {top_words_df.pct.sum()} of the words using our top letters")
        # Stats on the % of words for each of the top words:
        self.top_word_stats = top_words_df.pct.to_dict()

    def crack_letters(self, top_letters, n_top=3, n_mult=3, known_enc_dict=None):
        """
        Try all relevant permutations for top_letters. For each option score the result and select the best scoring one
        :param top_letters: The letters to try and crack. Typically you would try 5-10 letters that are next most frequent
        :param n_top: Number of top options to try for each letter
        :param n_mult: The multiplier of the letter score - we only consider letters with similarity n_mult times more
            than the most likely option. This limits the amount we need to brute force.
        :param known_enc_dict: A dict of letters we already cracked {plain_letter: enc_letter}
        :return: A dict of letters we now cracked {plain_letter: enc_letter}
        """
        if known_enc_dict:
            letters_left = [l for l in letters if l not in known_enc_dict.values()]
            enc_dict = known_enc_dict.copy()
        else:
            letters_left = None
            enc_dict = {}

        options = []
        for l in top_letters:
            candidates = self.get_top_candidates(l, n_top=n_top, n_mult=n_mult, letters_left=letters_left)
            options.append(candidates)
            print(f"We have {len(candidates)} for letter {l}: {list(candidates)}")

        # opt_cnt is the index of letters we are now trying for each of the letters we are permuting over.
        # It is like a counter, only it counts to the number of options for each of the letters.
        opt_cnt = [0 for k in range(len(top_letters))]
        cnt = 0
        best = None
        best_score = -1
        done = False

        while not (done):
            # Get the letters for this option:
            opt_enc_l = [options[k][opt_cnt[k]] for k in range(len(top_letters))]
            # If we have the same letter replacing more than one source letter this can't be. So check we have unique letter option:
            if len(set(opt_enc_l)) == len(top_letters):
                # Score the option:
                if known_enc_dict is None:
                    # Just score the letters we have:
                    score = self.score_option(dict(zip(top_letters, opt_enc_l)))
                else:
                    # Add the letters we are trying to the ones we know and then score:
                    enc_dict.update(dict(zip(top_letters, opt_enc_l)))
                    score = self.score_option(enc_dict)

                if score > best_score:
                    print(f"Found new best with score {score}!:")
                    print(opt_enc_l)
                    best = opt_enc_l
                    best_score = score

            # Add 1 to the letter order choice to try the next permutation
            for k in range(len(top_letters)):
                opt_cnt[k] += 1
                if opt_cnt[k] < len(options[k]):
                    # No overflow. Exit.
                    break
                elif k < len(top_letters) - 1:
                    # Overflow. Put 0:
                    opt_cnt[k] = 0
                else:
                    # Got to the last value in the highest "digit". We can finish the loop.
                    done = True
            cnt += 1
        print(f"We tried {cnt} options and found best result")
        return dict(zip(top_letters, best))

    def try_switch(self, enc_dict):
        """
        Score the existing enc_dict, then try to switch a pair of letters to improve the score.
        We try to switch both a pair of letters in the letters we found, and one letter we found with an unused letter
        :param enc_dict: A dict of letters we already cracked {plain_letter: enc_letter}
        :return: A dict of letters we already cracked after the revision (if any made) {plain_letter: enc_letter}
        """
        # First try inside the letters in enc_dict:
        score = self.score_option(enc_dict)
        enc_dict_letters = list(enc_dict.keys())
        replace = None

        # Get all pairs of letters in enc_dict_letters:
        for pair in combinations(enc_dict_letters, 2):
            # Try to switch the letters:
            orig_vals = (enc_dict[pair[0]], enc_dict[pair[1]])
            enc_dict[pair[0]] = orig_vals[1]
            enc_dict[pair[1]] = orig_vals[0]
            new_score = self.score_option(enc_dict)
            if new_score > score:
                print(f"Found a better score with switch {pair}!")
                score = new_score
                replace = ('in', pair[0], pair[1])
            # Put the letter back:
            enc_dict[pair[0]] = orig_vals[0]
            enc_dict[pair[1]] = orig_vals[1]

        # Now try to replace the enc_letters (the values in the dict) with the letters we don't know:
        unused_letters = [l for l in letters if l not in enc_dict.values()]
        for enc_l in enc_dict.keys():
            for l in unused_letters:
                orig_l = enc_dict[enc_l]
                enc_dict[enc_l] = l
                new_score = self.score_option(enc_dict)
                if new_score > score:
                    print(f"Found a better score with switch {l}!")
                    score = new_score
                    replace = ('out', enc_l, l)
                # Put the letter back:
                enc_dict[enc_l] = orig_l

        if replace:
            print(f"We replaced {replace[1]} with {replace[2]} in mode {replace[0]}")
            if replace[0]=='in':
                orig_vals = (enc_dict[replace[1]], enc_dict[replace[2]])
                enc_dict[replace[1]] = orig_vals[1]
                enc_dict[replace[2]] = orig_vals[0]
            elif replace[0]=='out':
                enc_dict[replace[1]] = replace[2]
        return enc_dict

    def crack_code(self, group_sizes=5, n_top=5, n_mult=3):
        """
        Crack the code iteratively. In each round we crack the next group_size letters. We also increase the
        number of top options we allow, as the space gets smaller.
        :param group_sizes: The number of letters to crack each time. If it is a list then we use the sizes in the list items
            as the size of each group.
        :param n_top: The top number of letters to try each time. Also can be a list, if group_sizes is a list
        :param n_mult: The multiplier to use. Also can be a list, if group_sizes is a list.
        :return: The result
        """
        enc_dict = {}
        if not isinstance(group_sizes, list):
            group_sizes = [group_sizes for k in range(len(letters)//group_sizes + 1)]

        if not isinstance(n_top, list):
            n_top = [int(n_top * (1+k/2)) for k in range(len(group_sizes))]

        if not isinstance(n_mult, list):
            n_mult = [n_mult * (1+k/2) for k in range(len(group_sizes))]

        # Now run the cracking of the letters by their order:
        self._calc_similarity()
        for k in range(len(group_sizes)):
            group_size = group_sizes[k]
            letters_found = len(enc_dict)
            next_letters = self.letter_stats.iloc[letters_found:(letters_found+group_size)].index.values
            print(f'{"-"*20}\nWe are now cracking {next_letters}')

            # The set of letters we either know already or are trying to find:
            top_set = set(next_letters) | enc_dict.keys()
            self.get_top_words(top_set)
            round_res = self.crack_letters(next_letters, n_top[k], n_mult[k], enc_dict)
            # If None, we couldn't find a good enough result. Try increasing n_top and n_mult:
            if round_res is None:
                print("No result, retrying...")
                round_res = self.crack_letters(next_letters, n_top[k]*3, n_mult[k]*10, enc_dict)

            enc_dict.update(round_res)

            # Try to switch one pair of letters:
            enc_dict = self.try_switch(enc_dict)

        return enc_dict

    def decipher(self, enc_dict):
        """
        Decipher the code using the enc_dict
        :param enc_dict: A dict of letters we cracked {plain_letter: enc_letter}
        :return: The decipher
        """
        dec_trans = str.maketrans({v:k for k,v in enc_dict.items()})
        return self.enc_text.translate(dec_trans)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--enc_file', default="Data/cipher.txt", help='Encrypted text file')
    parser.add_argument('-d', '--dec_file', default="Data/out.txt", help='Decrypted text output file')
    parser.add_argument('-k', '--key_file', default=None, help='Key file to write the result decrypt key to')
    args = parser.parse_args()

    cr = Cracker(args.enc_file)
    group_sizes = [7, 7, 7, 5]
    n_top = [5, 6, 6, 6]
    n_mult = [3, 6, 10, 1000]
    enc_res = cr.crack_code(group_sizes, n_top, n_mult)

    # Print the decipher:
    dec_text = cr.decipher(enc_res)
    print(dec_text)

    if args.key_file:
        with open(args.key_file, 'w') as f:
            f.write(json.dumps(enc_res))
        print(f"Wrote key to {args.key_file}")

    if args.dec_file:
        with open(args.dec_file, 'w') as f:
            f.write(dec_text)
        print(f"Wrote decipher to {args.dec_file}")


