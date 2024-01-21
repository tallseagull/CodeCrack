# Generate statistics about letters and words from a corpus of articles
import re
from collections import defaultdict
import pandas as pd
from gzip import GzipFile

letters = 'abcdefghijklmnopqrstuvwxyz'


class TextStats:
    def __init__(self):
        pass

    def read_text(self, text_file):
        # Read the text file. Get rid of lines that start with 'URL' since they are the location of the article
        # Also get rid of any non-letters (not a-z or A-Z):
        with GzipFile(text_file, 'r') as fp:
            txt = fp.read().decode('utf8')

        # Filter out the lines with no text, or with the URL pointing to the text:
        txt_lines = list(filter(lambda l: (len(l) > 0) & (~l.startswith('URL')), txt.split('\n')))

        # Remove non-letter data:
        non_letters = re.compile(r'[^a-zA-Z ]')
        txt_lines = [non_letters.sub('', line.lower()) for line in txt_lines]
        print(f"We have {len(txt_lines)} lines of text, with {sum(map(len, txt_lines))} chars in them in total")
        self.txt_lines = txt_lines

    def count_letters(self, letter_stats_file=None, word_cnt_file=None):
        # Count the letters in the text: Overall letter count, first letter in word, last letter in word
        # Inititalize the counters - all letters start with count=0:
        letters_cnt = {l: 0 for l in letters}
        first_letters_cnt = {l: 0 for l in letters}
        last_letters_cnt = {l: 0 for l in letters}
        letters_set = set(letters)
        words_count = defaultdict(int)

        # Iterate on the lines in the text:
        for line in self.txt_lines:
            # If there is any text in the line:
            if len(line) > 0:
                # Go over the words - use split():
                for word in line.lower().split():
                    # If the word is not empty (should not happen...):
                    if len(word) > 0:
                        words_count[word] += 1
                        for l in word:
                            # Go over the letters in the word, count only letters a-z:
                            if l in letters_set:
                                letters_cnt[l] += 1
                        # For words longer than one letter, count first letter and last letter:
                        if len(word) > 1:
                            # Look at the first and last letters:
                            if word[0] in letters_set:
                                first_letters_cnt[word[0]] += 1
                            if word[-1] in letters_set:
                                last_letters_cnt[word[-1]] += 1

        self.letters_cnt = letters_cnt
        self.first_letters_cnt = first_letters_cnt
        self.last_letters_cnt = last_letters_cnt
        self.words_df = pd.DataFrame.from_dict(words_count, orient='index', columns=['cnt']).sort_values('cnt', ascending=False)
        self.words_df['pct'] = self.words_df.cnt / self.words_df.cnt.sum()

        # Save the stats we found to a CSV file:
        # First create a DataFrame with the data in it, then save that dataframe to a CSV file:
        all_letters = pd.Series(self.letters_cnt).sort_values(ascending=False).rename('all_letters')
        last_letters = pd.Series(self.last_letters_cnt).sort_values(ascending=False).rename('last_letters')
        first_letters = pd.Series(self.first_letters_cnt).sort_values(ascending=False).rename('first_letters')
        stats_df = pd.concat((all_letters / all_letters.sum(),
                              first_letters / first_letters.sum(),
                              last_letters / last_letters.sum()), axis=1)
        self.stats_df = stats_df
        if letter_stats_file:
            self.stats_df.to_csv(letter_stats_file)
        if word_cnt_file:
            self.words_df.to_csv(word_cnt_file)

    def calc_text_stats(self, words):
        # For words longer than one letter, count first letter and last letter:
        letters_cnt = {l: 0 for l in letters}
        first_letters_cnt = {l: 0 for l in letters}
        last_letters_cnt = {l: 0 for l in letters}
        letters_set = set(letters)
        words_count = defaultdict(int)
        for word in words:
            words_count[word] += 1
            for l in word:
                # Go over the letters in the word, count only letters a-z:
                if l in letters_set:
                    letters_cnt[l] += 1
            if len(word) > 1:
                # Look at the first and last letters:
                if word[0] in letters_set:
                    first_letters_cnt[word[0]] += 1
                if word[-1] in letters_set:
                    last_letters_cnt[word[-1]] += 1

        # First create a DataFrame with the data in it, then save that dataframe to a CSV file:
        all_letters = pd.Series(letters_cnt).sort_values(ascending=False).rename('all_letters')
        last_letters = pd.Series(last_letters_cnt).sort_values(ascending=False).rename('last_letters')
        first_letters = pd.Series(first_letters_cnt).sort_values(ascending=False).rename('first_letters')
        stats_df = pd.concat((all_letters / all_letters.sum(),
                              first_letters / first_letters.sum(),
                              last_letters / last_letters.sum()), axis=1)

        words_df = pd.DataFrame.from_dict(words_count, orient='index', columns=['cnt']).sort_values('cnt',
                                                                                                    ascending=False)
        words_df['pct'] = words_df.cnt / words_df.cnt.sum()
        return stats_df, words_df

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--text_file', help='Text from NYTimes')
    args = parser.parse_args()

    text_stats = TextStats()
    text_stats.read_text(args.text_file)
    text_stats.count_letters('Data/letters.csv', 'Data/words_cnt.csv')
    print(text_stats.words_df.head(100))
    print(text_stats.stats_df)