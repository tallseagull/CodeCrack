# CodeCrack
Code breaker for replacement cipher

## Setup and Running
To generate the data you need to download the NYTimes articles data set (ask me for it :-)):
python text_stats -t /path/to/nytimes_news_articles.txt.gz

This will create the CSV files in the Data/ directory. They are already available in the repo, so no real need to rerun this part.

## Running on example
Now to run on a specific example user cracker.py. Options are:
* -e /path/to/enc_file.txt
* -d /path/to/decrypt_result.txt - will write the result to this file after decrypt
* -k /path/to/key.json - will write the encrypt key here as a JSON

