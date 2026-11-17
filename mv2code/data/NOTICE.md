# Data files

## `fa_words.tsv.gz`

Persian word list with frequencies (28k words) used by `mv2code.lexicon` to repair OCR errors.

* Source: the [`wordfreq`](https://github.com/rspeer/wordfreq) package by Robyn Speer (Persian list: Wikipedia, subtitles, news, web text), extracted with `tools/build_fa_words.py`.
* Licence of the data: **CC BY-SA 4.0** (https://creativecommons.org/licenses/by-sa/4.0/); wordfreq's code is Apache-2.0.
  This derived file keeps CC BY-SA 4.0 (attribution above, same licence if you redistribute it). The rest of mv2code is not affected: the word list is a
  separate data file that the code only reads.
* Format: one `word<TAB>zipf` per line (zipf = log10 of occurrences per billion words), UTF-8, gzip.
