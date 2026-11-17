"""Rebuild mv2code/data/fa_words.tsv.gz (the Persian word list used to repair OCR errors).

    pip install wordfreq && python tools/build_fa_words.py

Source: the `wordfreq` package (Robyn Speer) - Persian list: Wikipedia, subtitles, news, web text.  The data of wordfreq is released under
CC BY-SA 4.0 (https://github.com/rspeer/wordfreq/blob/master/README.md#license); this derived file keeps that licence (see mv2code/data/NOTICE.md).
Format: one `word<TAB>zipf` per line (zipf = log10 of occurrences per billion words; 3.0 is rare-ish, 5+ is common), UTF-8, gzip.
"""
import gzip
import os
import re
import unicodedata

from wordfreq import top_n_list, zipf_frequency

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mv2code", "data", "fa_words.tsv.gz")
AR = re.compile("^[ء-غف-يپچژکگی‌]+$")


def norm(w: str) -> str:
    w = unicodedata.normalize("NFKC", w).replace("ي", "ی").replace("ك", "ک").replace("ة", "ه")
    return w.strip("‌")


def main():
    words = {}
    for w in top_n_list("fa", 1_000_000):
        n = norm(w)
        if len(n.replace("‌", "")) < 2 or not AR.match(n):
            continue
        z = zipf_frequency(w, "fa")
        if z < 1.5:
            continue
        if n not in words or z > words[n]:
            words[n] = z
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8", compresslevel=9) as fh:
        for w, z in sorted(words.items(), key=lambda kv: (-kv[1], kv[0])):
            fh.write("%s\t%.2f\n" % (w, z))
    print(len(words), "words ->", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
