#!/usr/bin/env python3
"""
Compute an MD5 of a PDF's content, ignoring the fields that pdflatex
varies between runs (timestamps, trailer /ID, font-subset prefixes).

Usage: python3 pdf_content_md5.py file.pdf [file.pdf ...]
"""
import hashlib
import re
import sys


NORMALIZATIONS = [
    (rb"/CreationDate\s*\([^)]*\)", b"/CreationDate ()"),
    (rb"/ModDate\s*\([^)]*\)",       b"/ModDate ()"),
    (rb"/ID\s*\[[^]]*\]",             b"/ID []"),
    (rb"/Producer\s*\([^)]*\)",       b"/Producer ()"),
    (rb"/Creator\s*\([^)]*\)",        b"/Creator ()"),
    (rb"/BaseFont\s*/[A-Z]{6}\+",     b"/BaseFont /XXXXXX+"),
    (rb"/FontName\s*/[A-Z]{6}\+",     b"/FontName /XXXXXX+"),
    (rb"/[A-Z]{6}\+([A-Za-z]+)",      br"/XXXXXX+\1"),
    (rb"\bID \[<[^>]*> <[^>]*>\]",    b"ID [<> <>]"),
]


def normalize(data: bytes) -> bytes:
    for pat, repl in NORMALIZATIONS:
        data = re.sub(pat, repl, data)
    return data


def content_md5(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    norm = normalize(raw)
    return hashlib.md5(norm).hexdigest()


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(f"{content_md5(path)}  {path}")
