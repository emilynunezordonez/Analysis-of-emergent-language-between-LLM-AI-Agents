# Reference English corpus

`carroll-alice.txt` is Lewis Carroll's *Alice's Adventures in Wonderland* (1865), a public-domain text from Project Gutenberg, taken unchanged from the NLTK Gutenberg corpus package:

- URL: https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip (member `gutenberg/carroll-alice.txt`)
- SHA-256: `5ce9042c0d8536b072b61d4c7c4e6d70c72d9dbb5f3d816261326d9280749266`

It is a convenience baseline for "standard English". Narrative fiction is not descriptive scene language, so for the final study also try a modern, more neutral corpus. Pass any plain-text file with `python -m emergent.analysis.report --reference-corpus <file>`. For example, use the Brown corpus via `nltk.download("brown")`, but check its licence first.
