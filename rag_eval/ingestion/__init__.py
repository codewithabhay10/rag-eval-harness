"""ingestion — corpus download, PDF parsing, chunking, embedding, indexing.

Pipeline order: corpus.py -> parse.py -> chunk.py -> embed.py -> index.py, wired
together by run.py. Each stage is a separate module so it can be swapped or tested
in isolation.
"""
