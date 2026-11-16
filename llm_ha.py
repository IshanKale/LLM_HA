"""
LLM Home Assignment
Topic: Text Preprocessing and Tokenization Strategies for NLP Applications
Name: Ishan Kale | PRN: 12310654 | Roll No.: 166

Install dependencies if needed:
    pip install nltk sentencepiece matplotlib

Run:
    python llm_ha.py
"""

import os
import re
import tempfile
import unicodedata
from collections import Counter

import matplotlib.pyplot as plt
import sentencepiece as spm
from nltk.tokenize import TweetTokenizer, WordPunctTokenizer


TRAIN_SENTENCES = [
    "Large Language Models can generate text, answer questions, and summarize documents.",
    "I can't believe GPT-style models handle contractions so well!",
    "Visit https://example.com/docs for API-v2.1 details.",
    "Email support@example.com if login fails.",
    "The price is ₹1,299.50 — taxes included.",
    "AI/ML, NLP, and RAG are common terms in modern systems.",
    "Tokenization splits 'unbelievable' into useful units.",
    "User123 posted: #LLM #PromptEngineering 🚀",
    "Pune's weather can change quickly; carry an umbrella.",
    "The file name is report_final_v3.pdf.",
    "zero-shot, one-shot, and few-shot prompting are related ideas.",
    "C++, Python3, and Node.js appear in developer resumes.",
    "Numbers like 2026-11-17 and 3.14159 need careful handling.",
    "Don't remove negation words when sentiment matters.",
    "MixedCaseWords and snake_case identifiers are common in code.",
    "नमस्ते Pune! Multilingual text should not break the tokenizer.",
    "Emojis 😊🔥 can carry sentiment.",
    "Repeated spaces should be normalized.",
    "Quotation marks “like these” may be Unicode.",
    "A tokenizer should balance vocabulary size and sequence length.",
]

TEST_SENTENCES = [
    "The LLM-based chatbot crashed on API-v3.0.",
    "NewUser42 paid ₹999 via UPI.",
    "UnknownWordXYZ appears only in this test sentence.",
    "Try docs.example.org and email admin@demo.ai.",
    "Great job!!! 🤖✨ #NLP",
]


def normalize(text: str) -> str:
    """Apply minimal Unicode and whitespace normalization."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def basic_preprocess(text: str) -> str:
    """Example preprocessing used in the report."""
    text = normalize(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "<url>", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "<email>", text)
    return text


wordpunct_tokenizer = WordPunctTokenizer()
tweet_tokenizer = TweetTokenizer(preserve_case=True)


def whitespace_tokenize(text: str):
    return normalize(text).split()


def wordpunct_tokenize(text: str):
    return wordpunct_tokenizer.tokenize(normalize(text))


def tweet_tokenize(text: str):
    return tweet_tokenizer.tokenize(normalize(text))


def train_sentencepiece(train_sentences):
    """Train SentencePiece BPE only on the training corpus."""
    temp_dir = tempfile.TemporaryDirectory()
    corpus_path = os.path.join(temp_dir.name, "corpus.txt")
    model_prefix = os.path.join(temp_dir.name, "bpe")

    with open(corpus_path, "w", encoding="utf-8") as f:
        for sentence in train_sentences:
            f.write(normalize(sentence) + "\n")

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        model_type="bpe",
        vocab_size=800,
        character_coverage=1.0,
        byte_fallback=True,
        hard_vocab_limit=False,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
        minloglevel=2,
    )

    processor = spm.SentencePieceProcessor(model_file=model_prefix + ".model")
    return temp_dir, processor


def evaluate_word_tokenizer(name, tokenize_fn):
    train_tokens = [token for s in TRAIN_SENTENCES for token in tokenize_fn(s)]
    train_vocab = set(train_tokens)
    test_tokens = [token for s in TEST_SENTENCES for token in tokenize_fn(s)]
    oov_tokens = [token for token in test_tokens if token not in train_vocab]

    return {
        "Tokenizer": name,
        "Train Tokens": len(train_tokens),
        "Unique Train Tokens": len(train_vocab),
        "Avg Tokens/Sentence": len(train_tokens) / len(TRAIN_SENTENCES),
        "Test Tokens": len(test_tokens),
        "OOV Tokens": len(oov_tokens),
        "Test OOV Rate": 100 * len(oov_tokens) / len(test_tokens),
    }


def evaluate_sentencepiece(processor):
    def bpe(text):
        return processor.encode(normalize(text), out_type=str)

    train_tokens = [token for s in TRAIN_SENTENCES for token in bpe(s)]
    test_tokens = [token for s in TEST_SENTENCES for token in bpe(s)]
    unk_piece = processor.id_to_piece(processor.unk_id())
    oov_tokens = [token for token in test_tokens if token == unk_piece]

    return {
        "Tokenizer": "SentencePiece BPE",
        "Train Tokens": len(train_tokens),
        "Unique Train Tokens": len(set(train_tokens)),
        "Avg Tokens/Sentence": len(train_tokens) / len(TRAIN_SENTENCES),
        "Test Tokens": len(test_tokens),
        "OOV Tokens": len(oov_tokens),
        "Test OOV Rate": 100 * len(oov_tokens) / len(test_tokens),
    }, bpe


def print_results(results):
    print("\nTOKENIZATION COMPARISON")
    print("-" * 92)
    print(f"{'Tokenizer':<20}{'Train':>10}{'Unique':>10}{'Avg/Sent':>12}{'Test':>10}{'OOV':>8}{'OOV Rate':>12}")
    print("-" * 92)
    for r in results:
        print(
            f"{r['Tokenizer']:<20}"
            f"{r['Train Tokens']:>10}"
            f"{r['Unique Train Tokens']:>10}"
            f"{r['Avg Tokens/Sentence']:>12.2f}"
            f"{r['Test Tokens']:>10}"
            f"{r['OOV Tokens']:>8}"
            f"{r['Test OOV Rate']:>11.2f}%"
        )
    print("-" * 92)


def print_sample_tokens(bpe):
    sample = TEST_SENTENCES[-1]
    print("\nSAMPLE TOKENIZATION")
    print("Input:", sample)
    print("Whitespace      :", whitespace_tokenize(sample))
    print("WordPunct       :", wordpunct_tokenize(sample))
    print("TweetTokenizer  :", tweet_tokenize(sample))
    print("SentencePiece BPE:", bpe(sample))


def save_graphs(results):
    names = [r["Tokenizer"] for r in results]
    avg_lengths = [r["Avg Tokens/Sentence"] for r in results]
    oov_rates = [r["Test OOV Rate"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.bar(names, avg_lengths)
    plt.ylabel("Average tokens per training sentence")
    plt.title("Average Sequence Length by Tokenizer")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig("average_token_length.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(names, oov_rates)
    plt.ylabel("OOV rate on unseen test set (%)")
    plt.title("Out-of-Vocabulary Rate by Tokenizer")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig("oov_rate.png", dpi=150)
    plt.close()

    print("\nSaved graphs: average_token_length.png, oov_rate.png")


def main():
    print("LLM Home Assignment - Tokenization Experiment")
    print("Name: Ishan Kale | PRN: 12310654 | Roll No.: 166")

    raw_example = "Visit https://example.com/docs for API-v2.1 details."
    print("\nPREPROCESSING EXAMPLE")
    print("Raw          :", raw_example)
    print("Basic cleaned:", basic_preprocess(raw_example))

    results = [
        evaluate_word_tokenizer("Whitespace", whitespace_tokenize),
        evaluate_word_tokenizer("WordPunct", wordpunct_tokenize),
        evaluate_word_tokenizer("TweetTokenizer", tweet_tokenize),
    ]

    temp_dir, sp = train_sentencepiece(TRAIN_SENTENCES)
    try:
        sp_result, bpe = evaluate_sentencepiece(sp)
        results.append(sp_result)
        print_results(results)
        print_sample_tokens(bpe)
        save_graphs(results)
    finally:
        temp_dir.cleanup()


if __name__ == "__main__":
    main()
