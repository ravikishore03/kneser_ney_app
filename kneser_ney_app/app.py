from flask import Flask, render_template
from collections import Counter, defaultdict

app = Flask(__name__)

@app.route("/")
def kneser_ney():
    corpus = "I love NLP and I love machine learning"

    words = corpus.lower().split()

    # Unigram and Bigram counts
    unigram_counts = Counter(words)
    bigram_counts = Counter(zip(words[:-1], words[1:]))

    D = 0.75  # Discount value

    # Continuation counts
    continuation_counts = defaultdict(set)
    for (w1, w2) in bigram_counts:
        continuation_counts[w2].add(w1)

    total_unique_bigrams = len(bigram_counts)

    probabilities = []

    for (w1, w2), count in bigram_counts.items():
        # Discounted probability
        discounted = max(count - D, 0) / unigram_counts[w1]

        # Lambda (normalization factor)
        lambda_w1 = (D / unigram_counts[w1]) * len(
            [w for (x, w) in bigram_counts if x == w1]
        )

        # Continuation probability
        continuation_prob = len(continuation_counts[w2]) / total_unique_bigrams

        prob = discounted + lambda_w1 * continuation_prob

        probabilities.append({
            "bigram": f"{w1} {w2}",
            "probability": round(prob, 4)
        })

    return render_template("index.html", probabilities=probabilities)


if __name__ == "__main__":
    app.run(debug=True)
