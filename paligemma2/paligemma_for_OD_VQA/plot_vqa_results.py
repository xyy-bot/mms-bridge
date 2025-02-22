import json
import re
import difflib
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from sklearn.metrics import accuracy_score, precision_score
plt.rcParams.update({
    'font.size': 8,
    'font.family': 'Consolas'
})
# --- Helper Functions ---

def extract_classification(text):
    """
    Extracts the final classification label from text.
    It looks for a pattern like "it is empty/full/fake/real" and, if not found,
    falls back to keyword search.
    """
    text = text.lower()
    m = re.search(r"it is\s+(empty|full|fake|real)", text)
    if m:
        return m.group(1)
    else:
        for keyword in ["empty", "full", "fake", "real"]:
            if keyword in text:
                return keyword
    return None

def compute_metrics(results):
    """
    Given a list of VQA result dictionaries, compute evaluation metrics.
    Each result should have keys: "ground_truth" and "generated".
    Returns a dictionary of aggregated metrics.
    """
    smoothie = SmoothingFunction().method4

    # Process each sample.
    for r in results:
        gt = r["ground_truth"].strip().lower()
        gen = r["generated"].strip().lower()
        # Exact match.
        r["exact_match"] = (gt == gen)
        # Compute BLEU score (using tokenization based on whitespace).
        reference = gt.split()
        candidate = gen.split()
        r["bleu"] = sentence_bleu([reference], candidate, smoothing_function=smoothie)
        # Extract classification labels.
        r["gt_class"] = extract_classification(r["ground_truth"])
        r["gen_class"] = extract_classification(r["generated"])

    n_results = len(results)
    exact_match_accuracy = sum(1 for r in results if r["exact_match"]) / n_results
    avg_bleu = sum(r["bleu"] for r in results) / n_results

    # Compute BERT Score for the entire dataset.
    refs = [r["ground_truth"] for r in results]
    cands = [r["generated"] for r in results]
    # Set verbose=False to suppress intermediate output.
    P, R, F1 = bert_score(cands, refs, lang="en", verbose=False)
    avg_bert = F1.mean().item()

    # Collect classification labels where extraction succeeded.
    gt_labels = []
    gen_labels = []
    for r in results:
        if r["gt_class"] is not None and r["gen_class"] is not None:
            gt_labels.append(r["gt_class"])
            gen_labels.append(r["gen_class"])
    if gt_labels:
        classification_accuracy = accuracy_score(gt_labels, gen_labels)
        classification_precision = precision_score(
            gt_labels, gen_labels, average="macro", labels=["fake", "empty", "real", "full"]
        )
    else:
        classification_accuracy = 0
        classification_precision = 0

    return {
        "E. M.": exact_match_accuracy,
        "BLEU": avg_bleu,
        "BERT": avg_bert,
        "Acc. ": classification_accuracy,
        "Precis. ": classification_precision
    }

# --- Load Inference Results ---

with open("inference_results_10epoch_manually_split.json", "r") as f:
    paligemma_results = json.load(f)

with open("llava_inference_result.json", "r") as f:
    llava_results = json.load(f)

# --- Compute Metrics for Each Model ---

metrics_paligemma = compute_metrics(paligemma_results)
metrics_llava = compute_metrics(llava_results)

print("Paligemma Metrics:", metrics_paligemma)
print("Llava Metrics:", metrics_llava)

# --- Plot Comparison as a Grouped Bar Chart ---

metric_names = list(metrics_paligemma.keys())
paligemma_values = [metrics_paligemma[m] for m in metric_names]
llava_values = [metrics_llava[m] for m in metric_names]

x = np.arange(len(metric_names))  # label locations
width = 0.35  # width of each bar

fig, ax = plt.subplots(figsize=(3.5, 2.5))
rects1 = ax.bar(x - width/2, llava_values, width, label="Llava1.5", color="gray")
rects2 = ax.bar(x + width/2, paligemma_values, width, label="PaliGemma2", color="skyblue")


ax.set_ylabel("Score")
# ax.set_title("Comparison of VQA Evaluation Metrics: Paligemma vs. Llava")
ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 1.1)
ax.legend(loc="lower right")

# Function to annotate bars with their values.
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom")

def autolabel1(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height-0.05),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom")

autolabel1(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig("vqa_results_llava_pali.png", dpi=600)
plt.show()
