import json
import re

def extract_label(text):
    """
    Extracts the label (fake/real or empty/full) from a text string.
    The function searches for the pattern "it is <label>" (case insensitive)
    and returns the captured label.
    """
    text = text.lower()
    # Look for a word following "it is" (e.g., "it is fake")
    match = re.search(r"it is\s+(\w+)", text)
    if match:
        return match.group(1)
    else:
        return None

def load_results(filename):
    """
    Loads the results from a JSON file. It first tries to load it as a JSON array.
    If that fails, it reads the file line-by-line assuming each line is a JSON object.
    """
    results = []
    try:
        with open(filename, "r") as f:
            results = json.load(f)
    except json.JSONDecodeError:
        # Fallback to reading line by line if not a valid JSON list
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    return results

def compute_precision(results):
    """
    Computes the precision as the ratio of records where the ground truth label
    is equal to the generated label.
    """
    total = len(results)
    correct = 0

    for record in results:
        gt_text = record.get("ground_truth", "")
        gen_text = record.get("generated", "")
        gt_label = extract_label(gt_text)
        gen_label = extract_label(gen_text)
        # For debugging, you can uncomment the following lines:
        # print(f"GT: {gt_text} -> {gt_label}")
        # print(f"GEN: {gen_text} -> {gen_label}")
        if gt_label is not None and gen_label is not None:
            if gt_label == gen_label:
                correct += 1

    precision = correct / total if total > 0 else 0.0
    return precision, correct, total

if __name__ == "__main__":
    # Update the file name to your inference results file.
    results_filename = "inference_results_15epoch_manually_split.json"
    results = load_results(results_filename)
    precision, correct, total = compute_precision(results)
    print(f"Precision: {precision*100:.2f}% ({correct} out of {total} records are accurate)")
