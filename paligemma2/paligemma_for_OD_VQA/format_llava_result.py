import json

def process_llava_generation(generated_text):
    """
    Given a generated text from Llava, return only the text after 'ASSISTANT:'.
    If the delimiter is not found, return the original text.
    """
    if "ASSISTANT:" in generated_text:
        return generated_text.split("ASSISTANT:")[-1].strip()
    return generated_text.strip()

# Load the Llava inference results from file.
with open("inference_results_5epoch_llava.json", "r") as f:
    results = json.load(f)

# Process each result.
for r in results:
    if "generated" in r:
        r["generated"] = process_llava_generation(r["generated"])

# Optionally, save the processed results to a new file.
with open("llava_inference_result.json", "w") as f:
    json.dump(results, f, indent=2)

print("Processing complete. Processed results saved to inference_results_5epoch_llava_processed.json")
