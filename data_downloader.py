from datasets import load_dataset
import csv
import json

def download_dataset(dataset_name, output_file):
    """Download dataset from Hugging Face and save it as a CSV file."""
    dataset = load_dataset(dataset_name)

    with open(output_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["book_id", "summary_id", "summary"])
        for i in range(len(dataset["train"])):
            writer.writerow([dataset["train"][i]["book_id"], dataset["train"][i]["summary_id"], json.loads(dataset["train"][i]["summary"])["summary"]])

if __name__ == "__main__":
    download_dataset("kmfoda/booksum", "raw_data.csv")
