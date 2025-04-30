import csv
import matplotlib.pyplot as plt
from typing import Dict
import seaborn as sns

def parse_file(file_path: str) -> Dict[str, int]:
    """Parse a file and count occurrences of each file path.

    Args:
        file_path (str): Path to the input file.

    Returns:
        Dict[str, int]: A dictionary with file paths as keys and their occurrences as values.
    """
    file_occurrences = {}

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                # Split the line by commas and count occurrences
                for item in line.strip().split(','):
                    item = item.strip()  # Remove leading/trailing spaces
                    file_occurrences[item] = file_occurrences.get(item, 0) + 1
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        raise

    return file_occurrences

def write_csv(data: Dict[str, int], output_file: str) -> None:
    """ Write file occurrences data to a CSV file.

    Args:
        data (Dict[str, int]): A dictionary with file paths as keys and their occurrences as values.
        output_file (str): Path to the output CSV file.
    """
    try:
        with open(output_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["File", "Occurrence"])  # Write header

            for file_path, occurrence in data.items():
                writer.writerow([file_path, occurrence])

        print(f"CSV file created successfully at {output_file}")
    except Exception as e:
        print(f"Error writing to CSV file {output_file}: {e}")
        raise

def main():
    input_file = "paths.txt"
    output_file = "path_occurences_w_dal.csv"

    print("Starting file parsing...")
    file_occurrences = parse_file(input_file)

    print("Writing data to CSV...")
    write_csv(file_occurrences, output_file)

if __name__ == "__main__":
    main()

