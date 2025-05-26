import csv
import ast
from typing import Dict

def parse_file(file_path: str) -> Dict[str, int]:
    """Parse a CSV file and count occurrences of each file path in the 'Paths' column."""
    file_occurrences = {}

    try:
        with open(file_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                paths_str = row.get("Paths", "")
                try:
                    paths = ast.literal_eval(paths_str)
                    if isinstance(paths, list):
                        for path in paths:
                            path = path.strip()
                            if path:
                                file_occurrences[path] = file_occurrences.get(path, 0) + 1
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Could not parse row: {paths_str}. Error: {e}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        raise

    return file_occurrences

def write_csv(data: Dict[str, int], output_file: str) -> None:
    """ Write file occurrences data to a CSV file."""
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
    input_file = "extracted_bugs.csv"
    output_file = "path_occurences.csv"

    print("Starting file parsing...")
    file_occurrences = parse_file(input_file)

    print("Writing data to CSV...")
    write_csv(file_occurrences, output_file)

if __name__ == "__main__":
    main()

