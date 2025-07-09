input_file = "input.txt"
output_file = "output.csv"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        # Remove any quotes and strip spaces
        cleaned = line.strip().replace('"', '')
        # Split by whitespace and join with commas
        parts = cleaned.split()
        csv_line = ",".join(parts)
        outfile.write(csv_line + "\n")