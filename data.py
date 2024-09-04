import random

# Sample sentences for our corpus
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "Actions speak louder than words.",
    "Where there's a will, there's a way.",
    "The early bird catches the worm.",
    "Practice makes perfect.",
    "Knowledge is power.",
    "Time is money.",
]

def generate_text_file(filename, num_lines):
    with open(filename, "w") as f:
        for _ in range(num_lines):
            # Randomly combine a few sentences to form a line
            line = " ".join(random.sample(sentences, k=random.randint(2, 4)))
            f.write(line + "\n")

# Generate training data
generate_text_file("train.txt", 1000)

# Generate validation data
generate_text_file("val.txt", 200)

print("Sample data generated and saved to 'train.txt' and 'val.txt'")