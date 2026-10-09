def save_list_plaintext(values: list[str], file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        for v in values:
            f.write(v + "\n")

def load_list_plaintext(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

words: list[str] = []
words.extend(load_list_plaintext("en_top_1000.txt"))
words.extend(load_list_plaintext("en_top_3000.txt"))
words.extend(load_list_plaintext("en_top_6000.txt"))

words = [v for v in words if "!" not in v]
words = [v for v in words if " " not in v]
words = [v for v in words if v.lower() == v]

unique_words = list(set(words))
unique_words.sort()

save_list_plaintext(unique_words, "en_top.txt")
