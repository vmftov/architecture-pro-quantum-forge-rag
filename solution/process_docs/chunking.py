import re


def split_to_chunks(text: str, max_chunk_size_symbols: int) -> tuple[list[str], list[str]]:
    """ 
    Разбивает текст на чанки. Сначала делит на предложения, а если они превышают макс. длину то на более мелкие части. 
    Args:
        text (str): Исходный текст
        max_chunk_size_symbols (int): Максимальная длина чанка
    Returns:
        tuple[list[str], list[str]]: Кортеж из 2 списков. В первом - все чанки, во втором - длинные чанки, которые не удалось разделить
    """

    if len(text) <= max_chunk_size_symbols:
        return [text], []

    all_chunks: list[str] = []
    long_chunks: list[str] = []

    for chunk in re.split(r'(?<=[.!?])\s+', text):
        if not chunk.strip():
            continue

        while len(chunk) > max_chunk_size_symbols:
            space_index = find_left_space_index(chunk, max_chunk_size_symbols)
            if space_index <= 0:
                long_chunks.append(chunk)
                break
            sub_chunk = chunk[:space_index]
            all_chunks.append(sub_chunk)
            chunk = chunk[space_index:]

        if chunk:
            all_chunks.append(chunk)

    return all_chunks, long_chunks


def find_left_space_index(text: str, index: int):
    """ Ищет индекс пробела слева от индекса """

    if index < 0 or index >= len(text):
        return -1
    while index >= 0 and text[index] != " ":
        index -= 1
    return index
