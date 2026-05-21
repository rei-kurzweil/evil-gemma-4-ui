def consume_complete_sentences(text: str):
    sentences = []
    start = 0
    index = 0
    closers = "\"')]}* "

    while index < len(text):
        if text[index] in ".!?":
            end = index + 1
            while end < len(text) and text[end] in closers:
                end += 1
            sentence = text[start:end].strip()
            if sentence:
                sentences.append(sentence)
            start = end
            index = end
            continue
        index += 1

    return sentences, text[start:]
