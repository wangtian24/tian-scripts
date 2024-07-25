def combine_short_sentences(
    sentences: list[str], max_combined_length: int = 40, max_single_length: int = 10
) -> list[str]:
    """Combine short sentences into a single sentence."""
    combined_sentences = []
    current_sentence = ""

    for sentence in sentences:
        if (len(current_sentence) + len(sentence) <= max_combined_length) or (
            len(current_sentence) < max_single_length
        ):
            current_sentence += " " + sentence if current_sentence else sentence
        else:
            if current_sentence:
                combined_sentences.append(current_sentence.strip())
            current_sentence = sentence

    if current_sentence:
        combined_sentences.append(current_sentence.strip())

    return combined_sentences
