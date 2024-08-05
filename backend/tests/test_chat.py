from backend.llm.chat import highlight_llm_similarities_with_embeddings, prompt_difficulty

SAMPLE_RESPONSES: dict = {
    "prompt": "What is the difference between marriage license and marriage certificate?",
    "responses": {
        "response_a": "A marriage license is a legal document that allows a couple to get married. It is issued by a government agency, such as a county clerk's office or a state government, and is valid for a certain period of time, usually one year. After the marriage has taken place, the couple must obtain a marriage certificate, which is a document that records the marriage and is used to prove that the marriage took place. The marriage certificate is usually issued by the same government agency that issued the marriage license, and it is typically used for legal purposes, such as to change a name on a driver's license or to prove that a couple is married when applying for government benefits.",  # noqa
        "response_b": "A marriage license and a marriage certificate are two different legal documents that have separate purposes.\n\n1. Marriage License: A marriage license is a legal document that gives a couple permission to get married. It's usually obtained from local government or court officials before the wedding ceremony takes place. The couple is required to meet certain criteria, such as being of a certain age or not being closely related. Once the license is issued, there's often a waiting period before the marriage ceremony can take place. The marriage license has to be signed by the couple, their witnesses, and the officiant conducting the marriage ceremony, then returned to the license issuer for recording.\n\n2. Marriage Certificate: A marriage certificate, on the other hand, is a document that proves a marriage has legally taken place. It's issued after the marriage ceremony, once the signed marriage license has been returned and recorded. The marriage certificate includes details about the couple, like their names, the date and location of their wedding, and the names of their witnesses. This document serves as the official record of the marriage and is often needed for legal transactions like changing a name, adding a spouse to insurance, or proving marital status.",  # noqa
    },
}


def test_highlight_llm_similarities_with_embeddings() -> None:
    llm_sim = highlight_llm_similarities_with_embeddings(
        response_a=SAMPLE_RESPONSES["responses"]["response_a"],
        response_b=SAMPLE_RESPONSES["responses"]["response_b"],
    )

    assert llm_sim["high_similarity_pairs"] == [
        {
            "sentence_a": "A marriage license is a legal document that allows a couple to get married.",
            "sentence_b": "1. Marriage License: A marriage license is a legal document that gives a couple permission to get married.",  # noqa: E501
            "similarity": 0.9521,
        },
        {
            "sentence_a": "After the marriage has taken place, the couple must obtain a marriage certificate, which is a document that records the marriage and is used to prove that the marriage took place.",  # noqa: E501
            "sentence_b": "The marriage certificate includes details about the couple, like their names, the date and location of their wedding, and the names of their witnesses.",  # noqa: E501
            "similarity": 0.8272,
        },
    ]
    assert llm_sim["unique_sentences_a"] == [
        "It is issued by a government agency, such as a county clerk's office or a state government, and is valid for a certain period of time, usually one year.",  # noqa: E501
    ]
    assert llm_sim["unique_sentences_b"] == [
        "It's usually obtained from local government or court officials before the wedding ceremony takes place.",
        "The couple is required to meet certain criteria, such as being of a certain age or not being closely related.",  # noqa: E501
        "Once the license is issued, there's often a waiting period before the marriage ceremony can take place.",
        "The marriage license has to be signed by the couple, their witnesses, and the officiant conducting the marriage ceremony, then returned to the license issuer for recording.",  # noqa: E501
        "It's issued after the marriage ceremony, once the signed marriage license has been returned and recorded.",
    ]


def test_prompt_difficulty() -> None:
    prompt = SAMPLE_RESPONSES["prompt"]
    responses = [
        SAMPLE_RESPONSES["responses"]["response_a"],
        SAMPLE_RESPONSES["responses"]["response_b"],
    ]

    result = prompt_difficulty(prompt, responses)

    assert isinstance(result, dict)
    assert "prompt_difficulty" in result
    assert "embedding_similarity" in result
    assert "structure_similarity" in result
    assert "content_similarity" in result

    assert isinstance(result["prompt_difficulty"], float)
    assert isinstance(result["embedding_similarity"], float)
    assert isinstance(result["structure_similarity"], float)
    assert isinstance(result["content_similarity"], float)

    assert 0 <= result["prompt_difficulty"] <= 1
    assert 0 <= result["embedding_similarity"] <= 1
    assert 0 <= result["structure_similarity"] <= 1
    assert 0 <= result["content_similarity"] <= 1

    # Test with similar responses (should have high similarity scores)
    similar_responses = [
        "A dog is a domesticated mammal.",
        "A dog is a domesticated animal that belongs to the mammal class.",
    ]
    similar_result = prompt_difficulty("What is a dog?", similar_responses)
    assert similar_result["prompt_difficulty"] < 0.15
    assert similar_result["embedding_similarity"] > 0.9
    assert similar_result["content_similarity"] > 0.4

    # Test with dissimilar responses (should have low similarity scores)
    dissimilar_responses = [
        "Python is a programming language.",
        "An apple is a type of fruit.",
    ]
    dissimilar_result = prompt_difficulty("Random prompt", dissimilar_responses)
    assert dissimilar_result["prompt_difficulty"] > 0.35
    assert dissimilar_result["embedding_similarity"] < 0.6
    assert dissimilar_result["content_similarity"] < 0.3
