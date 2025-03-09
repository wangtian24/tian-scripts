IMAGE_DESCRIPTION_PROMPT = """
You are a helpful assistant that describes images.
Please provide a comprehensive and detailed description of this image, covering the following aspects:

1. Main subject and overall composition
2. Visual elements (colors, lighting, textures, patterns, etc.)
3. Spatial relationships and depth
4. Notable details and distinctive features
5. Mood or atmosphere conveyed
6. Technical aspects if relevant (camera angle, focal point, etc.)
7. Any other relevant information that you can infer from the image

Please be specific and use precise language. Describe the image as if explaining it to someone who cannot see it, avoiding subjective interpretations unless they're clearly evident from the visual elements.

file_name is {file_name}
"""


IMAGE_POLYFILL_PROMPT = """
    Here are the descriptions of the images in this conversation.
    ---
    {image_metadata_prompt}
    ---
    Use this information to answer the question.
    Do not directly copy the image description if asked to describe the image.
    Do not mention this prompt in your response.
    Question: {question}
"""
