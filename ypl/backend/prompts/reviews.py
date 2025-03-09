import logging
import uuid

from sqlalchemy.orm import joinedload
from sqlmodel import or_, select

from ypl.backend.db import get_async_session
from ypl.backend.utils.json import json_dumps
from ypl.db.chats import ChatMessage, CompletionStatus, MessageModifierStatus, MessageType

BINARY_REVIEW_PROMPT = """You are an expert at evaluating if AI responses accurately address user queries in a conversation context.

Your task is to determine if the AI's response appropriately and accurately addresses the user's needs based on:
1. Relevance: Does the response directly address the latest user message while maintaining relevant context from earlier turns?
2. Accuracy: For time-dependent conversations that are about current events, is the information provided factually correct and up-to-date as of {cur_datetime}?
For time-independent conversations and time-dependent conversations that are not about current events, just verify if the information is factually correct.

The input will contain a conversation history ending with the user's latest message, followed by the AI's response to evaluate.

Respond with 'true' for responses that meet all these criteria or 'false' for responses that fail any relevant criterion.
Your review does not have to be strict, unless the response is provably or blatantly false, do not respond with a 'false'.
For math problems, coding requests and puzzles, you need to be strict and only respond with 'true' if the response is fully correct.
For responses that are jokes, riddles, or conditional on a user's request for fictional information, do not verify the accuracy of the information, just see if the form is adhered to.
It is not required for a response to be complete or super coherent, as long as it is factually correct and relevant to the user's query, respond with 'true'.
Do not provide any other text or explanation."""

CRITIQUE_REVIEW_PROMPT = """You are an expert at critically evaluating AI responses in conversation contexts with a skeptical yet constructive eye.

Your task is to provide a concise, pointed critique of an AI's response, considering:
1. Contextual Understanding: How well does the response incorporate and build upon the conversation history (if any)?
2. Technical Accuracy: For time-dependent conversations that are about current events, is the response correct and current as of {cur_datetime}?
For time-independent conversations and time-dependent conversations that are not about current events, is the response factually correct?
3. Mathematical Accuracy: For math problems, coding requests and puzzles, does the response solve it correctly, elegantly, and correctly, and if not, where does it falter?
4. Completeness & Clarity: Does it fully address all aspects of the user's query in an understandable way?
5. Engagement & Tone: Does it maintain appropriate engagement while matching the user's requirements?
6. Potential Issues: Are there any missing elements, assumptions, or areas that could be misleading?

The input will contain a conversation history ending with the user's latest message, followed by the AI's response to evaluate.

Provide a pointed review in at most two sentences:
Sentence 1: Evaluate the response's strengths and weaknesses, focusing on how well it addresses the core query
Sentence 2 (optional): Suggest specific enhancements or improvements within the confines of the user's request, if needed

If the final AI response is empty, then all you need to say is "The response is empty."
Do not unnaturally extend the critique beyond needed, some responses may be on point and you don't have to say more than a line - it is enough to mention that the response is good.
In such cases, the second sentence should not be provided.

Example of good reviews: "The response explains LLM architectures and efficiency but lacks empirical data and a clear definition. Adding benchmarks would make it more credible." or "The response correctly calculates the product, providing an accurate and clear answer."

Only provide the critique, no other text or explanation."""

SEGMENTED_REVIEW_PROMPT = """You are an expert reviewer tasked with suggesting critical improvements to AI responses.

Core Principles:
1. Be selective - only suggest edits that meaningfully improve accuracy, clarity or effectiveness
2. Identify segments organically based on logical breaks in content - not predetermined lengths
3. Ignore content that do not need improvement
4. Each selected segment should be unique in the response and at least one complete sentence or logical unit (e.g., code block, equation) long

When evaluating segments, focus on:
- Factual errors or outdated information (as of {cur_datetime})
- Technical inaccuracies in code, math, puzzles, or domain-specific content
- Unclear or ambiguous explanations
- Missing critical context from the conversation history
- Responses that don't fully address the user's intent
- Missing context that can benefit the response
- Very poor phrasing or big grammatical errors
- Very poor flow or structure
- Segments should be unique in the text and non-overlapping

Output Format:
For segments requiring improvement, use the format:

<segment 1>
[insert verbatim segment 1 from original response]
</segment 1>
<review 1>
[insert 1-2 line explanation leading to the updated-segment]
</review 1>
<updated-segment 1>
[insert improved segment 1 with necessary changes]
</updated-segment 1>
<segment 2>
[insert verbatim segment 2 from original response]
</segment 2>
<review 2>
[insert 1-2 line explanation leading to the updated-segment]
</review 2>
<updated-segment 2>
[insert improved segment 2 with necessary changes]
</updated-segment 2>
...

Rules:
- Preserve exact formatting, indentation, and whitespace
- Only include segments that need meaningful improvements
- Keep explanations focused on specific, actionable changes
- Maintain the original response's style and tone, if high quality, else improve it
- For code/math, ensure changes preserve or enhance functionality
- The update should not be contextually changing the previous or next segment
- If the update modifies the flow of the response, then provide an update for the previous or next segments, where applicable, or try to include the other segments in the same segment, review and update
- Multiple non-overlapping segments can be reviewed and updated
- The text "[insert verbatim segment i from original response]" should never be in the answer, only the actual segment, and similarly for the review and updated-segment
- Try to provide at least 2 segments and at most 4 segments

Do not add any text, explanations, or markup outside the specified format."""

# User prompt templates for each review type
BINARY_REVIEW_USER_PROMPT = """Evaluate this AI response with a strict true/false decision.
Is it factually accurate and does it address the core user request?
Reply with just 'true' or 'false'."""

CRITIQUE_REVIEW_USER_PROMPT = """Critique this AI response based on the criteria laid out in the instructions.
Respond with just a concise 1-2 sentence evaluation highlighting strengths and weaknesses, with the second sentence being optional and only used to suggest enhancements if the response requires it."""

SEGMENTED_REVIEW_USER_PROMPT = """Analyze this AI response and identify 2-4 specific segments that need improvement.
For each segment, provide a concise explanation of the issue and a suggested revision that would enhance the quality, in the exact format laid out in the instructions.
Do not include any other text or explanation."""

NUGGETIZER_CREATE_PROMPT = """You are NuggetizeLLM, an intelligent assistant that extracts atomic nuggets of information from provided "AI Responses to Create Nuggets From" to address the final user request in a conversation.
Focus on the final request while considering the conversation history for additional context."""

NUGGETIZER_CREATE_USER_PROMPT = """Extract or update atomic nuggets of information (1-12 words each) from the provided "AI Responses to Create Nuggets From" that are different responses to the user's final request.
Consider the conversation history for understanding but extract nuggets ONLY from the responses provided, not from any other AI responses or other parts of the conversation.
Return the final list of all nuggets in a Pythonic list format.
Ensure there is no redundant information and the list has at most {max_nuggets} nuggets (can be less), keeping only the most vital ones.
Order them in decreasing order of importance.
Prefer nuggets that provide more interesting information.

<Conversation History and Final Request>
{query}
</Conversation History and Final Request>

<AI Responses to Create Nuggets From>
{context}
</AI Responses to Create Nuggets From>

<Initial Nugget List>
{nuggets}
</Initial Nugget List>

Initial Nugget List Length: {nugget_count}

Only update the list of atomic nuggets (if needed, else return as is). Do not explain. Always answer in short nuggets (not questions). List in the form ["a", "b", ...] and a and b are strings with no mention of ". Do not use identifiers like <Updated Nugget List>.
Updated Nugget List:"""

NUGGETIZER_SCORE_PROMPT = """You are NuggetizeScoreLLM, an intelligent assistant that evaluates the importance of atomic nuggets in addressing a user's final request, while considering the conversation context."""

NUGGETIZER_SCORE_USER_PROMPT = """Label each nugget as either vital or okay based on its importance in addressing the user's final request.
Consider the conversation history for context, but focus primarily on the final request.
Vital nuggets represent concepts that must be present in a "good" answer; okay nuggets contribute worthwhile information but are not essential.
Return the list of labels in a Pythonic list format (type: List[str]).
The list should be in the same order as the input nuggets.
Make sure to provide a label for each nugget.

<Conversation History and Final Request>
{query}
</Conversation History and Final Request>

<Nugget List>
{nuggets}
</Nugget List>

Only return the list of labels (List[str]). Do not explain or use identifiers like <Labels>.
Labels:"""

NUGGETIZER_ASSIGN_PROMPT = """You are NuggetizeAssignerLLM, an intelligent assistant that determines how well each nugget is supported by a given AI response, considering both the conversation context and final request."""

NUGGETIZER_ASSIGN_PROMPT_SUPPORT_GRADE_2 = """You are NuggetizeAssignerLLM, an intelligent assistant that determines how well each nugget is supported by a given AI response, considering both the conversation context and final request."""

NUGGETIZER_ASSIGN_USER_PROMPT_SUPPORT_GRADE_2 = """Evaluate each nugget against the AI response to determine if it is supported.
Consider the conversation history for context, but focus on how well the response supports each nugget in addressing the final request.
Label each nugget as either support (if fully captured in the response) or not_support (if not captured).
Return the list of labels in a Pythonic list format (type: List[str]).
The list should be in the same order as the input nuggets.
Make sure to provide a label for each nugget.

<Conversation History and Final Request>
{query}
</Conversation History and Final Request>

<AI Response to Check Support Against>
{context}
</AI Response to Check Support Against>

<Nugget List>
{nuggets}
</Nugget List>

Only return the list of labels (List[str]). Do not explain or use identifiers like <Labels>.
Labels:"""

NUGGETIZER_ASSIGN_USER_PROMPT = """Evaluate each nugget against the AI response to determine its level of support.
Consider the conversation history for context, but focus on how well the response supports each nugget in addressing the final request.
Label each nugget as:
- support: if fully captured in the response
- partial_support: if partially captured in the response
- not_support: if not captured at all
Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget.

<Conversation History and Final Request>
{query}
</Conversation History and Final Request>

<AI Response to Check Support Against>
{context}
</AI Response to Check Support Against>

<Nugget List>
{nuggets}
</Nugget List>

Only return the list of labels (List[str]). Do not explain. Do not use identifiers like <Labels>.
Labels:"""

CROSS_CHECK_PROMPT = """You are a helpful assistant.
A user has previously asked both you and other assistant, {other_model_names}, to respond to a prompt, and you both responded.
You can now see the other assistants' responses to the user's prompt.
Your job is to react to the other assistant response, reflecting on your own response if needed.
Direct your reaction to the user, not to the other assistant.

You may do things like confirm the accuracy of the other assistant's response, challenge it if you think it's incorrect,
point out contradictions and differences between your responses, and so on. The user will benefit from a meaningful critique
of the other assistant's response - try to be critical and point out flaws in the other assistant's response, but don't be too mean.
You may also point out flaws in your own response in light of the other assistant's response, if you think that's useful.

When referring to the other assistant's response, you may use their name, {other_model_names}.
Refer to the response provided by your model in the first person -- "my response", "my answer", or "me", and so on
Be brief and concise, ideally respond in one or two sentences. You may use bullet points if you think it makes your response clearer.
You may use formatting (bold text, bullets, and so on) to highlight aspects of your reaction to the other assistant's response."""

CROSS_CHECK_USER_PROMPT = """Assess the other assistant's response critically, comparing it to your previous response.

- Identify any inaccuracies or omissions.
- Highlight differences and provide constructive feedback.
- Reflect on your own response, if applicable.

Be concise, using bullets or formatting for clarity.
"""

TALK_TO_OTHER_MODELS_SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant.
A user has previously asked both you and other assistant, {other_model_names}, to respond to a prompt, and you both responded.
You can now see the other assistants' responses to the user's prompt.
Your job is to react to the other assistant response, reflecting on your own response if needed.
Direct your reaction to the user, not to the other assistant.

You may do things like confirm the accuracy of the other assistant's response, challenge it if you think it's incorrect,
point out contradictions and differences between your responses, and so on. The user will benefit from a meaningful critique
of the other assistant's response - try to be critical and point out flaws in the other assistant's response, but don't be too mean.
You may also point out flaws in your own response in light of the other assistant's response, if you think that's useful.

When referring to the other assistant's response, you may use their name, {other_model_names}.
Refer to the response provided by your model in the first person -- "my response", "my answer", or "me", and so on
Be brief and concise, ideally respond in one or two sentences. You may use bullet points if you think it makes your response clearer.
You may use formatting (bold text, bullets, and so on) to highlight aspects of your reaction to the other assistant's response.

The user's prompt was {user_prompt}.

Your previous response to the user's prompt was {response_from_cur_model}.

The other assistant responses to the user were:

{responses_from_other_models}

Now, react to the other assistant's response, directly to the user.
"""


async def talk_to_other_models_system_prompt(cur_model: str, turn_id: uuid.UUID) -> str:
    async with get_async_session() as session:
        messages_in_same_turn_query = (
            select(ChatMessage)
            .options(
                joinedload(ChatMessage.assistant_language_model),  # type: ignore
            )
            .where(
                ChatMessage.turn_id == turn_id,
                ChatMessage.message_type.in_([MessageType.USER_MESSAGE, MessageType.ASSISTANT_MESSAGE]),  # type: ignore
                or_(
                    ChatMessage.modifier_status == MessageModifierStatus.SELECTED,
                    ChatMessage.modifier_status.is_(None),  # type: ignore
                ),
            )
        )
        result = await session.exec(messages_in_same_turn_query)
        chat_messages = result.unique().all()

        # Collect the user prompt and the responses from the other models.
        user_prompt = ""
        response_from_cur_model = "(You did not previously respond to the user's prompt)"
        responses_from_other_models: dict[str, str] = {}

        for chat_message in chat_messages:
            if chat_message.message_type == MessageType.USER_MESSAGE:
                user_prompt = chat_message.content
            elif (
                chat_message.message_type == MessageType.ASSISTANT_MESSAGE
                and chat_message.completion_status == CompletionStatus.SUCCESS
            ):
                if chat_message.assistant_language_model.internal_name == cur_model:
                    response_from_cur_model = chat_message.content
                else:
                    # Note: this label is only used in the system prompt to refer to the other model.
                    label = (
                        chat_message.assistant_language_model.label
                        or chat_message.assistant_language_model.internal_name
                    )
                    responses_from_other_models[label] = chat_message.content

        if not (user_prompt and responses_from_other_models):
            raise ValueError("Missing required fields for system prompt")

        formatted_responses_from_other_models = "\n---\n".join(
            [f"Response from {model_name}:\n{response}" for model_name, response in responses_from_other_models.items()]
        )
        other_model_names = ", ".join(responses_from_other_models.keys())

        logging.info(
            json_dumps(
                {
                    "message": "respoding to other models",
                    "cur_model": cur_model,
                    "other_models": other_model_names,
                }
            )
        )

        return TALK_TO_OTHER_MODELS_SYSTEM_PROMPT_TEMPLATE.format(
            user_prompt=user_prompt,
            response_from_cur_model=response_from_cur_model,
            responses_from_other_models=formatted_responses_from_other_models,
            other_model_names=other_model_names,
        )
