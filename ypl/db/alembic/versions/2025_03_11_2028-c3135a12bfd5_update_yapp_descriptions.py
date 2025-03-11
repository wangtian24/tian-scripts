"""update yapp descriptions

Revision ID: c3135a12bfd5
Revises: e5e9c1cf445d
Create Date: 2025-03-11 20:28:01.111566+00:00

"""
from collections.abc import Sequence

from alembic import op
from sqlmodel import Session, update, select
from ypl.db.language_models import LanguageModel
from ypl.db.yapps import Yapp

# revision identifiers, used by Alembic.
revision: str = 'c3135a12bfd5'
down_revision: str | None = 'e5e9c1cf445d'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

YAPPS_DESC = [
    (
        "weather-yapp",
        """gathers and summarizes real-time weather information and meteorological data to deliver accurate forecasts and insights.
This agent does not have access to historical weather data.
Example questions: 'What is the weather in Tokyo?', 'Is it sunny in Chennai right now?', 'Should I wear a coat today in Waterloo?', 'Contrast the weather in Bengaluru and Delhi today'.
Not suitable for:
- 'What is the usual climate in Beijing?' (historical climate patterns not available)
- 'What was the temperature last summer?' (historical data not accessible)
- 'What is the weather forecast for NYC in 2026?' (future predictions beyond current forecast window)
- 'How does the climate vary through the year in the Himalayas?' (seasonal climate patterns not a fit)""",
    ),
    (
        "news-yapp",
        """aggregates, filters, and synthesizes current news articles from diverse sources to deliver concise, real-time insights on ongoing events.
This agent focuses on recent news and current events from reliable news sources.
Example questions: 'What is the latest news on the stock market?', 'What are the new developments in AI startups?', 'What happened in today's tech industry news?'.
Not suitable for:
- 'What were the major events of World War II?' (historical events outside current news scope)
- 'Who won the 1995 Super Bowl?' (past events not in current news coverage)
- 'What will happen in the 2030 elections?' (future predictions beyond current news)""",
    ),
    (
        "wikipedia-yapp",
        """searches, retrieves, and synthesizes Wikipedia content to help users find information on a wide range of factoid topics.
This agent only accesses Wikipedia content and does not include real-time information, news, or specialized research papers.
Example questions: 'What is the capital of France?', 'Who invented the telephone?', 'What is photosynthesis?', "Who discovered the electron?".
Not suitable for:
- 'What is the latest research on quantum computing hinting at?' (current research unlikely to be on Wikipedia)
- 'Tell me about what lead to Pink Floyd's split?' (long-form responses not suitable for agent)
- 'What happened in yesterday's football match?' (real-time sports results unlikely to be on Wikipedia)
- 'What are today's NVDA shares trading at?' (real-time financial data unlikely to be on Wikipedia)""",
    ),
    (
        "youtube-transcript-yapp",
        """retrieves YouTube video transcripts, summarizes key points, and answers questions about video content based on transcript data.
This agent only processes video transcripts and cannot analyze visual content or audio.
Example questions: 'What are the main points from this video: [URL]?', 'Can you summarize this lecture: [URL]?', 'What did the speaker say about AI at 5:20 in this video: [URL]?', 'summarize the main ideas in this video: [URL]'.
Not suitable for:
- 'what is up in the new mr beast video' (requires searching for the video)
- 'what dogs are they showing in the video youtube.com/watch?v=abc123 at 1:42' (requires visual content analysis, unless spoken about in the transcript)
- 'how many speakers are there in this video?' (cannot analyze audio characteristics)""",
    ),
]


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with Session(op.get_bind()) as session:
        for yapp_model_name, yapp_desc in YAPPS_DESC:
            # Update existing Yapp descriptions
            lang_id = session.exec(
                select(LanguageModel.language_model_id)
                .where(LanguageModel.internal_name == yapp_model_name)
            ).first()
            if lang_id:
                session.exec(
                    update(Yapp)
                    .where(Yapp.language_model_id == lang_id)
                    .values(description=yapp_desc)
                )
            else:
                print(f"Warning: Language model {yapp_model_name} not found, description not updated")
        session.commit()
    # ### end Alembic commands ###


def downgrade() -> None:
    # No downgrade needed as we're just updating descriptions
    pass
    # ### end Alembic commands ###
