"""fill taxonomy table

Revision ID: 996d3b884f49
Revises: 99537bdbb8b8
Create Date: 2025-02-18 19:51:54.832519+00:00

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlmodel import Session


# revision identifiers, used by Alembic.
revision: str = '996d3b884f49'
down_revision: str | None = '99537bdbb8b8'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Taxonomy tree data extracted from the spreadsheet: https://docs.google.com/spreadsheets/d/1k-EhCCSxxnLWaxPFSSxRx0uA1pYJKPdCdWuYSFxqCeI/edit?gid=892740398#gid=892740398
# Fields:            is_pickable, is_leaf_node, taxo_label, model_publisher, model_family, model_class, model_version, model_release,
#                    is_strong, is_pro, is_live, parameter_count, context_window_tokens, supported_attachment_mime_types,
#                    knowledge_cutoff_date, avatar_url)

TAXONOMY_DATA = [
  (True, True, 'Claude 3 Haiku', 'Claude', 'Claude', 'Haiku', '3', '20240307', False, False, False, None, 200000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/qwen-5Jz2blSb03vSB8pTdfJ4yppXQEp00P.svg'),
  (True, True, 'Claude 3 Opus', 'Claude', 'Claude', 'Opus', '3', '20240229', False, True, False, None, 200000, '{image/*}', None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gryphe-AvLBWAam6ANMneC064efbJLFOMalw6.png'),
  (True, True, 'Claude 3 Sonnet', 'Claude', 'Claude', 'Sonnet', '3', '', False, False, False, None, 200000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/qwen-5Jz2blSb03vSB8pTdfJ4yppXQEp00P.svg'),
  (True, True, 'Claude 3.5 Haiku', 'Claude', 'Claude', 'Haiku', '3.5', '20241022', False, False, False, None, 200000, '{""}', None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/qwen-5Jz2blSb03vSB8pTdfJ4yppXQEp00P.svg'),
  (True, True, 'Claude 3.5 Sonnet', 'Claude', 'Claude', 'Sonnet', '3.5', '20241022', False, True, False, None, 200000, '{image/*,application/pdf}', None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gryphe-AvLBWAam6ANMneC064efbJLFOMalw6.png'),
  (True, True, 'Command R', 'Cohere', 'Command', 'R', '', '202408', False, False, False, None, 128000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/qwen-5Jz2blSb03vSB8pTdfJ4yppXQEp00P.svg'),
  (True, True, 'Command R+', 'Cohere', 'Command', 'R+', '', '202408', False, False, False, None, 128000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/qwen-5Jz2blSb03vSB8pTdfJ4yppXQEp00P.svg'),
  (True, True, 'DBRX Instruct', 'Databricks', 'DBRX', 'Instruct', '', '', False, False, False, None, 32768, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/qwen-5Jz2blSb03vSB8pTdfJ4yppXQEp00P.svg'),
  (True, True, 'DeepSeek Chat', 'DeepSeek', 'DeepSeek', 'Chat', '', '', True, False, False, None, 64000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (True, True, 'DeepSeek Coder', 'DeepSeek', 'DeepSeek', 'Coder', '', '', True, False, False, 236000000000, 128000, None, '2023-09-01', 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (True, True, 'DeepSeek R1', 'DeepSeek', 'DeepSeek', 'R1', '', '', False, False, False, None, 64000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/qwen-5Jz2blSb03vSB8pTdfJ4yppXQEp00P.svg'),
  (True, True, 'DeepSeek R1 distill llama 70B', 'DeepSeek', 'DeepSeek', 'R1 distill llama 70b', '', '', True, True, False, 70000000000, 128000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (True, True, 'Gemini', 'Google', 'Gemini', None, '', '20241114', False, False, False, None, 32000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/qwen-5Jz2blSb03vSB8pTdfJ4yppXQEp00P.svg'),
  (True, True, 'Gemini 1.5 Flash', 'Google', 'Gemini', 'Flash', '1.5', '20240827', True, False, False, None, 1000000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (True, True, 'Gemini 1.5 Flash 8B', 'Google', 'Gemini', 'Flash 8b', '1.5', '20240827', False, False, False, 8000000000, 1000000, '{image/*,application/pdf}', None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/qwen-5Jz2blSb03vSB8pTdfJ4yppXQEp00P.svg'),
  (True, True, 'Gemini 1.5 Pro', 'Google', 'Gemini', 'Pro', '1.5', '20240827', False, True, False, None, 2000000, '{image/*,application/pdf}', None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/ai21-yI1CWcuqmijuCBQOqss8Gl5f4TM0NH.svg'),
  (True, True, 'Gemini 1.5 Pro online', 'Google', 'Gemini', 'Pro online', '1.5', '', False, True, True, None, 127072, '{image/*,application/pdf}', None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gryphe-AvLBWAam6ANMneC064efbJLFOMalw6.png'),
  (True, True, 'Gemini 2.0 Flash', 'Google', 'Gemini', 'Flash', '2.0', '', False, False, False, None, 1048576, '{image/*,application/pdf}', '2024-11-28', 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/qwen-5Jz2blSb03vSB8pTdfJ4yppXQEp00P.svg'),
  (True, True, 'Gemini 2.0 Flash Lite', 'Google', 'Gemini', 'Flash Lite', '2.0', '', False, False, False, None, 1048576, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/qwen-5Jz2blSb03vSB8pTdfJ4yppXQEp00P.svg'),
  (True, True, 'Gemini 2.0 Flash Thinking', 'Google', 'Gemini', 'Flash Thinking', '2.0', '', False, False, False, None, 1000000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/qwen-5Jz2blSb03vSB8pTdfJ4yppXQEp00P.svg'),
  (True, True, 'Gemini 2.0 Pro', 'Google', 'Gemini', 'Pro', '2.0', '20250205', False, True, False, None, 2097152, '{image/*,application/pdf}', None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/ai21-yI1CWcuqmijuCBQOqss8Gl5f4TM0NH.svg'),
  (True, True, 'Gemma 2 27B', 'Google', 'Gemma', '27b', '2', '', False, False, False, 27000000000, 8192, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (True, True, 'Gemma 2 9B', 'Google', 'Gemma', '9b', '2', '', True, False, False, 9000000000, 8192, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (True, True, 'Gemma 7B', 'Google', 'Gemma', '7b', '', '', True, False, False, 7000000000, 8192, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (True, True, 'Gemma 9B', 'Google', 'Gemma', '9b', '', '', False, False, False, 9000000000, 8192, None, '2023-09-01', 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (True, True, 'GPT 4o', 'OpenAI', 'GPT', '4o', '', '', False, True, False, 200000000000, 128000, '{image/*}', '2023-09-01', 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (True, True, 'GPT 4o-mini', 'OpenAI', 'GPT', '4o mini', '', '', False, False, False, None, 128000, '{image/*}', None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/databricks-nttMiUaLFk0zvkGxFiGvx86D7PjAtf.svg'),
  (True, True, 'GPT Models', 'OpenAI', 'GPT', None, '', '', False, False, False, None, 200000, None, '2023-10-01', 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/databricks-nttMiUaLFk0zvkGxFiGvx86D7PjAtf.svg'),
  (True, True, 'GPT o1', 'OpenAI', 'GPT', 'o1', '', '20241217', False, False, False, 200000000000, 200000, None, '2024-10-01', 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/deepseek-7hPhPINL13UAqe1ny30pTIY2OMrYbj.svg'),
  (True, True, 'GPT o1-mini', 'OpenAI', 'GPT', 'o1 mini', '', '20240912', True, False, False, 200000000000, 200000, None, '2024-09-12', 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (True, True, 'GPT o1-preview', 'OpenAI', 'GPT', 'o1 preview', '', '20240912', False, True, False, 200000000000, 200000, None, '2024-09-12', 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (True, True, 'GPT o3-mini', 'OpenAI', 'GPT', 'o3 mini', '', '20250131', False, True, False, None, 200000, None, '2023-10-01', 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (True, True, 'GPT-3.5 Turbo', 'OpenAI', 'GPT', 'turbo', '3.5', '20240125', False, False, False, None, 16000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/deepseek-7hPhPINL13UAqe1ny30pTIY2OMrYbj.svg'),
  (True, True, 'GPT-4 Turbo', 'OpenAI', 'GPT', 'turbo', '4', '', False, True, False, 200000000000, 128000, None, '2023-09-01', 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (True, True, 'Grok 2', 'xAI', 'Grok', None, '2', '20241212', True, False, False, None, 131072, None, '2024-12-12', 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (True, True, 'Grok 2 Beta', 'xAI', 'Grok', None, 'beta', '', True, False, False, None, 131072, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (True, True, 'Hermes 3 llama 3.1 70B', 'Nous Research', 'Hermes', 'llama 3.1 70b', '3', '', False, False, False, 70000000000, 128000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/cohere-ukaHosE0YpsqSARzTdGm8W7aho6dWn.svg'),
  (True, True, 'Jamba 1.5 Large', 'Jamba', 'Jamba', 'Large', '1.5', '', False, False, False, None, 256000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/amazon-bedrock-bpkCePwqlRFTRC19tPzXbSJ39Ntv5B.svg'),
  (True, True, 'Jamba 1.5 Mini', 'Jamba', 'Jamba', 'Mini', '1.5', '', False, False, False, None, 256000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/amazon-bedrock-bpkCePwqlRFTRC19tPzXbSJ39Ntv5B.svg'),
  (True, True, 'Llama 3 70B', 'Meta', 'Llama', '70b', '3', '', True, False, False, 70000000000, 8192, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (True, True, 'Llama 3 70B Chat HF', 'Meta', 'Llama', '70b chat hf', '3', '', False, False, False, 70000000000, 8192, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/claude-4R13TWqlQQbNC7HAM5HJTp0Zq1Hjz6.svg'),
  (True, True, 'Llama 3 8B', 'Meta', 'Llama', '8b', '3', '', True, False, False, 8000000000, 8192, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (True, True, 'Llama 3 8B Chat HF', 'Meta', 'Llama', '8b chat hf', '3', '', False, False, False, 8000000000, 8192, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/claude-4R13TWqlQQbNC7HAM5HJTp0Zq1Hjz6.svg'),
  (True, True, 'Llama 3.1 405B Instruct', 'Meta', 'Llama', '405b instruct', '3.1', '', True, True, False, 405000000000, 128000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (True, True, 'Llama 3.1 70B', 'Meta', 'Llama', '70b', '3.1', '', True, True, False, 70000000000, 128000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (True, True, 'Llama 3.1 70B Instruct', 'Meta', 'Llama', '70b instruct', '3.1', '', True, True, False, 70000000000, 128000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (True, True, 'Llama 3.1 70B Instruct Turbo', 'Meta', 'Llama', '70b instruct turbo', '3.1', '', False, False, False, 70000000000, 128000, None, '2023-09-01', 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/claude-4R13TWqlQQbNC7HAM5HJTp0Zq1Hjz6.svg'),
  (True, True, 'Llama 3.1 8B', 'Meta', 'Llama', '8b', '3.1', '', True, True, False, 8000000000, 128000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (True, True, 'Llama 3.1 8B Instant', 'Meta', 'Llama', '8b instant', '3.1', '', True, False, False, 8000000000, 128000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (True, True, 'Llama 3.1 8B Instruct', 'Meta', 'Llama', '8b instruct', '3.1', '', False, True, False, 8000000000, 128000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (True, True, 'Llama 3.1 Tulu 3 405B', 'Meta', 'Llama', 'tulu 3 405b', '3.1', '', True, True, False, 405000000000, 128000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (True, True, 'Llama 3.2 11B Vision Instruct', 'Meta', 'Llama', '11b vision instruct', '3.2', '', False, False, False, 11000000000, 131072, '{image/*}', None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/amazon-bedrock-bpkCePwqlRFTRC19tPzXbSJ39Ntv5B.svg'),
  (True, True, 'Llama 3.2 1B Instruct', 'Meta', 'Llama', '1b instruct', '3.2', '', False, True, False, 1000000000, 128000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/ai21-yI1CWcuqmijuCBQOqss8Gl5f4TM0NH.svg'),
  (True, True, 'Llama 3.2 3B Instruct', 'Meta', 'Llama', '3b instruct', '3.2', '', False, True, False, 3000000000, 128000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/ai21-yI1CWcuqmijuCBQOqss8Gl5f4TM0NH.svg'),
  (True, True, 'Llama 3.2 3B Instruct Turbo', 'Meta', 'Llama', '3b instruct turbo', '3.2', '', False, False, False, 3000000000, 131072, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/claude-4R13TWqlQQbNC7HAM5HJTp0Zq1Hjz6.svg'),
  (True, True, 'Llama 3.2 90B Vision Instruct', 'Meta', 'Llama', '90b vision instruct', '3.2', '', True, True, False, 90000000000, 128000, '{image/*}', None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (True, True, 'Llama 3.2 90B Vision Instruct Turbo', 'Meta', 'Llama', '90b vision instruct turbo', '3.2', '', True, False, False, 90000000000, 131072, '{image/*}', None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (True, True, 'Llama 3.3 70B', 'Meta', 'Llama', '70b', '3.3', '', True, True, False, 70000000000, 8192, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (True, True, 'Llama 3.3 70B Instruct', 'Meta', 'Llama', '70b instruct', '3.3', '', True, True, False, 70000000000, 128000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (True, True, 'Llama 3.3 70B Versatile', 'Meta', 'Llama', '70b versatile', '3.3', '', True, False, False, 70000000000, 128000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (True, True, 'Mistral Codestral', 'Mistral', 'Mistral', 'Codestral', '', '202405', False, False, False, None, 32000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/claude-4R13TWqlQQbNC7HAM5HJTp0Zq1Hjz6.svg'),
  (True, True, 'Mistral Large', 'Mistral', 'Mistral', 'Large', '', '202402', False, False, False, None, 32000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/claude-4R13TWqlQQbNC7HAM5HJTp0Zq1Hjz6.svg'),
  (True, True, 'Mistral Medium', 'Mistral', 'Mistral', 'Medium', '', '', False, False, False, None, 32000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/claude-4R13TWqlQQbNC7HAM5HJTp0Zq1Hjz6.svg'),
  (True, True, 'Mistral Ministral 3B', 'Mistral', 'Mistral', 'Ministral 3b', '', '202410', False, False, False, 3000000000, 128000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/claude-4R13TWqlQQbNC7HAM5HJTp0Zq1Hjz6.svg'),
  (True, True, 'Mistral Ministral 8B', 'Mistral', 'Mistral', 'Ministral 8b', '', '202410', False, False, False, 8000000000, 128000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/claude-4R13TWqlQQbNC7HAM5HJTp0Zq1Hjz6.svg'),
  (True, True, 'Mistral Mixtral 8x22B', 'Mistral', 'Mistral', 'Mixtral 8x22b', '', '', False, False, False, 176000000000, 8192, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/cohere-ukaHosE0YpsqSARzTdGm8W7aho6dWn.svg'),
  (True, True, 'Mistral Mixtral 8x7B', 'Mistral', 'Mistral', 'Mixtral 8x7b', '', '', True, False, False, 56000000000, 32768, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (True, True, 'Mistral Nemo', 'Mistral', 'Mistral', 'Nemo', '', '202407', False, False, False, None, 128000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/cohere-ukaHosE0YpsqSARzTdGm8W7aho6dWn.svg'),
  (True, True, 'Mistral Pixtral 12B', 'Mistral', 'Mistral', 'Pixtral 12b', '', '202409', False, False, False, 12000000000, 128000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/cohere-ukaHosE0YpsqSARzTdGm8W7aho6dWn.svg'),
  (True, True, 'Mistral Small', 'Mistral', 'Mistral', 'Small', '', '202501', False, False, False, 24000000000, 32000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/cohere-ukaHosE0YpsqSARzTdGm8W7aho6dWn.svg'),
  (True, True, 'MythoMax L2 13B', 'Gryphe', 'MythoMax', 'L2 13b', '', '', False, False, False, 13000000000, 4096, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/amazon-bedrock-bpkCePwqlRFTRC19tPzXbSJ39Ntv5B.svg'),
  (True, True, 'Nemotron 3.1 70B Instruct', 'NVIDIA', 'Nemotron', '70b instruct', '3.1', '', False, False, False, None, 128000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/databricks-nttMiUaLFk0zvkGxFiGvx86D7PjAtf.svg'),
  (True, True, 'Nemotron 4 340B Instruct', 'NVIDIA', 'Nemotron', '340b instruct', '4', '', False, False, False, 340000000000, 4096, None, '2023-09-01', 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/cohere-ukaHosE0YpsqSARzTdGm8W7aho6dWn.svg'),
  (True, True, 'News', 'Yupp', 'Yapp', 'News', '', '', False, False, False, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (True, True, 'Nova Pro 1', 'Amazon', 'Nova', 'Pro', '1', '', False, True, False, None, 300000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (True, True, 'Phi 3 Medium 128K Instruct', 'Microsoft', 'Phi', 'medium 128k instruct', '3', '', False, False, False, None, 128000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/claude-4R13TWqlQQbNC7HAM5HJTp0Zq1Hjz6.svg'),
  (True, True, 'Phi 3 Mini 4K Instruct', 'Microsoft', 'Phi', 'mini 4k instruct', '3', '', False, False, False, 3800000000, 4096, None, '2023-09-01', 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/claude-4R13TWqlQQbNC7HAM5HJTp0Zq1Hjz6.svg'),
  (True, True, 'Phi 3.5 mini 128K Instruct', 'Microsoft', 'Phi', 'mini 128k instruct', '3.5', '', False, False, False, None, 128000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/claude-4R13TWqlQQbNC7HAM5HJTp0Zq1Hjz6.svg'),
  (True, True, 'Phi 4', 'Microsoft', 'Phi', None, '4', '', False, False, False, None, 16384, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/claude-4R13TWqlQQbNC7HAM5HJTp0Zq1Hjz6.svg'),
  (True, True, 'Qwen 1.5 110B Chat', 'Alibaba', 'Qwen', '110b chat', '1.5', '', False, False, False, 110000000000, None, None, None, 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDgiIHZpZXdCb3g9IjAgMCA2MDAgNjAwIiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxjaXJjbGUgY3g9IjMwMCIgY3k9IjMwMCIgcj0iMzAwIiBmaWxsPSIjMDAzNDI1Ii8+PHJlY3QgeD0iNDA5LjczMyIgeT0iMzQwLjAzMSIgd2lkdGg9IjQyLjM4NiIgaGVpZ2h0PSIxNTEuNjQ4IiByeD0iMjEuMTkzIiBmaWxsPSIjZmZmIi8+PHBhdGggZD0iTTQyMi4wMDUgMTMzLjM1NGMtOC45MTYtNy41ODMtMjIuMjkxLTYuNTAzLTI5Ljg3NCAyLjQxM0wyNzMuNjk5IDI3NS4wMjFhMjEuMSAyMS4xIDAgMCAwLTUuMDAxIDEyLjI4MSAyMS40IDIxLjQgMCAwIDAtLjI1MiAzLjI3OXYxNzguMDIyYzAgMTEuNzA1IDkuNDg4IDIxLjE5MyAyMS4xOTMgMjEuMTkzczIxLjE5My05LjQ4OCAyMS4xOTMtMjEuMTkzVjI5Ni43ODRsMTEzLjU4Ny0xMzMuNTU2YzcuNTgzLTguOTE2IDYuNTAyLTIyLjI5MS0yLjQxNC0yOS44NzQiIGZpbGw9IiNmZmYiLz48cmVjdCB4PSIxMTMuOTcyIiB5PSIxMzQuMjUiIHdpZHRoPSI0Mi4zODYiIGhlaWdodD0iMTc0Ljc0NSIgcng9IjIxLjE5MyIgdHJhbnNmb3JtPSJyb3RhdGUoLTM5LjM0NCAxMTMuOTcyIDEzNC4yNSkiIGZpbGw9IiNmZmYiLz48Y2lyY2xlIGN4PSI0NjAuMTI2IiBjeT0iMjc5LjI3OCIgcj0iMjUuOTAzIiBmaWxsPSIjMDBGRjI1Ii8+PC9zdmc+'),
  (True, True, 'Qwen 1.5 72B Chat', 'Alibaba', 'Qwen', '72b chat', '1.5', '', False, False, False, 72000000000, 32768, None, '2023-09-01', 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/qwen-5Jz2blSb03vSB8pTdfJ4yppXQEp00P.svg'),
  (True, True, 'Qwen 2 72B Instruct', 'Alibaba', 'Qwen', '72b instruct', '2', '', False, False, False, 72000000000, 32768, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/qwen-5Jz2blSb03vSB8pTdfJ4yppXQEp00P.svg'),
  (True, True, 'Qwen 2.5 14B Instruct', 'Alibaba', 'Qwen', '14b instruct', '2.5', '', False, False, False, 14000000000, 1000000, None, '2023-10-01', 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDgiIHZpZXdCb3g9IjAgMCA2MDAgNjAwIiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxjaXJjbGUgY3g9IjMwMCIgY3k9IjMwMCIgcj0iMzAwIiBmaWxsPSIjMDAzNDI1Ii8+PHJlY3QgeD0iNDA5LjczMyIgeT0iMzQwLjAzMSIgd2lkdGg9IjQyLjM4NiIgaGVpZ2h0PSIxNTEuNjQ4IiByeD0iMjEuMTkzIiBmaWxsPSIjZmZmIi8+PHBhdGggZD0iTTQyMi4wMDUgMTMzLjM1NGMtOC45MTYtNy41ODMtMjIuMjkxLTYuNTAzLTI5Ljg3NCAyLjQxM0wyNzMuNjk5IDI3NS4wMjFhMjEuMSAyMS4xIDAgMCAwLTUuMDAxIDEyLjI4MSAyMS40IDIxLjQgMCAwIDAtLjI1MiAzLjI3OXYxNzguMDIyYzAgMTEuNzA1IDkuNDg4IDIxLjE5MyAyMS4xOTMgMjEuMTkzczIxLjE5My05LjQ4OCAyMS4xOTMtMjEuMTkzVjI5Ni43ODRsMTEzLjU4Ny0xMzMuNTU2YzcuNTgzLTguOTE2IDYuNTAyLTIyLjI5MS0yLjQxNC0yOS44NzQiIGZpbGw9IiNmZmYiLz48cmVjdCB4PSIxMTMuOTcyIiB5PSIxMzQuMjUiIHdpZHRoPSI0Mi4zODYiIGhlaWdodD0iMTc0Ljc0NSIgcng9IjIxLjE5MyIgdHJhbnNmb3JtPSJyb3RhdGUoLTM5LjM0NCAxMTMuOTcyIDEzNC4yNSkiIGZpbGw9IiNmZmYiLz48Y2lyY2xlIGN4PSI0NjAuMTI2IiBjeT0iMjc5LjI3OCIgcj0iMjUuOTAzIiBmaWxsPSIjMDBGRjI1Ii8+PC9zdmc+'),
  (True, True, 'Qwen 2.5 72B Instruct', 'Alibaba', 'Qwen', '72b instruct', '2.5', '', False, False, False, 72000000000, 32000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/qwen-5Jz2blSb03vSB8pTdfJ4yppXQEp00P.svg'),
  (True, True, 'Qwen 2.5 7B Instruct', 'Alibaba', 'Qwen', '7b instruct', '2.5', '', False, False, False, 7000000000, 1000000, None, '2023-10-01', 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/qwen-5Jz2blSb03vSB8pTdfJ4yppXQEp00P.svg'),
  (True, True, 'Qwen 2.5 Coder 32B Instruct', 'Alibaba', 'Qwen', 'coder 32b instruct', '2.5', '', False, False, False, 32000000000, 131072, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/qwen-5Jz2blSb03vSB8pTdfJ4yppXQEp00P.svg'),
  (True, True, 'Qwen 2.5 VL 3B Instruct', 'Alibaba', 'Qwen', 'vl 3b instruct', '2.5', '', False, False, False, 3000000000, 128000, '{image/*}', '2021-12-31', 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/qwen-5Jz2blSb03vSB8pTdfJ4yppXQEp00P.svg'),
  (True, True, 'Qwen 2.5 VL 72B Instruct', 'Alibaba', 'Qwen', 'vl 72b instruct', '2.5', '', False, False, False, 72000000000, 128000, '{image/*}', '2021-12-31', 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/qwen-5Jz2blSb03vSB8pTdfJ4yppXQEp00P.svg'),
  (True, True, 'Qwen 2.5 VL 7B Instruct', 'Alibaba', 'Qwen', 'vl 7b instruct', '2.5', '', False, False, False, 7000000000, 128000, '{image/*}', '2021-12-31', 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/qwen-5Jz2blSb03vSB8pTdfJ4yppXQEp00P.svg'),
  (True, True, 'Qwen Max', 'Alibaba', 'Qwen', 'max', '', '20250125', True, False, False, 100000000000, 32000, None, '2024-12-01', 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (True, True, 'Qwen Plus', 'Alibaba', 'Qwen', 'plus', '', '', False, False, False, None, 30000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/qwen-5Jz2blSb03vSB8pTdfJ4yppXQEp00P.svg'),
  (True, True, 'QwQ 32B Preview', 'Alibaba', 'Qwen', 'QwQ 32b preview', '', '', False, True, False, 32000000000, 32768, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (True, True, 'Sona Large Chat', 'Perplexity', 'Sonar', 'large chat', '', '', False, False, False, None, 127072, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/sambanova%20logo-DQufRZ9YknZF3bE2WoH6y0LQ2zlyuJ.webp'),
  (True, True, 'Sonar', 'Perplexity', 'Sonar', None, '', '', False, False, False, None, 127000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/deepseek-7hPhPINL13UAqe1ny30pTIY2OMrYbj.svg'),
  (True, True, 'Sonar Huge Online', 'Perplexity', 'Sonar', 'huge online', '', '', False, False, True, None, 127072, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/deepseek-7hPhPINL13UAqe1ny30pTIY2OMrYbj.svg'),
  (True, True, 'Sonar Large Online', 'Perplexity', 'Sonar', 'large online', '', '', False, False, True, None, 127072, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/deepseek-7hPhPINL13UAqe1ny30pTIY2OMrYbj.svg'),
  (True, True, 'Sonar Pro', 'Perplexity', 'Sonar', 'Pro', '', '', False, False, False, None, 200000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (True, True, 'Sonar Reasoning', 'Perplexity', 'Sonar', 'Reasoning', '', '', False, False, False, None, 127000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (True, True, 'Sonar Small Chat', 'Perplexity', 'Sonar', 'small chat', '', '', False, False, False, None, 127072, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (True, True, 'Sonar Small Online', 'Perplexity', 'Sonar', 'small online', '', '', False, False, True, None, 127072, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (True, True, 'Weather', 'Yupp', 'Yapp', 'Weather', '', '', False, False, False, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (True, True, 'Wikipedia', 'Yupp', 'Yapp', 'Wikipedia', '', '', False, False, False, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (True, True, 'Yi Large', '01-ai', 'Yi', 'Large', '', '', False, False, False, 405000000000, 32768, None, '2023-09-01', 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDgiIHZpZXdCb3g9IjAgMCA2MDAgNjAwIiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxjaXJjbGUgY3g9IjMwMCIgY3k9IjMwMCIgcj0iMzAwIiBmaWxsPSIjMDAzNDI1Ii8+PHJlY3QgeD0iNDA5LjczMyIgeT0iMzQwLjAzMSIgd2lkdGg9IjQyLjM4NiIgaGVpZ2h0PSIxNTEuNjQ4IiByeD0iMjEuMTkzIiBmaWxsPSIjZmZmIi8+PHBhdGggZD0iTTQyMi4wMDUgMTMzLjM1NGMtOC45MTYtNy41ODMtMjIuMjkxLTYuNTAzLTI5Ljg3NCAyLjQxM0wyNzMuNjk5IDI3NS4wMjFhMjEuMSAyMS4xIDAgMCAwLTUuMDAxIDEyLjI4MSAyMS40IDIxLjQgMCAwIDAtLjI1MiAzLjI3OXYxNzguMDIyYzAgMTEuNzA1IDkuNDg4IDIxLjE5MyAyMS4xOTMgMjEuMTkzczIxLjE5My05LjQ4OCAyMS4xOTMtMjEuMTkzVjI5Ni43ODRsMTEzLjU4Ny0xMzMuNTU2YzcuNTgzLTguOTE2IDYuNTAyLTIyLjI5MS0yLjQxNC0yOS44NzQiIGZpbGw9IiNmZmYiLz48cmVjdCB4PSIxMTMuOTcyIiB5PSIxMzQuMjUiIHdpZHRoPSI0Mi4zODYiIGhlaWdodD0iMTc0Ljc0NSIgcng9IjIxLjE5MyIgdHJhbnNmb3JtPSJyb3RhdGUoLTM5LjM0NCAxMTMuOTcyIDEzNC4yNSkiIGZpbGw9IiNmZmYiLz48Y2lyY2xlIGN4PSI0NjAuMTI2IiBjeT0iMjc5LjI3OCIgcj0iMjUuOTAzIiBmaWxsPSIjMDBGRjI1Ii8+PC9zdmc+'),
  (True, True, 'YouTube Transcript', 'Yupp', 'Yapp', 'YouTube Transcript', '', '', False, False, False, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (False, True, 'Claude 3.5 Sonnet', 'Claude', 'Claude', 'Sonnet', '3.5', '20240620', False, False, False, None, 200000, '{image/*,application/pdf}', '2023-09-01', 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/qwen-5Jz2blSb03vSB8pTdfJ4yppXQEp00P.svg'),
  (False, True, 'Command R', 'Cohere', 'Command', 'R', '', '', False, False, False, None, 128000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/qwen-5Jz2blSb03vSB8pTdfJ4yppXQEp00P.svg'),
  (False, True, 'Command R+', 'Cohere', 'Command', 'R+', '', '', True, False, False, None, 128000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (False, True, 'Gemini 1.5 Flash', 'Google', 'Gemini', 'Flash', '1.5', '002', True, False, False, None, 1000000, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (False, True, 'Gemini 1.5 Flash 8B', 'Google', 'Gemini', 'Flash 8b', '1.5', '', False, False, False, 8000000000, 1000000, '{image/*,application/pdf}', '2021-09-01', 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/qwen-5Jz2blSb03vSB8pTdfJ4yppXQEp00P.svg'),
  (False, True, 'Gemini 1.5 Pro', 'Google', 'Gemini', 'Pro', '1.5', '', False, True, False, 1500000000000, 2000000, '{image/*,application/pdf}', '2023-09-01', 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/ai21-yI1CWcuqmijuCBQOqss8Gl5f4TM0NH.svg'),
  (False, True, 'Gemini 1.5 Pro', 'Google', 'Gemini', 'Pro', '1.5', '002', False, True, False, None, 2000000, '{image/*,application/pdf}', None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/ai21-yI1CWcuqmijuCBQOqss8Gl5f4TM0NH.svg'),
  (False, True, 'GPT o1', 'OpenAI', 'GPT', 'o1', '', '', True, True, False, None, 200000, None, '2023-10-01', 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (False, True, 'Mistral Large', 'Mistral', 'Mistral', 'Large', '', 'latest', True, False, False, 123000000000, 128000, None, '2023-09-01', 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/gemini-RHIQEDlCoRx6dHCtGdITc5enBLT1by.svg'),
  (False, True, 'Qwen Max', 'Alibaba', 'Qwen', 'max', '', '', False, False, False, 100000000000, 32768, None, '2023-09-01', 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/qwen-5Jz2blSb03vSB8pTdfJ4yppXQEp00P.svg'),
  (False, False, '01.ai Models', '01-ai', None, None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (False, False, '01.ai Yi', '01-ai', 'Yi', None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (False, False, 'Alibaba Models', 'Alibaba', None, None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (False, False, 'Alibaba Qwen', 'Alibaba', 'Qwen', None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (False, False, 'Claude', 'Claude', 'Claude', None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (False, False, 'Claude', 'Claude', None, None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (False, False, 'Claude 3 Opus', 'Claude', 'Claude', 'Opus', '3', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (False, False, 'Claude 3.5 Sonnet', 'Claude', 'Claude', 'Sonnet', '3.5', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/Tulu3-logo-SLi8N7J6e5qeU9pqDt29zlzT1XMHoK.png'),
  (False, False, 'Claude Haiku', 'Claude', 'Claude', 'Haiku', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (False, False, 'Claude Opus', 'Claude', 'Claude', 'Opus', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (False, False, 'Claude Sonnet', 'Claude', 'Claude', 'Sonnet', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (False, False, 'Codestral', 'Mistral', 'Mistral', 'Codestral', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/openai-Fi8fMfiZMAz0tnFvMkLG9Ds6opA3M2.svg'),
  (False, False, 'Cohere Command', 'Cohere', 'Command', None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/microsoft-384pL0XSEAnJ9cT4dCnBvzh1zbJ0tS.svg'),
  (False, False, 'Cohere Models', 'Cohere', None, None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/Tulu3-logo-SLi8N7J6e5qeU9pqDt29zlzT1XMHoK.png'),
  (False, False, 'Databricks DBRX', 'Databricks', 'DBRX', None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/microsoft-384pL0XSEAnJ9cT4dCnBvzh1zbJ0tS.svg'),
  (False, False, 'Databricks Models', 'Databricks', None, None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/microsoft-384pL0XSEAnJ9cT4dCnBvzh1zbJ0tS.svg'),
  (False, False, 'DeepSeek Models', 'DeepSeek', 'DeepSeek', None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/microsoft-384pL0XSEAnJ9cT4dCnBvzh1zbJ0tS.svg'),
  (False, False, 'DeepSeek Models', 'DeepSeek', None, None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/microsoft-384pL0XSEAnJ9cT4dCnBvzh1zbJ0tS.svg'),
  (False, False, 'Gemini 1.5 Flash', 'Google', 'Gemini', 'Flash', '1.5', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/mistral-cIgeIEH6DUd9DlLlYdAn9wMFMQ81s4.svg'),
  (False, False, 'Gemini 2.0 Pro', 'Google', 'Gemini', 'Pro', '2.0', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/mistral-cIgeIEH6DUd9DlLlYdAn9wMFMQ81s4.svg'),
  (False, False, 'Gemini Flash', 'Google', 'Gemini', 'Flash', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/mistral-cIgeIEH6DUd9DlLlYdAn9wMFMQ81s4.svg'),
  (False, False, 'Gemini Flash 8B', 'Google', 'Gemini', 'Flash 8b', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/microsoft-384pL0XSEAnJ9cT4dCnBvzh1zbJ0tS.svg'),
  (False, False, 'Gemini Flash Lite', 'Google', 'Gemini', 'Flash Lite', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/microsoft-384pL0XSEAnJ9cT4dCnBvzh1zbJ0tS.svg'),
  (False, False, 'Gemini Flash Thinking', 'Google', 'Gemini', 'Flash Thinking', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/mistral-cIgeIEH6DUd9DlLlYdAn9wMFMQ81s4.svg'),
  (False, False, 'Gemini Models', 'Google', 'Gemini', None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/microsoft-384pL0XSEAnJ9cT4dCnBvzh1zbJ0tS.svg'),
  (False, False, 'Gemini Pro', 'Google', 'Gemini', 'Pro', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/mistral-cIgeIEH6DUd9DlLlYdAn9wMFMQ81s4.svg'),
  (False, False, 'Gemini Pro online', 'Google', 'Gemini', 'Pro online', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/mistral-cIgeIEH6DUd9DlLlYdAn9wMFMQ81s4.svg'),
  (False, False, 'Gemma', 'Google', 'Gemma', None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/mistral-cIgeIEH6DUd9DlLlYdAn9wMFMQ81s4.svg'),
  (False, False, 'Gemma 27B', 'Google', 'Gemma', '27b', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/mistral-cIgeIEH6DUd9DlLlYdAn9wMFMQ81s4.svg'),
  (False, False, 'Google Models', 'Google', None, None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/microsoft-384pL0XSEAnJ9cT4dCnBvzh1zbJ0tS.svg'),
  (False, False, 'GPT o1-mini', 'OpenAI', 'GPT', 'o1 mini', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/xai-DX9AeEsLL0QYBhE4yIjHPEf1v3zgyP.svg'),
  (False, False, 'GPT o1-preview', 'OpenAI', 'GPT', 'o1 preview', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/xai-DX9AeEsLL0QYBhE4yIjHPEf1v3zgyP.svg'),
  (False, False, 'GPT o3-mini', 'OpenAI', 'GPT', 'o3 mini', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/xai-DX9AeEsLL0QYBhE4yIjHPEf1v3zgyP.svg'),
  (False, False, 'GPT Turbo', 'OpenAI', 'GPT', 'turbo', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/xai-DX9AeEsLL0QYBhE4yIjHPEf1v3zgyP.svg'),
  (False, False, 'GPT-3.5 Turbo', 'OpenAI', 'GPT', 'turbo', '3.5', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/xai-DX9AeEsLL0QYBhE4yIjHPEf1v3zgyP.svg'),
  (False, False, 'Grok 2', 'xAI', 'Grok', None, '2', '', None, None, None, None, None, None, None, None),
  (False, False, 'Grok Models', 'xAI', 'Grok', None, '', '', None, None, None, None, None, None, None, None),
  (False, False, 'Gryphe Models', 'Gryphe', None, None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/mistral-cIgeIEH6DUd9DlLlYdAn9wMFMQ81s4.svg'),
  (False, False, 'Hermes llama 3.1 70B', 'Nous Research', 'Hermes', 'llama 3.1 70b', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/perplexity-ypgfj9eJ4t5KnUOta27ob70hVoKh3X.svg'),
  (False, False, 'Hermes Models', 'Nous Research', 'Hermes', None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/perplexity-ypgfj9eJ4t5KnUOta27ob70hVoKh3X.svg'),
  (False, False, 'Jamba Large', 'Jamba', 'Jamba', 'Large', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/mistral-cIgeIEH6DUd9DlLlYdAn9wMFMQ81s4.svg'),
  (False, False, 'Jamba Mini', 'Jamba', 'Jamba', 'Mini', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/mistral-cIgeIEH6DUd9DlLlYdAn9wMFMQ81s4.svg'),
  (False, False, 'Jamba Models', 'Jamba', 'Jamba', None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/mistral-cIgeIEH6DUd9DlLlYdAn9wMFMQ81s4.svg'),
  (False, False, 'Jamba Models', 'Jamba', None, None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/mistral-cIgeIEH6DUd9DlLlYdAn9wMFMQ81s4.svg'),
  (False, False, 'Llama 11B Vision Instruct', 'Meta', 'Llama', '11b vision instruct', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/mistral-cIgeIEH6DUd9DlLlYdAn9wMFMQ81s4.svg'),
  (False, False, 'Llama 1B Instruct', 'Meta', 'Llama', '1b instruct', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/mistral-cIgeIEH6DUd9DlLlYdAn9wMFMQ81s4.svg'),
  (False, False, 'Llama 3B Instruct', 'Meta', 'Llama', '3b instruct', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/mistral-cIgeIEH6DUd9DlLlYdAn9wMFMQ81s4.svg'),
  (False, False, 'Llama 3B Instruct Turbo', 'Meta', 'Llama', '3b instruct turbo', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/mistral-cIgeIEH6DUd9DlLlYdAn9wMFMQ81s4.svg'),
  (False, False, 'Llama 405B Instruct', 'Meta', 'Llama', '405b instruct', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/nousresearch-qi0XACnz8pVS5g6rkt07frBTHIvt0T.png'),
  (False, False, 'Llama 70B', 'Meta', 'Llama', '70b', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/nvidia-sK64oFOCgdZgh9Yg2HEgXdwep08L1Z.svg'),
  (False, False, 'Llama 70B Chat hf', 'Meta', 'Llama', '70b chat hf', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/nousresearch-qi0XACnz8pVS5g6rkt07frBTHIvt0T.png'),
  (False, False, 'Llama 70B Instruct', 'Meta', 'Llama', '70b instruct', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/nousresearch-qi0XACnz8pVS5g6rkt07frBTHIvt0T.png'),
  (False, False, 'Llama 70B Instruct Turbo', 'Meta', 'Llama', '70b instruct turbo', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/nousresearch-qi0XACnz8pVS5g6rkt07frBTHIvt0T.png'),
  (False, False, 'Llama 70B versatile', 'Meta', 'Llama', '70b versatile', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/nvidia-sK64oFOCgdZgh9Yg2HEgXdwep08L1Z.svg'),
  (False, False, 'Llama 8B', 'Meta', 'Llama', '8b', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/nvidia-sK64oFOCgdZgh9Yg2HEgXdwep08L1Z.svg'),
  (False, False, 'Llama 8B Chat hf', 'Meta', 'Llama', '8b chat hf', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/nvidia-sK64oFOCgdZgh9Yg2HEgXdwep08L1Z.svg'),
  (False, False, 'Llama 8B instant', 'Meta', 'Llama', '8b instant', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/nvidia-sK64oFOCgdZgh9Yg2HEgXdwep08L1Z.svg'),
  (False, False, 'Llama 8B Instruct', 'Meta', 'Llama', '8b instruct', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/nvidia-sK64oFOCgdZgh9Yg2HEgXdwep08L1Z.svg'),
  (False, False, 'Llama 90B Vision Instruct', 'Meta', 'Llama', '90b vision instruct', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/openai-Fi8fMfiZMAz0tnFvMkLG9Ds6opA3M2.svg'),
  (False, False, 'Llama 90B Vision Instruct Turbo', 'Meta', 'Llama', '90b vision instruct turbo', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/openai-Fi8fMfiZMAz0tnFvMkLG9Ds6opA3M2.svg'),
  (False, False, 'Llama Models', 'Meta', 'Llama', None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/mistral-cIgeIEH6DUd9DlLlYdAn9wMFMQ81s4.svg'),
  (False, False, 'Llama Tulu 3 405B', 'Meta', 'Llama', 'tulu 3 405b', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/openai-Fi8fMfiZMAz0tnFvMkLG9Ds6opA3M2.svg'),
  (False, False, 'Meta Models', 'Meta', None, None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/mistral-cIgeIEH6DUd9DlLlYdAn9wMFMQ81s4.svg'),
  (False, False, 'Microsoft Models', 'Microsoft', None, None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/openai-Fi8fMfiZMAz0tnFvMkLG9Ds6opA3M2.svg'),
  (False, False, 'Mistral Large', 'Mistral', 'Mistral', 'Large', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/openai-Fi8fMfiZMAz0tnFvMkLG9Ds6opA3M2.svg'),
  (False, False, 'Mistral Ministral 3B', 'Mistral', 'Mistral', 'Ministral 3b', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/openai-Fi8fMfiZMAz0tnFvMkLG9Ds6opA3M2.svg'),
  (False, False, 'Mistral Ministral 8B', 'Mistral', 'Mistral', 'Ministral 8b', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/openai-Fi8fMfiZMAz0tnFvMkLG9Ds6opA3M2.svg'),
  (False, False, 'Mistral Models', 'Mistral', 'Mistral', None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/openai-Fi8fMfiZMAz0tnFvMkLG9Ds6opA3M2.svg'),
  (False, False, 'Mistral Models', 'Mistral', None, None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/openai-Fi8fMfiZMAz0tnFvMkLG9Ds6opA3M2.svg'),
  (False, False, 'Mistral Nemo', 'Mistral', 'Mistral', 'Nemo', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/openai-Fi8fMfiZMAz0tnFvMkLG9Ds6opA3M2.svg'),
  (False, False, 'Mistral Pixtral 12B', 'Mistral', 'Mistral', 'Pixtral 12b', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/openai-Fi8fMfiZMAz0tnFvMkLG9Ds6opA3M2.svg'),
  (False, False, 'Mistral Small', 'Mistral', 'Mistral', 'Small', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/perplexity-ypgfj9eJ4t5KnUOta27ob70hVoKh3X.svg'),
  (False, False, 'MythoMax Models', 'Gryphe', 'MythoMax', None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/mistral-cIgeIEH6DUd9DlLlYdAn9wMFMQ81s4.svg'),
  (False, False, 'Nemotron 340B Instruct', 'NVIDIA', 'Nemotron', '340b instruct', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/perplexity-ypgfj9eJ4t5KnUOta27ob70hVoKh3X.svg'),
  (False, False, 'Nemotron 70B Instruct', 'NVIDIA', 'Nemotron', '70b instruct', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/perplexity-ypgfj9eJ4t5KnUOta27ob70hVoKh3X.svg'),
  (False, False, 'Nemotron Models', 'NVIDIA', 'Nemotron', None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/perplexity-ypgfj9eJ4t5KnUOta27ob70hVoKh3X.svg'),
  (False, False, 'Nous Research Models', 'Nous Research', None, None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/perplexity-ypgfj9eJ4t5KnUOta27ob70hVoKh3X.svg'),
  (False, False, 'Nova', 'Amazon', 'Nova', None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (False, False, 'Nova', 'Amazon', None, None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (False, False, 'Nova Pro', 'Amazon', 'Nova', 'Pro', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (False, False, 'NVIDIA Models', 'NVIDIA', None, None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/perplexity-ypgfj9eJ4t5KnUOta27ob70hVoKh3X.svg'),
  (False, False, 'OpenAI Models', 'OpenAI', None, None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/perplexity-ypgfj9eJ4t5KnUOta27ob70hVoKh3X.svg'),
  (False, False, 'Perplexity Models', 'Perplexity', None, None, '', '', None, None, None, None, None, None, None, None),
  (False, False, 'Phi Medium 128k Instruct', 'Microsoft', 'Phi', 'medium 128k instruct', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/openai-Fi8fMfiZMAz0tnFvMkLG9Ds6opA3M2.svg'),
  (False, False, 'Phi mini 128k Instruct', 'Microsoft', 'Phi', 'mini 128k instruct', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/openai-Fi8fMfiZMAz0tnFvMkLG9Ds6opA3M2.svg'),
  (False, False, 'Phi mini 4k Instruct', 'Microsoft', 'Phi', 'mini 4k instruct', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/openai-Fi8fMfiZMAz0tnFvMkLG9Ds6opA3M2.svg'),
  (False, False, 'Phi Models', 'Microsoft', 'Phi', None, '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/openai-Fi8fMfiZMAz0tnFvMkLG9Ds6opA3M2.svg'),
  (False, False, 'Qwen 110B Chat', 'Alibaba', 'Qwen', '110b chat', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (False, False, 'Qwen 14B Instruct', 'Alibaba', 'Qwen', '14b instruct', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (False, False, 'Qwen 72B Chat', 'Alibaba', 'Qwen', '72b chat', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (False, False, 'Qwen 72B Instruct', 'Alibaba', 'Qwen', '72b instruct', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (False, False, 'Qwen 7B Instruct', 'Alibaba', 'Qwen', '7b instruct', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (False, False, 'Qwen coder 32B Instruct', 'Alibaba', 'Qwen', 'coder 32b instruct', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (False, False, 'Qwen VL 3B Instruct', 'Alibaba', 'Qwen', 'vl 3b instruct', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (False, False, 'Qwen VL 72B Instruct', 'Alibaba', 'Qwen', 'vl 72b instruct', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (False, False, 'Qwen VL 7B Instruct', 'Alibaba', 'Qwen', 'vl 7b instruct', '', '', None, None, None, None, None, None, None, 'https://7s8qtap8fabg3zyw.public.blob.vercel-storage.com/meta-DLQGfcPlz3g39tXl4ivbzb2fVgTlni.svg'),
  (False, False, 'xAI Models', 'xAI', None, None, '', '', None, None, None, None, None, None, None, None),
  (False, False, 'Yapps', 'Yupp', 'Yapp', None, '', '', None, None, None, None, None, None, None, None),
  (False, False, 'Yapps', 'Yupp', None, None, '', '', None, None, None, None, None, None, None, None),
]


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('language_model_taxonomy', 'parameter_count',
               existing_type=sa.INTEGER(),
               type_=sa.BigInteger(),
               existing_nullable=True)
    op.alter_column('language_model_taxonomy', 'context_window_tokens',
               existing_type=sa.INTEGER(),
               type_=sa.BigInteger(),
               existing_nullable=True)

    # ### commands auto generated by Alembic - please adjust! ###
    with Session(op.get_bind()) as session:
        # insert all new rows in the language_model_taxonomy table
        insert_query = sa.text(
                """
                INSERT INTO language_model_taxonomy (
                    language_model_taxonomy_id, created_at, status,
                    is_pickable, is_leaf_node, taxo_label, model_publisher, model_family, model_class, model_version, model_release,
                    is_strong, is_pro, is_live, parameter_count, context_window_tokens, supported_attachment_mime_types,
                    knowledge_cutoff_date, avatar_url)
                VALUES (
                    GEN_RANDOM_UUID(), NOW(), 'ACTIVE',
                    :is_pickable, :is_leaf_node, :taxo_label, :model_publisher, :model_family, :model_class, :model_version, :model_release,
                    :is_strong, :is_pro, :is_live, :parameter_count, :context_window_tokens, :supported_attachment_mime_types,
                    :knowledge_cutoff_date, :avatar_url
                )
                """)
        for (is_pickable, is_leaf_node, taxo_label, model_publisher, model_family, model_class, model_version, model_release,
             is_strong, is_pro, is_live, parameter_count, context_window_tokens, supported_attachment_mime_types,
             knowledge_cutoff_date, avatar_url) in TAXONOMY_DATA:
                        
            session.exec(insert_query, params={
                "is_pickable": is_pickable,
                "is_leaf_node": is_leaf_node,
                "taxo_label": taxo_label,
                "model_publisher": model_publisher,
                "model_family": model_family,
                "model_class": model_class,
                "model_version": model_version,
                "model_release": model_release,
                "is_strong": is_strong,
                "is_pro": is_pro,
                "is_live": is_live,
                "parameter_count": parameter_count,
                "context_window_tokens": context_window_tokens,
                "supported_attachment_mime_types": supported_attachment_mime_types,
                "knowledge_cutoff_date": knowledge_cutoff_date,
                "avatar_url": avatar_url
            })
        session.commit()

        # update the language_model table and link to the right taxonomy id, this is the best effort.
        # the Null and Empty will match
        update_query = sa.text(
            """
                UPDATE language_models lm
                SET taxonomy_id = lmt.language_model_taxonomy_id
                FROM language_model_taxonomy lmt
                WHERE 
                COALESCE(NULLIF(lmt.model_family, ''), '') = COALESCE(NULLIF(lm.family, ''), '')
                AND COALESCE(NULLIF(lmt.model_class, ''), '') = COALESCE(NULLIF(lm.model_class, ''), '')
                AND COALESCE(NULLIF(lmt.model_version, ''), '') = COALESCE(NULLIF(lm.model_version, ''), '')
                AND COALESCE(NULLIF(lmt.model_release, ''), '') = COALESCE(NULLIF(lm.model_release, ''), '');
            """
        )
        session.exec(update_query)
        session.commit()
    # ### end Alembic commands ###


def downgrade() -> None:
    with Session(op.get_bind()) as session:
        # delete all taxonomy id in the language_models table
        session.exec(sa.text("UPDATE language_models SET taxonomy_id = NULL"))
        # delete the taxonomy table rows themselves.
        session.exec(sa.text("DELETE FROM language_model_taxonomy"))
        session.commit()

    pass
    # ### end Alembic commands ###
