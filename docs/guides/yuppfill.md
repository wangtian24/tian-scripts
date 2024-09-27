# Yuppfill: LLM-Generated Backfill for Yupp

_Haven't installed Yupp-LLMs as a Python package? Follow the instructions in the [main README](/README.md)._

## Introduction

Yuppfill is a suite of library routines and CLI utilities that generate synthetic user prompts, responses, and
LLM-pair evaluations for Yupp. A few use cases include but are not limited to
- Generating and inserting backfill data into the Yupp database, resolving the cold-start problem 
- Autocompleting and suggesting user prompts for Yupp users
- Generating synthetic user prompts and responses for testing and evaluation

This guide covers the first case, the most common one.

## Using Yuppfill

The high-level workflow for Yuppfill is as follows:
1. Fine-tune an LLM (presently `gpt-4o-mini`) on WildChat to generate user prompts, optionally conditioned on personas.
2. Create a configuration file specifying the number of prompts to generate, the LLM responders to use, the number
  of turns per conversation, the personas, etc.
3. Run a CLI utility to generate conversations, writing to a JSON file.
4. Run a CLI utility to judge the conversations using the LLMs specified in the configuration file.
5. Run another CLI utility to bulk insert the JSON file into the Yupp database.

Let's go through each step in detail.

### 1. Fine-tuning the LLM

The fine-tuning routines are currently in the Jupyter notebook [finetune_wildchat_openai.ipynb](/notebooks/finetune_wildchat_openai.ipynb)
for now. The notebook is self-explanatory and should be run in a Jupyter environment with the proper dependencies
installed. You should check that we have sufficient OpenAI credit. Note that the best results are obtained with
personaless fine-tuning on WildChat using at least 5k examples, two epochs, and some deduplication. At the end of the
notebook, you should receive the name of the fine-tuned model from OpenAI in the format `ft:...:yupp::...`.

### 2. Creating a Configuration File

The configuration file is a JSON file that specifies the parameters for the backfill generation. Here is an example:

```json
{
        "use_personas": false,  // Whether to use personas; recommended is false      
        "personas": [],  // list of persona objects with the format `{"persona": str, "interests": str[], "style": str}`, e.g.,
`{"persona": "scientist", "interests": ["physics", "math"], "style": "formal"}`
        "generate_num_personas": 0,  // Number of personas to generate
        "persona_llm_provider": "openai",
        "persona_llm_name": "gpt-4o-mini",
        "persona_llm_api_key": "...",  // fill in
        "user_llm_provider": "openai",
        "user_llm_name": "ft:gpt-4o-mini-2024-07-18:yupp::A7H2sxeE",  // the fine-tuned LLM from the previous step
        "user_llm_temperature": 0.7,  // the temperature to use for sampling from the user LLM; 0.7 is recommended
        "user_llm_api_key": "...",  // fill in
        "eval_llms": [
                {
                        "info": {
                                "provider": "openai",
                                "model": "gpt-4o-mini",
                                "api_key": "..."
                        },
                        "sample_weight": 8  // the relative weight of the LLM to be sampled at random
                }, {
                        "info": {
                                "provider": "openai",
                                "model": "gpt-4o",
                                "api_key": "..."
                        },
                        "sample_weight": 2
                }
        ],
        "num_turns_min": 2,  // the number of minimum turns to generate
        "num_turns_max": 2  // the number of maximum turns to generate
}
```

### 3. Generating Conversations and User Judgements

Check again that enough credit remains in OpenAI. A rough rule of thumb is $0.01 per example (assuming GPT-4o) -- use
our cost estimation utility `ypl.cli estimate-cost` for a better estimate. Afterwards, run the CLI utility as follows:
```bash
python -um ypl.cli synthesize-backfill-data -o </path/to/output.json> -c </path/to/config.json> -n <number of examples>
```

After it finishes, you should spotcheck the JSON file before further processing.

### 4. Judging the Conversations

Create this configuration file for the judgement step:
```json
{
    "llms": [  // the LLMs to use for judging. The utility will always use LLMs not in the conversation
        {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "your-api-key-here",
            "temperature": 0.7
        }, {
            "provider": "anthropic",
            "model": "claude-2",
            "api_key": "your-api-key-here",
            "temperature": 0.5
        }
    ],
    "choice_strategy": "min_cost",  // the strategy to use for routing; "min_cost" is recommended. "random" is the other
    "timeout": 5  // the timeout in seconds
}
```

Then run the CLI utility
```bash
python -um ypl.cli judge-yupp-llm-outputs -c judge-config.json -i </path/to/output.json> -o </path/to/judged-output.json>  --limit <number of examples to judge>
```

### 5. Bulk Inserting the JSON File into the Yupp Database

Use the CLI utility as follows, making sure to populate `.env` with the correct credentials:
```bash
python -um ypl.cli convert-backfill-data -c </path/to/config.json> -it json -i </path/to/judged-output.json>  -ot db -n <number of examples to insert>
```

Congrats, you've successfully backfilled the Yupp database! You may view the conversations on https://chaos.yupp.ai or in DBeaver.
