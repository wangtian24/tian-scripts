#!/bin/bash

# Check if output directory is provided
if [ -n "$1" ] && [[ "$1" != "-c" ]]; then
  OUTPUT_DIR="$1"
  shift
else
  # Default directory to store model lists
  OUTPUT_DIR="./llm-monitor-log"
fi

mkdir -p "$OUTPUT_DIR"

# Set default provider to "all" if no argument provided
MODEL_PROVIDER="all"
if [ -n "$1" ]; then
  # Convert argument to lowercase for case-insensitive matching
  MODEL_PROVIDER=$(echo "$1" | tr '[:upper:]' '[:lower:]')
fi

fetch_and_compare() {
  local provider=$1
  local command=$2
  local output_file="$OUTPUT_DIR/${provider}_models.txt"
  local temp_file="$OUTPUT_DIR/${provider}_models_new.txt"
  
  echo "Fetching $provider models..."
  # Execute command and store in temporary file
  eval "$command" > "$temp_file"
  
  # Check if the command succeeded by checking if the temp file has content
  if [ ! -s "$temp_file" ]; then
    echo "Error: Failed to fetch $provider models. Keeping previous data."
    rm -f "$temp_file"
    return 1
  fi
  
  # Check if previous file exists
  if [ -f "$output_file" ]; then
    # Compare with previous results
    if ! diff -q "$output_file" "$temp_file" >/dev/null; then
      echo "Changes detected in $provider models:"
      # Store diff output in a variable
      diff_output=$(diff "$output_file" "$temp_file")
      
      # Display diff on screen
      echo "$diff_output"
      
      # Send diff to Slack
      # Escape quotes and other special characters in diff_output for JSON
      escaped_diff=$(echo "$diff_output" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g' | sed 's/\n/\\n/g')
      slack_message=":rainbow: *$provider* just added some new models:\\n\`\`\`\\n$escaped_diff\\n\`\`\`"
      
      curl -s -X POST "https://slack.com/api/chat.postMessage" \
        -H "Authorization: Bearer $SLACK_MODEL_MANAGEMENT_APP_BOT_TOKEN" \
        -H "Content-Type: application/json" \
        --data "{
          \"channel\": \"#alert-model-discovery\",
          \"text\": \"$slack_message\"
        }"
    else
      echo "No changes detected in $provider models."
    fi
  else
    echo "First run for $provider models. Created initial file."
  fi
  
  # Update the file with new results only if we have valid data
  mv "$temp_file" "$output_file"
  
  # Display current models
  cat "$output_file"
}

fetch_openai() {
  fetch_and_compare "openai" "curl -s https://api.openai.com/v1/models -H \"Authorization: Bearer \$OPENAI_API_KEY\" | grep \"\\\"id\\\"\" | sort"
}

fetch_google() {
  fetch_and_compare "google" "curl -s -X GET \"https://generativelanguage.googleapis.com/v1/models?key=\$GOOGLE_API_KEY\" | grep \"\\\"name\\\"\" | sort"
}

fetch_openrouter() {
  fetch_and_compare "openrouter" "curl -s -X GET \"https://openrouter.ai/api/v1/models\" -H \"Authorization: Bearer \$OPENROUTER_API_KEY\" | jq | grep \"\\\"id\\\"\" | sort"
}

fetch_anthropic() {
  fetch_and_compare "anthropic" "curl -s -X GET \"https://api.anthropic.com/v1/models\" -H \"anthropic-version: 2023-06-01\" -H \"x-api-key: \$ANTHROPIC_API_KEY\" | jq | grep \"\\\"id\\\"\" | sort"
}

fetch_groq() {
  fetch_and_compare "groq" "curl -s -X GET \"https://api.groq.com/openai/v1/models\" -H \"Authorization: Bearer \$GROQ_API_KEY\" | jq | grep \"\\\"id\\\"\" | sort"
}

fetch_together() {
  fetch_and_compare "together" "curl -s -X GET \"https://api.together.xyz/v1/models\" -H \"Authorization: Bearer \$TOGETHERAI_API_KEY\" | jq | grep \"\\\"id\\\"\" | sort"
}

fetch_deepseek() {
  fetch_and_compare "deepseek" "curl -s -X GET \"https://api.deepseek.com/models\" -H \"Authorization: Bearer \$DEEPSEEK_API_KEY\" | jq | grep \"\\\"id\\\"\" | sort"
}

fetch_mistral() {
  fetch_and_compare "mistral" "curl -s -X GET \"https://api.mistral.ai/v1/models\" -H \"Authorization: Bearer \$MISTRAL_API_KEY\" | jq | grep \"\\\"id\\\"\" | sort"
}

case "$MODEL_PROVIDER" in
  "openai")
    fetch_openai
    ;;
  "google")
    fetch_google
    ;;
  "openrouter")
    fetch_openrouter
    ;;
  "anthropic")
    fetch_anthropic
    ;;
  "groq")
    fetch_groq
    ;;
  "together")
    fetch_together
    ;;
  "deepseek")
    fetch_deepseek
    ;;
  "mistral")
    fetch_mistral
    ;;
  "all")
    fetch_openai
    fetch_google
    fetch_openrouter
    fetch_anthropic
    fetch_groq
    fetch_together
    fetch_deepseek
    fetch_mistral
    ;;  
  *)
    echo "Unknown provider: $MODEL_PROVIDER"
    echo "Supported providers: openai, google, openrouter, anthropic, groq, together, deepseek, mistral, all"
    echo "Usage: $0 [OUTPUT_DIR] [PROVIDER]"
    exit 1
    ;;
esac
