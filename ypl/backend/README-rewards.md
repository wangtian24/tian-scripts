# Reward Rules System

## TL;DR
Reward rules are defined in `data/reward_rules.yml`. To update rules:
1. Modify the YAML file
2. Run the [refresh_reward_rules.yml](https://github.com/yupp-ai/yupp-mind/actions/workflows/refresh_reward_rules.yml) action

## Overview
The reward rules system distributes points for user actions by matching action properties against defined rules:
- **Amount Rules**: Define reward amounts/ranges
- **Probability Rules**: Define likelihood of reward distribution

Rules are stored in `data/reward_rules.yml` and synced to SQL tables (`reward_amount_rules` and `reward_probability_rules`).

### Rule Evaluation
- Rules are evaluated in priority order (highest first)
- First matching rule determines reward outcome
- Rules auto-reload from database every few minutes

## Configuration

### Global Constants
```yaml
constants:
  daily_points_limit: 50000
  weekly_points_limit: 100000
  monthly_points_limit: 200000
  zero_turn_based_reward_probability: 0.25
  min_ever_high_value_reward_amount: 15
```

### Rule Properties
- `name`: Unique identifier
- `priority`: Evaluation order (higher = earlier)
- `is_active`: Enable/disable rule
- `action_type`: Applicable action (TURN, FEEDBACK, QT_EVAL)
- `conditions`: Rule matching logic
- `probability` or `min/max_value`: Reward parameters

## Rule Updates
Rules are refreshed when:
1. New deployments occur via `deploy.yml` action
2. Rule file changes (staging only)
3. Manual refresh via `refresh_reward_rules.yml` action

## Database Queries

### View Active Rules
```sql
-- Amount Rules
SELECT *
FROM reward_amount_rules rar 
WHERE is_active = true
AND deleted_at IS NULL
ORDER BY action_type, priority DESC;

-- Probability Rules
SELECT *
FROM reward_probability_rules rpr
WHERE is_active = true
AND deleted_at IS NULL
ORDER BY action_type, priority DESC;
```

### View Rule Matches
```sql
SELECT *
FROM reward_action_logs ral
JOIN turns t ON ral.turn_id = t.turn_id 
LEFT JOIN rewards r ON r.reward_id = ral.associated_reward_id
LEFT JOIN reward_amount_rules rar ON r.reward_amount_rule_id = rar.reward_amount_rule_id
LEFT JOIN reward_probability_rules rpr ON r.reward_probability_rule_id = rpr.reward_probability_rule_id
WHERE ral.action_type = 'TURN'
ORDER BY ral.created_at DESC
LIMIT 10;
```
