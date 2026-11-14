| Entity | Feature | Type | Definition | Source |
|---|---|---|---|---|
| user | activity_count | int | Number of training interactions | Training history |
| user | average_rating | float | Mean training rating | Training history |
| user | days_since_last_interaction | float | Cutoff minus latest training timestamp in days | Training history |
| user | preferred_category | string | Category with most positive training interactions; Unknown fallback | Training positives and item metadata |
| user_category | preference | float | Smoothed category positive share; numerator count plus 1, denominator positives plus category count | Training positives only |
| item | interaction_count | int | Number of training interactions | Training history |
| item | average_rating | float | Mean training rating | Training history |
| item | positive_count | int | Training ratings at or above positive threshold | Training history |
| item | category | string | Cleaned item category | Static catalog metadata |
| interaction | is_positive | int | Rating at or above fixed positive threshold | Training history |
| interaction | days_before_cutoff | float | Cutoff minus interaction timestamp in days | Training history |
