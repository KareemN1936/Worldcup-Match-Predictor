# Automated World Cup Match Predictor

## 1. Project Summary

This project is an educational Machine Learning project that predicts FIFA World Cup match outcomes.

The model should output:

```text
Team A Win %
Draw %
Team B Win %
```

Example:

```text
France vs Senegal

France Win: 52%
Draw: 26%
Senegal Win: 22%
```

The goal is not to build a gambling/betting tool. Do not include betting odds, betting markets, or bookmaker data. This is a football analytics and Machine Learning learning project.

The user is new to Machine Learning, so every implementation should be beginner-friendly, clean, modular, and explained.

---

## 2. Core Philosophy

Do not build a huge broken “supercomputer.”

Build a serious but manageable system in phases:

```text
Phase 1: Historical match predictor
Phase 2: Add Elo and recent form
Phase 3: Add automated World Cup fixtures
Phase 4: Add tournament squad features
Phase 5: Add Game 1 lineups and stats
Phase 6: Add injuries, suspensions, rest, and fatigue
Phase 7: Add dashboard only after the model works
```

Important rule:

```text
Data pipeline first.
Simple model second.
Advanced features third.
Dashboard last.
```

Do not start with UI.

Do not manually collect data.

Do not scrape websites unless the site allows it and the scraper is reliable.

Do not fake unavailable stats like xG or market value.

If a feature is missing from the API, handle it safely as missing or skip it.

---

## 3. Main Data Provider Rule

Use one primary football API at first.

Recommended primary provider:

```text
API-Football / API-Sports
```

Use it for:

```text
fixtures
teams
players
squads if available
lineups
formations
fixture statistics
player statistics
injuries
suspensions if available
```

Backup provider:

```text
Sportmonks
```

Only use Sportmonks if API-Football does not provide enough World Cup coverage.

Important:

Do not mix API providers early.

Reason:

Different providers use different IDs for:

```text
teams
players
fixtures
competitions
seasons
```

Mixing sources too early creates mapping problems.

---

## 4. Historical Data Source

For historical international match results, use a Kaggle international football results dataset.

The historical data should include:

```text
date
home_team
away_team
home_score
away_score
tournament
city
country
neutral
```

This data trains the base model.

The project should support loading historical data from a local CSV file inside:

```text
data/raw/historical_matches.csv
```

Later, the project can automate downloading through the Kaggle API if credentials are set up.

---

## 5. No Manual Data Collection Requirement

The user does not want manual data collection.

Allowed manual steps:

```text
1. Create API account
2. Add API key to .env
3. Download historical dataset once if Kaggle API is not configured
4. Run scripts
```

Not allowed:

```text
Manually typing squad lists
Manually typing player stats
Manually copying lineups
Manually copying match stats
Manually copying injuries
Manually copying market values
```

If a data point cannot be automated, either:

```text
1. Exclude it from the first version
2. Create a script for it
3. Add a TODO explaining what provider is needed
```

---

## 6. Project Folder Structure

Create this exact structure:

```text
worldcup-match-predictor/
│
├── data/
│   ├── raw/
│   │   ├── historical_matches.csv
│   │   ├── fixtures.csv
│   │   ├── teams.csv
│   │   ├── squads.csv
│   │   ├── lineups.csv
│   │   ├── team_match_stats.csv
│   │   ├── player_match_stats.csv
│   │   ├── injuries.csv
│   │   └── raw_api_json/
│   │
│   ├── processed/
│   │   ├── elo_history.csv
│   │   ├── team_form_features.csv
│   │   ├── squad_features.csv
│   │   ├── lineup_features.csv
│   │   ├── tournament_features.csv
│   │   ├── injury_features.csv
│   │   └── training_dataset.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── model_metadata.json
│
├── notebooks/
│   ├── 01_data_check.ipynb
│   ├── 02_feature_engineering_check.ipynb
│   └── 03_model_evaluation.ipynb
│
├── reports/
│   ├── evaluation_report.txt
│   ├── feature_importance.csv
│   └── sample_predictions.csv
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── api_client.py
│   ├── collect_historical_matches.py
│   ├── collect_fixtures.py
│   ├── collect_teams.py
│   ├── collect_squads.py
│   ├── collect_lineups.py
│   ├── collect_match_stats.py
│   ├── collect_player_stats.py
│   ├── collect_injuries.py
│   ├── build_elo.py
│   ├── build_features.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict_match.py
│   └── run_pipeline.py
│
├── tests/
│   ├── test_elo.py
│   ├── test_features.py
│   └── test_prediction.py
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── PROJECT_CONTEXT.md
```

---

## 7. Required Python Packages

Use Python.

Create `requirements.txt`:

```text
pandas
numpy
scikit-learn
xgboost
requests
python-dotenv
joblib
matplotlib
```

Optional later:

```text
streamlit
sqlalchemy
pytest
```

Use these packages for:

```text
pandas        → data cleaning and feature engineering
numpy         → calculations
requests      → API calls
python-dotenv → loading API keys
scikit-learn  → baseline models and evaluation
xgboost       → main model
joblib        → saving/loading trained models
matplotlib    → simple charts
pytest        → tests
streamlit     → optional dashboard later
```

---

## 8. Environment Variables

Create `.env.example`:

```text
FOOTBALL_API_KEY=your_api_key_here
FOOTBALL_API_BASE_URL=https://v3.football.api-sports.io
```

Create `.env` locally but do not commit it.

`.gitignore` must include:

```text
.env
__pycache__/
*.pkl
data/raw/raw_api_json/
```

The code must never hardcode API keys.

---

## 9. Data Tables

### 9.1 historical_matches.csv

Purpose:

Train the base model on previous international matches.

Required columns:

```text
match_id
date
home_team
away_team
home_score
away_score
tournament
city
country
neutral
```

Create target later:

```text
result
```

Encoding:

```text
0 = Team A loss
1 = Draw
2 = Team A win
```

Important:

The model should treat the first listed team as `team_a`.

For historical matches:

```text
team_a = home_team
team_b = away_team
```

For neutral World Cup games, home/away is not true home advantage, so include `neutral`.

---

### 9.2 fixtures.csv

Purpose:

Store current World Cup fixtures.

Required columns:

```text
fixture_id
date
competition
season
round
group
home_team_id
away_team_id
home_team
away_team
stadium
city
country
status
```

Use this file to know what matches need predictions.

---

### 9.3 teams.csv

Purpose:

Stable team ID and team-name mapping.

Required columns:

```text
team_id
team_name
country
fifa_code
api_provider
```

Important:

Team names are messy.

Examples:

```text
USA
United States
United States of America
```

Use one standardized name everywhere.

---

### 9.4 squads.csv

Purpose:

Store official tournament squad players.

Required columns:

```text
team_id
team_name
player_id
player_name
age
position
club
league
nationality
caps
international_goals
market_value
player_rating
season_minutes
season_appearances
```

Not every API will provide all of these.

Rules:

```text
If market_value is missing, do not fake it.
If caps are missing, leave null.
If player_rating is missing, leave null.
If season_minutes are missing, leave null.
```

The feature builder must handle missing values safely.

---

### 9.5 lineups.csv

Purpose:

Store starting XI and substitutes, especially from Game 1.

Required columns:

```text
fixture_id
team_id
team_name
player_id
player_name
is_starting
is_substitute
position
formation
coach
shirt_number
grid_position
```

Game 1 lineups matter because they show who the coach actually trusts.

---

### 9.6 team_match_stats.csv

Purpose:

Store team-level match stats.

Required columns:

```text
fixture_id
team_id
team_name
opponent_id
opponent_name
goals_for
goals_against
shots
shots_on_target
possession
passes
pass_accuracy
corners
fouls
yellow_cards
red_cards
offsides
saves
xg
xga
```

If xG is unavailable:

```text
Leave xg and xga as null.
Do not invent xG.
```

---

### 9.7 player_match_stats.csv

Purpose:

Store player-level match stats.

Required columns:

```text
fixture_id
team_id
team_name
player_id
player_name
minutes
rating
position
is_substitute
goals
assists
shots_total
shots_on_target
passes_total
passes_key
pass_accuracy
tackles
interceptions
duels_total
duels_won
dribbles_attempts
dribbles_success
fouls_drawn
fouls_committed
yellow_card
red_card
```

Do not feed every player row directly into the beginner model.

Aggregate player stats into team-level features.

---

### 9.8 injuries.csv

Purpose:

Store injured, suspended, or questionable players.

Required columns:

```text
fixture_id
team_id
team_name
player_id
player_name
reason
status
date_updated
```

Useful statuses:

```text
injured
suspended
questionable
missing fixture
```

---

## 10. Feature Engineering Rules

Most model features should be difference features.

Formula:

```text
feature_diff = team_a_feature - team_b_feature
```

Examples:

```text
elo_diff = team_a_elo - team_b_elo
points_last_5_diff = team_a_points_last_5 - team_b_points_last_5
squad_caps_diff = team_a_squad_caps - team_b_squad_caps
```

Why?

The model should learn the advantage of Team A over Team B.

---

## 11. Target Variable

For each match:

```text
if team_a_goals > team_b_goals:
    result = 2

elif team_a_goals == team_b_goals:
    result = 1

else:
    result = 0
```

Meaning:

```text
0 = Team A loss
1 = Draw
2 = Team A win
```

Prediction output should be transformed into:

```text
Team A Win %
Draw %
Team B Win %
```

---

## 12. Historical Features

For each team before a match, calculate:

```text
elo_rating
matches_played_last_2_years
win_rate_last_2_years
draw_rate_last_2_years
loss_rate_last_2_years
goals_for_per_match_last_2_years
goals_against_per_match_last_2_years
goal_difference_per_match_last_2_years
clean_sheet_rate_last_2_years
failed_to_score_rate_last_2_years
```

Create match features:

```text
elo_diff
win_rate_2y_diff
goals_for_2y_diff
goals_against_2y_diff
goal_difference_2y_diff
clean_sheet_rate_diff
failed_to_score_rate_diff
```

Highest-priority historical features:

```text
elo_diff
goal_difference_2y_diff
goals_against_2y_diff
win_rate_2y_diff
```

---

## 13. Recent Form Features

Recent form is one of the most important parts.

Calculate before each match:

```text
last 5 matches
last 10 matches
```

For each team:

```text
points_last_5
points_last_10
wins_last_5
draws_last_5
losses_last_5
goals_for_last_5
goals_against_last_5
goal_difference_last_5
clean_sheets_last_5
failed_to_score_last_5
```

Points system:

```text
Win  = 3
Draw = 1
Loss = 0
```

Create difference features:

```text
points_last_5_diff
points_last_10_diff
goal_difference_last_5_diff
goals_for_last_5_diff
goals_against_last_5_diff
clean_sheets_last_5_diff
failed_to_score_last_5_diff
```

Highest-priority form features:

```text
points_last_5_diff
points_last_10_diff
goal_difference_last_5_diff
goals_against_last_5_diff
```

---

## 14. Match Importance

Not all matches are equal.

Create:

```text
match_importance
```

Suggested weights:

```text
World Cup knockout: 5.0
World Cup group: 4.5
Continental tournament knockout: 4.0
Continental tournament group: 3.5
World Cup qualifier: 3.0
Nations League: 2.5
Friendly: 1.0
Other: 1.5
```

Use this for weighted form:

```text
weighted_points_last_5
weighted_goal_difference_last_5
```

A World Cup qualifier should count more than a friendly.

---

## 15. Elo System

Build custom Elo from historical matches.

Initial Elo:

```text
1500
```

Expected result formula:

```text
expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
```

Actual result:

```text
win  = 1
draw = 0.5
loss = 0
```

Update formula:

```text
new_elo_a = old_elo_a + K * (actual_a - expected_a)
```

Suggested K values:

```text
World Cup: 45
Continental tournament: 35
World Cup qualifier: 30
Nations League: 25
Friendly: 15
Other: 20
```

Critical rule:

Save pre-match Elo into the feature row.

Do not use post-match Elo to predict the same match.

That is data leakage.

---

## 16. Squad Features

From `squads.csv`, create one row per team.

Basic features:

```text
squad_avg_age
squad_median_age
squad_total_caps
squad_avg_caps
squad_total_international_goals
squad_avg_international_goals
num_goalkeepers
num_defenders
num_midfielders
num_forwards
```

Experience features:

```text
players_with_50_plus_caps
players_with_75_plus_caps
players_with_100_plus_caps
```

Club/league quality features:

```text
players_in_top5_leagues
players_in_champions_league_clubs
players_in_top_20_clubs
domestic_league_players
avg_club_strength_score
total_club_strength_score
```

If available:

```text
total_squad_market_value
avg_squad_market_value
top_11_market_value
top_5_market_value
bench_market_value
```

If market value is unavailable:

Use proxies:

```text
club_strength_score
league_strength_score
season_minutes
season_appearances
player_rating
international_caps
international_goals
```

Do not manually copy market values.

---

## 17. Starting XI Features

From Game 1 lineups:

```text
starting_xi_avg_age
starting_xi_total_caps
starting_xi_avg_caps
starting_xi_total_international_goals
starting_xi_top5_league_players
starting_xi_avg_club_strength
starting_xi_total_club_strength
formation
```

Bench features:

```text
bench_avg_caps
bench_total_caps
bench_avg_club_strength
bench_total_club_strength
bench_top5_league_players
```

Starting XI matters because it shows who actually plays, not just who was called up.

---

## 18. Game 1 Team Performance Features

After each team’s first match, create:

```text
wc_points
wc_goals_for
wc_goals_against
wc_goal_difference
wc_shots
wc_shots_on_target
wc_shot_accuracy
wc_possession
wc_corners
wc_yellow_cards
wc_red_cards
wc_xg
wc_xga
wc_xg_difference
```

Efficiency features:

```text
goals_minus_xg
goals_against_minus_xga
goals_per_shot
goals_per_shot_on_target
shots_on_target_rate
```

If xG is missing, skip xG features.

Do not fake xG.

Most important Game 1 features:

```text
wc_goal_difference
wc_xg_difference
wc_shots_on_target
wc_goals_against
wc_red_cards
```

Possession alone should not be treated as very important.

---

## 19. Game 1 Player Performance Features

Aggregate player stats into team features.

Create:

```text
starting_xi_avg_rating
starting_xi_min_rating
starting_xi_max_rating
team_total_player_minutes
team_goals_by_players
team_assists_by_players
team_shots_on_target_by_players
team_key_passes_by_players
team_tackles_by_players
team_interceptions_by_players
team_duels_won_rate
substitutes_avg_rating
substitutes_goal_contributions
```

Do not create 26 separate player columns in the first version.

Aggregate them.

---

## 20. Injury and Suspension Features

From `injuries.csv`, create:

```text
injured_players_count
questionable_players_count
suspended_players_count
injured_starters_count
suspended_starters_count
missing_key_players_count
```

Define key player as:

```text
A player who:
- Started Game 1
or
- Is top 5 in squad by market value / club strength / rating
or
- Has 50+ caps
```

Create:

```text
missing_key_player_flag
missing_starter_flag
```

Raw injury count is not enough.

One missing starter matters more than three missing bench players.

---

## 21. Rest and Fatigue Features

For each team before a match:

```text
days_since_last_match
players_with_90_minutes_previous_match
players_with_75_plus_minutes_previous_match
same_starting_xi_count
extra_time_previous_match
penalty_shootout_previous_match
```

Optional later:

```text
travel_distance_from_previous_city
```

Simple fatigue score:

```text
fatigue_score =
(players_with_90_minutes_previous_match * 1)
+ (players_with_75_plus_minutes_previous_match * 0.5)
+ (2 if days_since_last_match < 4 else 0)
+ (3 if extra_time_previous_match else 0)
+ (1 if penalty_shootout_previous_match else 0)
```

Keep fatigue simple at first.

---

## 22. Final Feature Priority

### Tier 1: Must Have

```text
elo_diff
points_last_5_diff
points_last_10_diff
goal_difference_last_5_diff
goals_for_last_5_diff
goals_against_last_5_diff
weighted_points_last_5_diff
neutral
match_importance
```

### Tier 2: Strong Additions

```text
squad_avg_age_diff
squad_total_caps_diff
squad_total_international_goals_diff
players_in_top5_leagues_diff
avg_club_strength_score_diff
```

### Tier 3: Tournament-Specific

```text
starting_xi_total_caps_diff
starting_xi_avg_club_strength_diff
bench_avg_club_strength_diff
wc_points_diff
wc_goal_difference_diff
wc_shots_on_target_diff
wc_red_cards_diff
```

### Tier 4: Advanced

```text
wc_xg_difference_diff
starting_xi_avg_rating_diff
substitutes_avg_rating_diff
injured_starters_count_diff
missing_key_players_count_diff
fatigue_score_diff
```

---

## 23. Models to Train

Train three models.

### Model 1: Logistic Regression

Purpose:

Simple baseline.

If advanced models cannot beat it, feature engineering may be bad.

### Model 2: Random Forest

Purpose:

Good beginner nonlinear model.

### Model 3: XGBoost

Purpose:

Main final model for tabular data.

Save all models, but use the best one based on validation log loss.

---

## 24. Train/Test Split

Do not use random split.

Use time-based split.

Correct:

```text
Train: older matches
Validation: newer matches
Test: newest matches
```

Example:

```text
Train: before 2022
Validation: 2022–2024
Test: 2025 onward
```

Reason:

In real life, we predict the future using the past.

Random split leaks time patterns.

---

## 25. Evaluation Metrics

Use:

```text
accuracy
log_loss
confusion_matrix
classification_report
brier_score
```

Most important:

```text
log_loss
```

Why?

Because we care about probabilities, not just the predicted class.

A model saying:

```text
Brazil 90% win
```

and Brazil loses should be punished heavily.

---

## 26. Data Leakage Rules

Never use information that would not be known before the match.

Bad:

```text
Using final score as a feature
Using post-match Elo to predict that same match
Using Match 2 stats to predict Match 2
Using final group standings to predict group matches
```

Good:

```text
Using previous historical matches
Using pre-match Elo
Using official squad before match
Using Game 1 stats to predict Game 2
Using lineups only when officially released
```

Every feature must be available before the predicted match starts.

---

## 27. Required Scripts

### 27.1 src/config.py

Responsibilities:

```text
Load environment variables
Store API base URL
Store paths
Store constants
```

Must include:

```text
RAW_DATA_DIR
PROCESSED_DATA_DIR
MODELS_DIR
FOOTBALL_API_KEY
FOOTBALL_API_BASE_URL
```

---

### 27.2 src/api_client.py

Responsibilities:

```text
Make API requests
Attach API key
Handle rate limits
Retry failed requests
Save raw JSON responses
Return parsed JSON
```

Must include:

```text
get(endpoint, params)
save_json(data, filename)
```

The API client should never crash the whole pipeline because of one failed request.

It should log the error and continue when safe.

---

### 27.3 src/collect_historical_matches.py

Responsibilities:

```text
Load historical match CSV
Standardize column names
Convert date to datetime
Sort by date
Save cleaned file
```

Output:

```text
data/raw/historical_matches.csv
```

---

### 27.4 src/collect_fixtures.py

Responsibilities:

```text
Call fixtures endpoint
Filter World Cup season/competition
Save fixture IDs, teams, dates, stadiums, status
```

Output:

```text
data/raw/fixtures.csv
```

---

### 27.5 src/collect_teams.py

Responsibilities:

```text
Fetch team IDs and team metadata
Create teams.csv
Standardize team names
```

Output:

```text
data/raw/teams.csv
```

---

### 27.6 src/collect_squads.py

Responsibilities:

```text
Fetch squad/player list for each World Cup team
Save player ID, name, age, position, club, league if available
Save caps/goals/rating/season stats if available
```

Output:

```text
data/raw/squads.csv
```

If the primary API does not provide official World Cup squads, create a TODO and either:

```text
1. Use an allowed official source parser
2. Use backup provider
3. Skip squad features temporarily
```

Do not manually copy squads.

---

### 27.7 src/collect_lineups.py

Responsibilities:

```text
For each fixture, fetch lineups
Save formation, coach, starting XI, substitutes
Mark Game 1 lineups
```

Output:

```text
data/raw/lineups.csv
```

---

### 27.8 src/collect_match_stats.py

Responsibilities:

```text
For each completed fixture, fetch team statistics
Save shots, possession, cards, corners, saves, xG if available
```

Output:

```text
data/raw/team_match_stats.csv
```

---

### 27.9 src/collect_player_stats.py

Responsibilities:

```text
For each completed fixture, fetch player statistics
Save minutes, rating, goals, assists, passes, tackles, cards
```

Output:

```text
data/raw/player_match_stats.csv
```

---

### 27.10 src/collect_injuries.py

Responsibilities:

```text
For upcoming fixtures, fetch injuries/suspensions/questionable players
Save missing players
```

Output:

```text
data/raw/injuries.csv
```

---

### 27.11 src/build_elo.py

Responsibilities:

```text
Sort historical matches by date
Initialize every team at 1500 Elo
Before each match, save both teams' pre-match Elo
Update Elo after match
Save elo history
```

Output:

```text
data/processed/elo_history.csv
```

---

### 27.12 src/build_features.py

Responsibilities:

```text
Load historical matches
Load Elo history
Build recent form features
Build weighted form features
Build squad features
Build lineup features
Build tournament features
Build injury/fatigue features
Merge all features
Create final training dataset
```

Output:

```text
data/processed/training_dataset.csv
```

Must handle missing values safely.

Do not drop too much data without explanation.

---

### 27.13 src/train_model.py

Responsibilities:

```text
Load training_dataset.csv
Select feature columns
Use time-based train/validation/test split
Train Logistic Regression
Train Random Forest
Train XGBoost
Evaluate all models
Save best model
Save metadata
```

Outputs:

```text
models/logistic_regression.pkl
models/random_forest.pkl
models/xgboost.pkl
models/model_metadata.json
reports/evaluation_report.txt
```

---

### 27.14 src/evaluate_model.py

Responsibilities:

```text
Load trained model
Run evaluation on test set
Save confusion matrix
Save classification report
Save feature importance
```

Outputs:

```text
reports/evaluation_report.txt
reports/feature_importance.csv
```

---

### 27.15 src/predict_match.py

Responsibilities:

```text
Accept Team A and Team B
Load latest processed features
Build one feature row
Load best trained model
Predict probabilities
Print readable output
```

Example command:

```bash
python src/predict_match.py "France" "Senegal"
```

Example output:

```text
France vs Senegal

France Win: 52%
Draw: 26%
Senegal Win: 22%

Top factors:
- France higher Elo
- France stronger squad depth
- Senegal lower recent form
```

---

### 27.16 src/run_pipeline.py

Responsibilities:

Run scripts in correct order.

Order:

```text
1. collect_historical_matches.py
2. collect_teams.py
3. collect_fixtures.py
4. collect_squads.py
5. collect_lineups.py
6. collect_match_stats.py
7. collect_player_stats.py
8. collect_injuries.py
9. build_elo.py
10. build_features.py
11. train_model.py
12. evaluate_model.py
```

The pipeline should print progress clearly.

---

## 28. Build Phases

### Phase 1: Historical Model Only

Build first.

Use only:

```text
historical matches
Elo
recent form
match importance
neutral venue
```

Features:

```text
elo_diff
points_last_5_diff
points_last_10_diff
goal_difference_last_5_diff
goals_for_last_5_diff
goals_against_last_5_diff
neutral
match_importance
```

Goal:

```text
A working model predicts Win/Draw/Loss from historical data.
```

Do not add squads yet.

Do not add Game 1 yet.

Do not build dashboard yet.

---

### Phase 2: World Cup Fixtures

Add automated fixture collection.

Goal:

```text
predict_match.py can predict real upcoming World Cup fixtures.
```

Still use only historical features.

---

### Phase 3: Squad Features

Add automated squad collection.

Goal:

```text
The model knows which players are in each tournament squad.
```

Features:

```text
squad_avg_age_diff
squad_total_caps_diff
squad_total_goals_diff
players_in_top5_leagues_diff
avg_club_strength_score_diff
```

---

### Phase 4: Game 1 Lineups and Team Stats

Add:

```text
Game 1 starting XI
Game 1 formation
Game 1 team stats
```

Goal:

```text
The model updates after each team has played once.
```

Features:

```text
starting_xi_caps_diff
starting_xi_club_strength_diff
wc_points_diff
wc_goal_difference_diff
wc_shots_on_target_diff
wc_red_cards_diff
```

---

### Phase 5: Player Stats, Injuries, and Fatigue

Add:

```text
player match stats
injuries
suspensions
rest days
fatigue score
```

Goal:

```text
The model accounts for player performance and availability.
```

---

### Phase 6: Dashboard

Only after predictions work.

Optional dashboard with Streamlit:

```text
Predict Match
View Team Features
Compare Squads
View Feature Importance
View Model Evaluation
```

---

## 29. What ChatGPT/Codex Should Do

When helping with this project, ChatGPT should:

```text
1. Work phase by phase.
2. Do not generate the whole giant project at once.
3. Explain code in beginner-friendly terms.
4. Keep code modular.
5. Avoid overengineering.
6. Use clear function names.
7. Add comments where useful.
8. Test each script after creating it.
9. Never hardcode API keys.
10. Never manually insert football data.
11. Never use future match information as features.
12. Never include betting odds.
```

If the user asks for full code, provide it only for the current phase/script.

Do not jump to dashboard or advanced features before Phase 1 works.

---

## 30. First Prompt to Give ChatGPT/Codex

Use this prompt in a new coding chat:

```text
I am building an educational Machine Learning project called worldcup-match-predictor.

Read PROJECT_CONTEXT.md first and follow it strictly.

Start with Phase 1 only.

Do not build the dashboard.
Do not add squads yet.
Do not add Game 1 stats yet.
Do not add injuries yet.

Create the project structure and implement:

- requirements.txt
- .env.example
- src/config.py
- src/collect_historical_matches.py
- src/build_elo.py
- src/build_features.py
- src/train_model.py
- src/evaluate_model.py
- src/predict_match.py

The Phase 1 model should use:

- elo_diff
- points_last_5_diff
- points_last_10_diff
- goal_difference_last_5_diff
- goals_for_last_5_diff
- goals_against_last_5_diff
- neutral
- match_importance

Use a time-based train/test split, not random split.

Train Logistic Regression, Random Forest, and XGBoost.

Evaluate with accuracy and log loss.

Save the best model.

Explain each step clearly because this is my first Machine Learning project.
```

---

## 31. Definition of Done

The project is successful when this works:

```bash
python src/run_pipeline.py
```

Then:

```bash
python src/predict_match.py "Argentina" "France"
```

And the output looks like:

```text
Argentina vs France

Argentina Win: __%
Draw: __%
France Win: __%
```

Also required:

```text
No manual data entry
No future data leakage
No betting odds
Model evaluation report exists
Feature importance report exists
Missing data is handled safely
Scripts are modular
The user understands what each phase does
```

---

## 32. Final Rule

The correct order is:

```text
1. Make historical model work.
2. Add World Cup fixtures.
3. Add squads.
4. Add Game 1 lineups and stats.
5. Add injuries and fatigue.
6. Add dashboard.
```

Do not skip steps.

Do not build the fancy version before the basic version works.

A simple correct model is better than a complicated broken one.

---

## 33. Phase 4 Experimental PyFotMob Enrichment

PyFotMob/FotMob may be used as an optional enrichment provider for football analytics only:

```text
team info
match details
lineups
formations
player match stats
team match stats
```

Rules:

```text
Do not remove the Phase 1 historical model.
Do not remove football-data.org fixtures/results.
Do not use betting odds or betting-related features.
Do not crash the core pipeline if PyFotMob fails.
Do not use same-match stats to predict the same match.
Use tournament stats only after they happened.
```

Run optional enrichment with:

```bash
python src/run_pipeline.py --fotmob-only
python src/run_pipeline.py --with-fixtures --with-fotmob
```
