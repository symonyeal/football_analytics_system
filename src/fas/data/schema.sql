-- Canonical PostgreSQL schema (Part 0.2).
-- The `actions` table is the spine of the system; everything else references it.

CREATE TABLE IF NOT EXISTS competitions (
    competition_id INT PRIMARY KEY,
    season_id      INT,
    name           VARCHAR(128)
);

CREATE TABLE IF NOT EXISTS teams (
    team_id   INT PRIMARY KEY,
    team_name VARCHAR(128),
    league    VARCHAR(64)
);

CREATE TABLE IF NOT EXISTS players (
    player_id   INT PRIMARY KEY,
    player_uid  INT,                       -- unified cross-source id (Part 0.2)
    full_name   VARCHAR(128),
    dob         DATE,
    nationality VARCHAR(64),
    position    VARCHAR(32)
);

CREATE TABLE IF NOT EXISTS matches (
    match_id       INT PRIMARY KEY,
    competition_id INT REFERENCES competitions(competition_id),
    home_team_id   INT REFERENCES teams(team_id),
    away_team_id   INT REFERENCES teams(team_id),
    home_score     SMALLINT,
    away_score     SMALLINT,
    match_date     DATE
);

CREATE TABLE IF NOT EXISTS actions (
    action_id     BIGSERIAL PRIMARY KEY,
    match_id      INT REFERENCES matches(match_id),
    period        SMALLINT,
    timestamp_ms  INT,
    player_id     INT REFERENCES players(player_id),
    team_id       INT REFERENCES teams(team_id),
    action_type   VARCHAR(32),
    x_start       FLOAT,  y_start FLOAT,
    x_end         FLOAT,  y_end   FLOAT,
    outcome       BOOLEAN,
    freeze_json   JSONB
);

CREATE INDEX IF NOT EXISTS idx_actions_match  ON actions(match_id);
CREATE INDEX IF NOT EXISTS idx_actions_player ON actions(player_id);
CREATE INDEX IF NOT EXISTS idx_actions_type   ON actions(action_type);

-- Per-source valuation / rating side-tables joined via player_uid.
CREATE TABLE IF NOT EXISTS transfermarkt_values (
    player_uid   INT,
    as_of        DATE,
    market_value NUMERIC,    -- in EUR
    PRIMARY KEY (player_uid, as_of)
);
