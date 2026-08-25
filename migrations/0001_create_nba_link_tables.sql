-- Migration: create initial NBA Link schema
-- Run this in your Postgres database to create the core tables required

BEGIN;

CREATE TABLE players (
  id BIGSERIAL PRIMARY KEY,
  nba_player_id INTEGER,
  name TEXT NOT NULL,
  slug TEXT,
  birth_year INTEGER,
  active_from INTEGER,
  active_to INTEGER,
  primary_position TEXT,
  metadata JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);
CREATE INDEX players_name_trgm ON players USING gin (lower(name) gin_trgm_ops);

CREATE TABLE teams (
  id BIGSERIAL PRIMARY KEY,
  nba_team_id INTEGER,
  name TEXT,
  abbrev TEXT,
  metadata JSONB
);

CREATE TABLE games (
  id BIGSERIAL PRIMARY KEY,
  source_game_id TEXT UNIQUE NOT NULL,
  game_date DATE NOT NULL,
  season TEXT,
  home_team_id BIGINT REFERENCES teams(id),
  away_team_id BIGINT REFERENCES teams(id),
  source TEXT,
  source_url TEXT,
  raw_payload JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);
CREATE INDEX games_game_date_idx ON games(game_date);

CREATE TABLE appearances (
  id BIGSERIAL PRIMARY KEY,
  game_id BIGINT REFERENCES games(id) ON DELETE CASCADE,
  team_id BIGINT REFERENCES teams(id),
  player_id BIGINT REFERENCES players(id),
  player_nba_id INTEGER,
  minutes TEXT,
  starter BOOLEAN DEFAULT false,
  listed_in_boxscore BOOLEAN DEFAULT true,
  dnp_reason TEXT,
  raw_row JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  UNIQUE(game_id, player_id)
);
CREATE INDEX appearances_player_idx ON appearances(player_id);
CREATE INDEX appearances_game_idx ON appearances(game_id);

CREATE TABLE player_links (
  id BIGSERIAL PRIMARY KEY,
  player_a BIGINT NOT NULL,
  player_b BIGINT NOT NULL,
  games_together_count INTEGER DEFAULT 0,
  first_game_id BIGINT REFERENCES games(id),
  last_game_id BIGINT REFERENCES games(id),
  sample_game_ids BIGINT[],
  last_updated TIMESTAMP WITH TIME ZONE DEFAULT now(),
  UNIQUE (player_a, player_b)
);
-- enforce canonical ordering when inserting (player_a < player_b)

CREATE TABLE puzzles (
  id BIGSERIAL PRIMARY KEY,
  seed_player_id BIGINT REFERENCES players(id) NOT NULL,
  target_player_id BIGINT REFERENCES players(id) NOT NULL,
  optimal_length INTEGER,
  alt_path_count INTEGER,
  difficulty TEXT,
  generated_ts TIMESTAMP WITH TIME ZONE DEFAULT now(),
  stats JSONB,
  archived BOOLEAN DEFAULT false
);

CREATE TABLE daily_puzzles (
  puzzle_date DATE PRIMARY KEY,
  puzzle_id BIGINT REFERENCES puzzles(id),
  published_ts TIMESTAMP WITH TIME ZONE DEFAULT now()
);

CREATE TABLE puzzle_attempts (
  id BIGSERIAL PRIMARY KEY,
  puzzle_id BIGINT REFERENCES puzzles(id),
  attempt_ts TIMESTAMP WITH TIME ZONE DEFAULT now(),
  submitted_path BIGINT[],
  submitted_length INTEGER,
  score INTEGER,
  client_metadata JSONB,
  validation_result JSONB
);

COMMIT;
