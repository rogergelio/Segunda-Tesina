# Consolidated Imports
import sys
import json
import time
import numpy as np
import pandas as pd
import re
import ast
from bs4 import BeautifulSoup
from pydantic import BaseModel
from typing import List, Optional, Dict
from selenium import webdriver
from supabase import create_client, Client
from uuid import UUID, uuid4
from tqdm import tqdm
import psycopg2
from dotenv import load_dotenv
import os
#from config import APIKEY, URL, PASSWORD
from sqlalchemy import create_engine, MetaData
from sqlalchemy.dialects.postgresql import insert
from uuid import uuid4
from pydantic import BaseModel
from typing import Optional, Dict

from sqlalchemy import create_engine, MetaData

# Lightweight logging helper for progress markers
def _log(component: str, message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [{component}] {message}", flush=True)

def calculate_xg_xgot(df, xg_model, xgot_model):
    _log('utils.calculate_xg_xgot', 'Start')
    original_df=df.copy()
    _log('utils.calculate_xg_xgot', 'Coercing coordinates to numeric')
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    _log('utils.calculate_xg_xgot', 'test log')
    def extract_display_names(qualifiers_string):
        if pd.isna(qualifiers_string):
            return []
        try:
            qualifiers = ast.literal_eval(qualifiers_string)
            return [q['type']['displayName'] for q in qualifiers if 'type' in q and 'displayName' in q['type']]
        except Exception:
            return []
    
    def calculate_shot_angle(x, y):
        goal_x, goal_y_top, goal_y_bottom = 100, 56, 44
        dx = goal_x - x
        dy_top = goal_y_top - y
        dy_bottom = goal_y_bottom - y
        
        # Angles from (x,y) to top and bottom posts
        angle_top = np.arctan2(dy_top, dx)
        angle_bottom = np.arctan2(dy_bottom, dx)
        
        # Angle between lines (always positive)
        angle = abs(angle_top - angle_bottom)
        
        # Convert to degrees
        return np.degrees(angle)
    
    qualifiers_of_interest = [
        'IntentionalAssist', 'IntentionalGoalAssist', 'Assisted', 'KeyPass', 'IndividualPlay',
        'RightFoot', 'LeftFoot', 'Head', 'OtherBodyPart', 'Volley', 'SetPiece', 'DirectFreekick', 'OneOnOne'
        ,'Penalty', 'DirectCorner']
    
    _log('utils.calculate_xg_xgot', 'Filtering shots only')
    df=original_df[original_df['is_shot'] == True].copy()

    # Step 2: Apply to your DataFrame
    _log('utils.calculate_xg_xgot', 'Extracting qualifier display names')
    df['display_names'] = df['qualifiers'].apply(extract_display_names)

    # Step 3: Flatten and get unique display names
    all_display_names = set(name for sublist in df['display_names'] for name in sublist)

    _log('utils.calculate_xg_xgot', f"Unique display names found: {len(all_display_names)}")

    # Get list of display names per row
    df['display_names'] = df['qualifiers'].apply(extract_display_names)

    # Create binary columns for each qualifier
    _log('utils.calculate_xg_xgot', 'Creating qualifier one-hot columns')
    for qualifier in qualifiers_of_interest:
        df[f'qual_{qualifier}'] = df['display_names'].apply(lambda x: 1 if qualifier in x else 0)

    # Optional: drop the intermediate column if you don’t need it
    # df.drop(columns=['display_names'], inplace=True)

    _log('utils.calculate_xg_xgot', 'Feature engineering (angles, distances)')
    df['centered_y'] = 50 - abs(df['y'] - 50)
    df['centered_goalmouth_y'] = 50 - abs(df['goal_mouth_y'] - 50)
    df['shot_angle'] = df.apply(lambda row: calculate_shot_angle(row['x'], row['y']), axis=1)

    # Distance to center (you already have this):
    df['distance_to_goal_center'] = np.sqrt(
        (100 - df['x'].fillna(0))**2 + (50 - df['y'].fillna(0))**2
    )

    # Distance to left post
    df['distance_to_left_post'] = np.sqrt((100 - df['x'])**2 + (44 - df['y'])**2)

    # Distance to right post
    df['distance_to_right_post'] = np.sqrt((100 - df['x'])**2 + (56 - df['y'])**2)

    # 1. Distance × Angle — close shots at bad angles vs far shots at good angles
    df['dist_angle'] = df['distance_to_goal_center'] * df['shot_angle']

    # 2. One-on-one within 10m — high conversion zone for 1v1
    df['one_on_one_near'] = df['qual_OneOnOne'] * (df['distance_to_goal_center'] < 10).astype(int)

    # 3. Strong foot at tight angle — can the player score from a sharp position?
    df['tight_angle_strong_foot'] = df['shot_angle'] * (
        (df['qual_RightFoot'] | df['qual_LeftFoot']).astype(int)
    )

    # 4. Header far from goal — headers lose power with distance
    df['header_far'] = df['qual_Head'] * (df['distance_to_goal_center'] > 12).astype(int)

    # 5. Volley in central position — more dangerous than wide volleys
    df['central_volley'] = df['qual_Volley'] * (df['centered_y'] < 2).astype(int)

    # 6. Header from corner — headers following a corner kick
    df['header_from_corner'] = df['qual_Head'] * df['qual_DirectCorner']

    # 7. Sharp angle with opposite foot — usually worse finishing
    df['sharp_angle_weak_foot'] = (df['shot_angle'] > 30).astype(int) * (
        (~df['qual_RightFoot'] & ~df['qual_LeftFoot']).astype(int)
    )

    # 8. Cross + header — classic aerial danger pattern
    df['cross_header'] = (df['qual_Assisted'] | df['qual_IntentionalAssist']).astype(int) * df['qual_Head']

    # 9. Set piece danger zone — set piece + close distance
    df['setpiece_close'] = df['qual_SetPiece'] * (df['distance_to_goal_center'] < 12).astype(int)

    # 10. Angle efficiency — ratio of angle to distance
    df['angle_efficiency'] = df['shot_angle'] / (df['distance_to_goal_center'] + 1e-6)

    all_possible_features = [ #Miami Horror reference
        'x', 'y', 'id',
        'qual_IntentionalAssist', 'qual_IntentionalGoalAssist', 'qual_Assisted',
        'qual_KeyPass', 'qual_IndividualPlay', 'qual_RightFoot', 'qual_LeftFoot',
        'qual_Head', 'qual_OtherBodyPart', 'qual_Volley', 'qual_SetPiece',
        'qual_DirectFreekick', 'qual_OneOnOne', 'qual_Penalty', 'is_goal',
        'qual_DirectCorner', 'shot_angle', 'centered_y', 'distance_to_goal_center', 'distance_to_left_post', 'distance_to_right_post', 'goal_mouth_y', 'goal_mouth_z', 'centered_goalmouth_y',
        'dist_angle',
        'one_on_one_near',
        'tight_angle_strong_foot',
        'header_far',
        'central_volley',
        'header_from_corner',
        'sharp_angle_weak_foot',
        'cross_header',
        'setpiece_close',
        'angle_efficiency']

    target = 'is_goal'
    df = df[all_possible_features]

    nan_counts = df.isna().sum()
    nan_counts = nan_counts[nan_counts > 0]  # Filter columns with any NaNs

    _log('utils.calculate_xg_xgot', f"Dropping rows with NaNs in selected features (before: {len(df)})")
    df = df.dropna()
    _log('utils.calculate_xg_xgot', f"Rows after dropna: {len(df)}")

    xg_features = [
        'x', 'y',
        'qual_IntentionalAssist', 'qual_IntentionalGoalAssist', 'qual_Assisted',
        'qual_KeyPass', 'qual_IndividualPlay', 'qual_RightFoot', 'qual_LeftFoot',
        'qual_Head', 'qual_OtherBodyPart', 'qual_Volley', 'qual_SetPiece',
        'qual_DirectFreekick', 'qual_OneOnOne', 'qual_Penalty',
        'qual_DirectCorner', 'shot_angle', 'centered_y', 'distance_to_goal_center', 'distance_to_left_post', 'distance_to_right_post',
        'dist_angle',
        'one_on_one_near',
        'tight_angle_strong_foot',
        'header_far',
        'central_volley',
        'header_from_corner',
        'sharp_angle_weak_foot',
        'cross_header',
        'setpiece_close',
        'angle_efficiency'
        ]

    xgot_features = [
        'x', 'y',
        'qual_IntentionalAssist', 'qual_IntentionalGoalAssist', 'qual_Assisted',
        'qual_KeyPass', 'qual_IndividualPlay', 'qual_RightFoot', 'qual_LeftFoot',
        'qual_Head', 'qual_OtherBodyPart', 'qual_Volley', 'qual_SetPiece',
        'qual_DirectFreekick', 'qual_OneOnOne', 'qual_Penalty',
        'qual_DirectCorner', 'shot_angle', 'centered_y', 'distance_to_goal_center', 'distance_to_left_post', 'distance_to_right_post', 'goal_mouth_y', 'goal_mouth_z', 'centered_goalmouth_y',
        'dist_angle',
        'one_on_one_near',
        'tight_angle_strong_foot',
        'header_far',
        'central_volley',
        'header_from_corner',
        'sharp_angle_weak_foot',
        'cross_header',
        'setpiece_close',
        'angle_efficiency'
        ]

    target = 'is_goal'

    final_columns_xg = xg_features + [target]
    final_columns_xgot = xgot_features + [target]

    _log('utils.calculate_xg_xgot', 'Preparing model input frames')
    df_xgot = df[final_columns_xgot].copy()
    df_xg = df[final_columns_xg].copy()

    _log('utils.calculate_xg_xgot', 'Predicting xG and xGoT')
    xg_preds=xg_model.predict_proba(df_xg[xg_features])[:, 1]
    xgot_preds=xgot_model.predict_proba(df_xgot[xgot_features])[:, 1]

    _log('utils.calculate_xg_xgot', 'Assigning predictions back to working frame')
    df.loc[df_xg.index, 'xG_pred'] = xg_preds
    df.loc[df_xgot.index, 'xGoT_pred'] = xgot_preds
    _log('utils.calculate_xg_xgot', 'Applying penalty rule (xG=0.72 for penalties)')
    df.loc[df['qual_Penalty'] == 1, 'xG_pred'] = 0.72

    # Create a DataFrame with ids and predictions for both models
    _log('utils.calculate_xg_xgot', 'Merging predictions back to original frame')
    preds_df = pd.DataFrame({
        'id': df['id'],
        'xG_pred': df['xG_pred'],
        'xGoT_pred': df['xGoT_pred']
    })

    merged_df = original_df.merge(preds_df, on='id', how='left')

    _log('utils.calculate_xg_xgot', 'Applying not-on-target rule for xGoT')
    merged_df.loc[~merged_df['type_display_name'].isin(['Goal', 'SavedShot']), 'xGoT_pred'] = 0

    _log('utils.calculate_xg_xgot', 'Done')
    return merged_df

def scrape_match_events(whoscored_url, driver):
    _log('utils.scrape_match_events', 'Start')
    #-------------------------------------------- declarartions
    engine = create_engine("postgresql+psycopg2://postgres:admin@localhost:5432/postgres")
    
    def insert_match_events(df):

        df.to_sql(
            name='match_events',
            con=engine,
            if_exists='append',  # use 'replace' to drop and recreate, or 'append' to insert new rows
            index=False
        )

    

    class Match(BaseModel):
        match_id: int
        home_team: str
        away_team: str
        score: str
        home_goals: int
        away_goals: int
        competition: str
        season: str
        attendance: Optional[int]
        venueName: Optional[str]
        referee: Optional[str]
        weatherCode: Optional[str]
        startTime: str
        startDate: str
        htScore: Optional[str]
        ftScore: Optional[str]
        statusCode: int
        periodCode: Optional[str]
        maxMinute: int
        minuteExpanded: str
        maxPeriod: int
        home_id: int
        away_id: int

    class Player(BaseModel):
        id: str
        player_id: int
        shirt_no: int
        name: str
        age: int
        position: str
        team_id: int
        ds: str

    class MatchEvent(BaseModel):
        id: int
        event_id: int
        minute: int
        second: Optional[float] = None
        team_id: int
        player_id: int
        x: float
        y: float
        end_x: Optional[float] = None
        end_y: Optional[float] = None
        qualifiers: List[dict]
        is_touch: bool
        blocked_x: Optional[float] = None
        blocked_y: Optional[float] = None
        goal_mouth_z: Optional[float] = None
        goal_mouth_y: Optional[float] = None
        is_shot: bool
        card_type: bool
        is_goal: bool
        type_display_name: str
        outcome_type_display_name: str
        period_display_name: str
        match_id: int

    def insert_match_events(df):
        # Create engine
        engine = create_engine("postgresql+psycopg2://postgres:admin@localhost:5432/postgres")

        # Reflect existing table
        metadata = MetaData()
        metadata.reflect(bind=engine)
        match_events = metadata.tables['match_events']

        total_inserted = 0
        # Insert in batches
        with engine.begin() as conn:
            for chunk_start in range(0, len(df), 1000):
                chunk = df.iloc[chunk_start:chunk_start+1000]
                records = chunk.to_dict(orient='records')

                stmt = insert(match_events).values(records)
                stmt = stmt.on_conflict_do_nothing(index_elements=['id'])

                result = conn.execute(stmt)
                total_inserted += result.rowcount  # Number of rows inserted in this batch

        #print(f"Total rows inserted (excluding duplicates): {total_inserted}")
    def insert_players(team_info, matchdict):
        players = []
        for team in team_info:
            for player in team['players']:
                players.append({
                    'id': str(uuid4()),
                    'player_id': player['playerId'],
                    'team_id': team['team_id'],
                    'shirt_no': player['shirtNo'],
                    'name': player['name'],
                    'position': player['position'],
                    'age': player['age'],
                    'ds': matchdict.get('startDate'),
                })
        
        engine = create_engine("postgresql+psycopg2://postgres:admin@localhost:5432/postgres")

        metadata = MetaData()
        metadata.reflect(bind=engine)
        players_table = metadata.tables['players']

        batch_size = 1000
        total_inserted = 0

        with engine.begin() as conn:
            for i in range(0, len(players), batch_size):
                chunk = players[i:i+batch_size]  # slice list directly

                stmt = insert(players_table).values(chunk)
                stmt = stmt.on_conflict_do_nothing(index_elements=['id'])

                result = conn.execute(stmt)
                total_inserted += result.rowcount

        #print(f"Total rows inserted (excluding duplicates): {total_inserted}")
    def insert_matches(matches):
        # Accept either a single Match or a list of Match models
        if not isinstance(matches, list):
            matches = [matches]

        # Prepare list of dicts for upsert
        match_records = []
        for match in matches:
            match_records.append({
                'match_id': match.match_id,
                'home_team': match.home_team,
                'away_team': match.away_team,
                'score': match.score,
                'home_goals': match.home_goals,
                'away_goals': match.away_goals,
                'competition': match.competition,
                'season': match.season,
                'attendance': match.attendance,
                'venuename': match.venueName,
                'referee': match.referee,
                'weathercode': match.weatherCode,
                'starttime': match.startTime,
                'startdate': match.startDate,
                'htscore': match.htScore,
                'ftscore': match.ftScore,
                'statuscode': match.statusCode,
                'periodcode': match.periodCode,
                'maxminute': match.maxMinute,
                'minuteexpanded': match.minuteExpanded,
                'maxperiod': match.maxPeriod,
                'home_id': match.home_id,
                'away_id': match.away_id
            })

        # Create engine
        engine = create_engine("postgresql+psycopg2://postgres:admin@localhost:5432/postgres")

        # Reflect existing table
        metadata = MetaData()
        metadata.reflect(bind=engine)
        matches_table = metadata.tables['matches']

        total_inserted = 0
        batch_size = 1000

        with engine.begin() as conn:
            for i in range(0, len(match_records), batch_size):
                chunk = match_records[i:i + batch_size]  # slice list directly

                stmt = insert(matches_table).values(chunk)

                # Use the actual primary key or unique constraint column name
                stmt = stmt.on_conflict_do_nothing(index_elements=['match_id'])

                result = conn.execute(stmt)
                total_inserted += result.rowcount

        #print(f"Total rows inserted (excluding duplicates): {total_inserted}")



    #---------------------------------------------------------
    print('Fetching URL...')
    _log('utils.scrape_match_events', 'Navigating to URL')
    driver.get(whoscored_url)
    matchdict=dict()
    match = re.search(r'/matches/(\d+)/', whoscored_url)
    match_id = int(match.group(1) if match else 0)
    _log('utils.scrape_match_events', 'Parsing page source')
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    _log('utils.scrape_match_events', 'Locating matchCentreData script tag')
    element = soup.select_one('script:-soup-contains("matchCentreData")')

    if not element:
        _log('utils.scrape_match_events', f"Skipping: 'matchCentreData' script tag not found at {whoscored_url}")
        return

    text = element.text.strip()

    if "matchCentreData: " not in text:
        _log('utils.scrape_match_events', f"Skipping: 'matchCentreData' not in script text at {whoscored_url}")
        return

    try:
        _log('utils.scrape_match_events', 'Extracting JSON payload')
        json_str = text.split("matchCentreData: ")[1].split(',\n')[0]
        matchdict = json.loads(json_str)
    except (IndexError, json.JSONDecodeError) as e:
        _log('utils.scrape_match_events', f"Skipping due to parsing error: {e}")
        return

    try:
        match_events = matchdict.get('events')
        if match_events is None:
            _log('utils.scrape_match_events', "Skipping: no 'events' key in matchCentreData")
            return
    except AttributeError:
        _log('utils.scrape_match_events', 'Skipping: matchdict is None or invalid')
        return

    match_events = matchdict['events']

    ## Get the match info

    _log('utils.scrape_match_events', 'Extracting match metadata from <title>')
    test_element=str(soup.select_one('title'))
    title_text = re.sub(r'</?title>', '', test_element).strip()  # remove leading/trailing whitespace and newlines

    # Split on ' - '
    parts = title_text.split(' - ')
    if len(parts) < 2:
        raise ValueError("Unexpected format")

    match_part = parts[0]
    comp_part = parts[1]

    # Extract score
    score_match = re.search(r'(\d+-\d+)', match_part)
    if not score_match:
        raise ValueError("Score not found")

    score = score_match.group(1)

    # Extract teams
    teams = match_part.split(score)
    if len(teams) != 2:
        raise ValueError("Unexpected teams format")

    home_team = teams[0].strip()
    away_team = teams[1].strip()

    # Extract season
    season_match = re.search(r'(\d{4}/\d{4})', comp_part)
    if not season_match:
        raise ValueError("Season not found")

    season = season_match.group(1)

    # Competition
    competition = comp_part.split(season)[0].strip()

    # Split goals into integers
    home_goals, away_goals = map(int, score.split('-'))

    #print(f"Home team: {home_team} vs {away_team}")
    try:
        if matchdict['referee']['name'] is None:
            matchdict['referee']['name'] = 'No Referee'
    except (KeyError, TypeError):
        matchdict['referee'] = {'name': 'No Referee'}


    _log('utils.scrape_match_events', 'Building Match model')
    match = Match(
        match_id=int(match_id),
        home_team=home_team,
        away_team=away_team,
        score=score,
        home_goals=home_goals,
        away_goals=away_goals,
        competition=competition,
        season=season,
        attendance=matchdict.get('attendance'),
        venueName=matchdict.get('venueName'),
        referee=matchdict['referee']['name'],
        weatherCode=str(matchdict.get('weatherCode')),
        startTime=matchdict.get('startTime'),
        startDate=matchdict.get('startDate'),
        htScore=matchdict.get('htScore'),
        ftScore=matchdict.get('ftScore'),
        statusCode=matchdict.get('statusCode'),
        periodCode=str(matchdict.get('periodCode')),
        maxMinute=matchdict.get('maxMinute'),
        minuteExpanded=str(matchdict.get('minuteExpanded')),
        maxPeriod=matchdict.get('maxPeriod'),
        home_id=int(matchdict['home']['teamId']),
        away_id=int(matchdict['away']['teamId'])
    )


    ## Get the actaul event data
    
    _log('utils.scrape_match_events', 'Creating events DataFrame')
    df = pd.DataFrame(match_events)
    
    _log('utils.scrape_match_events', 'Dropping rows with missing playerId')
    df.dropna(subset='playerId', inplace=True)
    
    df = df.where(pd.notnull(df), None)
    
    _log('utils.scrape_match_events', 'Renaming and normalizing columns')
    df = df.rename(
    {
        'eventId': 'event_id',
        'expandedMinute': 'expanded_minute',
        'outcomeType': 'outcome_type',
        'isTouch': 'is_touch',
        'playerId': 'player_id',
        'teamId': 'team_id',
        'endX': 'end_x',
        'endY': 'end_y',
        'blockedX': 'blocked_x',
        'blockedY': 'blocked_y',
        'goalMouthZ': 'goal_mouth_z',
        'goalMouthY': 'goal_mouth_y',
        'isShot': 'is_shot',
        'cardType': 'card_type',
        'isGoal': 'is_goal'
    },
        axis=1
    )
    
    _log('utils.scrape_match_events', 'Deriving display name columns')
    df['period_display_name'] = df['period'].apply(lambda x: x['displayName'])
    df['type_display_name'] = df['type'].apply(lambda x: x['displayName'])
    df['outcome_type_display_name'] = df['outcome_type'].apply(lambda x: x['displayName'])
    
    df.drop(columns=["period", "type", "outcome_type"], inplace=True)
    
    if 'is_goal' not in df.columns:
        df['is_goal'] = False
        
    if 'is_card' not in df.columns:
        df['is_card'] = False
        df['card_type'] = False
        
    df = df[~(df['type_display_name'] == "OffsideGiven")]

    expected_schema = {
        'id': 'Int64',
        'event_id': 'Int64',
        'minute': 'Int64',
        'second': 'float64',
        'team_id': 'Int64',
        'player_id': 'Int64',
        'x': 'float64',
        'y': 'float64',
        'end_x': 'float64',
        'end_y': 'float64',
        'qualifiers': 'object',  # for List[dict]
        'is_touch': 'boolean',
        'blocked_x': 'float64',
        'blocked_y': 'float64',
        'goal_mouth_z': 'float64',
        'goal_mouth_y': 'float64',
        'is_shot': 'boolean',
        'card_type': 'boolean',
        'is_goal': 'boolean',
        'type_display_name': 'string',
        'outcome_type_display_name': 'string',
        'period_display_name': 'string',
        'match_id': 'string'
    }

    _log('utils.scrape_match_events', 'Ensuring expected schema and dtypes')
    for col, dtype in expected_schema.items():
        if col not in df.columns:
            if dtype == 'object':
                df[col] = [[] for _ in range(len(df))]
            elif dtype == 'string':
                df[col] = pd.Series([pd.NA] * len(df), dtype='string')
            elif dtype == 'boolean':
                df[col] = pd.Series([pd.NA] * len(df), dtype='boolean')
            elif dtype == 'Int64':
                df[col] = pd.Series([pd.NA] * len(df), dtype='Int64')
            else:
                df[col] = pd.Series([np.nan] * len(df))  # Use np.nan for float64

        try:
            df[col] = df[col].astype(dtype)
        except Exception as e:
            print(f"Could not cast column '{col}' to {dtype}: {e}")

    df = df[list(expected_schema.keys())]


    _log('utils.scrape_match_events', 'Casting core numeric columns')
    df[['id', 'event_id', 'minute', 'team_id', 'player_id']] = df[['id', 'event_id', 'minute', 'team_id', 'player_id']].astype(np.int64)
    df[['second', 'x', 'y', 'end_x', 'end_y']] = df[['second', 'x', 'y', 'end_x', 'end_y']].astype(float)

    boolean_cols = ['is_shot', 'is_goal', 'card_type']
    df[boolean_cols] = df[boolean_cols].fillna(False).astype(bool)

    df[['is_shot', 'is_goal', 'card_type']] = df[['is_shot', 'is_goal', 'card_type']].astype(bool)

    df['is_goal'] = df['is_goal'].fillna(False)
    df['is_shot'] = df['is_shot'].fillna(False)
    
    _log('utils.scrape_match_events', 'Converting NaNs to None for floats')
    for column in df.columns:
        if df[column].dtype == np.float64 or df[column].dtype == np.float32:
            df[column] = np.where(
                np.isnan(df[column]),
                None,
                df[column]
            )
    df['match_id']=str(match_id)
    if "qualifiers" in df.columns:
        df["qualifiers"] = df["qualifiers"].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else str(x))
    _log('utils.scrape_match_events', 'Inserting match events into DB')
    insert_match_events(df)
    _log('utils.scrape_match_events', 'Upserting match record')
    insert_matches(match)
    team_info = []
    _log('utils.scrape_match_events', 'Preparing team/player payloads')
    team_info.append({
        'id':str(uuid4()),
        'team_id': matchdict['home']['teamId'],
        'name': matchdict['home']['name'],
        'country_name': matchdict['home']['countryName'],
        'manager_name': matchdict['home']['managerName'],
        'players': matchdict['home']['players'],
        'ds': matchdict.get('startDate')
    })

    team_info.append({
        'id':str(uuid4()),
        'team_id': matchdict['away']['teamId'],
        'name': matchdict['away']['name'],
        'country_name': matchdict['away']['countryName'],
        'manager_name': matchdict['away']['managerName'],
        'players': matchdict['away']['players'],
        'ds': matchdict.get('startDate')
    })
    _log('utils.scrape_match_events', 'Inserting players into DB')
    insert_players(team_info, matchdict)
    _log('utils.scrape_match_events', 'Success')
    return df, match, team_info, matchdict