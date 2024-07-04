# [⭕] 1. 포켓몬의 'speed'가 70이상이면 '빠름', 그렇지 않으면 '느림'으로 표시하는 새로운 컬럼 'Speed_Category'를 만들어주세요

## mine

SELECT
  speed,
  CASE
    WHEN speed >= 70 THEN '빠름'
    ELSE '느림'
  END AS Speed_Catergory
FROM basic.pokemon;


## A

SELECT
  id,
  kor_name,
  speed,
  IF(speed >= 70, '빠름', '느림') AS Speed_Category 
FROM basic.pokemon;



# [⭕] 2. 포켓몬의 'type1'에 따라 'Water', 'Fire', 'Electric' 타입은 각각 '물', '불', '전기'로, 그 외 타입은 '기타'로 분류하는 새로운 컬럼 'type_Korean'을 만들어주세요

# mine

SELECT
  type1,
  CASE
    WHEN type1 = 'Water' THEN '물'
    WHEN type1 = 'Fire' THEN '불'
    WHEN type1 = 'Electric' THEN '전기'
    ELSE '기타'
  END AS type_Korean
FROM basic.pokemon;


## A

SELECT
  id,
  kor_name,
  type1,
  CASE
    WHEN type1 = 'Water' THEN '물'
    WHEN type1 = 'Fire' THEN '불'
    WHEN type1 = 'Electric' THEN '전기'
    ELSE '기타'
  END AS type1_Korean
FROM basic.pokemon;



# [⭕] 3. 각 포켓몬의 총점(total)을 기준으로, 300 이하면 'Low', 301에서 500 사이면 
-- 'Medium', 501 이상이면 'High'로 분류해주세요

## mine

SELECT
  total,
  CASE
   WHEN total <= 300 THEN 'Low'
   WHEN total <= 500 THEN 'Medium'
   ELSE 'High'
  END AS total_level
FROM basic.pokemon;


## A

SELECT
  id,
  kor_name,
  total,
  CASE
   WHEN total >= 501 THEN 'High'
   WHEN total BETWEEN 300 AND 500 THEN 'Medium'
   ELSE 'Low'
  END AS total_grade
FROM basic.pokemon;



# [⭕] 4. 각 트레이너의 배지 개수(badge_count)를 기준으로, 5개 이하면 'Beginner', 6개에서 8개 사이면 'Intermediate', 그 이상이면 'Advanced'로 분류해주세요.

## mine

SELECT
  badge_count,
  CASE
    WHEN badge_count <= 5 THEN 'Beginner'
    WHEN badge_count <= 8 THEN 'Intermediate'
    ELSE 'Advanced'
  END AS badge_count_level
FROM basic.trainer;


## A

SELECT
  trainer_level,
  COUNT(DISTINCT id) AS trainer_cnt
FROM (
  SELECT
    id,
    name,
    badge_count,
    CASE
      WHEN badge_count >= 9 THEN 'Advanced'
      WHEN badge_count BETWEEN 6 AND 8 THEN 'Intermediate'
      ELSE 'Beginner'
    END AS trainer_level
  FROM basic.trainer
)
GROUP BY 
  trainer_level;



# [⭕] 5. 트레이너가 포켓몬을 포획한 날짜(catch_date)가 '2023-01-01' 이후이면 'Recent', 그렇지 않으면 'Old'로 분류해주세요.

## mine

SELECT
  catch_date,
  CASE
    WHEN catch_date >= '2023-01-01' THEN 'RECENT'
    ELSE 'OLD'
  END AS catch_date_level,
  catch_datetime_kr,
  CASE
    WHEN catch_datetime_kr >= '2023-01-01' THEN 'RECENT'
    ELSE 'OLD'
  END AS catch_datetime_kr_level
FROM(
  SELECT
    id,
    catch_date,
    DATE(DATETIME(catch_datetime, 'Asia/Seoul')) AS catch_datetime_kr
  FROM basic.trainer_pokemon
);


## A

SELECT
  recent_or_old,
  COUNT(id) AS cnt
FROM(
  SELECT
    id,
    trainer_id,
    pokemon_id,
    catch_datetime,
    IF(DATE(catch_datetime, 'Asia/Seoul') > '2023-01-01', 'Recent', 'Old') AS recent_or_old
  FROM basic.trainer_pokemon
)
GROUP BY
  recent_or_old;

SELECT
  id,
  trainer_id,
  pokemon_id,
  catch_datetime,
  IF(DATE(catch_datetime, 'Asia/Seoul') > '2023-01-01', 'Recent', 'Old') AS recent_or_old,
  'Recent' AS recent_value # 모든 값이 한 가지 조건에 성립될 때. 컬럼의 모든 값을 하나로 입력할 때.
FROM basic.trainer_pokemon;



# [⭕] 6. 배틀에서 승자(winner_id)가 player1_id와 같으면 'Player 1 Wins', player2_id와 같으면 'Player 2 Wins', 그렇지 않으면 'Draw'로 결과가 나오게 해주세요

## mine

SELECT
  id,
  player1_id,
  player2_id,
  winner_id,
  CASE
    WHEN winner_id = player1_id THEN 'Player 1 Wins'
    WHEN winner_id = player2_id THEN 'Player 2 Wins'
    ELSE 'Draw'
  END AS result
FROM basic.battle
ORDER BY id;


## A

SELECT
  id,
  winner_id,
  player1_id,
  player2_id,
  CASE
    WHEN winner_id = player1_id THEN 'Player 1 Wins'
    WHEN winner_id = player2_id THEN 'Player 2 Wins'
    ELSE 'Draw'
  END AS battle_result
FROM basic.battle;




