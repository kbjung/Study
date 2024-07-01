# [❌] 1. 트레이너가 포켓몬을 포획한 날짜(catch_date)를 기준으로, 2023년 1월에 포획한 포켓몬의 수를 계산해주세요.

## mine(서울 시간으로 변환 필요)
-- SELECT
--   COUNT(id) AS pokemon_cnt
-- FROM (
--   SELECT
--     id,
--     EXTRACT(YEAR FROM catch_date) AS year,
--     EXTRACT(MONTH FROM catch_date) AS month
--   FROM basic.trainer_pokemon
-- )
-- WHERE
--   year = 2023 AND
--   month = 1;
-- 결과 : 86개

## A

-- catch_date : DATE 타입
-- catch_datetime : UTC, TIMESTAMP 타입 => 컬럼의 이름은 datetime인데 TIMESTAMP 타입으로 저장되어 있다.
-- 데이터가 잘못 저장된 경우. 컬럼의 이름만 믿지 말고, 데이터를 확인해야 한다.
-- catch_date => KR 기준? UTC 기준? 확인 필요

-- catch_date를 사용하면 안됨
-- SELECT
--   id,
--   catch_date,
--   DATE(DATETIME(catch_datetime, 'Asia/Seoul')) AS catch_datetime_kr_date
-- FROM basic.trainer_pokemon;

-- SELECT
--   COUNT(DISTINCT id) as cnt
-- FROM basic.trainer_pokemon
-- WHERE
--   EXTRACT(YEAR FROM DATETIME(catch_datetime, 'Asia/Seoul')) = 2023 # catch_datetime은 TIMESTAMP로 저장되어 있으므로, DATETIME으로 변경해야 함
--   AND EXTRACT(MONTH FROM DATETIME(catch_datetime, 'Asia/Seoul')) = 1;
-- 결과 : 85개


# [⭕] 2. 배틀이 일어난 시간(battle_datetime)을 기준으로, 오전6시에서 오후6시 사이에 일어난 배틀의 수를 계산해주세요.

## mine
-- SELECT
--   COUNT(id) AS battle_cnt
-- FROM(
--   SELECT
--     id,
--     EXTRACT(HOUR FROM battle_datetime) as hour
--   FROM basic.battle  
-- )
-- WHERE
--   hour >= 6 AND hour <= 18;
-- 결과 : 44개


## A
-- battle_datetime, battle_timestamp 검증
-- SELECT
--   -- id,
--   -- battle_datetime,
--   -- DATETIME(battle_timestamp, 'Asia/Seoul') AS battle_timestamp_kr
--   COUNTIF(battle_datetime = DATETIME(battle_timestamp, 'Asia/Seoul')) AS same,
--   COUNTIF(battle_datetime != DATETIME(battle_timestamp, 'Asia/Seoul')) AS different
-- FROM basic.battle;
-- battle_timstamp를 서울 시간으로 변경하여 battle_datetime으로 저장한 것으로 확인

-- SELECT
--   COUNT(DISTINCT id) AS battle_cnt
-- FROM basic.battle
-- WHERE
-- --   EXTRACT(HOUR FROM battle_datetime) >= 6
-- --   AND EXTRACT(HOUR FROM battle_datetime) <= 18;
--   EXTRACT(HOUR FROM battle_datetime) BETWEEN 6 AND 18;
  # BETWEEN a AND b : a, b 사이에 있는 것을 반환
-- 결과 : 44개

# 2-1. 시간대별로 몇 건이 있는가?
-- SELECT
--   hour,
--   COUNT(DISTINCT id) as battle_cnt
-- FROM(
--   SELECT
--     *,
--     EXTRACT(HOUR FROM battle_datetime) AS hour
--     # hour라는 새로운 컬럼 생
--   FROM basic.battle
-- )
-- GROUP BY
--   hour
-- ORDER BY
--   hour



# [❌] 3. 각 트레이너별로 그들이 포켓몬을 포획한 첫 날(catch_date)를 찾고, 그 날짜를 'DD/MM/YYYY' 형식으로 출력해주세요.
--   (2024-01-01 => 01/01/2024)

## mine(서울 시간으로 변환 필요)
-- SELECT
--   trainer_id,
--   FORMAT_DATETIME('%d/%m/%Y', first_catch_date) AS formatted
-- FROM (
--   SELECT
--     trainer_id,
--     MIN(catch_date) AS first_catch_date
--   FROM basic.trainer_pokemon
--   GROUP BY
--     trainer_id
-- );

-- 트레이너별 최초 포획 날짜 제대로 출력 되었는지 확인
-- select
--   trainer_id,
--   catch_date
-- from `basic.trainer_pokemon`
-- where
--   trainer_id = 5 # 2023-01-07 확인
-- order by catch_date asc


## A
-- SELECT
--   trainer_id,
--   min_catch_date,
--   FORMAT_DATE('%d/%m/%Y', min_catch_date) AS new_min_catch_date
--   # FORMAT_DATE, FORMAT_DATETIME 함수들은 FORMAT_DATE, FORMAT_DATETIME 중 맞는 것으로 내부적으로 판단하여 알아서 적용.
-- FROM (
--   SELECT
--   -- 포획한 첫 날 + 날짜를 변경
--     trainer_id,
--     MIN(DATE(catch_datetime, 'Asia/Seoul')) AS min_catch_date
--   FROM basic.trainer_pokemon
--   GROUP BY
--     trainer_id
-- )
-- ORDER BY
--   trainer_id;
  ### ORDER BY : SELECT 제일 바깥에서 1번만 실행 추천.(모든 ROW를 확인해서 재정렬. 시간이 많이 걸림.)



# [⭕] 4. 배틀이 일어난 날짜(battle_date)를 기준으로,요일별로 배틀이 얼마나 자주 일어났는지 계산해주세요.

## mine
-- SELECT
--   day_of_week,
--   COUNT(id) AS battle_cnt
-- FROM(
--   SELECT
--     id,
--     EXTRACT(DAYOFWEEK FROM battle_date) AS day_of_week
--   FROM basic.battle
-- )
-- GROUP BY
--   day_of_week
-- ORDER BY
--   day_of_week ASC;


## A
-- SELECT
--   day_of_week,
--   COUNT(DISTINCT id) AS battle_cnt
-- FROM(
--   SELECT
--     *,
--     EXTRACT(DAYOFWEEK FROM battle_date) AS day_of_week
--   FROM basic.battle
-- )
-- GROUP BY
--   day_of_week
-- ORDER BY
--   day_of_week;

# [❌] 5. 트레이너가 포켓몬을 처음으로 포획한 날짜와 마지막으로 포획한 날짜의 간격이 큰 순으로 정렬하는 쿼리를 작성해주세요.

## mine(서울 시간으로 변환 필요)
-- SELECT
--   trainer_id,
--   DATETIME_DIFF(last_catch_date, first_catch_date, DAY) AS day_diff
-- FROM (
--   SELECT
--     trainer_id,
--     MIN(catch_datetime) AS first_catch_date,
--     MAX(catch_datetime) AS last_catch_date
--   FROM basic.trainer_pokemon
--   GROUP BY
--     trainer_id
-- )
-- ORDER BY
--   day_diff DESC;


## A
-- SELECT
--   *,
--   DATETIME_DIFF(max_catch_datetime, min_catch_datetime, DAY) AS diff
-- FROM(
--   SELECT
--     trainer_id,
--     MIN(DATETIME(catch_datetime, 'Asia/Seoul')) AS min_catch_datetime,
--     MAX(DATETIME(catch_datetime, 'Asia/Seoul')) AS max_catch_datetime,
--   FROM basic.trainer_pokemon
--   GROUP BY
--     trainer_id
-- )
-- ORDER BY
--   diff DESC;

