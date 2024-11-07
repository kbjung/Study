-- Active: 1728371458573@@127.0.0.1@3306@inflearn-bigquery

# 1. 트레이너가 포켓몬을 포획한 날짜(catch_date)를 기준으로, 2023년 1월에 포획한 포켓몬의 수를 계산해주세요.(trainer_pokemon)

# BigQuery
SELECT
  count(id) as cnt
FROM(
  SELECT
    id,
    catch_date,
    DATE(DATETIME(catch_datetime, 'Asia/Seoul')) AS  catch_datetime_kr
  FROM basic.trainer_pokemon
)
WHERE
  EXTRACT(YEAR FROM catch_datetime_kr) = 2023 and
  EXTRACT(MONTH FROM catch_datetime_kr) = 1

# MariaDB
SELECT *
FROM trainer_pokemon

SELECT
    COUNT(id)
FROM(
    SELECT
        id,
        catch_datetime,
        DATE_ADD(catch_datetime, INTERVAL 9 HOUR) AS catch_datetime_kr # catch_datetime은 UTC 기준. 서울은 UTC+9.
    FROM trainer_pokemon
) AS subquery
WHERE
    YEAR(catch_datetime_kr) = 2023 AND
    MONTH(catch_datetime_kr) = 1
-- result : 85

SELECT
    COUNT(id)
FROM trainer_pokemon
WHERE
    YEAR(DATE_ADD(catch_datetime, INTERVAL 9 HOUR)) = 2023 AND
    MONTH(DATE_ADD(catch_datetime, INTERVAL 9 HOUR)) = 1
-- result : 85



# 2. 배틀이 일어난 시간(battle_datetime)을 기준으로, 오전6시에서 오후6시 사이에 일어난 배틀의 수를 계산해주세요.(battle)
SELECT *
FROM battle # battle_date, battle_datetime 은 서울 시간대, battle_timestamp은 UTC.

SELECT
    COUNT(id)
FROM battle
WHERE
    HOUR(battle_datetime) BETWEEN 6 AND 18
-- result : 44



# 2-1. 시간대별로 몇 건이 있는가?
SELECT
    hour,
    COUNT(id) AS battle_cnt
FROM (
    SELECT
        id,
        HOUR(battle_datetime) AS hour
    FROM battle
) AS subquery
GROUP BY 1
ORDER BY 1



SELECT
    HOUR(battle_datetime) AS hour, 
    COUNT(id) AS battle_cnt
FROM battle
GROUP BY 1
ORDER BY 1



# 3. 각 트레이너별로 그들이 포켓몬을 포획한 첫 날(catch_date)를 찾고, 그 날짜를 'DD/MM/YYYY' 형식으로 출력해주세요.(trainer_pokemon)
-- (2024-01-01 => 01/01/2024)
SELECT *
FROM trainer_pokemon
-- catch_datetime : UTC.

SELECT
    trainer_id,
    min_catch_date,
    DATE_FORMAT(min_catch_date, '%d/%m/%Y') AS new_min_catch_date
FROM (
    SELECT
        trainer_id,
        MIN(DATE(DATE_ADD(catch_datetime, INTERVAL 9 HOUR))) AS min_catch_date
    FROM trainer_pokemon
    GROUP BY
        1
) AS subquery
ORDER BY
    1




# 4. 배틀이 일어난 날짜(battle_date)를 기준으로,요일별로 배틀이 얼마나 자주 일어났는지 계산해주세요.(battle)
SELECT
    *
FROM battle
LIMIT 1

SELECT
    DAYOFWEEK(battle_date) AS day,
    COUNT(id) as battle_cnt
FROM battle
GROUP BY
    1



# 5. 트레이너가 포켓몬을 처음으로 포획한 날짜와 마지막으로 포획한 날짜의 간격이 큰 순으로 정렬하는 쿼리를 작성해주세요.(trainer_pokemon)
SELECT *
FROM trainer_pokemon
LIMIT 1

SELECT
    trainer_id,
    DATEDIFF(MAX(DATE_ADD(catch_datetime, INTERVAL 9 HOUR)), MIN(DATE_ADD(catch_datetime, INTERVAL 9 HOUR))) AS diff
FROM trainer_pokemon
GROUP BY
    1
ORDER BY
    2 DESC


SELECT
    trainer_id,
    DATEDIFF(max_catch_datetime, min_catch_datetime) AS diff
FROM (
    SELECT
        trainer_id,
        MIN(DATE_ADD(catch_datetime, INTERVAL 9 HOUR)) AS min_catch_datetime,
        MAX(DATE_ADD(catch_datetime, INTERVAL 9 HOUR)) AS max_catch_datetime
    FROM trainer_pokemon
    GROUP BY
        1
) AS subquery
GROUP BY
    1
ORDER BY
    diff DESC