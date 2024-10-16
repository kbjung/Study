-- Active: 1728371458573@@127.0.0.1@3306@inflearn-bigquery
-- 예시
WITH PlayerStats AS (
  SELECT 'Adams' as LastName, 'Noam' as FirstName, 3 as PointsScored UNION ALL
  SELECT 'Buchanan', 'Jie', 0 UNION ALL
  SELECT 'Coolidge', 'Kiran', 1 UNION ALL
  SELECT 'Adams', 'Noam', 4 UNION ALL
  SELECT 'Buchanan', 'Jie', 13)
SELECT
    *
FROM PlayerStats

WITH PlayerStats AS (
  SELECT 'Adams' as LastName, 'Noam' as FirstName, 3 as PointsScored UNION ALL
  SELECT 'Buchanan', 'Jie', 0 UNION ALL
  SELECT 'Coolidge', 'Kiran', 1 UNION ALL
  SELECT 'Adams', 'Noam', 4 UNION ALL
  SELECT 'Buchanan', 'Jie', 13)
SELECT
    -- *
  SUM(PointsScored) AS total_points,
  FirstName AS first_name,
  LastName AS last_name
FROM PlayerStats
GROUP BY
    first_name,
    last_name

/*--------------+------------+-----------+
 | total_points | first_name | last_name |
 +--------------+------------+-----------+
 | 7            | Noam       | Adams     |
 | 13           | Jie        | Buchanan  |
 | 1            | Kiran      | Coolidge  |
 +--------------+------------+-----------*/


-- test
SELECT
  trainer_id,
  COUNT(pokemon_id) AS pokemon_cnt
FROM trainer_pokemon
WHERE
  status = 'Released'
GROUP BY
  trainer_id
ORDER BY
  pokemon_cnt DESC
LIMIT 1

-- 동일한 기능
SELECT
  trainer_id,
  COUNT(pokemon_id) AS pokemon_cnt
FROM basic.trainer_pokemon
WHERE
  status = 'Released'
GROUP BY ALL # ALL : SELECT에서 집계할 컬럼을 알아서 인식하여 지정해준다(bigquery에서 새로 생긴 함수, mariaDB에서는 사용 불가)
ORDER BY
  pokemon_cnt DESC
LIMIT 1
