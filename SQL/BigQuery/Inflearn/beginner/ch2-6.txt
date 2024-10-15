-- Active: 1728371458573@@127.0.0.1@3306@inflearn-bigquery
# 1. 포켓몬 중에 type2가 없는 포켓몬의 수를 작성하는 쿼리를 작성해주세요
# (힌트) ~가 없다 : 컬럼 IS NULL

SELECT 
    COUNT(id)
FROM pokemon
WHERE
    type2 IS NULL
# result : 135



# 2. type2가 없는 포켓몬의 type1과 type1의 포켓몬 수를 알려주는 쿼리를 작성해주세요. 단, type1의 포켓몬 수가 큰 순으로 정렬해주세요
SELECT
    type1,
    COUNT(id)
FROM pokemon
WHERE
    type2 IS NULL
GROUP BY
    type1
ORDER BY
    2 DESC



# 3. type2 상관없이 type1의 포켓몬 수를 알 수 있는 쿼리를 작성해주세요
SELECT
    type1,
    COUNT(id)
FROM pokemon
GROUP BY
    1



# 4.전설 여부에 따른 포켓몬 수를 알 수 있는 쿼리를 작성해주세요
SELECT
    is_legendary,
    COUNT(id)
FROM pokemon
GROUP BY
    1
ORDER BY
    1 DESC


# 5. 동명 이인이 있는 이름은 무엇일까요? (한번에 찾으려고 하지 않고 단계적으로 가도 괜찮아요)
SELECT
    name,
    COUNT(id) AS cnt
FROM trainer
GROUP BY
    1
HAVING
    cnt > 1



# 6. trainer 테이블에서 “Iris” 트레이너의 정보를 알 수 있는 쿼리를 작성해주세요
SELECT *
FROM trainer
WHERE
    name = 'Iris'



# 7. trainer 테이블에서 “Iris”, ”Whitney”, “Cynthia” 트레이너의 정보를 알 수 있는 쿼리를 작성해주세요
# (힌트)컬럼IN("Iris", "Whitney", "Cynthia")
SELECT *
FROM trainer
WHERE
    name IN ('Iris', 'Whitney', 'Cynthia')


# 8. 전체 포켓몬 수는 얼마나 되나요?
SELECT
    COUNT(id)
FROM pokemon



# 9. 세대(generation)별로 포켓몬 수가 얼마나 되는지 알 수 있는 쿼리를 작성해주세요
SELECT
    generation,
    COUNT(id)
FROM pokemon
GROUP BY
    1
ORDER BY
    2 DESC



# 10. type2가 존재하는 포켓몬의 수는 얼마나 되나요?
SELECT
    COUNT(id)
FROM pokemon
WHERE
    type2 IS NOT NULL



# 11. type2가 있는 포켓몬 중에 제일 많은 type1은 무엇인가요?
SELECT
    type1,
    COUNT(id)
FROM pokemon
WHERE
    type2 IS NOT NULL
GROUP BY
    1
ORDER BY
    2 DESC
LIMIT 1



# 12. 단일(하나의 타입만 있는)타입 포켓몬 중 많은 type1은 무엇일까요?
SELECT
    type1,
    COUNT(id)
FROM pokemon
WHERE
    type2 IS NULL
GROUP BY
    1
ORDER BY
    2 DESC
LIMIT 1



# 13. 포켓몬의 이름에 "파"가 들어가는 포켓몬은 어떤 포켓몬이 있을까요?
# (힌트) 컬럼 LIKE "파%"
SELECT
    kor_name
FROM pokemon
WHERE
    kor_name LIKE "%파%"



# 14. 뱃지가 6개 이상인 트레이너는 몇 명이 있나요?
SELECT
    COUNT(id)
FROM trainer
WHERE
    badge_count >= 6


# 15. 트레이너가 보유한 포켓몬(trainer_pokemon)이 제일 많은 트레이너는 누구일까요?
SELECT 
    trainer_id,
    COUNT(pokemon_id)
FROM trainer_pokemon
WHERE
    status <> 'Released'
GROUP BY
    1
ORDER BY
    2 DESC



# 16. 포켓몬을 많이 풀어준 트레이너는 누구일까요?
SELECT
    trainer_id,
    COUNT(pokemon_id)
FROM trainer_pokemon
WHERE
    status = 'Released'
GROUP BY
    1
ORDER BY
    2 DESC


# 17. 트레이너 별로 풀어준 포켓몬의 비율이 20%가 넘는 포켓몬 트레이너는 누구일까요?
# 풀어준 포켓몬의 비율=(풀어준 포켓몬 수/전체 포켓몬의 수)
# (힌트)COUNTIF(조건)
SELECT
    trainer_id,
    SUM(CASE WHEN status = 'Released' THEN 1 ELSE 0 END) / COUNT(pokemon_id) AS released_ratio,
    SUM(CASE WHEN status = 'Released' THEN 1 ELSE 0 END) AS released_pokemon_cnt,
    COUNT(pokemon_id) AS total_pokemon_cnt
FROM trainer_pokemon
GROUP BY
    1
HAVING
    released_ratio > 0.2
