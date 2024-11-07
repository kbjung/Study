-- Active: 1728371458573@@127.0.0.1@3306@inflearn-bigquery

SELECT *
FROM trainer
LIMIT 1

SELECT
    COUNT(id),
    COUNT(DISTINCT id)
FROM trainer

SELECT *
FROM trainer_pokemon
LIMIT 1

SELECT
    COUNT(trainer_id),
    COUNT(DISTINCT trainer_id)
FROM trainer_pokemon

SELECT
    * # EXCEPT 구문 사용 불가
FROM trainer_pokemon AS tp
LEFT JOIN trainer AS t
ON tp.trainer_id = t.id

# trainer.id 제외(trainer_pokemon.trainer_id와 trainer.id는 join의 기준으로 쓰여서 중복되므로 trainer.id 제거)
SELECT
    tp.id,
    trainer_id,
    pokemon_id,
    level,
    experience_point,
    current_health,
    catch_date,
    catch_datetime,
    location,
    status,
    name,
    age,
    hometown,
    preferred_pokemon_type,
    badge_count,
    achievement_level
FROM trainer_pokemon AS tp
LEFT JOIN trainer AS t
ON tp.trainer_id = t.id