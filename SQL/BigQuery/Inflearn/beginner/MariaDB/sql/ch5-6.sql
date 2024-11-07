-- Active: 1728371458573@@127.0.0.1@3306@inflearn-bigquery

# 1. 트레이너가 보유한 포켓몬들은 얼마나 있는지 알 수 있는 쿼리를 작성해주세요
    # 참고. 보유했다의 정의는 status가 Active, Training인 경우를 의미. Released는 방출했다는 것을 의미

### mine
    # Active, Training 된 포켓몬 이름별 포켓몬 수
    # 참고 테이블 : trainer_pokemon, pokemon
    # join key : trainer_pokemon.pokemon_id, pokemon.id
    # 조건 : trainer_pokemon.status가 Active 또는 Training
  
SELECT *
FROM trainer_pokemon
LIMIT 1

SELECT
    COUNT(pokemon_id),
    COUNT(DISTINCT(pokemon_id))
FROM trainer_pokemon
WHERE status IN ('Active', 'Training')


SELECT *
FROM pokemon
LIMIT 1

-- sol1
SELECT
    p.kor_name,
    COUNT(tp.pokemon_id) AS pokemon_cnt
FROM trainer_pokemon AS tp
LEFT JOIN pokemon AS p
ON tp.pokemon_id = p.id
WHERE
    tp.status IN ('Active', 'Training')
GROUP BY 1
ORDER BY 2 DESC



-- sol2
SELECT
    p.kor_name,
    COUNT(tp.pokemon_id) AS pokemon_cnt
FROM(
    SELECT
        pokemon_id,
        status
    FROM trainer_pokemon
    WHERE
        status IN ('Active', 'Training')
) AS tp
LEFT JOIN pokemon AS p
ON tp.pokemon_id = p.id
GROUP BY 1
ORDER BY 2 DESC



# 2. 각 트레이너가 가진 포켓몬 중에서 'Grass' 타입의 포켓몬 수를 계산해주세요(단, 편의를 위해 type1 기준으로 계산해주세요)

### mine
    # Grass 타입의 포켓몬중 Active, Training인 포켓몬의 총 수
    # 참고 테이블 : trainer_pokemon, pokemon
    # join key : trainer_pokemon.pokemon_id, pokemon.id
    # 조건 : trainer_pokemon.status가 Active 또는 Training이면서, pokemon.type1이 Grass

SELECT *
FROM trainer_pokemon
LIMIT 1

SELECT *
FROM pokemon
LIMIT 1

-- sol1
SELECT
    COUNT(tp.pokemon_id) AS pokemon_cnt
FROM trainer_pokemon AS tp
LEFT JOIN pokemon AS p
ON tp.pokemon_id = p.id
WHERE
    tp.status IN ('Active', 'Training') AND
    p.type1 = 'Grass'
# 결과 23

-- sol2
SELECT
    COUNT(tp.pokemon_id) AS pokemon_cnt
FROM(
    SELECT
        pokemon_id
    FROM trainer_pokemon
    WHERE
        status IN ('Active', 'Training')
) AS tp
LEFT JOIN pokemon AS p
ON tp.pokemon_id = p.id
WHERE p.type1 = 'Grass'
# 결과 23



# 3. 트레이너의 고향(hometown)과 포켓몬을 포획한 위치(location)를 비교하여, 자신의 고향에서 포켓몬을 포획한 트레이너의 수를 계산해주세요.
    # 참고. status 상관없이 구해주세요

### mine
    # 트레이너 고향과 포켓몬 포획한 위치가 같은 트레이너의 수
    # 참고 테이블 : trainer_pokemon, trainer
    # join key : trainer_pokemon.trainer_id, trainer.id
    # 조건 : trainer.hometown과 trainer_pokemon.location이 같은 트레이너

SELECT *
FROM trainer_pokemon
LIMIT 1

SELECT *
FROM trainer
LIMIT 1

## 각 조건의 컬럼들 결측값 확인
SELECT
    COUNT(trainer_id) AS cnt
FROM trainer_pokemon
WHERE 
    location IS NULL
# location 빈값 : 0

SELECT
    COUNT(hometown) AS cnt
FROM trainer
WHERE 
    hometown IS NULL
# hometown 빈값 : 0

SELECT
    COUNT(DISTINCT tp.trainer_id) AS trainer_uniq
FROM trainer_pokemon AS tp
LEFT JOIN trainer AS t
ON tp.trainer_id = t.id
WHERE 
    t.hometown = tp.location AND
    tp.location IS NOT NULL AND # 각 조건의 컬럼들 결측값 확인
    t.hometown IS NOT NULL # 각 조건의 컬럼들 결측값 확인

# 결과 : 28

# 결과에 영향을 미치는 컬럼들의 결측값 확인 방법
    # 1. 각 조건의 컬럼들 결측값을 따로 확인
    # 2. 결과 출력시 조건을 건다



# 4. Master 등급인 트레이너들은 어떤 타입(type1)의 포켓몬을 제일 많이 보유하고 있을까요?
    ## 참고. 보유했다의 정의는 1번 문제의 정의와 동일

### mine
    # Master 등급의 트레이너가 보유한 포켓몬의 type1별 포켓몬의 수
    # 참고 테이블 : trainer_pokemon, trainer, pokemon
    # join key
        # trainer_pokemon.trainer_id, trainer.id
        # trainer_pokemon.pokemon_id, pokemon.id
    # 조건
        # trainer_pokemon.status가 Active 또는 Training
        # trainer.achievement_level가 Master
        # group by : pokemon.type1

-- sol1
SELECT *
FROM trainer_pokemon
LIMIT 1 # trainer_id, pokemon_id, status

SELECT *
FROM trainer
LIMIT 1 # id, achievement_level

SELECT *
FROM pokemon
LIMIT 1 # id, type1

-- sol1
SELECT
    p.type1,
    COUNT(tp.pokemon_id) AS pokmon_cnt
FROM trainer_pokemon AS tp
LEFT JOIN trainer AS t
ON tp.trainer_id = t.id
LEFT JOIN pokemon AS p
ON tp.pokemon_id = p.id
WHERE
    tp.status IN ('Active', 'Training') AND
    t.achievement_level = 'Master'
GROUP BY 1
ORDER BY 2 DESC
LIMIT 1

-- sol2
SELECT
    p.type1,
    COUNT(tp.pokemon_id) AS pokemon_cnt
FROM(
    SELECT
        trainer_id,
        pokemon_id,
        status
    FROM trainer_pokemon
    WHERE
        status IN ('Active', 'Training')
) AS tp
LEFT JOIN trainer AS t
ON tp.trainer_id = t.id
LEFT JOIN pokemon AS p
ON tp.pokemon_id = p.id
WHERE
    t.achievement_level = 'Master'
GROUP BY 1
ORDER BY 2 DESC
LIMIT 1



# 5. Incheon 출신 트레이너들은 1세대, 2세대 포켓몬을 각각 얼마나 보유하고 있나요?

### mine
    # Incheon 출신 트레이너들이 보유한 세대별 포켓몬의 수
    # 참고 테이블 : trainer_pokemon, trainer, pokemon
    # join key
        # trainer_pokemon.trainer_id, trainer.id
        # trainer_pokemon.pokemon_id, pokemon.id
    # 조건
        # trainer.hometown = Incheon
        # group by : pokemon.generation

SELECT *
FROM trainer_pokemon
LIMIT 1 # trainer_id, pokemon_id, status

SELECT *
FROM trainer
LIMIT 1 # id, hometown

SELECT *
FROM pokemon
LIMIT 1 # id, generation

-- sol1
SELECT
    p.generation,
    COUNT(tp.pokemon_id) AS pokemon_cnt
FROM trainer_pokemon AS tp
LEFT JOIN trainer AS t
ON tp.trainer_id = t.id
LEFT JOIN pokemon AS p
ON tp.pokemon_id = p.id
WHERE
    tp.status IN ('Active', 'Training') AND
    t.hometown = 'Incheon'
GROUP BY 1
ORDER BY 2 DESC

-- sol2
SELECT
    p.generation,
    COUNT(tp.pokemon_id) AS pokemon_cnt
FROM (
    SELECT
        trainer_id,
        pokemon_id
    FROM trainer_pokemon
    WHERE
        status IN ('Active', 'Training')
) AS tp
LEFT JOIN trainer AS t
ON tp.trainer_id = t.id
LEFT JOIN pokemon AS p
ON tp.pokemon_id = p.id
WHERE
    t.hometown = 'Incheon'
GROUP BY 1
ORDER BY 2 DESC