# [⭕] 1. 각 트레이너별로 가진 포켓몬의 평균 레벨을 계산하고, 그 중 평균 레벨이 높은 TOP3 트레이너의 이름과 보유한 포켓몬의 수, 평균 레벨을 출력해주세요.
-- 필요한 정보
    -- trainer_pokemon > trainer_id, pokemon_id, level, status(Released 이외)
    -- trainer > id, name

-- sol
-- 검증 코드(name, mean)
    -- Blue, 74.5
    -- Caitlin, 71.5
SELECT
    tp.trainer_id,
    tp.pokemon_id,
    tp.level,
    tp.status,
    t.name
FROM trainer_pokemon AS tp
LEFT JOIN trainer AS t
ON tp.trainer_id = t.id
WHERE
    status != 'Released' AND
    name = 'Caitlin'

-- 풀이 코드
SELECT
    name,
    SUM(level) / COUNT(trainer_id) AS pokemon_level_mean
FROM(
    SELECT
        tp.trainer_id,
        tp.pokemon_id,
        tp.level,
        tp.status,
        t.name
    FROM trainer_pokemon AS tp
    LEFT JOIN trainer AS t
    ON tp.trainer_id = t.id
    WHERE
        status != 'Released'
) AS subquery
GROUP BY
    1
ORDER BY
    2 DESC
LIMIT 3
-- 결과(name, pokemon_level_mean)
    -- Blue, 74.5
    -- Caitlin, 71.5
    -- Spenser, 약 68.7





# [❌] 2. 각 포켓몬 타입1을 기준으로 가장 많이 포획된(방출 여부 상관없음) 포켓몬의 타입1, 포켓몬의 이름과 포획 횟수를 출력해주세요.
    # 내가 이해한 내용 : 타입1별 포켓몬 수 파악한 후 가장 많은 타입1의 이름과 개수
    # 문제 : 타입1, 포켓몬 이름 별 포획 횟수 파악 후 가장 많은 횟수
-- 필요한 정보
    -- trainer_pokemon > pokemon_id
    -- pokemon > id, kor_name, type1
-- 해결 방법
    -- type1별 포켓몬의 수 파악
    -- top1 type1의 포켓몬의 이름, type1, 포획 횟수 계산

-- sol
-- 검증 코드(type1, 개수)
    -- Water, 74
SELECT
    type1,
    COUNT(pokemon_id) AS pk_cnt
FROM trainer_pokemon AS tp
LEFT JOIN pokemon AS p
ON tp.pokemon_id = p.id
GROUP BY
    1
ORDER BY
    2 DESC


-- 풀이 코드
SELECT
    type1,
    kor_name,
    COUNT(pokemon_id) AS pk_cnt
FROM(
    SELECT
        pokemon_id,
        kor_name,
        type1
    FROM trainer_pokemon AS tp
    LEFT JOIN pokemon AS p
    ON tp.pokemon_id = p.id
) AS subquery
WHERE
    type1 = 'Water'
GROUP BY
    1, 2
ORDER BY
    3 DESC
-- 결과(kor_name, 개수)
    -- 잉어킹, 9
    -- 쥬쥬, 6


-- 수정 코드
SELECT
    type1,
    kor_name,
    COUNT(pokemon_id) AS pk_cnt
FROM trainer_pokemon AS tp
LEFT JOIN pokemon AS p
ON tp.pokemon_id = p.id
GROUP BY
    1, 2
ORDER BY
    pk_cnt DESC
LIMIT 5
-- result(type1, kor_name, cnt)
    -- Bug, 버터플, 9
    -- Water, 잉어킹, 9
    -- Normal, 잠만보, 9
    -- 동일한 개수이므로 가장많이 포획된 포켓몬은 3가지.





# [⭕] 3. 전설의 포켓몬을 보유한 트레이너들은 전설의 포켓몬과 일반 포켓몬을 얼마나 보유하고 있을까요? (트레이너의 이름을 같이 출력해주세요)
-- 필요한 정보
    -- trainer_pokemon > trainer_id, pokemon_id, status
    -- trainer > id, name
    -- pokemon > id, is_legendary
-- 해결 방법
    -- 1. trainer_pokemon(status가 Released가 아닌)과 pokemon 테이블 병합
    -- 2. 1번 결과 테이블에 trainer 테이블 병합
    -- 3. is_legendary가 1개 이상인 name별 is_legendary의 true, false 개수 출력

-- 검증 코드
SELECT
    id,
    is_legendary
FROM pokemon
WHERE
    is_legendary = True
ORDER BY
    id
-- 9개(pokemon_id) : 144, 145, 146, 150, 243, 244, 245, 249, 250

SELECT
    pokemon_id,
    trainer_id,
    is_legendary
FROM trainer_pokemon AS tp
LEFT JOIN pokemon AS p
ON tp.pokemon_id = p.id
WHERE
    tp.pokemon_id = 144
ORDER BY
    trainer_id
-- 결과(pokemon_id : trainer_id)
    -- 144 : 7, 22, 70
    -- 145 : 33

SELECT
    trainer_id,
    pokemon_id,
    is_legendary
FROM trainer_pokemon AS tp
LEFT JOIN pokemon AS p
ON tp.pokemon_id = p.id
WHERE
    tp.trainer_id = 33
ORDER BY
    pokemon_id
-- 결과(trainer_id : legendary 개수, not_legendary 개수)
    -- 7 : 1, 9
    -- 33 : 1, 3



-- 풀이 코드
SELECT
    trainer_id,
    name,
    SUM(CASE 
            WHEN is_legendary = True THEN 1
            ELSE 0
        END
    ) AS lg_cnt,
    SUM(CASE 
            WHEN is_legendary = False THEN 1 
            ELSE 0 
        END
    ) AS not_lg_cnt
FROM trainer_pokemon AS tp
LEFT JOIN pokemon AS p
ON tp.pokemon_id = p.id
LEFT JOIN trainer AS t
ON tp.trainer_id = t.id
WHERE
    status != 'Released' 
    -- AND is_legendary = True
GROUP BY
    1
HAVING
    lg_cnt != 0
ORDER BY
    1

-- 결과
-- 행	trainer_id	name	lg_cnt	not_lg_cnt
-- 1	7	Serena	1	9
-- 2	8	Lumios	1	6
-- 3	22	Cilan	1	5
-- 4	33	Mallow	1	3
-- 5	60	Barry	1	1
-- 6	70	Skyla	1	2
-- 7	88	Whitney	1	3





# 4. 가장 승리가 많은 트레이너 ID, 트레이너의 이름, 승리한 횟수, 보유한 포켓몬의 수, 평균 포켓몬의 레벨을 출력해주세요. 단, 포켓몬의 레벨은 소수점 2째 자리에서 반올림해주세요
-- 참고
    -- 반올림 함수 : ROUND



# 5. 트레이너가 잡았던 포켓몬의 총 공격력(attack)과 방어력(defense)의 합을 계산하고, 이 합이 가장 높은 트레이너를 찾으세요.



# 6. 각 포켓몬의 최고 레벨과 최저 레벨을 계산하고, 레벨 차이가 가장 큰 포켓몬의 이름을 출력하세요.



# 7. 각 트레이너가 가진 포켓몬 중에서 공격력(attack)이 100 이상인 포켓몬과 100 미만인 포켓몬의 수를 각각 계산해주세요. 트레이너의 이름과 두 조건에 해당하는 포켓몬의 수를 출력해주세요


