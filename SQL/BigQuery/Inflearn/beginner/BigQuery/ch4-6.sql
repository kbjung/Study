# CASE WHEN

# Rock이나 Ground 타입을 Rock&Ground 타입으로 변경하여 새로운 타입 분류 통계(pokemon)

SELECT
  new_type1,
  COUNT(DISTINCT id) as pokemon_cnt
FROM(
  SELECT
    *, 
    CASE 
      WHEN (type1 IN ('Rock', 'Ground')) OR (type2 IN ('Rock', 'Ground')) THEN 'Rock&Ground'
    ELSE type1
    END AS new_type1
  FROM basic.pokemon
)
GROUP BY
  new_type1;

# 포켓몬의 공격력을 50이상이면 Strong, 100이상이면 Very Strong, 나머진 Weak으로 분류(pokemon)
-- 조건문 순서 중요성
-- case문의 첫번째 조건에 확인된 줄은 다음 조건에서 확인하지 않는다
-- 주의! 조건문 순서에 따라 결과가 다라진다

SELECT
  eng_name,
  attack,
  CASE
    WHEN attack >= 100 THEN 'Very Strong'
    WHEN attack >= 50 THEN 'Strong'
  ELSE 'Weak'
  END AS attack_level
FROM basic.pokemon;

-- 올바르지 않은 결과

SELECT
  eng_name,
  attack,
  CASE
    WHEN attack >= 50 THEN 'Strong'
    WHEN attack >= 100 THEN 'Very Strong'
  ELSE 'Weak'
  END AS attack_level
FROM basic.pokemon;

# IF 조건문
-- 단일 조건일 경우 유용
-- 문법
-- IF(조건문, True일 때의 값, False일 때의 값) AS 새로운_컬럼_이름

# 조건문 정리
-- CASE WHEN : 여러 조건이 있을 경유 사용. 조건의 순서에 주의
-- IF : 단일 조건일 경우 사용
