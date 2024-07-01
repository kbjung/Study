# 1. pokemon 테이블에 있는 포켓몬 수를 구하는 쿼리를 작성해주세요
SELECT 
  COUNT(DISTINCT kor_name) 
FROM basic.pokemon

  -- A
SELECT 
  COUNT(id) AS cnt
FROM basic.pokemon


# 2. 포켓몬의 수가 세대별로 얼마나 있는지 알 수 있는 쿼리를작성해주세요
SELECT 
  generation, count(kor_name) AS cnt
FROM basic.pokemon
GROUP BY 
  generation

  -- A
SELECT
  generation,
  COUNT(id) AS cnt
FROM basic.pokemon
GROUP BY
  generation


# 3. 포켓몬의 수를 타입 별로 집계하고, 포켓몬의 수가 10 이상인 타입만 남기는 쿼리를 작성해주세요. 포켓몬의 수가 많은 순으로 정렬해주세요
SELECT 
  type1, 
  count(*) AS cnt
FROM basic.pokemon
GROUP BY 
  type1
HAVING 
  cnt >= 10
ORDER BY cnt DESC

  -- A
SELECT 
  type1,
  COUNT(id) AS cnt
FROM basic.pokemon
GROUP BY
  type1
HAVING cnt >= 10
ORDER BY cnt DESC
