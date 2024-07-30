# JOIN 쿼리 작성하기

# left : trainer_pokemon
# right : trainer
# right : pokemon

SELECT
  tp.*,
  t.* except(id), # trainer_id => tp에 있는 컬럼이라서 제외
  p.* except(id) # pokemon_id => tp에 있는 컬럼이라서 제외
FROM basic.trainer_pokemon AS tp
LEFT JOIN basic.trainer AS t
ON tp.trainer_id = t.id
LEFT JOIN basic.pokemon AS p
ON tp.pokemon_id = p.id