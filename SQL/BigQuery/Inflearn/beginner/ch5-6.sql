# [⭕] 1. 트레이너가 보유한 포켓몬들은 얼마나 있는지 알 수 있는 쿼리를 작성해주세요
  ## 참고. 보유했다의 정의는 status가 Active, Training인 경우를 의미. Released는 방출했다는 것을 의미

### mine
  # 참고 테이블 : trainer_pokemon, pokemon
  # join key : trainer_pokemon.pokemon_id, pokemon.id
  # 조건 : trainer_pokemon.status가 Active 또는 Training

select
  kor_name,
  count(pokemon_id) as pokemon_cnt
from basic.trainer_pokemon as tp
left join basic.pokemon as p
on tp.pokemon_id = p.id
where 
  tp.status in ('Active', 'Training')
group by kor_name
order by pokemon_cnt desc;



### A
select
  -- tp.*,
  -- p.id,
  -- p.kor_name
  kor_name,
  count(tp.id) as pokemon_cnt
  -- join할 때 자주 나올 수 있는 에러 <Column name id is ambiguous as> : id가 모호하다. 더 구체적으로(Specific하게) 말해달라
  -- join에서 사용하는 테이블에 중복된 컬럼의이름이 있으면 꼭 어떤 테이블의 컬럼인지 명시해야 함
  -- id -> tp.id
from(
  -- 복잡하다 => 가독성 있는 쿼리 => with 문
select
  id,
  trainer_id,
  pokemon_id,
  status
from basic.trainer_pokemon
where
  status in ('Active', 'Training') # 먼저 행수를 줄인다(계산량을 줄이기 위해)
) as tp
left join basic.pokemon as p
on tp.pokemon_id = p.id
-- where
--   status in ('Active', 'Training')
--   1=1 # 1=1은 무조건 TRUE를 반환, 모든 행을 출력
--   and status = 'Active'
--   and status = 'Training'
--   쿼리를 작성할 때 값을 바꿔가면서 실행해야 함 => 빨리 주석처리하기 위해서 앞에선 TRUE인 1=1을 넣고, AND 쓰고 빠르게 주석 처리
--   그러면 여기서 where 쓰는게 더 좋은거 아니었나? => 먼저 데이터를 줄이고 하는 것이 더 좋다!
group by 
  kor_name
order by 
  pokemon_cnt desc







# [⭕] 2. 각 트레이너가 가진 포켓몬 중에서 'Grass' 타입의 포켓몬 수를 계산해주세요(단, 편의를 위해 type1 기준으로 계산해주세요)

### mine
  # 참고 테이블 : trainer_pokemon, pokemon
  # join key : trainer_pokemon.pokemon_id, pokemon.id
  # 조건 : trainer_pokemon.status가 Active 또는 Training이면서, pokemon.type1이 Grass

select
  count(pokemon_id) as pokemon_cnt
from basic.trainer_pokemon as tp
left join basic.pokemon as p
on tp.pokemon_id = p.id
where 
  p.type1 = 'Grass'
  and
  tp.status in ('Active', 'Training'); # 결과 23



### A
select
  -- tp.*,
  p.type1,
  count(tp.id) as pokemon_cnt
from(
select
  id,
  trainer_id,
  pokemon_id,
  status
from basic.trainer_pokemon
where
  status in ('Active', 'Training')
) as tp
left join basic.pokemon as p
on tp.pokemon_id = p.id
where
  type1 = 'Grass'
group by
  type1
order by
  2 desc # 2 대신에 pokemon_cnt도 가능
  # 결과 23







# [⭕] 3. 트레이너의 고향(hometown)과 포켓몬을 포획한 위치(location)를 비교하여, 자신의 고향에서 포켓몬을 포획한 트레이너의 수를 계산해주세요.
  ## 참고. status 상관없이 구해주세요

### mine
  # 참고 테이블 : trainer_pokemon, trainer
  # join key : trainer_pokemon.trainer_id, trainer.id
  # 조건 : trainer.hometown과 trainer_pokemon.location이 같은 트레이너

select
  count(distinct trainer_id) as trainer_cnt
from basic.trainer_pokemon as tp
left join basic.trainer as t
on tp.trainer_id = t.id
where tp.location = t.hometown; # 결과 : 28



### A
select
  count(distinct tp.trainer_id) as trainer_uniq,# 트레이너의 수 => 28명
  -- count(tp.trainer_id) as trainer_cnt, # 트리이너의 고향과 포켓몬이 잡힌 위치가 같은 건 43개
from basic.trainer as t
left join basic.trainer_pokemon as tp
on t.id = tp.trainer_id
where
  location is not null # 현재 데이터에서는 없어도 결과는 동일. 현업에서는 점검 필수
  and t.hometown = tp.location








# [⭕] 4. Master 등급인 트레이너들은 어떤 타입(type1)의 포켓몬을 제일 많이 보유하고 있을까요?
  ## 참고. 보유했다의 정의는 1번 문제의 정의와 동일

### mine
  # 참고 테이블 : trainer_pokemon, trainer, pokemon
  # join key
    # trainer_pokemon.trainer_id, trainer.id
    # trainer_pokemon.pokemon_id, pokemon.id
  # 조건
    # trainer_pokemon.status가 Active 또는 Training
    # trainer.achievement_level가 Master
    # group by : pokemon.type1

select
  type1,
  count(pokemon_id) as pokemon_cnt
  -- tp.id,
  -- trainer_id,
  -- pokemon_id,
  -- status,
  -- type1,
  -- achievement_level
from basic.trainer_pokemon as tp
left join basic.trainer as t
on tp.trainer_id = t.id
left join basic.pokemon as p
on tp.pokemon_id = p.id
where
  achievement_level = 'Master'
  and
  status in ('Active', 'Training')
group by type1
order by pokemon_cnt desc
limit 1; # 결과 : Water, 14마리



### A
select
  type1,
  count(tp.id) as pokemon_cnt
from(
select
  id,
  trainer_id,
  pokemon_id,
  status
from basic.trainer_pokemon
where
  status in ('Active', 'Training')
) as tp
left join basic.pokemon as p
on tp.pokemon_id = p.id
left join basic.trainer as t
on tp.trainer_id = t.id
where
  t.achievement_level = 'Master'
group by
  type1
order by
  2 desc
limit 1






# [❌] 5. Incheon 출신 트레이너들은 1세대, 2세대 포켓몬을 각각 얼마나 보유하고 있나요?

### mine
  # 참고 테이블 : trainer_pokemon, trainer, pokemon
  # join key
    # trainer_pokemon.trainer_id, trainer.id
    # trainer_pokemon.pokemon_id, pokemon.id
  # 조건
    # trainer.hometown = Incheon
    # group by : pokemon.generation


select
  generation,
  count(pokemon_id) as pokemon_cnt
from basic.trainer_pokemon as tp
left join basic.trainer as t
on tp.trainer_id = t.id
left join basic.pokemon as p
on tp.pokemon_id = p.id
where 
  hometown = 'Incheon'
group by 
  generation; # 결과 : 1세대 40마리, 2세대 14마리
-- 보유 조건 빠짐.


### A
select
  generation,
  count(tp.id) as pokemon_cnt
from(
select
  id,
  trainer_id,
  pokemon_id,
  status
from basic.trainer_pokemon
where
  status in ('Active', 'Training')
) as tp
left join basic.trainer as t
on tp.trainer_id = t.id
left join basic.pokemon as p
on tp.pokemon_id = p.id
where
  hometown = 'Incheon'
group by
  generation
order by
  2 desc # 결과 : 1세대 33, 2세대 10

