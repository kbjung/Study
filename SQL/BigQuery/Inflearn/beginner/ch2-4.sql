# 1. trainer 테이블의 모든 데이터를 보여주는 쿼리 작성
SELECT 
    * 
FROM basic.trainer

# 2. trainer 테이블에 있는 트레이너의 name을 출력하는 쿼리를 작성해주세요
SELECT 
    name 
FROM basic.trainer

# 3. trainer 테이블에 있는 트레이너의 name, age를 출력하는 쿼리를 작성해주세요
SELECT 
    name, 
    age 
FROM basic.trainer

# 4. trainer의 테이블에서 id가 3인 트레이너의 name, age, hometown을 출력하는 쿼리를 작성해주세요
SELECT 
    name, 
    age, 
    hometown 
FROM basic.trainer
WHERE 
    id = 3

# 5. pokemon 테이블에서 “피카츄”의 공격력과 체력을 확인할 수 있는 쿼리를 작성해주세요
SELECT 
    attack, 
    hp 
FROM basic.pokemon
WHERE 
    kor_name = '피카츄'