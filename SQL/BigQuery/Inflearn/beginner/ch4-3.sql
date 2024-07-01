-- 문자열 붙이기 : CONCAT
SELECT
  CONCAT('안녕', '하세요', '!') AS result;
-- FROM 이 없는데 어떻게 동작하지?
-- CONCAT 인자로 STRING이나 숫자를 넣을 때는 데이터를 직접 넣어준 것 => FROM이 없어도 실행



-- 문자열 분리하기 : SPLIT
SELECT
  SPLIT('가, 나, 다, 라', ', ') AS result;
-- SPLIT(문자열 원본, 나눌 기준이 되는 문자)



-- 특정 단어 수정하기 : REPLACE
SELECT
  REPLACE('안녕하세요', '안녕', '실천') AS result;
-- REPLACE(문자열 원본, 찾을 단어, 바꿀 단어)



-- 문자열 자르기 : TRIM
SELECT
  TRIM('안녕하세요', '하세요') AS result;
-- TRIM(문자열 원본, 자를 단어)



-- 영어 소문자를 대문자로 변경 : UPPER
SELECT
  UPPER('abc') AS result;

SELECT
  UPPER('aBc') AS result;
