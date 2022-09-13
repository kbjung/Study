use market_db;

CREATE TABLE hongong4 (
    tinyint_col TINYINT,
    smallint_col SMALLINT,
    int_col INT,
    bigint_col BIGINT
);

insert into hongong4 values(127, 32767, 2147483647, 9000000000000000000);
insert into hongong4 values(128, 32768, 2147483648, 90000000000000000000);-- 범위 밖의 숫자가 입력되어 오류 발생

CREATE TABLE big_table (
    data1 CHAR(255)
);

drop table big_table;
CREATE TABLE big_table (
    data1 CHAR(256)
); -- 오류발생, char는 255까지 표현가능.

drop table big_table;
CREATE TABLE big_table (
    data2 VARCHAR(16383)
);

drop table big_table;
CREATE TABLE big_table (
    data2 VARCHAR(16384)
); -- 오류발생, varchar는 16383까지 표현가능.

create database netflix_db;
use netflix_db;
CREATE TABLE movie (
    movie_id INT,
    movie_title VARCHAR(30),
    movie_director VARCHAR(20),
    movie_star VARCHAR(20),
    movie_script LONGTEXT,
    movie_film LONGBLOB
);

-- 변수의 사용

use market_db;
set @myVar1 = 5;
set @myVar2 = 4.25;

SELECT @myVar1;
SELECT @myVar1 + @myVar2; -- 변수에 임시 저장됨.

set @txt = '가수 이름 ==> ';
set @height = 166;
SELECT 
    @txt, mem_name
FROM
    member
WHERE
    height > @height;

set @count = 3;
select mem_name, height from member order by height limit @count; -- 오류. limit에 변수사용은 문법적으로 지원하지 않음.

set @count = 3;
prepare mySQL from 'select mem_name, height from member order by height limit ?';
execute mySQL using @count;

SELECT AVG(price) '평균 가격'
	FROM buy;-- 결과 : 142.9167

-- 명시적 형변환
SELECT CAST(AVG(price) AS SIGNED) '평균 가격'
	FROM buy;-- 부호가 있는 정수로 나타냄. 결과 : 143

SELECT CAST('2022$12$12' AS DATE);-- 2022-12-12 출력
SELECT CAST('2022/12/12' AS DATE);-- 동일한 결과
SELECT CAST('2022%12%12' AS DATE);-- 동일한 결과
SELECT CAST('2022@12@12' AS DATE);-- 동일한 결과

SELECT 
    num,
    CONCAT(CAST(price AS CHAR), 'X', CAST(amount AS CHAR), '=') '가격X수량', price * amount '구매액'
    FROM buy;
    
-- 암시적 형변환
SELECT '100' + '200';-- 문자와 문자를 더함(정수로 변환되서 연산됨)
SELECT CONCAT('100', '200');-- 문자와 문자를 연결(문자로 처리)
SELECT CONCAT(100, '200');-- 정수와 문자를 연결(정수가 문자로 변환되서 처리)
SELECT 1 > '2mega';-- 정수 2로 변환되어서 비교
SELECT 3 > '2mega';-- 정수 2로 변환되어서 비교
SELECT 0 = 'mega2'; -- 문자는 0으로 변환됨