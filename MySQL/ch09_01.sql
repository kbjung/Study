use market_db;

create table hongong4(tinyint_col tinyint, smallint_col smallint, int_col int, bigint_col bigint);

insert into hongong4 values(127, 32767, 2147483647, 9000000000000000000);
insert into hongong4 values(128, 32768, 2147483648, 90000000000000000000); -- 범위 밖의 숫자가 입력되어 오류 발생

create table big_table ( data1 char(255) );

drop table big_table;
create table big_table ( data1 char(256) ); -- 오류발생, char는 255까지 표현가능.

drop table big_table;
create table big_table ( data2 varchar(16383) );

drop table big_table;
create table big_table ( data2 varchar(16384) ); -- 오류발생, varchar는 16383까지 표현가능.

create database netflix_db;
use netflix_db;
create table movie( 
	movie_id int, 
    movie_title varchar(30), 
    movie_director varchar(20), 
    movie_star varchar(20), 
    movie_script longtext, 
    movie_film longblob
    );

-- 변수의 사용

use market_db;
set @myVar1 = 5;
set @myVar2 = 4.25;

select @myVar1;
select @myVar1 + @myVar2; -- 변수에 임시 저장됨.

set @txt = '가수 이름 ==> ';
set @height = 166;
select @txt, mem_name from member where height > @height;

set @count = 3;
select mem_name, height from member order by height limit @count; -- 오류. limit에 변수사용은 문법적으로 지원하지 않음.

set @count = 3;
prepare mySQL from 'select mem_name, height from member order by height limit ?';
execute mySQL using @count;

select avg(price) '평균 가격' from buy; -- 결과 : 142.9167
-- 명시적 형변환
select cast(avg(price) as signed) '평균 가격' from buy; -- 부호가 있는 정수로 나타냄. 결과 : 143

select cast('2022$12$12' as date); -- 2022-12-12 출력
select cast('2022/12/12' as date); -- 동일한 결과
select cast('2022%12%12' as date); -- 동일한 결과
select cast('2022@12@12' as date); -- 동일한 결과

select num, concat(cast(price as char), 'X', cast(amount as char), '=') '가격X수량', price*amount '구매액' from buy;
-- 암시적 형변환
select '100' + '200'; -- 문자와 문자를 더함(정수로 변환되서 연산됨)
select concat('100', '200'); -- 문자와 문자를 연결(문자로 처리)
select concat(100, '200'); -- 정수와 문자를 연결(정수가 문자로 변환되서 처리)
select 1 > '2mega'; -- 정수 2로 변환되어서 비교
select 3 > '2mega'; -- 정수 2로 변환되어서 비교
select 0 = 'mega2'; -- 문자는 0으로 변환됨
