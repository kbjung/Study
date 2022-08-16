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