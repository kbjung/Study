-- 인덱스 생성과 제거

-- 인덱스 생성 문법

-- CREATE [UNIQUE] INDEX 인덱스_이름
--     ON 테이블_이름(열_이름) [ASC|DESC]

-- 인덱스 제거 문법

-- DROP INDEX 익덱스_이름 ON 테이블_이름

-- 기본 키, 고유 키로 자동 생성된 인덱스는 DROP INDEX로 제거 할 수 없다.

use market_db;
select * from member;

show index from member;

show table status like 'member'; 
-- Data_length : 한 페이지의 크기. 16384 => 약 16 kbyte
-- Index_length : 보조인덱스의 크기

create index idx_member_addr on member(addr);

show index from member;

show table status like 'member'; -- 인덱스 적용이 안되어 Index_length 가 0.

analyze table member;
show table status like 'member'; -- Index_lengh : 16384 => 약 16 kbyte

create unique index idx_member_mem_number on member(mem_number); -- 오류. 중복된 값이 존재함.

create unique index idx_member_mem_name on member(mem_name); -- 중복된 값이 없기에 잘 작동.

show index from member;

insert into member values('MOO', '마마무', 2, '태국', '001', '12341234', 155, '2020.10.10');
-- 오류. mem_name의 인덱스가 unique로 설정되어 중복된 값은 입력 불가.

analyze table member; -- 지금까지 만든 인덱스를 모두 적용
show index from member;

select * from member;
-- execution plan 탭에서 full table scan이 나오면 인덱스 안사용한 것.

select mem_id, mem_name, addr from member;
-- full table scan.
-- select 문은 인덱스 사용하지 않음. where절 조건식이 있어야 인덱스 사용.

select mem_id, mem_name, addr from member where mem_name = '에이핑크';
-- single row. 인덱스 사용.

create index idx_member_mem_number on member(mem_number);
analyze table member;

select mem_name, mem_number from member where mem_number >= 7;
-- index range scan. 인덱스 사용.

select mem_name, mem_number from member where mem_number >= 1;
-- full table scan. 인덱스 사용 안함. 모든 값에 해당되는 조건이기에 MySQL이 인덱스 사용 여부 판단해서 사용 안함.

select mem_name, mem_number from member where mem_number*2 >= 14;
-- full table scan. 인덱스 사용 안함. where 조건문에서 열을 가공하면 인덱스를 사용하지 않는다.

select mem_name, mem_number from member where mem_number >= 14/2;
-- index range scan. 인덱스 사용. 조건문에서 열을 가공하지 않아 인덱스 사용.

show index from member;

drop index idx_member_mem_name on member;
drop index idx_member_addr on member;
drop index idx_member_mem_number on member;

alter table member drop primary key;
-- 오류. PK-FK 로 묶여있어서 지울 수 없다.

-- 외래키 찾는 방법
select table_name, constraint_name
	from information_schema.referential_constraints
    where constraint_schema = 'market_db';
    
alter table buy drop foreign key buy_ibfk_1; -- 외래키 삭제
alter table member drop primary key; -- 기본키 삭제
