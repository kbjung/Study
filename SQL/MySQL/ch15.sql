-- 인덱스
-- 책의 '찾아보기' 같은 기능.

-- 인덱스의 종류
-- 1. 클러스터형 인덱스
-- 2. 보조 인덱스

-- 클러스터형 인덱스 : 영어사전처럼 내용자체가 인덱스 기능을 해서 따로 찾아보기가 없는 경우.
-- 보조 인덱스 : 일반적인 책처럼 찾아보기가 따로 있는 경우.



-- 1. 클러스터형 인덱스

use market_db;
create table table1(
col1 int primary key, -- PK설정시 클러스터형 인덱스가 생성.
col2 int,
col3 int
);
show index from table1;

create table table2(
col1 int primary key,
col2 int unique,
col3 int unique
);
show index from table2;



drop table if exists buy, member;
create table member(
mem_id char(8),
mem_name varchar(10),
mem_number int,
addr char(2)
);

insert into member values('TWC', '트와이스', 9, '서울');
insert into member values('BLK', '블랙핑크', 4, '경남');
insert into member values('WMN', '여자친구', 6, '경기');
insert into member values('OMY', '오마이걸', 7, '서울');
select * from member;

alter table member
	add constraint
    primary key(mem_id); -- mem_id를 PK로 지정하면 클러스터형 인덱스로 설정되어 알파벳 순으로 정렬된다.
select * from member;

alter table member drop primary key; -- 기본키 제거
alter table member
	add constraint
    primary key(mem_name); -- mem_name의 가나다 순으로 정렬 된다.
select * from member;

insert into member values('GRL', '소녀시대', 8, '서울');
select * from member;



-- 2. 보조 인덱스

drop table if exists member;
create table member(
mem_id char(8),
mem_name varchar(10),
mem_number int,
addr char(2)
);

insert into member values('TWC', '트와이스', 9, '서울');
insert into member values('BLK', '블랙핑크', 4, '경남');
insert into member values('WMN', '여자친구', 6, '경기');
insert into member values('OMY', '오마이걸', 7, '서울');
select * from member;

alter table member
	add constraint
    unique(mem_id);
select * from member;

alter table member
	add constraint
    unique(mem_name);
select * from member;

insert into member values('GRL', '소녀시대', 8, '서울');
select * from member;