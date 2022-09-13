-- 제약조건
-- 데이터의 무결점을 위한 방법


-- PRIMARY KEY 제약조건
-- FOREIGN KEY 제약조건
-- UNIQUE 제약조건
-- CHECK 제약조건
-- DEFAULT 정의
-- NULL 값 허용



-- PRIMARY KEY 제약조건
-- 중복 불가, null 값 허용 불가
-- 테이블은 기본키 1개만 가질 수 있다
-- 기본키로 생성한 것은 자동으로 클러스터형 인덱스가 생성

use naver_db;
-- #1. 테이블을 만드는 방법(기본키 설정)
drop table if exists buy, member;
create table member(
mem_id char(8) not null primary key,
mem_name varchar(10) not null,
height tinyint unsigned null
);

describe member;

-- #2. 테이블을 만드는 방법(기본키 설정)
drop table if exists member;
create table member(
mem_id char(8) not null,
mem_name varchar(10) not null,
height tinyint unsigned null,
primary key(mem_id)
);

describe member;

-- #3. 기본키 따로 설정하는 방법
drop table if exists member;
create table member(
mem_id char(8) not null,
mem_name varchar(10) not null,
height tinyint unsigned null
);
alter table member -- 테이블을 수정
	add constraint -- 제약 조건 추가
    primary key(mem_id);
    
describe member;



-- FOREIGN KEY 제약조건
-- PK 테이블 = 기준 테이블, FK 테이블 = 참조 테이블
-- FK의 값은 PK에 반드시 있어야 한다

drop table if exists buy, member;
create table member(
mem_id char(8) not null primary key,
mem_name varchar(10) not null,
height tinyint unsigned null
);
create table buy(
num int auto_increment not null primary key,
mem_id char(8) not null,
prod_name char(6) not null,
foreign key(mem_id) references member(mem_id)
);

-- pk, fk 열 이름 다르게도 설정 가능.
-- 추천하지 않음. 헷갈리기 쉬움
drop table if exists buy, member;
create table member(
mem_id char(8) not null primary key,
mem_name varchar(10) not null,
height tinyint unsigned null
);
create table buy(
num int auto_increment not null primary key,
user_id char(8) not null,
prod_name char(6) not null,
foreign key(user_id) references member(mem_id)
);

-- FK 키 따로 지정하는 방법
drop table if exists buy;
create table buy(
num int auto_increment not null primary key,
mem_id char(8) not null,
prod_name char(6) not null
);
alter table buy
	add constraint
    foreign key(mem_id) references member(mem_id);
    
describe buy;
describe member;

insert into member values('BLK', '블랙핑크', 163);
insert into buy values(null, 'BLK', '지갑');
insert into buy values(null, 'BLK', '맥북');

select M.mem_id, M.mem_name, B.prod_name
	from buy B
		inner join member M
		on B.mem_id = M.mem_id;

update member set mem_id = 'PINK' where mem_id = 'BLK'; -- PK-FK 관계로 묶여있어서 수정 불가능. 오류 발생.

delete from member where mem_id = 'BLK'; -- 삭제도 불가능. 오류 발생.

-- PK, FK 같이 바꾸기
drop table if exists buy;
create table buy(
num int auto_increment not null primary key,
mem_id char(8) not null,
prod_name char(6) not null
);
alter table buy
	add constraint
    foreign key(mem_id) references member(mem_id)
    on update cascade
    on delete cascade;
    
insert into buy values(null, 'BLK', '지갑');
insert into buy values(null, 'BLK', '맥북');

update member set mem_id = 'PINK' where mem_id = 'BLK';

select M.mem_id, M.mem_name, B.prod_name
	from buy B
		inner join member M
		on B.mem_id = M.mem_id;

delete from member where mem_id = 'PINK';

select * from buy;



-- UNIQUE 제약조건
-- 유일한 값 입력 조건, null 값 허용.

drop table if exists buy, member;
create table member(
mem_id char(8) not null primary key,
mem_name varchar(10) not null,
height tinyint unsigned null,
email char(30) null unique
);

insert into member values('BLK', '블랙핑크', 163, 'pink@gmail.com');
insert into member values('TWC', '트와이스', 167, null);
insert into member values('APN', '에이핑크', 164, 'pink@gmail.com'); -- 오류. 'pink@gmail.com' 중복값.

select * from member;



-- CHECK 제약조건
-- 특정 범위, 값만 입력되도록 제한

drop table if exists member;
create table member(
mem_id char(8) not null primary key,
mem_name varchar(10) not null,
height tinyint unsigned null check(height >= 100),
phone1 char(3) null
);

insert into member values('BLK', '블랙핑크', 163, null);
insert into member values('TWC', '트와이스', 99, null); -- 오류. 99는 check 조건에 부합하지 않는다.

alter table member
	add constraint
    check ( phone1 in ('02', '031', '032', '054', '055', '061') );
    
insert into member values('TWC', '트와이스', 167, '02');
insert into member values('OMY', '오마이걸', 167, '010'); -- 오류. '010'은 check 조건에 없다.



-- DEFAULT 정의
-- 입력값이 없을 때, 자동으로 입력될 값 지정

drop table if exists member;
create table member(
mem_id char(8) not null primary key,
mem_name varchar(10) not null,
height tinyint unsigned null default 160,
phone1 char(3) null
);
alter table member
	alter column phone1 set default '02';
    
insert into member values('RED', '레드벨벳', 164, '054');
insert into member values('SPC', '우주소녀', default, default); -- 기본으로 설정된 160, '02'가 입려됨

select * from member;


-- NULL 값 허용
-- '아무 것도 없다'라는 의미