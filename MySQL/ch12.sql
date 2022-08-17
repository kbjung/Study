create database naver_db;

drop database if exists naver_db;
create database naver_db;

use naver_db;
drop table if exists member;
create table member
(
mem_id char(8) not null primary key,
mem_name varchar(8) not null,
mem_number tinyint not null,
addr char(2) not null,
phone1 char(3) null,
phone2 char(8) null,
height tinyint unsigned null,
debut_date date null
);

drop table if exists buy;
create table buy(
num int auto_increment not null primary key,
mem_id char(8) not null,
prod_name char(6) not null,
group_name char(4) null,
price int unsigned not null,
amount smallint unsigned not null,
foreign key(mem_id) references member(mem_id)
);

insert into member values('TWC', '트와이스', 9, '서울', '02', '11111111', 167, '2015-10-19');
insert into member values('BLK', '블랙핑크', 4, '경남', '055', '22222222', 163, '2016-08-08');
insert into member values('WMN', '여자친구', 6, '경기', '031', '33333333', 166, '2015-01-15');

insert into buy values(null, 'BLK', '지갑', NULL, 30, 2);
insert into buy values(null, 'BLK', '맥북프로', '디지털', 1000, 1);
insert into buy values(null, 'APN', '아이폰', '디지털', 200, 1); -- 외래키가 참조하는 member의 mem_id가 없기에 오류 발생