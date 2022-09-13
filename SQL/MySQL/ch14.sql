USE market_db;
SELECT mem_id, mem_name, addr FROM member;

USE market_db;
CREATE VIEW v_member
AS
    SELECT mem_id, mem_name, addr FROM member;

SELECT * FROM v_member;

select mem_name, addr from v_member
	where addr in ('서울', '경기');

create view v_memberbuy
as
	select B.mem_id, M.mem_name, B.prod_name, M.addr,
		concat(M.phone1, M.phone2) '연락처'
	from buy B
    inner join member M
    on B.mem_id = M.mem_id;
    
select * from v_memberbuy;

select * from v_memberbuy where mem_name = '블랙핑크';

create view v_viewtest1
as
	select B.mem_id 'Member ID', M.mem_name as 'Member Name', B.prod_name 'product Name',
		concat(M.phone1, M.phone2) as 'Office Phone'
	from buy B
        inner join member M
		on B.mem_id = M.mem_id;
        
select distinct `Member ID`, `Member Name` from v_viewtest1; -- 열이름에 띄어쓰기가 있으면 백틱(`)을 사용.

alter view v_viewtest1
as
	select B.mem_id '회원 아이디', M.mem_name as '회원 이름', B.prod_name '제품 이름',
		concat(M.phone1, M.phone2) as '연락처'
	from buy B
        inner join member M
		on B.mem_id = M.mem_id;
        
select distinct `회원 아이디`, `회원 이름` from v_viewtest1;

drop view v_viewtest1;
-- my code : 중간에 as 빼고 작성하기 테스트
-- 결과 동일하다.
create view v_viewtest1
as
	select B.mem_id 'Member ID', M.mem_name 'Member Name', B.prod_name 'product Name',
		concat(M.phone1, M.phone2) 'Office Phone'
	from buy B
        inner join member M
		on B.mem_id = M.mem_id;
        
select * from v_viewtest1;

drop view v_viewtest1;

create or replace view v_viewtest2 -- 뷰가 없으면 만들고, 있으면 덮어쓴다.
as
	select mem_id, mem_name, addr from member;
    
describe v_viewtest2; -- PK는 확인 안됨.

describe member;

show create view v_viewtest2;

select * from v_member;
update v_member set addr = '부산' where mem_id = 'BLK';
select * from v_member where mem_id = 'BLK';

insert into v_member(mem_id, mem_name, addr) values('BTS', '방탄소년단', '경기'); 
-- 오류. 실제 테이블에 not null인 열이 존재해서 반드시 값이 입력되어야 함.
-- 이러한 문제 때문에 view를 통한 새로운 데이터 추가는 적절하지 못하다.

create view v_height167
as
	select * from member where height >= 167;
    
select * from v_height167;

delete from v_height167 where height < 167;

insert into v_height167 values('TRA', '티아라', 6, '서울', null, null, 159, '2005-01-01');
select * from v_height167;
-- 키가 167이상인 데이터만 보이는 뷰에 입력은 가능하지만 조건은 만족하지 않는 데이터라 보이지 않는다.

alter view v_height167
as
	select * from member where height >= 167
     with check option; -- 조건 만족하는 데이터만 입력되도록 설정

insert into v_height167 values('TOB', '텔로토비', 4, '영국', null, null, 140, '1955-01-01');
-- 오류. 키가 설정된 조건에 부합하지 않음.

-- 뷰가 있어도 참조하고 있는 테이블 삭제 가능.
-- 참조하는 테이블 삭제되면 뷰도 조회 불가능.