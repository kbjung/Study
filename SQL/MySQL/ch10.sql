-- 내부 조인(조인)
-- PK-FK 관계 : 기본키와 외래키의 관계, 일대다 관계의 대부분 PK와 FK 관계로 맺어져 있다.

use market_db;
select * 
	from buy -- 첫번째 테이블
		inner join member -- 두번째 테이블
		on buy.mem_id = member.mem_id -- 조인될 조건
    where buy.mem_id = 'GRL'; -- 검색 조건
    
select *
	from buy
		inner join member
        on buy.mem_id = member.mem_id;
        
select mem_id, mem_name, prod_name, addr, concat(phone1, phone2) as '연락처' -- mem_id가 buy 테이블인지, member 테이블인지 알 수 없어서 오류 발생
	from buy
		inner join member
        on buy.mem_id = member.mem_id;
        
select buy.mem_id, mem_name, prod_name, addr, concat(phone1, phone2) as '연락처'
	from buy
		inner join member
		on buy.mem_id = member.mem_id;
        
select B.mem_id, M.mem_name, B.prod_name, M.addr, concat(M.phone1, M.phone2) as '연락처'
	from buy B -- 테이블 별명 설정
		inner join member M -- 테이블 별명 설정
        on B.mem_id = M.mem_id;        

-- 외부 조인
        
select M.mem_id, M.mem_name, B.prod_name, M.addr
	from member M
		left outer join buy B -- 왼쪽 테이블 기준(첫번째 테이블 member) 외부 조인
        on M.mem_id = B.mem_id
	order by M.mem_id;
-- 동일한 결과의 코드
select M.mem_id, M.mem_name, B.prod_name, M.addr
	from buy B
		right outer join member M -- 오른쪽 테이블 기준(두번째 테이블 member) 외부 조인
        on M.mem_id = B.mem_id
	order by M.mem_id;

-- 상호 조인(테스트용으로 대용량 데이터를 생성할 때 사용)
select * from buy cross join member;

select count(*) '데이터 개수' 
	from sakila.inventory 
		cross join world.city;
        
-- drop table cross_table
create table cross_table
	select *
		from sakila.actor
			cross join world.country;
            
select * from cross_table limit 5;

-- 자체 조인

USE market_db;
CREATE TABLE emp_table (emp CHAR(4), manager CHAR(4), phone VARCHAR(8));

INSERT INTO emp_table VALUES('대표', NULL, '0000');
INSERT INTO emp_table VALUES('영업이사', '대표', '1111');
INSERT INTO emp_table VALUES('관리이사', '대표', '2222');
INSERT INTO emp_table VALUES('정보이사', '대표', '3333');
INSERT INTO emp_table VALUES('영업과장', '영업이사', '1111-1');
INSERT INTO emp_table VALUES('경리부장', '관리이사', '2222-1');
INSERT INTO emp_table VALUES('인사부장', '관리이사', '2222-2');
INSERT INTO emp_table VALUES('개발팀장', '정보이사', '3333-1');
INSERT INTO emp_table VALUES('개발주임', '정보이사', '3333-1-1');

SELECT * FROM emp_table;

SELECT A.emp '직원', B.emp '직속상관', B.phone '직속상관연락처'
	FROM emp_table A
     INNER JOIN emp_table B
		ON A.manager = B.emp
	WHERE A.emp = '경리부장';