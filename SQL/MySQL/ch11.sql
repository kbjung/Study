-- SQL 프로그래밍

-- 스토어드 프로시저
-- DELIMITER $$
-- CREATE PROCEDURE 스토어드_프로시저_이름()
-- BEGIN
-- 	   SQL 프로그래밍 코딩
-- END $$ 스토어드 프로시저 종료
-- DELIMITER ; 종료 문자를 다시 세미콜론(;)으로 변경
--     CALL 스토어드_프로시저_이름(); 스토어드 프로시저 실행

-- 스토어드 프로시저는 DELIMITER $$ ~ END $$ 안에 작성하고 CALL로 호출합니다.

-- if문

-- IF <조건식> THEN
--     SQL문장들
-- END IF;

USE market_db;
DROP PROCEDURE IF EXISTS ifProc1; -- 기존에 만든적이 있다면 삭제
DELIMITER $$
CREATE PROCEDURE ifProc1()
BEGIN
	IF 100 = 100 THEN
		SELECT '100은 100과 같습니다.';
	END IF;
END $$
DELIMITER ;
CALL ifProc1();

DROP PROCEDURE IF EXISTS ifProc2;
DELIMITER $$
CREATE PROCEDURE ifProc2()
BEGIN
	DECLARE myNum INT; -- myNum 변수 선언
    SET myNum = 200; -- 변수에 값 대입
    IF myNum = 100 THEN
		SELECT '100입니다.';
	ELSE
		SELECT '100이 아닙니다.';
	END IF;
END $$
DELIMITER ;
CALL ifProc2();

DROP PROCEDURE IF EXISTS ifProc3;
DELIMITER $$
CREATE PROCEDURE ifProc3()
BEGIN
	DECLARE debutDate DATE;
    DECLARE curDate DATE;
    DECLARE days INT;
    
    SELECT debut_date INTO debutDate -- debut_date 결과를 debutDate에 대입
		FROM market_db.member
        WHERE mem_id = 'APN';
	
    SET curDate = CURRENT_DATE(); -- 현재 날짜
    SET days = DATEDIFF(curDate, debutDate); -- 날짜의 차이, 일 단위
    
    IF (days/365) >= 5 THEN
		SELECT CONCAT('데뷔한지 ', days, '일이나 지났습니다. 핑순이들 축하합니다!');
	ELSE
		SELECT '데뷔한지 ' + days + '일밖에 안되엇네요. 핑순이들 화이팅~';
	END IF;
END $$
DELIMITER ;
CALL ifProc3();

-- CASE문 : 여러가지 조건 중에서 선택. 
-- IF문은 '2중 분기', CASE문은 '다중 분기'라 부른다.

-- CASE
--   WHEN 조건1 THEN
--     SQL문장들1
--   WHEN 조건2 THEN
--     SQL문장들2 
--   WHEN 조건3 THEN
--     SQL문장들3
--   ELSE
--     SQL문장들4
-- END CASE;

DROP PROCEDURE IF EXISTS caseProc;
DELIMITER $$
CREATE PROCEDURE caseProc()
BEGIN
	DECLARE point INT;
    DECLARE credit CHAR(1);
    SET point = 88;
    
    CASE
		WHEN point >=90 THEN
			SET credit = 'A';
		WHEN point >= 80 THEN
			SET credit = 'B';
		WHEN point >= 70 THEN
			SET credit = 'C';
		WHEN point >= 60 THEN
			SET credit = 'D';
		ELSE
			SET credit = 'F';
	END CASE;
    SELECT CONCAT('취득점수 : ', point), CONCAT('학점 : ', credit);
    -- SELECT '취득점수 : ' +  point, '학점 : ' + credit; 학점이 0으로 출력됨.
END $$
DELIMITER ;
CALL caseProc();



select mem_id, sum(price*amount) '총 구매액'
	from buy
    group by mem_id;
    
select mem_id, sum(price*amount) '총 구매액'
	from buy
    group by mem_id
    order by sum(price*amount) desc;
    
select B.mem_id, M.mem_name, sum(price*amount) '총 구매액'
	from buy B
		inner join member M
        on B.mem_id = M.mem_id
	group by B.mem_id
    order by sum(price*amount) desc;
    
select B.mem_id, M.mem_name, sum(price*amount) '총 구매액'
	from buy B
		right outer join member M
        on B.mem_id = M.mem_id
	group by M.mem_id
    order by sum(price*amount) desc;
    
select B.mem_id, M.mem_name, sum(price*amount) '총 구매액',
	case
		when (sum(price*amount) >= 1500) then '최우수고객'
        when (sum(price*amount) >= 1000) then '우수고객'
        when (sum(price*amount) >= 1) then '일반고객'
        else '유령고객'
	end '회원등급'
    
	from buy B
		right outer join member M
        on B.mem_id = M.mem_id
	group by M.mem_id
    order by sum(price*amount) desc;
    
    -- while 문
    
    -- WHILE <조건식> DO
    --   SQL 문장들
    -- END WHILE;
    
drop procedure if exists whileProc;
delimiter $$
create procedure whileProc()
begin
	declare i int;
	declare hap int;
	set i = 1;
	set hap = 0;
	
	while (i <= 100) do
		set hap = hap + i;
		set i = i + 1;
	end while;
	
	select '1부터 100까지의 합 : ', hap;
end $$
delimiter ;
call whileProc();

-- ITERATE(레이블) : 지정한 레이블로 가서 계속 진행합니다.
-- LEAVE(레이블) : 지정한 레이블을 빠져나갑니다. 즉 WHILE문이 종료됩니다.

drop procedure if exists whileProc2;
delimiter $$
create procedure whileProc2()
begin
	declare i int;
    declare hap int;
    set i = 1;
    set hap = 0;
    
    mywhile: -- while 문의 이름 설정
    while (i <= 100) do
		if (i%4 = 0) then
			set i = i + 1;
            iterate mywhile; -- 지정한 label 문으로 가서 계속 진행
		end if;
        set hap = hap + i;
		if (hap > 1000) then
			leave mywhile; -- 지정한 label 문을 떠남. 즉 while 문 종료
		end if;
        set i = i + 1;
	end while;
    
    select '1부터 100까지의 합(4의 배수 제외), 합이 1000 넘으면 종료 : ', hap;
end $$
delimiter ;
call whileProc2();

-- 동적 SQL
-- 변경되는 내용을 실시간으로 적용

drop table if exists gate_table;
create table gate_table (id int auto_increment primary key, entry_time datetime);

set @curDate = current_timestamp(); -- 현재 날짜와 시간

prepare myQuery from 'insert into gate_table values(null, ?)';
execute myQuery using @curDate;
deallocate prepare myQuery;

select * from gate_table;