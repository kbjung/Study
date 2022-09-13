/* 스토어드 함수
- 직접 만들어서 사용하는 함수
- 형식
DELIMITER $$
CREATE FUNCTION 스토어드_함수_이름(매개변수)
	RETURNS 반환형식
BEGIN
	이 부분에 프로그래밍 코딩
    RETURN 반환값;
END $$
DELIMITER ;
SELECT 스토어드_함수_이름();
*/

SET GLOBAL log_bin_trust_function_creators = 1;

use market_db;
drop function if exists sumFunc;
delimiter //
create function sumFunc(number1 int, number2 int)
	returns int
begin
	return number1 + number2;
end //
delimiter ;
select sumFunc(100, 200) as '합계';
select sumFunc(100, 200) '합계'; -- 위 코드와 동일한 출력


DROP FUNCTION IF EXISTS calcYearFunc;
DELIMITER $$
CREATE FUNCTION calcYearFunc(dYear INT)
    RETURNS INT
BEGIN
    DECLARE runYear INT; -- 활동기간(연도)
    SET runYear = YEAR(CURDATE()) - dYear;
    RETURN runYear;
END $$
DELIMITER ;

SELECT calcYearFunc(2010) AS '활동햇수';

SELECT calcYearFunc(2007) INTO @debut2007;
SELECT calcYearFunc(2013) INTO @debut2013;
SELECT @debut2007-@debut2013 AS '2007과 2013 차이' ;

SELECT mem_id, mem_name, calcYearFunc(YEAR(debut_date)) AS '활동 햇수' 
    FROM member;


SHOW CREATE FUNCTION calcYearFunc;

DROP FUNCTION calcYearFunc; -- 함수 삭제


/* 커서
- 한 행씩 처리한다
- 전체적인 흐름
1. 커서 선언
2. 반복 조건 선언
3. 커서 열기
4. 데이터 가져오기
5. 데이터 처리하기
6. 커서 닫기
- 4~5과정을 반복
*/

USE market_db;
DROP PROCEDURE IF EXISTS cursor_proc;
DELIMITER $$
CREATE PROCEDURE cursor_proc()
BEGIN
    DECLARE memNumber INT; -- 회원의 인원수
    DECLARE cnt INT DEFAULT 0; -- 읽은 행의 수
    DECLARE totNumber INT DEFAULT 0; -- 인원의 합계
    DECLARE endOfRow BOOLEAN DEFAULT FALSE; -- 행의 끝 여부(기본을 FALSE)

    DECLARE memberCuror CURSOR FOR-- 커서 선언
        SELECT mem_number FROM member;

    DECLARE CONTINUE HANDLER -- 행의 끝이면 endOfRow 변수에 TRUE를 대입 
        FOR NOT FOUND SET endOfRow = TRUE;

    OPEN memberCuror;  -- 커서 열기

    cursor_loop: LOOP
        FETCH  memberCuror INTO memNumber; 

        IF endOfRow THEN 
            LEAVE cursor_loop;
        END IF;

        SET cnt = cnt + 1;
        SET totNumber = totNumber + memNumber;        
    END LOOP cursor_loop;

    SELECT (totNumber/cnt) AS '회원의 평균 인원 수';

    CLOSE memberCuror; 
END $$
DELIMITER ;

CALL cursor_proc();