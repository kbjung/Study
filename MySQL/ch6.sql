USE market_db; -- market_db 데이터베이스를 사용한다.

SELECT * FROM member WHERE mem_name = '블랙핑크';

SELECT * 
FROM member 
WHERE mem_name = '블랙핑크';

USE sys;
SELECT * FROM market_db.member WHERE mem_name = '블랙핑크'; -- 다른 데이터 베이스를 사용하는 중에 market_db의 member 테이블을 사용하는 방법.

SELECT addr, height, debut_date FROM member WHERE mem_name = '블랙핑크';
SELECT addr, height, debut_date FROM member;
SELECT height, debut_date, addr FROM member; -- 열 순서 바꾸어 출력

SELECT height 키, debut_date "데뷔 일자", addr FROM member; -- 열 이름 부여(참조, 별명)

SELECT * FROM member WHERE mem_number = 4;
SELECT * FROM member WHERE mem_number = 40;

SELECT mem_id, mem_name 
FROM member WHERE height <= 162;

SELECT mem_name, height, mem_number 
FROM member WHERE height >= 165 AND mem_number > 6;

SELECT mem_name, height, mem_number 
FROM member WHERE height >= 165 OR mem_number > 6;

SELECT mem_name, height
FROM member WHERE height >= 163 AND height <= 165;

SELECT mem_name, height
FROM member WHERE height BETWEEN 163 AND 165;

SELECT mem_name, addr
FROM member WHERE addr = '경기' OR addr = '전남' OR addr = '경남';

SELECT mem_name, addr
FROM member WHERE addr IN ('경기', '전남', '경남');

SELECT * FROM member WHERE mem_name LIKE '우%'; -- 앞글자가 매칭되는 데이터 출력, % : 여러글자
SELECT * FROM member WHERE mem_name LIKE '__핑크'; -- 뒷글자가 매칭되는 데이터 출력, _: 한글자
SELECT * FROM member WHERE mem_name LIKE '%핑크';
