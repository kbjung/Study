-- 인덱스 내부 작동 구조

-- 인덱스는 모두 내부적으로 균형 트리로 만들어진다.

-- 균형 트리
-- 노드(node) : 데이터가 저장되는 공간, MySQL에서는 페이지(page)라 부른다.
-- 루트 노드(root node) : 가장 상위 노드
-- 리프 노드(leaf node) : 가장 마지막 노드
-- 중간 노드(internal node) : 중간에 끼인 노드

-- 인덱스 구성
-- SELECT 속도 향상
-- 데이터 변경 작업(INSERT, UPDATE, DELETE) 속도 저하

use market_db;
create table cluster(
mem_id char(8),
mem_name varchar(10)
);
INSERT INTO cluster VALUES('TWC', '트와이스');
INSERT INTO cluster VALUES('BLK', '블랙핑크');
INSERT INTO cluster VALUES('WMN', '여자친구');
INSERT INTO cluster VALUES('OMY', '오마이걸');
INSERT INTO cluster VALUES('GRL', '소녀시대');
INSERT INTO cluster VALUES('ITZ', '잇지');
INSERT INTO cluster VALUES('RED', '레드벨벳');
INSERT INTO cluster VALUES('APN', '에이핑크');
INSERT INTO cluster VALUES('SPC', '우주소녀');
INSERT INTO cluster VALUES('MMU', '마마무');

select * from cluster;

alter table cluster
	add constraint
	primary key(mem_id);
    
select * from cluster;


create table second( -- 보조 인덱스 테이블
mem_id char(8),
mem_name varchar(10)
);
INSERT INTO second VALUES('TWC', '트와이스');
INSERT INTO second VALUES('BLK', '블랙핑크');
INSERT INTO second VALUES('WMN', '여자친구');
INSERT INTO second VALUES('OMY', '오마이걸');
INSERT INTO second VALUES('GRL', '소녀시대');
INSERT INTO second VALUES('ITZ', '잇지');
INSERT INTO second VALUES('RED', '레드벨벳');
INSERT INTO second VALUES('APN', '에이핑크');
INSERT INTO second VALUES('SPC', '우주소녀');
INSERT INTO second VALUES('MMU', '마마무');

alter table second
	add constraint
    unique(mem_id);
    
select * from second;

-- 클러스터형 인덱스가 조금 더 빠름