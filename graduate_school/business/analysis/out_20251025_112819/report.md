# 네이버 플러스 스토어 탈모 카테고리 시장 분석 보고서

## 요약 KPI
| 항목 | 값 | 의미 |
|:------:|------:|------|
| 총 상품 수 | 10,095 | 전체 데이터셋에 포함된 상품의 개수 |
| 고유 제조사 수 | 1,669 | 상품 정보에 명시된 고유(nunique) 제조사의 개수 |
| 고유 브랜드 수 | 1,942 | 상품 정보에 명시된 고유(nunique) 브랜드의 개수 |
| 고유 스토어 수 | 3,452 | 네이버 플러스 스토어에서 ‘탈모’ 관련 상품을 판매 중인 고유(nunique) 스토어(판매자)의 수 |
| 가격 중앙값 | 27,000.0 | 상품의 1개당 가격(`final_price_per_unit`) 중앙값 — 시장의 ‘중간 가격대’를 대표함 |
| 가격 1사분위수(Q1) | 16,980.0 | 상품의 1개당 가격(`final_price_per_unit`) 분포의 하위 25% 지점 — 저가 상품의 경계선 수준 |
| 가격 3사분위수(Q3) | 43,732.5 | 상품의 1개당 가격(`final_price_per_unit`) 분포의 상위 25% 지점 — 고가 상품의 경계선 수준 |

## 차트
![chart_category_count_bar.png](chart_category_count_bar.png)
- **카테고리별 상품 수 (chart_category_count_bar.png)** : `category` 컬럼만 사용하여 각 카테고리의 상품 개수를 집계한 막대그래프입니다.


![chart_price_per_ml_box.png](chart_price_per_ml_box.png)
- **카테고리별 ml당 가격 (chart_price_per_ml_box.png)** : `price_per_ml` 박스플롯, 박스 위 숫자는 중앙값.


![chart_peptide_counts.png](chart_peptide_counts.png)
- **펩타이드 포함 여부별 상품 수 (chart_peptide_counts.png)** : `ING_펩타이드` True/False 개수를 집계한 막대그래프입니다.


![chart_skin_penetration_keywords.png](chart_skin_penetration_keywords.png)
- **피부 투과 기술 키워드 (chart_skin_penetration_keywords.png)** : `제목`에서 추출한 키워드(`피부 투과 기술 키워드`) 분포입니다. 관련 제품 목록은 `skin_penetration_products.xlsx`에 저장됩니다.


![chart_price_per_unit_hist.png](chart_price_per_unit_hist.png)
- **1개 기준 가격 분포 (chart_price_per_unit_hist.png)** : `final_price_per_unit`(= `final_price` ÷ `unit_count`) 히스토그램. 막대 위 숫자는 각 구간의 상품 개수.


![chart_ingredients_top15.png](chart_ingredients_top15.png)
- **성분 키워드 TOP15 (chart_ingredients_top15.png)** : `ING_*` 합계(텍스트에서 키워드 매칭). 우측 숫자는 키워드 포함 상품 수.


![chart_whitespace_matrix.png](chart_whitespace_matrix.png)
- **화이트스페이스(펩타이드×저자극) (chart_whitespace_matrix.png)** : `final_price_per_unit` 평균 히트맵. 셀 숫자는 평균 1개 가격과 표본 개수(n)입니다.



## 참고
- `data_category_counts.xlsx` : 카테고리별 상품 수 집계표 데이터
- `data_price_per_ml_by_category.xlsx` : 카테고리별 ml당 가격 데이터
- `data_peptide_counts.xlsx` : 펩타이드 포함 여부별 상품 수 데이터
- `data_skin_penetration_keywords.xlsx` : 피부 투과 기술 키워드 분포 데이터
- `data_price_per_unit_hist.xlsx` : 상품 1개당 가격 히스토그램 데이터
- `data_ingredient_top15.xlsx` : 성분 키워드 TOP15 데이터
- `data_ingredient_all.xlsx` : 전체 성분 키워드 등장 상품 수 데이터
- `data_whitespace_matrix.xlsx` : 펩타이드 × 저자극 화이트스페이스 매트릭스 데이터