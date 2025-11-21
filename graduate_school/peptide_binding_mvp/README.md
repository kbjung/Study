# ai 기반 펩타이드-단백질 결합 후보 예측 시스템 구축 MVP
## ⌛ 오프라인 환경
- Ubuntu(Windows 10 WSL)
+ 펩타이드 생성 파이프라인 04
  - 01\. 펩타이드 생성 : PepMLM
  - 02\. 구조예측 : ColabFold
  - 03\. 결합력 평가 및 최종 평가 : ipTM(colabfold), Prodigy(파이프라인03보다 정확도+), Autoduck Vina, PLIP
  - 코드 [[py]](https://github.com/kbjung/Study/blob/main/graduate_school/peptide_binding_mvp/notebooks/pepbind_pipeline02.py)
  - 이슈 : Autodock Vina, PLIP 모델 계산 에러 발생

---

## Colab 환경
+ 펩타이드 생성 파이프라인 01
  - 01\. 펩타이드 생성 : PepMLM
  - 02\. 구조예측 : ColabFold
  - 03\. 결합력 평가 및 최종 평가 : Pafnucy, Autoduck Vina, PLIP
  - 코드 [[ipynb]](https://github.com/kbjung/Study/blob/main/graduate_school/peptide_binding_mvp/notebooks/%ED%8E%A9_%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B801(Vina_PLIP_PPI).ipynb)

+ 펩타이드 생성 파이프라인02
  - 01\. 펩타이드 생성 : PepMLM
  - 02\. 구조예측 : ColabFold
  - 03\. 결합력 평가 및 최종 평가 : PPI-Affinity(파이프라인01보다 정확도+25%), Autoduck Vina, PLIP
  - 코드 [[ipynb]](https://github.com/kbjung/Study/blob/main/graduate_school/peptide_binding_mvp/notebooks/%ED%8E%A9_%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B802(pTM_Vina_PLIP_PPI).ipynb)

+ 펩타이드 생성 파이프라인 03
  - 01\. 펩타이드 생성 : PepMLM
  - 02\. 구조예측 : ColabFold
  - 03\. 결합력 평가 및 최종 평가 : ipTM(colabfold), PPI-Affinity(파이프라인01보다 정확도+25%), Autoduck Vina, PLIP
  - 코드 [[ipynb]](https://github.com/kbjung/Study/blob/main/graduate_school/peptide_binding_mvp/notebooks/%ED%8E%A9_%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B803(pTM_Vina_PLIP_PPI).ipynb)
 
+ 펩타이드 생성 파이프라인 04
  - 01\. 펩타이드 생성 : PepMLM
  - 02\. 구조예측 : ColabFold
  - 03\. 결합력 평가 및 최종 평가 : ipTM(colabfold), Prodigy(파이프라인03보다 정확도+), Autoduck Vina, PLIP
  - 코드 [[ipynb]](https://github.com/kbjung/Study/blob/main/graduate_school/peptide_binding_mvp/notebooks/%ED%8E%A9_%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B804(ipTM_Vina_PLIP_Prodigy).ipynb)
 
---

+ 펩타이드 생성 ai
  - ESMFold [[ipynb]](https://github.com/kbjung/Study/blob/main/graduate_school/peptide_binding_mvp/notebooks/test_code/protein_folding.ipynb)
  - RFdiffusion Peptide Structure Generation(MVP Test) [[ipynb]](https://github.com/kbjung/Study/blob/main/graduate_school/peptide_binding_mvp/notebooks/test_code/rfdiffusion_peptide_generation.ipynb)
+ 결합 평가 ai
  - ZDOCK [[링크]](https://zdock.wenglab.org/)
    - 결합력 상위 10개 PDB 자료 수정 코드(ChimeraX에 사용위해 수정) [[py]](https://github.com/kbjung/Study/blob/main/graduate_school/peptide_binding_mvp/scripts/clean_pdb.py)
    - 결합력 상위 10개 전체 PDB 자료 수정 자동화 코드(ChimeraX에 사용위해 수정) [[py]](https://github.com/kbjung/Study/blob/main/graduate_school/peptide_binding_mvp/scripts/batch_clean_pdb.py)
    - 결합력 점수와 상위 10개 모델 이름 매칭 및 결과 출력 코드 [[py]](https://github.com/kbjung/Study/blob/main/graduate_school/peptide_binding_mvp/scripts/parse_zdock_scores.py)
