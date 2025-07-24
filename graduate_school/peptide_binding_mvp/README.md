# ai 시스템 구축 MVP
+ 펩타이드 생성 ai
  - ESMFold [[ipynb]](https://github.com/kbjung/Study/blob/main/graduate_school/peptide_binding_mvp/notebooks/protein_folding.ipynb)
  - RFdiffusion Peptide Structure Generation(MVP Test) [[ipynb]](https://github.com/kbjung/Study/blob/main/graduate_school/peptide_binding_mvp/notebooks/rfdiffusion_peptide_generation.ipynb)
+ 결합 평가 ai
  - ZDOCK [[링크]](https://zdock.wenglab.org/)
    - 결합력 상위 10개 PDB 자료 수정 코드(ChimeraX에 사용위해 수정) [[py]](https://github.com/kbjung/Study/blob/main/graduate_school/peptide_binding_mvp/scripts/clean_pdb.py)
    - 결합력 상위 10개 전체 PDB 자료 수정 자동화 코드(ChimeraX에 사용위해 수정) [[py]](https://github.com/kbjung/Study/blob/main/graduate_school/peptide_binding_mvp/scripts/batch_clean_pdb.py)
    - 결합력 점수와 상위 10개 모델 이름 매칭 및 결과 출력 코드 [[py]](https://github.com/kbjung/Study/blob/main/graduate_school/peptide_binding_mvp/scripts/parse_zdock_scores.py)
