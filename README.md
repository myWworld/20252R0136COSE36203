# 유튜브 영상 태그, 반응 분석 리포트 생성기

## https://github.com/sdouf5054/20252R0136COSE36203 최종 파이프라인

## 추가 bert_model.py YouTube_Comment_Sentiment_Unified.ipynb의 동일한 기능을 가진 py 스크립트

# 📁 Repository Structure
# 1. Jupyter Notebook
YouTube_Comment_Sentiment_Unified.ipynb

전체 프로젝트를 통합해 실행하는 메인 노트북입니다.

### 주요 기능

YouTube 댓글 전처리 및 데이터 정제

comments_labeled_for_training.csv 생성 및 검수

BERT 기반 sentiment classifier fine-tuning

학습된 모델(bert_sentiment/) 저장

새로운 댓글에 대한 inference 실행

#### 과거 기능 (현재는 주석 처리됨)

classical baseline(baseline_model.py) 실행

active learning 후보 추출(sentiment_utils.py)

# 2. Classical Baseline & Data Pipeline
## baseline_model.py -> active learning용으로 사용 현재 노트북에선 사용안함(주석처리) BERT 모델이 주모델임

classical ML baseline 전체 파이프라인을 자동 실행하는 스크립트입니다.

### 주요 역할
#### 🔹 라벨 병합 (merge_all_manual_golds)

기존 weak label + manual_label round 파일 병합

중복 comment → 최신 라벨 우선

weak 데이터는 weak_frac 비율로 downsample 후 병합

최종 출력: comments_labeled_merged.csv

#### 🔹 baseline 모델 학습 (baseline_once)

TF-IDF(word/char) + neg lexicon feature + Logistic Regression

valid set 기반 macro-F1 최적 hyperparam + tau(neutral threshold) 탐색

최종 모델 저장: artifacts/classic/

#### 🔹 옵션: inference + n-gram 분석 (optional_outputs)

새로운 CSV에 대한 batch prediction

chi-square 기반 top n-gram export

현재 프로젝트에서의 위치

classical baseline / active learning 용도의 레거시 파이프라인

BERT fine-tuning 이후로는 비교용 또는 보조 데이터 생성에 사용

# 3. Shared Utilities / Classic Model Logic
## sentiment_utils.py

데이터 처리, feature 생성, classical 모델 구성, active learning, inference까지
전반을 담당하는 공용 유틸리티 파일입니다.

### 🔹 Lexicon & Feature
NEG_LEXICON

한국어 + 영어 부정 표현 사전

YouTube 욕설/비하/비판 표현 다수 포함
(예: trash, garbage, sucks, cringe, disgusting, worst, waste of time, clickbait …)

neg_lexicon_features(texts)

lexicon 포함 여부를 0/1 sparse matrix로 변환

TF-IDF와 함께 부정 신호 강화 feature로 사용

### 🔹 Data Helpers
clean(t)

URL, @mention 제거

공백 정리 등 기본 텍스트 전처리

load_dataset(path)

CSV 로드 후 text, label 정제

label set 검증

split_train_valid_test

stratified 방식 train/valid/test 분리

### 🔹 Classic Model: ClassicSentiment

TF-IDF(word) + TF-IDF(char) + lexicon feature → hstack

LogisticRegression(saga) + CalibratedClassifierCV 사용

제공 메서드:

fit

predict

predict_proba

save(out_dir)

load(out_dir)

### 🔹 Neutral Policy & Grid Search
apply_neutral_policy

max proba < tau → 강제 neutral

pos/neg 차이 작으면 더 큰 쪽으로 재할당

small_grid_search

(C, l1_ratio, tau) 조합 탐색

valid 기반 macro-F1 최적 config 선택

### 🔹 Active Learning
select_active_learning_candidates

기존 baseline 모델로 low-confidence 샘플 선별

SVD + KMeans cluster로 그룹화

cluster별 대표 ambiguous 샘플 선택

### 🔹 Batch Inference & N-gram 분석
batch_predict

새로운 CSV에 대해 pred, p_neg, p_neu, p_pos 추가

neutral policy 적용

결과 CSV 저장

chi2_top_ngrams_from_df

label별 특징적 n-gram 추출

chi-square score 기반 ranking 후 CSV export

# 4. Data Files
## comments_labeled_for_training.csv

BERT fine-tuning에 사용된 최종 라벨링 데이터셋

포함 컬럼:

text

label (neg, neu, pos)

필요 시 metadata (video_id, comment_id, …)

comments_for_inference.csv

BERT 모델로 감정을 예측할 raw 댓글 데이터

노트북에서 batch inference 수행에 사용됨
