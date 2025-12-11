
import os, re, joblib, numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Tuple
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import chi2

# ------------neg비율 높이기 위한
NEG_LEXICON = [
    # --- 기존 + 비슷한 변형들 ---
    "별로", "별로네", "별로다", "별로임", "별로였", "그닥", "그다지", "마음에 안 든다", "맘에 안 듦",
    "노잼", "개노잼", "완전 노잼", "존노잼", "진짜 노잼", "노재미", "재미 한 개도 없",
    "재미없", "재미도 없", "별로 재미없", "하나도 안 웃기", "안 웃기", "하품 나온다",
    "최악", "진짜 최악", "개최악", "역대급 최악", "완전 최악", "최저", "최악의 영상", "최악의 컨텐츠",
    "형편없", "형편 없는", "퀄리티가 형편없", "엉망", "엉망진창", "난장판", "거지같", "개같이 만들", "구려", "구리네", "구리다",
    "쓰레기", "개쓰레기", "쓰레기 같", "쓰레기네", "쓰레기 영상", "쓰레기 컨텐츠", "쓰레기 방송",
    "망했네", "망했네요", "망했따", "개망했", "완전 망함", "망작", "이게 뭐냐 망작",
    "실망", "진짜 실망", "실망이 큼", "실망스럽", "완전 실망", "실망만 주네",
    "오글", "오글거려", "진짜 오글", "손발이 오그라", "손발이 오글", "오글오글",
    "불편", "불편하네", "기분 나쁘", "기분 더럽", "짜증", "짜증나", "개짜증", "존짜증", "짜증 유발", "빡치네", "빡친다", "열받", "열 받네",

    "한심", "한심하", "한심하네", "한심하다 진짜", "한심의 극치",
    "역겹", "역겨", "역겹다", "극혐", "진짜 극혐", "혐오", "혐오스럽", "혐오감", "끔찍", "끔찍하", "구역질", "토 나옴", "토나온다",
    "소름 돋는 수준", "민망하네", "민망하다", "창피하", "수치스럽",
    "바보 같", "바보냐", "멍청이", "멍청하네", "생각이 없", "머리가 비었", "멍청하다",
    "지랄", "지랄하네", "또라이", "정신 나간", "정신나갔", "미쳤냐", "미친 거 아님", "약한 거 아님", "뇌가 없",

    "헛소리", "개소리", "소리하네", "헛소리 작작", "개소리 그만", "말 같지도 않", "말이 되냐", "무슨 소리야", "헛소리 좀 그만",
    "꺼져", "꺼지라", "사라져", "눈앞에서 꺼져", "꺼지세요", "볼 가치도 없", "꺼지길",
    "시발", "씨발", "씹발", "ㅅㅂ", "ㅆㅂ", "ㅅㅂㅋㅋ", "씨ㅡ발", "시바", "시바알",
    "존나", "존나 싫", "존나 별로", "존나 구려", "존나 시끄러", "존내", "졸라 싫",
    "좆같", "ㅈ같", "개같", "개같네", "더럽게 못하네", "더럽게 재미없네",

    "개빡치", "빡치네", "현타온다", "현타 온다", "정 떨어지", "정떨어지네",
    "추하다", "추하네", "추잡하", "꼴보기 싫", "꼴불견",
    "불쾌하", "불쾌감", "기분 더럽", "기분 나쁘", "오만정 다 떨어지", "역대급 실망",

    # --- 영상/콘텐츠 디스 ---
    "보는 내내 고통", "보는 사람 생각 1도 없", "보고 있자니 현타", "보다 끔", "보다 껐", "보다가 나감",
    "시간 낭비", "내 시간 돌려줘", "시간이 아깝", "시간 버렸", "인생 낭비", "시간값 못하네",
    "볼 가치도 없", "이걸 왜 봄", "이걸 왜 만듦", "이딴 걸 왜 올리냐", "이따위로 만들",
    "퀄리티 바닥", "퀄리티가 쓰레기", "퀄리티 수준 뭐냐", "수준 낮네", "수준 떨어지", "저급하네", "저급하다",

    "민폐", "민폐 컨텐츠", "폐급", "폐급 영상", "민폐 유발", "민폐 그 자체",
    "차라리 안 올리는 게 낫", "차라리 조용히 있지", "그냥 가만히 있어라", "입 다물",
    "컨셉이 최악", "컨셉이 별로", "컨셉 구리네", "기획이 쓰레기", "기획자 뭐함",

    # --- 실력, 능력 디스 ---
    "실력이 없다", "실력 부족", "실력 너무 부족", "실력부터 키워라", "실력부터 쌓고 와라",
    "재능이 없", "재능 없는", "소질이 없", "안 어울린다", "어울리지도 않",
    "수준 미달", "중학생이 해도 이거보단 낫", "아마추어 냄새", "아마추어 티",
    "초딩이 편집했냐", "편집 수준 뭐냐", "편집 쓰레기", "연기 오글", "연기 너무 어색", "연기력 바닥",

    # --- 발언/사상/태도 비판 (특정 집단 타겟 X, 안전한 범위) ---
    "무례하네", "예의가 없", "이기적이네", "자기중심적", "무책임하", "양심 없",
    "내로남불", "내로남불이네", "위선자 같", "위선적", "위선자 냄새",
    "후안무치", "뻔뻔하네", "뻔뻔스럽", "창피한 줄 알아라",
    "남 생각 1도 안 하네", "배려가 없", "선 넘네", "선 넘었다", "도 넘네",

    # --- clickbait, 광고, 사기 느낌 ---
    "낚시 제목", "클릭 낚시", "클릭베이트", "제목 낚시", "제목이 사기", "제목 장난", "제목이랑 내용이 딴판",
    "사기네", "사기 같", "사기 치네", "구라네", "구라치네", "구라 자제", "거짓말하네", "뻥치지마", "거짓 정보",
    "광고네", "광고였네", "광고였잖아", "몰래 광고", "뒷광고", "스폰 티 너무 나", "광고질", "광고밖에 없네",

    # --- 유튜브형 감정 표현(비꼼, 짜증, 질림) ---
    "이게 웃기냐", "이게 재밌냐", "이게 콘텐츠냐", "이게 콘텐츠야", "이게 영상이냐",
    "뭐가 재밌어", "웃기지도 않", "억지로 웃기려 하지마", "억지웃음", "억지 리액션",
    "억지 감동", "감동 강요", "감성팔이", "감성팔이하네", "감성팔이 지겹", "역겨운 감성팔이",

    "이제 질린다", "질린다 진짜", "맨날 똑같", "컨텐츠가 똑같", "레퍼토리 똑같",
    "컨텐츠 고갈", "아이디어 고갈", "뇌절이네", "뇌절했다", "뇌절 그만",
    "오버하지마", "오바 좀 그만", "오버가 심하네", "말이 너무 많", "TMI 너무 심해",

    # --- 영어 표현들 (영상/채널 디스 포함) ---
    "trash", "total trash", "hot trash", "absolute trash", "literal trash",
    "garbage", "pure garbage", "complete garbage", "garbage content",
    "sucks", "this sucks", "so bad", "really bad", "so freaking bad",
    "horrible", "terrible", "awful", "pathetic", "miserable",
    "boring", "so boring", "super boring", "boring as hell",
    "cringe", "so cringe", "cringe af", "cringy",
    "disgusting", "gross", "makes me sick", "nauseating",
    "worst video", "worst ever", "worst content", "one of the worst",
    "unwatchable", "painful to watch", "waste of time",

    "clickbait", "bait title", "misleading title", "fake title", "fake thumbnail",
    "fake reaction", "overreacting", "overacting", "tryhard", "try hard",
    "you have no talent", "no talent", "zero talent", "lack of skill",
    "you suck", "this channel sucks", "your content sucks",

    # --- 짧은 감탄/슬랭/이모티콘류 (보통 부정 느낌) ---
    "ㅎㅂ", "ㅉㅉ", "ㅉㅉㅉ", "진짜ㅋㅋ", "이게 웃기냐ㅋㅋ", "아 ㅋㅋ", "하 ㅋㅋ",
    "하...진짜", "하..", "하아...", "휴 진짜", "하.. 뭐냐",
    "ㅡㅡ", "ㅡㅡ;", "어이없", "어이없네", "어이없다", "현웃 나옴(비꼼)",
    "구독 취소", "구취한다", "구독 취소함", "차단한다", "차단해야겠다",
    "싫어요 박고 간다", "싫어요 누르고 감", "바로 싫어요", "싫어요각",
    "신고각", "신고한다", "신고 박는다", "정지 먹어라",

    # --- 비교 디스 ---
    "다른 유튜버랑 비교되네", "다른 사람 보러 감", "이럴 거면 다른 채널 봄",
    "예전이 더 낫", "전성기 다 갔", "초창기가 그립다", "초반보다 퀄 떨어졌",
    "예전 영상이 훨씬 나았", "이번 영상 최악이다", "최근 영상 다 별로", "요즘 너무 재미없",

    # --- 음질/영상/편집 디스 ---
    "소리 개작", "소리 개작살", "음질 쓰레기", "소리 왜 이래", "마이크 뭐냐",
    "영상 흔들", "카메라 떨리", "화질 구려", "화질 쓰레기", "화질이 왜 이래",
    "편집 개판", "편집 이상함", "편집 왜 이렇게 했", "편집자 누구야", "편집 때문에 망했",

    # --- 진행/설명/정보 디스 ---
    "설명 개판", "설명도 못하네", "설명이 하나도 안 됨", "설명 전혀 이해 안됨",
    "정보가 틀렸", "정보가 부정확", "허위 정보", "틀린 소리만 하네",
    "도움이 1도 안 됨", "하나도 도움 안 됨", "배운 게 없음", "쓸모가 없네", "쓸데없는 영상",

    # --- 게임 플레이 디스(게임 유튜브용) ---
    "실력이 너무 구려", "손가락이 문제", "컨트롤이 쓰레기", "멘탈 나갔네", "못하니까 욕 먹지",
    "트롤이네", "트롤짓", "트롤 플레이", "게임 센스가 없음", "센스가 없네",
    "팀원만 고생", "팀플 망치네", "혼자 게임하냐", "피드백이 안 됨",

    # --- 리액션/리뷰/먹방 디스 ---
    "리액션이 가식", "가식 덩어리", "가식적인 리액션", "오바 리액션",
    "리뷰가 편향", "리뷰가 엉터리", "진정성이 없", "성의가 없", "성의 없이 찍었",
    "맛 표현도 못하네", "먹는 소리 역겨워", "쩝쩝 소리 역겨워", "쩝쩝 소리 그만",

    # --- 기타 자주 나오는 짧은 부정 표현들 ---
    "이딴 게 왜 추천", "이딴 게 왜 떠", "추천에 왜 떠", "이게 왜 인기냐", "인기 이유를 모르겠",
    "낯부끄럽", "차라리 삭제하", "삭제해라", "영상 내리세요", "내려라",
    "민망해서 못 보겠다", "보다 껐네요", "보다가 꺼버렸", "30초 컷", "10초 컷",
    "정신이 아득하네", "보는 내가 다 창피", "2분도 못 버텼",

    # --- 감정 강조형 (진짜, 개, 존나, 너무 등) 변형 ---
    "진짜 별로", "진짜 최악", "진짜 쓰레기", "진짜 역겹", "진짜 짜증", "진짜 실망", "진짜 구림",
    "개별로", "개실망", "개극혐", "개오글", "개민망", "개망신", "개노잼이네",
    "존별로", "존실망", "존극혐", "존망했네", "존나 못하네", "존나 노잼이네",
    "너무 별로다", "너무 최악이다", "너무 오바하네", "너무 시끄럽", "너무 오글거려", "너무 불편하네",
]


def neg_lexicon_features(texts):
    """
    texts: list[str]
    return: csr_matrix shape (N_samples, len(NEG_LEXICON))
            각 column은 해당 lexicon substring이 등장하면 1, 아니면 0
    """
    n = len(texts)
    m = len(NEG_LEXICON)
    data = np.zeros((n, m), dtype=np.float32)

    for i, t in enumerate(texts):
        if not isinstance(t, str):
            t = str(t)
        for j, pat in enumerate(NEG_LEXICON):
            if pat in t:
                data[i, j] = 1.0
    return csr_matrix(data)

# ---------- Cleaning ----------
url_pat = re.compile(r'https?://\S+|www\.\S+')
mention_pat = re.compile(r'@\w+')
def clean(t: str) -> str:
    t = url_pat.sub('', str(t))
    t = mention_pat.sub('', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# ---------- Data helpers ----------
def ensure_cols(df: pd.DataFrame, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

def load_dataset(path: str, labels=('neg','neu','pos')) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8')
    ensure_cols(df, ['text','label'])
    df['text'] = df['text'].astype(str).map(clean)
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    if not set(df['label'].unique()).issubset(set(labels)):
        raise ValueError(f'Labels must be within {labels}')
    return df

def split_train_valid_test(df: pd.DataFrame, test_size=0.1, valid_size=0.1, seed=42):
    dtrain, dtest = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=seed)
    dtrain, dvalid = train_test_split(dtrain, test_size=valid_size/(1-test_size),
                                      stratify=dtrain['label'], random_state=seed)
    return dtrain.reset_index(drop=True), dvalid.reset_index(drop=True), dtest.reset_index(drop=True)

# ---------- Model ----------
@dataclass
class ClassicCfg:
    word_min_df: int = 3
    word_max_features: int = 100_000
    word_token_pattern: str = r'(?u)\b\w+\b|[^\w\s]' # include symbols/emojis with separate tokens
    char_ngram_low: int = 3
    char_ngram_high: int = 7
    char_min_df: int = 3
    char_max_features: int = 100_000
    sublinear_tf: bool = True
    penalty: str = 'l2'             # 'l2' or 'elasticnet'
    C: float = 1.4
    l1_ratio: float = 0.5           # for elasticnet
    calibrate: str = 'sigmoid'      # 'sigmoid' or 'isotonic'
    cv: int = 3

class ClassicSentiment:
    def __init__(self, cfg: ClassicCfg):
        self.cfg = cfg
        self.word = TfidfVectorizer(analyzer='word',
                                    ngram_range=(1,5),
                                    min_df=cfg.word_min_df,
                                    max_features=cfg.word_max_features,
                                    token_pattern=cfg.word_token_pattern,
                                    sublinear_tf=cfg.sublinear_tf)
        self.char = TfidfVectorizer(analyzer='char',
                                    ngram_range=(cfg.char_ngram_low, cfg.char_ngram_high),
                                    min_df=cfg.char_min_df,
                                    max_features=cfg.char_max_features,
                                    sublinear_tf=cfg.sublinear_tf)
        base = LogisticRegression(max_iter=3000,
                                  class_weight='balanced',
                                  solver='saga',
                                  penalty=cfg.penalty,
                                  C=cfg.C,
                                  l1_ratio=(cfg.l1_ratio if cfg.penalty=='elasticnet' else None),
                                  n_jobs=-1)
        self.clf = CalibratedClassifierCV(base, method=cfg.calibrate, cv=cfg.cv)

    def fit(self, texts, labels):
        Xw = self.word.fit_transform(texts)
        Xc = self.char.fit_transform(texts)
        Xlex = neg_lexicon_features(texts)
        X  = hstack([Xw, Xc,Xlex])
        self.clf.fit(X, labels)
        return self

    def _transform(self, texts):
        Xw = self.word.transform(texts)
        Xc = self.char.transform(texts)
        Xlex = neg_lexicon_features(texts)
        return hstack([Xw, Xc, Xlex])

    def predict(self, texts):
        X = self._transform(texts)
        return self.clf.predict(X)

    def predict_proba(self, texts):
        X = self._transform(texts)
        return self.clf.predict_proba(X), self.clf.classes_

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(self.word, os.path.join(out_dir, 'tfidf_word.joblib'))
        joblib.dump(self.char, os.path.join(out_dir, 'tfidf_char.joblib'))
        joblib.dump(self.clf,  os.path.join(out_dir, 'lr_calibrated.joblib'))

    @staticmethod
    def load(out_dir: str):
        word = joblib.load(os.path.join(out_dir, 'tfidf_word.joblib'))
        char = joblib.load(os.path.join(out_dir, 'tfidf_char.joblib'))
        clf  = joblib.load(os.path.join(out_dir, 'lr_calibrated.joblib'))
        obj = object.__new__(ClassicSentiment)
        obj.word, obj.char, obj.clf = word, char, clf
        obj.cfg = None
        return obj

# ---------- Neutral policy ----------
def apply_neutral_policy(proba: np.ndarray, labels: np.ndarray, tau: float=0.42, gap: float=0.05):
    lab2i = {l:i for i,l in enumerate(labels)}
    pred = labels[proba.argmax(1)].astype(object)
    pmax = proba.max(1)
    pred[pmax < tau] = 'neu'
    if 'neu' in lab2i and 'pos' in lab2i and 'neg' in lab2i:
        neu_i = lab2i['neu']
        comp = np.maximum(proba[:, lab2i['pos']], proba[:, lab2i['neg']])
        mask = (pred=='neu') & ((proba[:, neu_i] - comp) < gap)
        pred[mask] = np.where(proba[mask, lab2i['pos']] >= proba[mask, lab2i['neg']], 'pos', 'neg')
    return np.array(pred)

# ---------- Grid search over valid ----------
from itertools import product
def small_grid_search(train_texts, y_train, valid_texts, y_valid) -> Tuple[ClassicCfg, float, float]: #just for once

    word_vec = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 4),
        min_df=4,
        max_features=150_000,
        token_pattern=r'(?u)\b\w+\b|[^\w\s]',
        sublinear_tf=True,
    )
    char_vec = TfidfVectorizer(
        analyzer='char',
        ngram_range=(1, 5),
        min_df=4,
        max_features=150_000,
        sublinear_tf=True,
    )

    Xw_tr = word_vec.fit_transform(train_texts)
    Xc_tr = char_vec.fit_transform(train_texts)
    Xlex_tr = neg_lexicon_features(train_texts)
    X_train = hstack([Xw_tr, Xc_tr, Xlex_tr])

    Xw_va = word_vec.transform(valid_texts)
    Xc_va = char_vec.transform(valid_texts)
    Xlex_va = neg_lexicon_features(valid_texts)
    X_valid = hstack([Xw_va, Xc_va, Xlex_va])

    C_list  = [0.9, 1.5]
    l1_list = [0.3, 0.5]
    tau_list = [0.42, 0.45]

    best_cfg = None
    best_tau = None
    best_score = -1.0

    for C, l1 in product(C_list, l1_list):
        base = LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            solver='saga',
            penalty='elasticnet',
            C=C,
            l1_ratio=l1,
            n_jobs=-1,
        )
        base.fit(X_train, y_train)
        proba = base.predict_proba(X_valid)
        labels = base.classes_

        for tau in tau_list:
            pred = apply_neutral_policy(proba, labels, tau=tau, gap=0.05)
            macro_f1 = f1_score(y_valid, pred, average='macro')

            if macro_f1 > best_score:
                best_score = macro_f1
                best_tau = tau
                best_cfg = ClassicCfg(
                    word_min_df=4,
                    char_ngram_low=1,
                    char_ngram_high=5,
                    char_min_df=4,
                    C=C,
                    penalty='elasticnet',
                    l1_ratio=l1,
                    calibrate='sigmoid',  # 최종 학습용
                )

    print(f'[Grid-fast] best macro-F1={best_score:.4f} | cfg={best_cfg} | tau={best_tau}')
    return best_cfg, best_tau, best_score

# ---------- Active learning ----------
def load_artifacts(art_dir: str):
    word = joblib.load(os.path.join(art_dir, 'tfidf_word.joblib'))
    char = joblib.load(os.path.join(art_dir, 'tfidf_char.joblib'))
    clf  = joblib.load(os.path.join(art_dir, 'lr_calibrated.joblib'))
    return word, char, clf

def select_active_learning_candidates(csv_path: str, art_dir: str, K=50, per_cluster=4, tau_for_low=0.6) -> pd.DataFrame:
    word, char, clf = load_artifacts(art_dir)
    df_new = pd.read_csv(csv_path, encoding='utf-8', dtype=str)
    text_col = 'text' if 'text' in df_new.columns else df_new.columns[0]
    df_new['text_clean'] = df_new[text_col].astype(str).map(clean)
    texts = df_new['text_clean'].tolist()
    Xw = word.transform(texts)
    Xc = char.transform(texts)
    Xlex = neg_lexicon_features(texts) 
    X  = hstack([Xw, Xc, Xlex])
    proba = clf.predict_proba(X)
    pmax = proba.max(1)
    idx_low = np.where(pmax < tau_for_low)[0]
    if len(idx_low) == 0:
        return df_new.iloc[:0]
    X_low = X[idx_low]
    svd = TruncatedSVD(n_components=min(100, max(2, X_low.shape[1]//50)), random_state=42)
    X_red = svd.fit_transform(X_low)
    k = min(K, len(idx_low))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labs = km.fit_predict(X_red)
    sel = []
    for lab in range(k):
        member_idx = np.where(labs == lab)[0]
        if len(member_idx) == 0: 
            continue
        # choose hardest per cluster
        order = member_idx[np.argsort(pmax[idx_low][member_idx])[:per_cluster]]
        sel.extend(idx_low[order].tolist())
    sel = sorted(set(sel))
    return df_new.iloc[sel].copy()

def merge_gold_and_weak(weak_path, gold_path, key=None, weak_frac=0.4) -> pd.DataFrame:
    df_weak = pd.read_csv(weak_path, encoding='utf-8')
    df_gold = pd.read_csv(gold_path, encoding='utf-8')
    ensure_cols(df_weak, ['text','label'])
    ensure_cols(df_gold, ['text','label'])
    for d in (df_weak, df_gold):
        d['text'] = d['text'].astype(str).map(clean)
        d['label'] = d['label'].astype(str).str.lower().str.strip()
    if key and key in df_gold.columns and key in df_weak.columns:
        df_weak = df_weak[~df_weak[key].isin(set(df_gold[key]))]
    if weak_frac < 1.0 and len(df_weak):
        df_weak = df_weak.sample(frac=weak_frac, random_state=42)
    df_merged = pd.concat([df_gold, df_weak], ignore_index=True).dropna(subset=['text','label'])
    return df_merged.reset_index(drop=True)

# ---------- Batch inference ----------
def batch_predict(csv_path: str, art_dir: str, encoding_try=('utf-8','cp949'), out_path='comments_with_pred.csv',
                  tau=0.42, gap=0.02):
    word, char, clf = load_artifacts(art_dir)
    last_err=None; df_new=None
    for enc in encoding_try:
        try:
            df_new = pd.read_csv(csv_path, encoding=enc, dtype=str)
            break
        except Exception as e:
            last_err=e
    if df_new is None:
        raise last_err
    text_col = 'text' if 'text' in df_new.columns else df_new.columns[0]
    df_new['text_clean'] = df_new[text_col].astype(str).map(clean)
    texts = df_new['text_clean'].tolist()
    from scipy.sparse import hstack
    Xw = word.transform(texts)
    Xc = char.transform(texts)
    Xlex = neg_lexicon_features(texts)  
    X  = hstack([Xw, Xc, Xlex])
    proba = clf.predict_proba(X)
    labels = clf.classes_
    pred = apply_neutral_policy(proba, labels, tau=tau, gap=gap)
    for i, c in enumerate(labels):
        df_new[f'p_{c}'] = proba[:, i]
    df_new['pred'] = pred
    df_new.to_csv(out_path, index=False, encoding='utf-8-sig')
    return out_path

# ---------- chi2 n-grams ----------
def chi2_top_ngrams_from_df(df_in: pd.DataFrame, label_col='pred', text_col=None,
                             topk=20, min_df=5, max_features=50_000, export_dir='exports_ngrams'):
    if text_col is None:
        text_col = 'text_clean' if 'text_clean' in df_in.columns else ('text' if 'text' in df_in.columns else df_in.columns[0])
    texts = df_in[text_col].astype(str).tolist()
    y = df_in[label_col].astype(str).values
    labels_order = list(pd.Series(y).unique())

    def chi2_top(texts, y, analyzer, ngram_range, lowercase=True):
        vec = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range,
                              min_df=min_df, max_features=max_features, lowercase=lowercase)
        X = vec.fit_transform(texts)
        vocab = np.array(vec.get_feature_names_out())
        res = []
        for lab in labels_order:
            y_bin = (y == lab).astype(int)
            chi2_scores, pvals = chi2(X, y_bin)
            order = np.argsort(-chi2_scores)[:topk]
            total_freq = np.asarray(X.sum(axis=0)).ravel()
            class_freq = np.asarray(X[y_bin == 1].sum(axis=0)).ravel()
            rows = []
            for idx in order:
                rows.append({
                    'label': lab,
                    'rank': len(rows)+1,
                    'ngram': vocab[idx],
                    'freq_in_class': int(class_freq[idx]),
                    'freq_total': int(total_freq[idx]),
                    'class_share(%)': round(100.0 * class_freq[idx] / total_freq[idx], 2) if total_freq[idx] else 0.0,
                    'chi2': float(chi2_scores[idx]),
                    'p_value': float(pvals[idx]),
                })
            res.append(pd.DataFrame(rows))
        return pd.concat(res, ignore_index=True)

    df_word = chi2_top(texts, y, analyzer='word', ngram_range=(1,2), lowercase=True)
    df_char = chi2_top(texts, y, analyzer='char', ngram_range=(3,5), lowercase=False)

    os.makedirs(export_dir, exist_ok=True)
    path_w = os.path.join(export_dir, 'top20_word_1_2.csv')
    path_c = os.path.join(export_dir, 'top20_char_3_5.csv')
    df_word.to_csv(path_w, index=False, encoding='utf-8-sig')
    df_char.to_csv(path_c, index=False, encoding='utf-8-sig')
    return path_w, path_c
