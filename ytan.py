import streamlit as st
import os
import glob
import json
import time
import re
import hashlib
import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import google.oauth2.credentials
import googleapiclient.discovery
import google.auth.transport.requests
import extra_streamlit_components as stx 
import google.generativeai as genai
from googleapiclient.errors import HttpError
import html
from pathlib import Path

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from datetime import datetime, timedelta
from github import Github, GithubException # PyGithub

# region [1. 설정 및 상수 (Config & Constants)]
# ==========================================
st.set_page_config(
    page_title="Drama YouTube Insight", 
    page_icon="📊",
    layout="wide", 
    initial_sidebar_state="expanded"
)
# endregion


# region [1-1. 입장게이트 (보안 인증)]
# ==========================================
# Dashboard.py에서 이식된 쿠키/비밀번호 인증 로직
# ==========================================

def _rerun():
    """스트림릿 버전 호환 리런 함수"""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

def get_cookie_manager():
    # 쿠키 매니저는 키(Key)가 고유해야 함
    return stx.CookieManager(key="yt_auth_cookie_manager")

def _hash_password(password: str) -> str:
    return hashlib.sha256(str(password).encode()).hexdigest()

def check_password_with_cookie() -> bool:
    """
    1. Secrets 비밀번호 확인
    2. 쿠키(과거 로그인) 확인
    3. 세션(현재 로그인) 확인
    4. 실패 시 로그인창 띄우고 False 반환 -> 앱 중단
    """
    cookie_manager = get_cookie_manager()
    
    # Secrets에서 비밀번호 가져오기 (없으면 에러)
    # secrets.toml 파일에 [general] 섹션 혹은 최상단에 DASHBOARD_PASSWORD = "..." 가 있어야 함
    secret_pwd = st.secrets.get("DASHBOARD_PASSWORD")
    if not secret_pwd:
        # 호환성을 위해 general 섹션도 체크
        if "general" in st.secrets:
            secret_pwd = st.secrets["general"].get("DASHBOARD_PASSWORD")
            
    if not secret_pwd:
        st.error("🔒 설정 오류: Secrets에 'DASHBOARD_PASSWORD'가 설정되지 않았습니다.")
        st.stop()
        
    hashed_secret = _hash_password(str(secret_pwd))
    
    # 쿠키 읽기
    cookies = cookie_manager.get_all()
    COOKIE_NAME = "yt_dashboard_auth"
    current_token = cookies.get(COOKIE_NAME)
    
    # 인증 검사
    is_cookie_valid = (current_token == hashed_secret)
    is_session_valid = st.session_state.get("auth_success", False)
    
    if is_cookie_valid or is_session_valid:
        if is_cookie_valid and not is_session_valid:
            st.session_state["auth_success"] = True
        return True

    # 로그인 UI
    st.markdown("#### 🔒 Access Restricted")
    st.caption("관계자 외 접근이 제한된 페이지입니다.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        input_pwd = st.text_input("Password", type="password", key="login_pw_input")
        login_btn = st.button("Login", type="primary", use_container_width=True)

    if login_btn:
        if _hash_password(input_pwd) == hashed_secret:
            # 쿠키 굽기 (1일 유효)
            expires = datetime.now() + timedelta(days=1)
            cookie_manager.set(COOKIE_NAME, hashed_secret, expires_at=expires)
            
            st.session_state["auth_success"] = True
            st.success("✅ 인증 성공")
            time.sleep(0.5)
            _rerun()
        else:
            st.error("❌ 비밀번호가 일치하지 않습니다.")
            
    return False

# 🛑 [중요] 여기서 인증 실패 시 앱 실행을 멈춤
if not check_password_with_cookie():
    st.stop()

# ==========================================
# 인증 통과 후 실행되는 영역
# ==========================================
# endregion


# region [1-2. 배포 환경 설정 (Secrets 복원)]
# ==========================================
# Streamlit Secrets에 저장된 토큰 정보를 읽어 로컬 파일로 복원
if "tokens" in st.secrets:
    for file_name, content in st.secrets["tokens"].items():
        if not file_name.endswith(".json"):
            file_name += ".json"
        # 파일이 없으면 생성
        if not os.path.exists(file_name):
            with open(file_name, "w", encoding='utf-8') as f:
                f.write(content)
# endregion


# region [1-3. 디자인 및 상수]
# ==========================================
# UI 디자인 CSS
custom_css = """
    <style>
        /* 헤더 투명화 */
        header[data-testid="stHeader"] { background: transparent; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stDecoration"] {display: none;}
        
        .block-container { padding-top: 1rem; padding-bottom: 3rem; }
        .stApp { background-color: #f8f9fa; }

        /* 카드 및 메트릭 스타일 */
        div[data-testid="stMetric"] {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #eee;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            text-align: center;
        }
        div[data-testid="stMetricLabel"] { font-size: 0.9rem; color: #6c757d; }
        div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; color: #2d3436; }

        [data-testid="stForm"] {
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border: 1px solid #e0e0e0;
            padding: 20px;
        }
        
        h1, h2, h3, h4 { color: #2d3436; font-weight: 700; }
        .stDataFrame { border: 1px solid #f0f0f0; border-radius: 8px; }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

MAX_WORKERS = 7
SCOPES = [
    'https://www.googleapis.com/auth/yt-analytics.readonly',
    'https://www.googleapis.com/auth/youtube.readonly'
]
DEFAULT_LIMIT_DATE = "2025-01-01"

# [지도용] 주요 국가 ISO-2 -> ISO-3 매핑
ISO_MAPPING = {
    'KR': 'KOR', 'US': 'USA', 'JP': 'JPN', 'VN': 'VNM', 'TH': 'THA', 
    'ID': 'IDN', 'TW': 'TWN', 'PH': 'PHL', 'MY': 'MYS', 'IN': 'IND',
    'BR': 'BRA', 'MX': 'MEX', 'RU': 'RUS', 'GB': 'GBR', 'DE': 'DEU',
    'FR': 'FRA', 'CA': 'CAN', 'AU': 'AUS', 'HK': 'HKG', 'SG': 'SGP'
}

def render_md_allow_br(text: str) -> str:
    # 1) 전부 escape 해서 HTML 주입 차단
    escaped = html.escape(text or "")

    # 2) <br> / <br/> / <br /> 만 다시 복원
    escaped = re.sub(r"&lt;br\s*/?&gt;", "<br>", escaped, flags=re.IGNORECASE)
    return escaped
# endregion


# region [2. 유틸리티 함수 (Utilities)]
# ==========================================
def normalize_text(text):
    if not text: return ""
    return re.sub(r'[^a-zA-Z0-9가-힣]', '', text).lower()

PROMPT_FILE_1ST = "1차 질문 프롬프트.md"

def load_text_file(filename: str) -> str:
    base_dir = Path(__file__).resolve().parent  # ytan.py가 있는 폴더
    path = base_dir / filename
    return path.read_text(encoding="utf-8")


def format_korean_number(num):
    if num == 0: return "0회"
    s = ""
    if num >= 100000000:
        eok = num // 100000000
        rem = num % 100000000
        s += f"{int(eok)}억 "
        num = rem
    if num >= 10000:
        man = num // 10000
        rem = num % 10000
        s += f"{int(man)}만 "
        num = rem
    if num > 0:
        s += f"{int(num)}"
    return s.strip() + "회"

TRAFFIC_MAP = {
    'YT_SEARCH': '유튜브 검색', 'RELATED_VIDEO': '추천 동영상',
    'BROWSE_FEATURES': '탐색 기능', 'EXT_URL': '외부 링크',
    'NO_LINK_OTHER': '기타', 'PLAYLIST': '재생목록',
    'VIDEO_CARD': '카드/최종화면', 'NOTIFICATION': '알림'
}

def map_traffic_source(key):
    return TRAFFIC_MAP.get(key, key)

def parse_utc_to_kst_date(utc_str):
    try:
        dt_utc = datetime.strptime(utc_str, "%Y-%m-%dT%H:%M:%SZ")
        dt_kst = dt_utc + timedelta(hours=9)
        return dt_kst.date()
    except: return None

def parse_duration_to_minutes(duration_str):
    if not duration_str: return 0.0
    pattern = re.compile(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?')
    match = pattern.match(duration_str)
    if not match: return 0.0
    h, m, s = match.groups()
    total_sec = (int(h or 0) * 3600) + (int(m or 0) * 60) + (int(s or 0))
    return round(total_sec / 60, 1)

# [수정] Github 업로드 함수 (한글 자모 분리 + 409 충돌 자동 복구)
def upload_to_github(file_path, content_list):
    """
    로컬 캐시 데이터를 GitHub에 저장합니다.
    1. 한글 자모 분리(NFC/NFD) 문제 해결
    2. 409 Conflict(SHA 불일치) 발생 시 자동 재시도
    """
    import unicodedata

    def normalize_str(s):
        return unicodedata.normalize('NFC', s) if s else ""

    try:
        if "github" not in st.secrets: 
            return False, "설정 오류: secrets.toml에 [github] 섹션이 없습니다."
        
        gh_token = st.secrets["github"]["token"]
        gh_repo = st.secrets["github"]["repo_name"]
        gh_branch = st.secrets["github"]["branch"]

        g = Github(gh_token)
        repo = g.get_repo(gh_repo)
        
        json_content = json.dumps(content_list, ensure_ascii=False, indent=2)
        commit_message = f"Update cache: {file_path} (via Streamlit App)"
        
        # 1. 저장소의 모든 파일 목록 가져오기 (파일명 정규화 비교를 위해)
        contents = repo.get_contents("", ref=gh_branch)
        
        target_sha = None
        target_path_on_git = file_path # 기본값은 로컬 파일명
        
        # 2. 정규화된 이름으로 파일 찾기
        search_name = normalize_str(file_path)
        for c in contents:
            if normalize_str(c.path) == search_name:
                target_sha = c.sha
                target_path_on_git = c.path # 깃허브 실존 경로
                break
        
        # 3. 저장 시도 (재시도 로직 포함)
        try:
            if target_sha:
                # 수정 (Update)
                repo.update_file(target_path_on_git, commit_message, json_content, target_sha, branch=gh_branch)
                return True, "Updated (SHA Found)"
            else:
                # 생성 (Create)
                repo.create_file(file_path, commit_message, json_content, branch=gh_branch)
                return True, "Created (New File)"

        except GithubException as e:
            # 🚨 [핵심] 409 Conflict (SHA 불일치) 발생 시 복구 로직
            if e.status == 409:
                print("⚠️ 409 Conflict 발생 -> 최신 SHA 재조회 후 1회 재시도")
                # 해당 파일만 콕 집어서 최신 상태를 다시 가져옴
                fresh_file = repo.get_contents(target_path_on_git, ref=gh_branch)
                # 재시도
                repo.update_file(target_path_on_git, commit_message, json_content, fresh_file.sha, branch=gh_branch)
                return True, "Updated (After 409 Retry)"
            else:
                # 409 외의 다른 에러는 그대로 보고
                raise e

    except Exception as e:
        return False, str(e)
# endregion

# region [3. 시각화 함수 (Visualization)]
# ==========================================
def get_pyramid_chart_and_df(stats_dict, total_views):
    if not stats_dict: return None, None, ""
    
    age_order = ['age13-17', 'age18-24', 'age25-34', 'age35-44', 'age45-54', 'age55-64', 'age65-']
    display_labels = [label.replace('age', '') for label in age_order]
    
    male_data = defaultdict(float); female_data = defaultdict(float)
    table_rows = []; total_male = 0; total_female = 0

    for key, count in stats_dict.items():
        parts = key.split('_')
        if len(parts) != 2: continue
        age_group, gender = parts[0], parts[1]
        if gender not in ['male', 'female']: continue
        if age_group not in age_order: continue 
        
        pct = (count / total_views) * 100 if total_views > 0 else 0
        clean_age = age_group.replace('age', '')
        
        if gender == 'male':
            male_data[clean_age] += pct; total_male += pct
        elif gender == 'female':
            female_data[clean_age] += pct; total_female += pct
            
        table_rows.append({
            "연령": clean_age, "성별": "남" if gender=='male' else "여", 
            "조회수": int(count), "비율": pct
        })

    male_vals = [male_data[l] for l in display_labels]
    female_vals = [female_data[l] for l in display_labels]
    male_vals_neg = [-v for v in male_vals] 

    fig = go.Figure()
    fig.add_trace(go.Bar(y=display_labels, x=male_vals_neg, name='남성', orientation='h',
        marker=dict(color='#5684D5'), text=[f"{v:.1f}%" if v>0 else "" for v in male_vals],
        textfont=dict(color='white'), textposition='auto', hoverinfo='text',
        hovertext=[f"남성 {a}: {v:.1f}%" for a, v in zip(display_labels, male_vals)]))
    fig.add_trace(go.Bar(y=display_labels, x=female_vals, name='여성', orientation='h',
        marker=dict(color='#FF7675'), text=[f"{v:.1f}%" if v>0 else "" for v in female_vals],
        textfont=dict(color='white'), textposition='auto', hoverinfo='text',
        hovertext=[f"여성 {a}: {v:.1f}%" for a, v in zip(display_labels, female_vals)]))
    
    max_val = max(max(male_vals) if male_vals else 0, max(female_vals) if female_vals else 0)
    max_range = max_val * 1.2 if max_val > 0 else 10

    fig.update_layout(
        barmode='overlay',
        xaxis=dict(tickvals=[-max_range, 0, max_range], ticktext=[f"{max_range:.0f}%", "0%", f"{max_range:.0f}%"], range=[-max_range, max_range]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=30, b=10),
        height=300,
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    df = pd.DataFrame(table_rows)
    if not df.empty:
        df['연령'] = pd.Categorical(df['연령'], categories=display_labels, ordered=True)
        df = df.sort_values(['연령', '성별'])

    title_str = f"👥 성별/연령 (남 {total_male:.1f}% vs 여 {total_female:.1f}%)"
    return fig, df, title_str

def get_traffic_chart(traffic_dict):
    if not traffic_dict: return None
    sorted_t = sorted(traffic_dict.items(), key=lambda x: x[1], reverse=True)
    labels = []; values = []
    for k, v in sorted_t[:5]:
        labels.append(map_traffic_source(k)); values.append(v)
    if len(sorted_t) > 5:
        labels.append("기타"); values.append(sum(v for k,v in sorted_t[5:]))
    
    if not values: return None
        
    teal_colors = ['#00b894', '#00cec9', '#55efc4', '#81ecec', '#b2bec3', '#dfe6e9']
    fig = px.pie(names=labels, values=values, hole=0.5, color_discrete_sequence=teal_colors)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        margin=dict(l=20, r=20, t=0, b=20), 
        height=300, 
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def get_keyword_bar_chart(keyword_dict):
    if not keyword_dict: return None
    
    sorted_k = sorted(keyword_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    if not sorted_k: return None
    
    words = [k[0] for k in sorted_k][::-1] 
    counts = [k[1] for k in sorted_k][::-1]
    
    fig = go.Figure(go.Bar(
        x=counts, y=words, orientation='h',
        marker=dict(color='#fdcb6e'),
        text=[f"{int(v):,}" for v in counts], textposition='auto'
    ))
    
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        xaxis=dict(showticklabels=False, visible=False),
        yaxis=dict(tickfont=dict(size=13)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def get_channel_share_chart(ch_details, highlight_channel=None):
    if not ch_details: return None
    sorted_ch = sorted(ch_details, key=lambda x: x['total_views'], reverse=True)
    labels = []; values = []
    top_list = sorted_ch[:5]
    others_sum = sum(ch['total_views'] for ch in sorted_ch[5:])
    
    for ch in top_list:
        labels.append(ch['channel_name']); values.append(ch['total_views'])
    if len(sorted_ch) > 5:
        labels.append("그 외 채널"); values.append(others_sum)
    
    if sum(values) == 0: return None
        
    colors = []
    if highlight_channel:
        for label in labels:
            colors.append('#6c5ce7' if label == highlight_channel else '#dfe6e9')
    else:
        colors = ['#6c5ce7', '#a29bfe', '#8e44ad', '#9b59b6', '#d6a2e8', '#dfe6e9']

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5, marker=dict(colors=colors))])
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        margin=dict(l=20, r=20, t=0, b=20), 
        height=300, 
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def get_engagement_chart(ch_details, highlight_channel=None):
    if not ch_details: return None
    sorted_ch = sorted(ch_details, key=lambda x: x['total_views'], reverse=True)[:10]
    names = [c['channel_name'] for c in sorted_ch]
    like_ratios = [(c['total_likes']/c['total_views']*100) if c['total_views']>0 else 0 for c in sorted_ch]
    share_ratios = [(c['total_shares']/c['total_views']*100) if c['total_views']>0 else 0 for c in sorted_ch]
    
    if not names: return None

    base_like_color = '#ff7675'; base_share_color = '#74b9ff'
    if highlight_channel:
        like_colors = [base_like_color if n == highlight_channel else '#dfe6e9' for n in names]
        share_colors = [base_share_color if n == highlight_channel else '#dfe6e9' for n in names]
    else:
        like_colors = base_like_color; share_colors = base_share_color

    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=like_ratios, name='좋아요(%)', marker_color=like_colors))
    fig.add_trace(go.Bar(x=names, y=share_ratios, name='공유(%)', marker_color=share_colors))

    fig.update_layout(
        barmode='group',
        margin=dict(l=20, r=20, t=10, b=40),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="비율(%)", tickformat=".2f"),
        xaxis=dict(tickangle=-45),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def get_country_map(country_stats):
    if not country_stats: return None
    data = []
    for k, v in country_stats.items():
        iso3 = ISO_MAPPING.get(k, k)
        c_name = k 
        if k == 'KR': c_name = '대한민국'
        elif k == 'US': c_name = '미국'
        elif k == 'JP': c_name = '일본'
        elif k == 'VN': c_name = '베트남'
        elif k == 'TH': c_name = '태국'
        data.append({'iso_alpha': iso3, 'views': v, 'country': c_name, 'fmt_views': format_korean_number(v)})
    
    if not data: return None
    df_map = pd.DataFrame(data)
    df_kr = df_map[df_map['iso_alpha'] == 'KOR']
    df_others = df_map[df_map['iso_alpha'] != 'KOR']
    
    fig = go.Figure()
    if not df_others.empty:
        fig.add_trace(go.Choropleth(
            locations=df_others['iso_alpha'], z=df_others['views'], text=df_others['country'],
            customdata=df_others[['country', 'fmt_views']], colorscale='Teal',
            marker_line_color='#ffffff', marker_line_width=0.5, showscale=True,
            colorbar=dict(title="조회수", x=1.0, len=0.8),
            hovertemplate="<b>%{customdata[0]}</b><br>조회수: %{customdata[1]}<extra></extra>"
        ))
    if not df_kr.empty:
        fig.add_trace(go.Choropleth(
            locations=df_kr['iso_alpha'], z=[1]*len(df_kr), text=df_kr['country'],
            customdata=df_kr[['country', 'fmt_views']], colorscale=[[0, '#5684D5'], [1, '#5684D5']], 
            marker_line_color='#ffffff', marker_line_width=1, showscale=False,
            hovertemplate="<b>%{customdata[0]} (Home)</b><br>조회수: %{customdata[1]}<extra></extra>"
        ))

    fig.update_geos(
        showcountries=True, countrycolor="#E0E0E0",
        showcoastlines=True, coastlinecolor="#E0E0E0",
        showframe=False, projection_type='natural earth',
        fitbounds="locations" if df_map.empty else False,
        bgcolor='rgba(0,0,0,0)'
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), 
        height=400, 
        dragmode='pan',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def get_daily_trend_chart(daily_stats, recent_gap=0):
    """
    daily_stats: Analytics API 일별 조회수 (딕셔너리)
    recent_gap: Data API(실시간) 총합 - Analytics 총합
    """
    if not daily_stats: return None
    
    # 1. 날짜순 정렬 및 데이터 추출
    sorted_dates = sorted(daily_stats.keys())
    daily_views = [daily_stats[d] for d in sorted_dates]
    
    # 2. 누적 조회수(Cumulative) 계산
    cumulative_views = []
    current_sum = 0
    for v in daily_views:
        current_sum += v
        cumulative_views.append(current_sum)
    
    fig = go.Figure()
    
    # [A] 확정된 과거 데이터 (실선)
    fig.add_trace(go.Scatter(
        x=sorted_dates, 
        y=cumulative_views, 
        mode='lines+markers',
        name='누적 조회수 (확정)',
        line=dict(color='#6c5ce7', width=3),
        marker=dict(size=6),
        hovertemplate='%{x}<br>누적: %{y:,}회<extra></extra>'
    ))
    
    # [B] 실시간 구간 연결 (최근 3일 이내일 때만)
    if recent_gap > 0 and sorted_dates:
        last_date_str = sorted_dates[-1]
        last_cum_val = cumulative_views[-1]
        
        today_dt = datetime.today()
        last_anl_dt = datetime.strptime(last_date_str, "%Y-%m-%d")
        
        # 차이가 3일 이내인 경우에만 실시간 점선 연결
        if (today_dt - last_anl_dt).days <= 3:
            target_date_str = today_dt.strftime("%Y-%m-%d")
            final_total_val = last_cum_val + recent_gap
            
            if last_date_str != target_date_str:
                fig.add_trace(go.Scatter(
                    x=[last_date_str, target_date_str],
                    y=[last_cum_val, final_total_val],
                    mode='lines+markers',
                    name='실시간 추이 (최근)',
                    line=dict(color='#ff7675', width=3, dash='dot'),
                    marker=dict(size=6, symbol='circle-open'),
                    hovertemplate=f'<b>실시간(추정)</b><br>현재 총합: %{{y:,}}회<br>(최근 +{recent_gap:,}회 증가)<extra></extra>'
                ))
                
                fig.add_annotation(
                    x=target_date_str, y=final_total_val,
                    text=f"Now (+{recent_gap:,})",
                    showarrow=True, arrowhead=2,
                    ax=0, ay=-20,
                    font=dict(color="#d63031", size=11, weight="bold")
                )

    # [X축 중복 방지 로직]
    # 조회 기간이 짧을 때(예: 30일 이내)는 강제로 '1일 1눈금(D1)'을 적용해 중복을 막습니다.
    # 기간이 길면(예: 1년) 자동(Auto)으로 둬야 겹치지 않습니다.
    dtick_setting = None
    if len(sorted_dates) <= 31: 
        dtick_setting = "D1"  # 1일 1눈금 강제 (중복 해결)

    fig.update_layout(
        title="📈 누적 조회수 성장 추이",
        margin=dict(l=20, r=20, t=40, b=20),
        height=350, 
        xaxis=dict(
            title=None, 
            tickformat="%m-%d", 
            dtick=dtick_setting  # [수정] 여기가 핵심입니다!
        ),
        yaxis=dict(title="총 조회수", tickformat=","),
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def get_efficiency_scatter(video_details):
    if not video_details: return None
    df = pd.DataFrame(video_details)
    if df.empty: return None
    df = df[(df['duration_min'] > 0) & (df['avg_pct'].notnull()) & (df['avg_pct'] > 0)]
    if df.empty: return None
    
    fig = px.scatter(
        df, x='duration_min', y='avg_pct', 
        size='views', color='avg_pct',
        hover_name='title',
        labels={'duration_min': '영상 길이(분)', 'avg_pct': '평균 지속률(%)'},
        color_continuous_scale='Viridis'
    )
    fig.add_vline(x=df['duration_min'].median(), line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=df['avg_pct'].median(), line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#eee'),
        yaxis=dict(showgrid=True, gridcolor='#eee')
    )
    return fig
# endregion


# region [4. API 및 데이터 처리 (API & Data Processing)]
# ==========================================
def get_creds_from_file(token_filename):
    creds = None
    if os.path.exists(token_filename):
        creds = google.oauth2.credentials.Credentials.from_authorized_user_file(token_filename, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(google.auth.transport.requests.Request())
                with open(token_filename, 'w') as token: token.write(creds.to_json())
            except: return None
        else: return None
    return creds

def process_sync_channel(token_file, limit_date, status_box, force_rescan):
    file_label = os.path.basename(token_file).replace("token_", "").replace(".json", "")
    creds = get_creds_from_file(token_file)
    if not creds: 
        status_box.error(f"❌ [{file_label}] 토큰 오류")
        return None
    try:
        youtube = googleapiclient.discovery.build('youtube', 'v3', credentials=creds)
        ch_res = youtube.channels().list(part='snippet,contentDetails', mine=True).execute()
        if not ch_res['items']: 
            status_box.warning(f"⚠️ [{file_label}] 정보 없음")
            return None
        ch_info = ch_res['items'][0]; ch_name = ch_info['snippet']['title']
        uploads_id = ch_info['contentDetails']['relatedPlaylists']['uploads']
        
        cache_file = f"cache_{token_file}"
        
        cached_videos = []
        cached_ids = set()
        
        if not force_rescan and os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f: cached_videos = json.load(f)
            cached_ids = {v['id'] for v in cached_videos}
            status_box.info(f"🔄 [{ch_name}] 확인 중...")
        else: 
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f: cached_videos = json.load(f)
            status_box.info(f"⏳ [{ch_name}] 스캔 시작")
        
        new_videos = []; next_page_token = None; stop_scanning = False
        
        while not stop_scanning:
            req = youtube.playlistItems().list(part='snippet', playlistId=uploads_id, maxResults=50, pageToken=next_page_token)
            res = req.execute()
            for item in res['items']:
                vid = item['snippet']['resourceId']['videoId']
                title = item['snippet']['title']
                desc = item['snippet']['description']
                p_at = item['snippet']['publishedAt']
                
                if p_at < limit_date: stop_scanning = True; break
                
                if not force_rescan and vid in cached_ids: 
                    stop_scanning = True; break
                
                new_videos.append({'id': vid, 'title': title, 'date': p_at, 'description': desc})
            if len(new_videos) > 0 and len(new_videos) % 50 == 0:
                status_box.markdown(f"🏃 **[{ch_name}]** +{len(new_videos)}")
            if not res.get('nextPageToken'): stop_scanning = True
            next_page_token = res.get('nextPageToken')
            if not next_page_token: stop_scanning = True
        
        if force_rescan:
            preserved_videos = [v for v in cached_videos if v['date'] < limit_date]
            final_list = new_videos + preserved_videos
        else:
            final_list = new_videos + cached_videos
        
        if new_videos or force_rescan:
            with open(cache_file, 'w', encoding='utf-8') as f: 
                json.dump(final_list, f, ensure_ascii=False, indent=2)
            
            # [수정] 업로드 결과를 받아서 실패 시 화면에 띄우기!
            is_ok, error_msg = upload_to_github(cache_file, final_list)
            
            if is_ok:
                status_box.success(f"✅ **[{ch_name}]** 깃허브 저장 완료 (+{len(new_videos)})")
            else:
                status_box.error(f"⚠️ **[{ch_name}]** 로컬만 저장됨 / 깃허브 실패:\n{error_msg}")
        else: 
            status_box.success(f"✅ **[{ch_name}]** 최신")
        
        return {'creds': creds, 'name': ch_name, 'videos': final_list}
    except Exception as e:
        status_box.error(f"❌ 에러: {str(e)}")
        return {'error': str(e)}

def process_analysis_channel(channel_data, keyword, vid_start, vid_end, anl_start, anl_end):
    # (기존 로직과 동일 - 생략 없이 그대로 유지)
    creds = channel_data['creds']; videos = channel_data['videos']
    norm_keyword = normalize_text(keyword)
    target_ids = []
    id_map = {} 
    video_date_map = {}
    
    # 1. 영상 필터링
    for v in videos:
        t_match = norm_keyword in normalize_text(v['title'])
        d_match = norm_keyword in normalize_text(v.get('description', ''))
        if not (t_match or d_match): continue
        
        v_dt_kst = parse_utc_to_kst_date(v['date'])
        if v_dt_kst and (vid_start <= v_dt_kst <= vid_end): 
            target_ids.append(v['id'])
            id_map[v['id']] = v['title']
            video_date_map[v['id']] = v['date']
            
    if not target_ids: return None
    
    yt_anl = googleapiclient.discovery.build('youtubeAnalytics', 'v2', credentials=creds)
    youtube = googleapiclient.discovery.build('youtube', 'v3', credentials=creds)
    
    total_views = 0; total_likes = 0; total_shares = 0
    demo = defaultdict(float); traffic = defaultdict(float)
    country = defaultdict(float); daily = defaultdict(float)
    keywords_count = defaultdict(float)
    w_avg_sum = 0; v_for_avg = 0
    over_1m_count = 0 
    
    top_video_stats = []
    
    today_date = datetime.now().date()
    
    if isinstance(anl_end, str):
        anl_end_date = datetime.strptime(anl_end, "%Y-%m-%d").date()
    else:
        anl_end_date = anl_end

    if isinstance(anl_start, str):
        anl_start_date = datetime.strptime(anl_start, "%Y-%m-%d").date()
    else:
        anl_start_date = anl_start
        
    use_hybrid_logic = anl_end_date >= (today_date - timedelta(days=2))
    
    batch_size = 50 
    for i in range(0, len(target_ids), batch_size):
        batch_ids = target_ids[i : i + batch_size]
        vid_str = ",".join(batch_ids)
        
        anl_views_map = {}; anl_likes_map = {}; anl_retention_map = {}
        
        try:
            r_v = yt_anl.reports().query(ids='channel==MINE', startDate=anl_start, endDate=anl_end, metrics='views,likes,averageViewPercentage', dimensions='video', filters=f'video=={vid_str}').execute()
            if 'rows' in r_v and r_v['rows']:
                for r in r_v['rows']:
                    anl_views_map[r[0]] = r[1]; anl_likes_map[r[0]] = r[2]; anl_retention_map[r[0]] = r[3]
            
            r_b = yt_anl.reports().query(ids='channel==MINE', startDate=anl_start, endDate=anl_end, metrics='shares', filters=f'video=={vid_str}').execute()
            if 'rows' in r_b and r_b['rows']: total_shares += r_b['rows'][0][0]

            r_d = yt_anl.reports().query(ids='channel==MINE', startDate=anl_start, endDate=anl_end, metrics='viewerPercentage', dimensions='ageGroup,gender', filters=f'video=={vid_str}').execute()
            batch_total_view_anl = sum(anl_views_map.values())
            if 'rows' in r_d and r_d['rows']:
                for r in r_d['rows']: demo[f"{r[0]}_{r[1]}"] += batch_total_view_anl * (r[2] / 100)

            r_t = yt_anl.reports().query(ids='channel==MINE', startDate=anl_start, endDate=anl_end, metrics='views', dimensions='insightTrafficSourceType', filters=f'video=={vid_str}').execute()
            if 'rows' in r_t and r_t['rows']:
                for r in r_t['rows']: traffic[r[0]] += r[1]
            
            try:
                r_k = yt_anl.reports().query(ids='channel==MINE', startDate=anl_start, endDate=anl_end, metrics='views', dimensions='insightTrafficSourceDetail', filters=f'video=={vid_str};insightTrafficSourceType==YT_SEARCH', maxResults=15, sort='-views').execute()
                if 'rows' in r_k and r_k['rows']:
                    for r in r_k['rows']:
                        if r[0] != 'GOOGLE_SEARCH': keywords_count[r[0]] += r[1]
            except: pass

            r_c = yt_anl.reports().query(ids='channel==MINE', startDate=anl_start, endDate=anl_end, metrics='views', dimensions='country', filters=f'video=={vid_str}', maxResults=50).execute()
            if 'rows' in r_c and r_c['rows']:
                for r in r_c['rows']: country[r[0]] += r[1]

            r_day = yt_anl.reports().query(ids='channel==MINE', startDate=anl_start, endDate=anl_end, metrics='views', dimensions='day', filters=f'video=={vid_str}', sort='day').execute()
            if 'rows' in r_day and r_day['rows']:
                for r in r_day['rows']: daily[r[0]] += r[1]
        except: pass

        try:
            rt_res = youtube.videos().list(part='statistics,contentDetails', id=vid_str).execute()
            rt_stats_map = {}; rt_content_map = {}
            if 'items' in rt_res:
                for item in rt_res['items']:
                    rt_stats_map[item['id']] = item['statistics']
                    rt_content_map[item['id']] = item['contentDetails']

            for vid_id in batch_ids:
                v_date_str = video_date_map.get(vid_id)
                if not v_date_str: continue
                v_upload_dt = parse_utc_to_kst_date(v_date_str)
                if isinstance(v_upload_dt, datetime): v_upload_dt = v_upload_dt.date()

                a_v = anl_views_map.get(vid_id, 0)
                a_l = anl_likes_map.get(vid_id, 0)
                a_pct = anl_retention_map.get(vid_id, 0)
                
                stats = rt_stats_map.get(vid_id, {})
                rt_v = int(stats.get('viewCount', 0))
                rt_l = int(stats.get('likeCount', 0))
                
                final_v = 0; final_l = 0
                
                if use_hybrid_logic:
                    if v_upload_dt >= anl_start_date:
                        final_v = rt_v; final_l = rt_l
                    else:
                        deduct_end_date = anl_start_date - timedelta(days=1)
                        deduct_start_str = v_upload_dt.strftime("%Y-%m-%d")
                        deduct_end_str = deduct_end_date.strftime("%Y-%m-%d")
                        
                        past_views = 0; past_likes = 0; is_past_data_fetched = False
                        try:
                            if v_upload_dt <= deduct_end_date:
                                r_past = yt_anl.reports().query(ids='channel==MINE', startDate=deduct_start_str, endDate=deduct_end_str, metrics='views,likes', filters=f'video=={vid_id}').execute()
                                if 'rows' in r_past and r_past['rows']:
                                    past_views = r_past['rows'][0][0]
                                    past_likes = r_past['rows'][0][1]
                                    is_past_data_fetched = True
                                else:
                                    past_views = 0; past_likes = 0; is_past_data_fetched = True
                        except: is_past_data_fetched = False

                        if is_past_data_fetched:
                            final_v = max(0, rt_v - past_views)
                            final_l = max(0, rt_l - past_likes)
                        else:
                            final_v = a_v; final_l = a_l

                        if final_v == rt_v and a_v > 0 and past_views == 0:
                             final_v = a_v; final_l = a_l
                else:
                    final_v = a_v; final_l = a_l

                if final_v == 0 and rt_v > 0 and use_hybrid_logic:
                    final_v = rt_v; final_l = rt_l
                
                total_views += final_v; total_likes += final_l
                
                if final_v > 0 and a_pct > 0:
                    w_avg_sum += (final_v * a_pct)
                    v_for_avg += final_v
                
                if rt_v >= 1000000: over_1m_count += 1

                if final_v > 0:
                    top_video_stats.append({
                        'id': vid_id, 'title': id_map.get(vid_id, 'Unknown'),
                        'views': rt_v, 'likes': rt_l, 'period_views': final_v, 'period_likes': final_l,
                        'avg_pct': a_pct if a_pct > 0 else None,
                        'duration_min': parse_duration_to_minutes(rt_content_map.get(vid_id, {}).get('duration'))
                    })

        except: pass
        time.sleep(0.05)

    if not top_video_stats and total_views == 0: return None
    top_video_stats.sort(key=lambda x: x['period_views'], reverse=True)

    return {
        'channel_name': channel_data['name'], 'video_count': len(target_ids),
        'total_views': total_views, 'total_likes': total_likes, 'total_shares': total_shares,
        'avg_view_pct': (w_avg_sum/v_for_avg) if v_for_avg > 0 else 0,
        'demo_counts': demo, 'traffic_counts': traffic,
        'country_counts': country, 'daily_stats': daily,
        'keywords_counts': keywords_count,
        'top_video_stats': top_video_stats,
        'over_1m_count': over_1m_count 
    }
# endregion

# region [5. 메인 UI 및 실행 로직 (Main UI & Execution)]
# ==========================================
st.title("📊 Drama YouTube Insight")

# --- 사이드바 ---
with st.sidebar:
    st.header("🎛️ 데이터 관리 센터")
    token_files = glob.glob("token_*.json")
    st.markdown("---")
    st.caption("데이터 동기화")
    if st.button("🔄 최신 영상 업데이트", type="primary", use_container_width=True):
        if not token_files: st.error("연동된 토큰 파일이 없습니다.")
        else:
            st.session_state['channels_data'] = []
            st.write("--- 업데이트 진행 중 ---")
            placeholders = {tf: st.empty() for tf in token_files}
            ready = []
            ctx = get_script_run_ctx()
            def sync_worker(tf, sb):
                add_script_run_ctx(ctx=ctx)
                return process_sync_channel(tf, DEFAULT_LIMIT_DATE, sb, False)
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futs = {ex.submit(sync_worker, tf, placeholders[tf]): tf for tf in token_files}
                for f in as_completed(futs):
                    res = f.result()
                    if res and 'name' in res: ready.append(res)
            st.session_state['channels_data'] = ready
            if ready: st.success("업데이트 완료!")
            
    st.markdown("---")
    with st.expander("🔒 고급 관리자 설정"):
        if 'admin_unlocked' not in st.session_state: st.session_state['admin_unlocked'] = False
        if not st.session_state['admin_unlocked']:
            if st.text_input("비밀번호", type="password", key="pw_input") == "dima1234":
                st.session_state['admin_unlocked'] = True
                st.rerun()
        if st.session_state['admin_unlocked']:
            st.success("✅ 관리자 모드 On")
            l_date = st.date_input("수집 마지노선", value=pd.to_datetime(DEFAULT_LIMIT_DATE))
            if st.button("🚨 전체 재수집", type="secondary"):
                st.session_state['channels_data'] = []
                placeholders = {tf: st.empty() for tf in token_files}
                ready = []
                ctx = get_script_run_ctx()
                def deep_worker(tf, sb):
                    add_script_run_ctx(ctx=ctx)
                    return process_sync_channel(tf, l_date.strftime("%Y-%m-%d"), sb, True)
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                    futs = {ex.submit(deep_worker, tf, placeholders[tf]): tf for tf in token_files}
                    for f in as_completed(futs):
                        res = f.result()
                        if res and 'name' in res: ready.append(res)
                st.session_state['channels_data'] = ready
                if ready: st.success("완료!")
            if st.button("🔒 잠금"):
                st.session_state['admin_unlocked'] = False
                st.rerun()

# --- 메인 ---
if 'channels_data' not in st.session_state or not st.session_state['channels_data']:
    token_files = glob.glob("token_*.json")
    temp_data = []
    for tf in token_files:
        cf = f"cache_{tf}"
        if os.path.exists(cf):
             with open(cf, 'r', encoding='utf-8') as f:
                try:
                    vids = json.load(f); creds = get_creds_from_file(tf)
                    if creds:
                        lbl = os.path.basename(tf).replace("token_", "").replace(".json", "")
                        temp_data.append({'creds': creds, 'name': lbl, 'videos': vids})
                except: pass
    if temp_data: st.session_state['channels_data'] = temp_data
    else: st.info("👋 환영합니다! 사이드바에서 [최신 영상 업데이트]를 먼저 진행해주세요.")

if 'channels_data' in st.session_state and st.session_state['channels_data']:
    data = st.session_state['channels_data']
    tv = sum(len(c['videos']) for c in data)
    
    st.markdown(f"""
    <div style='background-color:white; padding:10px 20px; border-radius:8px; border:1px solid #eee; margin-bottom:20px; display:flex; align-items:center; gap:10px;'>
        <span>✅ <b>연동 상태:</b> 채널 <span style='color:#2980b9; font-weight:bold'>{len(data)}개</span></span>
        <span style='color:#ddd'>|</span>
        <span>📁 <b>DB 영상:</b> <span style='color:#2980b9; font-weight:bold'>{tv:,}개</span></span>
    </div>
    """, unsafe_allow_html=True)

    today = datetime.today()
    first_day = today.replace(day=1)

    with st.form("analysis_form"):
        st.subheader("🔍 통합 분석 설정")
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1: keyword = st.text_input("분석 IP", placeholder="예: 눈물의 여왕")
        with c2: v_dates = st.date_input("영상 업로드 기간", value=(first_day, today))
        with c3: a_dates = st.date_input("데이터 산출 기간", value=(first_day, today))
        submit_btn = st.form_submit_button("분석 시작", type="primary", use_container_width=True)

    if submit_btn:
        if isinstance(v_dates, tuple):
            v_start = v_dates[0]; v_end = v_dates[1] if len(v_dates)>1 else v_dates[0]
        else: v_start = v_end = v_dates
        if isinstance(a_dates, tuple):
            a_start = a_dates[0]; a_end = a_dates[1] if len(a_dates)>1 else a_dates[0]
        else: a_start = a_end = a_dates

        if not keyword.strip(): st.error("⚠️ 분석 IP를 입력해주세요.")
        else:
            # ⬇️ [추가] 새 분석 시작 시 챗봇 상태 초기화 (대화 내용 삭제)
            st.session_state["chat_active"] = False
            st.session_state["chat_history"] = []
            st.session_state["chat_context_comments"] = ""

            # 1. 챗봇용 날짜 저장
            vs_str, ve_str = v_start.strftime("%Y-%m-%d"), v_end.strftime("%Y-%m-%d")
            st.session_state['analysis_dates'] = {'start': vs_str, 'end': ve_str}
            
            # 2. 분석용 날짜 준비
            as_str = a_start.strftime("%Y-%m-%d"); ae_str = a_end.strftime("%Y-%m-%d")
            
            prog_bar = st.progress(0, text="데이터 분석 중...")
            ch_details_results = []
            
            ctx = get_script_run_ctx()
            def worker(cd, kw, vs, ve, ast, aet):
                add_script_run_ctx(ctx=ctx)
                return process_analysis_channel(cd, kw, vs, ve, ast, aet)

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = {ex.submit(worker, ch, keyword, v_start, v_end, as_str, ae_str): ch for ch in data}
                done = 0
                for f in as_completed(futures):
                    done += 1
                    prog_bar.progress(done/len(data), text=f"채널 분석 중... ({done}/{len(data)})")
                    res = f.result()
                    if res: ch_details_results.append(res)

            prog_bar.empty()
            
            # 결과 저장 및 안내
            if ch_details_results:
                st.session_state['analysis_raw_results'] = ch_details_results
                st.session_state['analysis_keyword'] = keyword
                st.success(f"분석 완료! 총 {len(ch_details_results)}개 채널에서 데이터를 찾았습니다.")
            else:
                st.session_state['analysis_raw_results'] = []
                st.warning(f"⚠️ '{keyword}'에 대한 분석 결과가 없습니다.\n\n"
                           f"1. **영상 업로드 기간**이 초기화되었는지 확인해보세요. (현재 설정: {vs_str} ~ {ve_str})\n"
                           f"2. 키워드가 정확한지 확인해보세요.")

    if 'analysis_raw_results' in st.session_state and st.session_state['analysis_raw_results']:
        raw_data = st.session_state['analysis_raw_results']
        current_kw = st.session_state['analysis_keyword']
        
        st.divider()
        st.markdown(f"### 📊 분석 리포트: <span style='color:#2980b9;'>{current_kw}</span>", unsafe_allow_html=True)
        
        ch_names = sorted([d['channel_name'] for d in raw_data])
        sel_options = ["전체 채널 합산"] + ch_names
        
        c_sel_col, _ = st.columns([1, 2])
        with c_sel_col:
            selected_ch = st.selectbox("분석 대상 채널 선택", sel_options, label_visibility="collapsed")
        
        if selected_ch == "전체 채널 합산":
            target_data = raw_data
        else:
            target_data = [d for d in raw_data if d['channel_name'] == selected_ch]
            
        final_views = 0; final_likes = 0; final_shares = 0
        final_over_1m = 0; final_vid_count = 0
        final_stats = defaultdict(float); final_traffic = defaultdict(float)
        final_country = defaultdict(float); final_daily = defaultdict(float)
        final_keywords = defaultdict(float)
        w_avg_sum = 0; v_for_avg = 0
        final_top_videos = []
        
        for d in target_data:
            final_views += d['total_views']
            final_likes += d['total_likes']
            final_shares += d['total_shares']
            final_over_1m += d['over_1m_count']
            final_vid_count += d['video_count']
            
            if d['avg_view_pct'] > 0 and d['total_views'] > 0:
                w_avg_sum += (d['avg_view_pct'] * d['total_views'])
                v_for_avg += d['total_views']
            
            for k, v in d['demo_counts'].items(): final_stats[k] += v
            for k, v in d['traffic_counts'].items(): final_traffic[k] += v
            for k, v in d['country_counts'].items(): final_country[k] += v
            for k, v in d['daily_stats'].items(): final_daily[k] += v
            for k, v in d['keywords_counts'].items(): final_keywords[k] += v
            if 'top_video_stats' in d: final_top_videos.extend(d['top_video_stats'])
            
        final_avg_pct = (w_avg_sum / v_for_avg) if v_for_avg > 0 else 0
        
        anl_total_daily = sum(final_daily.values())
        recent_gap = final_views - anl_total_daily
        
        safe_anl_date = datetime.now() - timedelta(days=3)
        safe_str = safe_anl_date.strftime("%Y-%m-%d")
        
        if final_views > 0 or len(final_top_videos) > 0:
            st.caption(f"ℹ️ **데이터 기준**: 인구통계/경로 등은 **~{safe_str}** 확정치, **총 조회수/좋아요 및 리스트**는 **실시간(Realtime)** 데이터입니다.")

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("총 조회수", f"{int(final_views):,}")
            m2.metric("분석 영상", f"{final_vid_count:,}개")
            m3.metric("100만+ 영상", f"{final_over_1m:,}개")
            m4.metric("평균 시청지속률", f"{final_avg_pct:.1f}%")
            m5.metric("총 좋아요", f"{int(final_likes):,}")
            m6.metric("총 공유", f"{int(final_shares):,}")
            st.write("")

            fig_demo, df_table, _ = get_pyramid_chart_and_df(final_stats, final_views)
            if fig_demo:
                c1, c2 = st.columns([1.6, 1])
                with c1:
                    st.markdown("##### 👥 성별/연령 분포")
                    with st.container(border=True):
                        st.plotly_chart(fig_demo, use_container_width=True)
                with c2:
                    st.markdown("##### 📋 상세 데이터")
                    with st.container(border=True):
                        if not df_table.empty:
                            df_disp = df_table.copy()
                            df_disp['조회수'] = df_disp['조회수'].apply(lambda x: f"{x:,}")
                            df_disp['비율'] = df_disp['비율'].apply(lambda x: f"{x:.1f}%")
                            st.dataframe(df_disp, use_container_width=True, hide_index=True, height=300)
                st.write("")

            fig_trend = get_daily_trend_chart(final_daily, recent_gap)
            if fig_trend:
                st.markdown("##### 📈 일별 조회수 추이 (Analytics + Realtime Gap)")
                with st.container(border=True):
                    st.plotly_chart(fig_trend, use_container_width=True)
                st.write("")

            st.markdown("##### 🥇 인기 영상 TOP 100 (기간 내 성과 기준)")
            with st.container(border=True):
                if final_top_videos:
                    unique_vids_map = {v['id']: v for v in final_top_videos}
                    deduped_vids = list(unique_vids_map.values())
                    
                    top_vids = sorted(deduped_vids, key=lambda x: x['period_views'], reverse=True)[:100]
                    
                    df_top = pd.DataFrame(top_vids)
                    df_top['link'] = df_top['id'].apply(lambda x: f"https://youtu.be/{x}")
                    
                    df_show = df_top[['title', 'period_views', 'avg_pct', 'period_likes', 'link']].copy()
                    df_show.columns = ['제목', '조회수', '지속률(%)', '좋아요', '바로가기']
                    
                    df_show['조회수'] = df_show['조회수'].apply(lambda x: f"{int(x):,}")
                    df_show['좋아요'] = df_show['좋아요'].apply(lambda x: f"{int(x):,}")
                    
                    st.data_editor(
                        df_show,
                        column_config={
                            "바로가기": st.column_config.LinkColumn(display_text="Watch 🎬"),
                            "지속률(%)": st.column_config.NumberColumn(format="%.1f%%"),
                        },
                        hide_index=True, use_container_width=True, disabled=True
                    )
                else: st.caption("데이터가 없습니다.")
            st.write("")

            fig_traffic = get_traffic_chart(final_traffic)
            fig_keywords = get_keyword_bar_chart(final_keywords)
            
            if fig_traffic or fig_keywords:
                r2_1, r2_2 = st.columns(2)
                with r2_1:
                    if fig_traffic:
                        st.markdown("##### 🚦 유입 경로 Top 5")
                        with st.container(border=True):
                            st.plotly_chart(fig_traffic, use_container_width=True)
                with r2_2:
                    if fig_keywords:
                        st.markdown("##### 🔍 Top 10 검색어 (SEO)")
                        with st.container(border=True):
                            st.plotly_chart(fig_keywords, use_container_width=True)
                st.write("")
            
            show_share = (selected_ch == "전체 채널 합산") and (len(raw_data) > 1)
            fig_share = get_channel_share_chart(raw_data, highlight_channel=None) if show_share else None
            if selected_ch != "전체 채널 합산" and len(raw_data) > 1:
                fig_share = get_channel_share_chart(raw_data, highlight_channel=selected_ch)
            
            fig_engage = get_engagement_chart(target_data, highlight_channel=selected_ch if selected_ch!="전체 채널 합산" else None)

            if fig_share or fig_engage:
                r3_1, r3_2 = st.columns(2)
                with r3_1:
                    if fig_share:
                        st.markdown("##### 🏆 채널별 점유율")
                        with st.container(border=True):
                            st.plotly_chart(fig_share, use_container_width=True)
                with r3_2:
                    if fig_engage:
                        st.markdown("##### ❤️ 좋아요/공유 비율")
                        with st.container(border=True):
                            st.plotly_chart(fig_engage, use_container_width=True)
                st.write("")

            fig_map = get_country_map(final_country)
            if fig_map:
                st.markdown("##### 🌍 글로벌 조회수 분포")
                with st.container(border=True):
                    st.plotly_chart(fig_map, use_container_width=True)
                st.write("")

            if final_top_videos:
                valid_scatter_vids = [v for v in final_top_videos if v.get('avg_pct') is not None and v.get('avg_pct') > 0]
                fig_scatter = get_efficiency_scatter(valid_scatter_vids)
                if fig_scatter:
                    st.markdown("##### ⚡ 영상 효율성 매트릭스 (길이 vs 지속률)")
                    st.caption("우상단에 위치할수록 영상도 길고 끝까지 보는 고효율 콘텐츠입니다.")
                    with st.container(border=True):
                        st.plotly_chart(fig_scatter, use_container_width=True)

        else:
            st.warning("⚠️ 검색 결과가 없습니다.")
# endregion

# region [6. AI 댓글 분석 및 챗봇 (AI Chatbot Integration)]
# ==========================================
# 사용자 요청에 따라 애널리틱스 하단에 통합된 챗봇 모듈입니다.
# 기존 ytcc_chatbot.py의 핵심 기능을 이식했습니다.
# ==========================================

# [설정] API 키 로드 (secrets.toml에 설정 필요)
GEMINI_API_KEYS = st.secrets.get("GEMINI_API_KEYS", [])
YT_PUBLIC_KEYS = st.secrets.get("YT_API_KEYS", [])
GEMINI_MODEL = "gemini-2.5-flash-lite"

# [함수] Gemini 호출 (Rotating Key)
def call_gemini_integrated(system_prompt, user_prompt):
    if not GEMINI_API_KEYS: return "⚠️ 설정 오류: 'GEMINI_API_KEYS'가 secrets에 없습니다."
    
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold

    # 안전 설정 해제
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    # 키 순환 시도
    for key in GEMINI_API_KEYS:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system_prompt)
            resp = model.generate_content(
                user_prompt, 
                safety_settings=safety_settings,
                request_options={"timeout": 120}
            )
            if resp and hasattr(resp, 'text'):
                return resp.text
        except Exception as e:
            continue # 다음 키 시도
            
    return "❌ AI 응답 생성 실패 (API 키 할당량 초과 또는 오류)"

# [함수] UGC 영상 검색 (기간 필터 + OST 제외 + 중복 제외)
def search_ugc_videos(keyword, existing_ids, start_date=None, end_date=None, max_search=50):
    if not YT_PUBLIC_KEYS: return []
    
    youtube_pub = None
    for key in YT_PUBLIC_KEYS:
        try:
            youtube_pub = googleapiclient.discovery.build('youtube', 'v3', developerKey=key)
            # 키 유효성 테스트
            youtube_pub.search().list(part='id', q='test', maxResults=1).execute()
            break
        except: continue
    
    if not youtube_pub: return []

    ugc_ids = []
    try:
        # 날짜 포맷 변환 (YYYY-MM-DD -> RFC 3339 포맷)
        p_after = f"{start_date}T00:00:00Z" if start_date else None
        p_before = f"{end_date}T23:59:59Z" if end_date else None

        # 키워드로 검색 (기간 필터 적용)
        search_res = youtube_pub.search().list(
            q=keyword, 
            part="id,snippet", 
            type="video", 
            maxResults=max_search, 
            order="relevance",
            publishedAfter=p_after,
            publishedBefore=p_before
        ).execute()
        
        existing_set = set(existing_ids)
        
        for item in search_res.get('items', []):
            vid = item['id']['videoId']
            title = item['snippet']['title']
            
            # [필터링] 이미 수집된 영상이거나, 제목에 'OST'가 포함된 경우 제외
            if vid in existing_set: continue
            if "ost" in title.lower(): continue  # 대소문자 구분 없이 OST 제외
            
            ugc_ids.append(vid)
                
    except Exception as e:
        # 검색 실패 시 조용히 넘김 (빈 리스트 반환)
        pass
        
    return ugc_ids

# [함수] 댓글 수집 (제한 대폭 완화)
def collect_comments_fast(video_ids, max_comments=10000, max_workers=None):
    """
    - 영상별 commentThreads.list(최대 100개) 호출을 ThreadPool로 병렬화
    - Streamlit UI 객체(st.*)는 스레드 안에서 절대 건드리지 않음
    """
    if not YT_PUBLIC_KEYS or not video_ids:
        return ""

    # 중복 제거(순서 유지)
    seen = set()
    video_ids = [v for v in video_ids if not (v in seen or seen.add(v))]

    # 워커 수: 너무 크게 잡으면 429/쿼터/속도 역효과 날 수 있음
    # 기본값: MAX_WORKERS(파일 상수) 또는 8 중 작은 값 권장
    if max_workers is None:
        max_workers = min(8, max(1, MAX_WORKERS))  # MAX_WORKERS는 파일 상수(현재 7)
    max_workers = max(1, min(max_workers, len(video_ids)))

    # 스레드 로컬로 "키별 youtube client" 캐시 (build 비용 절감)
    import threading
    _tls = threading.local()

    def _get_client(dev_key: str):
        if not hasattr(_tls, "clients"):
            _tls.clients = {}
        if dev_key not in _tls.clients:
            # cache_discovery=False: discovery 캐시/IO 줄여서 build 조금 가벼워짐
            _tls.clients[dev_key] = googleapiclient.discovery.build(
                "youtube", "v3", developerKey=dev_key, cache_discovery=False
            )
        return _tls.clients[dev_key]

    def _fetch_one(vid: str, dev_key: str):
        out = []
        try:
            yt = _get_client(dev_key)
            req = yt.commentThreads().list(
                part="snippet",
                videoId=vid,
                maxResults=100,                 # 1회 최대
                textFormat="plainText",
                order="relevance"
            )
            res = req.execute()
            for item in res.get("items", []):
                snip = item["snippet"]["topLevelComment"]["snippet"]
                text = (snip.get("textDisplay") or "").replace("\n", " ")
                likes = int(snip.get("likeCount") or 0)
                out.append((likes, text))
        except Exception:
            # 댓글 비활성/차단/권한/일시적 에러 등은 그냥 스킵
            pass
        return out

    comments = []
    total = 0

    # 배치 단위로 future를 만들면 메모리/캔슬이 좀 더 수월함
    batch_size = max_workers * 6

    def _chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    submit_counter = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for batch in _chunks(video_ids, batch_size):
            futures = []
            for vid in batch:
                dev_key = YT_PUBLIC_KEYS[submit_counter % len(YT_PUBLIC_KEYS)]
                submit_counter += 1
                futures.append(ex.submit(_fetch_one, vid, dev_key))

            for fut in as_completed(futures):
                rows = fut.result() or []
                for likes, text in rows:
                    comments.append(f"[♥{likes}] {text}")
                    total += 1
                    if total >= max_comments:
                        return "\n".join(comments)

    return "\n".join(comments)


# ==========================================
# [UI] AI 대화하기 섹션
# ==========================================
st.divider()

# 세션 상태 초기화
if "chat_active" not in st.session_state: st.session_state["chat_active"] = False
if "chat_history" not in st.session_state: st.session_state["chat_history"] = []
if "chat_context_comments" not in st.session_state: st.session_state["chat_context_comments"] = ""

# 버튼: 분석 결과가 있을 때만 활성화
if 'analysis_raw_results' in st.session_state and st.session_state['analysis_raw_results']:
    
    # 1. 진입 버튼 (챗봇 비활성 상태일 때)
    if not st.session_state["chat_active"]:
        st.subheader("🤖 AI 심층 분석")
        st.caption("현재 분석된 데이터를 바탕으로 UGC(외부 반응)까지 포함하여 AI와 대화합니다.")
        
        if st.button("💬 AI와 대화하기 (UGC 포함)", type="primary"):
            with st.spinner("🔄 외부 여론 수집 및 AI 분석 중입니다... (데이터 양에 따라 시간이 소요될 수 있습니다)"):
                # A. 데이터 준비
                current_kw = st.session_state.get('analysis_keyword', '')
                raw_results = st.session_state['analysis_raw_results']
                
                # [적용] 저장해둔 날짜 꺼내오기
                dates = st.session_state.get('analysis_dates', {})
                v_start = dates.get('start')
                v_end = dates.get('end')
                
                # B. 채널 내 영상 ID 추출 (제한 없음!)
                channel_vids = []
                for ch in raw_results:
                    if 'top_video_stats' in ch:
                        channel_vids.extend([v['id'] for v in ch['top_video_stats']])
                
                # C. UGC(외부) 영상 추가 검색 (기간 필터 + OST 제외 적용)
                ugc_vids = search_ugc_videos(current_kw, channel_vids, start_date=v_start, end_date=v_end)
                
                # D. 댓글 수집 (채널 영상 전체 + UGC 영상 전체)
                # [수정] 슬라이싱([:20]) 제거 -> 전체 리스트 사용
                target_vids = channel_vids + ugc_vids
                
                # [수정] 넉넉하게 10,000개까지 수집 요청
                collected_text = collect_comments_fast(target_vids, max_comments=10000)
                st.session_state["chat_context_comments"] = collected_text
                
                # E. 프롬프트 구성 (전문성 강화 및 정형화된 포맷 적용)
                sys_prompt = ( load_text_file(PROMPT_FILE_1ST))

                
                video_info_str = f"채널 공식 영상 {len(channel_vids)}개 + UGC(외부) 영상 {len(ugc_vids)}개"
                if v_start and v_end:
                    video_info_str += f" (기간: {v_start} ~ {v_end})"

                user_payload = (
                    f"분석 주제(키워드): {current_kw}\n"
                    f"분석 대상: {video_info_str}\n"
                    f"댓글 데이터 샘플:\n{collected_text[:150000]}..." # 15만자 제한 (컨텍스트 윈도우 활용)
                )
                
                # F. 첫 분석 실행
                ai_response = call_gemini_integrated(sys_prompt, user_payload)
                
                # G. 세션 저장 및 챗봇 활성화
                st.session_state["chat_history"].append({"role": "assistant", "content": ai_response})
                st.session_state["chat_active"] = True
                st.rerun()

    # 2. 채팅 인터페이스 (활성화 시)
    else:
        st.subheader(f"💬 AI 챗봇: {st.session_state.get('analysis_keyword', '분석')}")
        
        # 닫기 버튼
        if st.button("❌ 대화 종료 및 초기화"):
            st.session_state["chat_active"] = False
            st.session_state["chat_history"] = []
            st.session_state["chat_context_comments"] = ""
            st.rerun()
            
        # 대화 기록 출력
        chat_container = st.container(border=True)
        with chat_container:
            for msg in st.session_state["chat_history"]:
                with st.chat_message(msg["role"]):
                    if msg["role"] == "assistant":
                        st.markdown(render_md_allow_br(msg["content"]), unsafe_allow_html=True)
                    else:
                        st.markdown(msg["content"]) 
        
        # 후속 질문 입력
        if prompt := st.chat_input("추가로 궁금한 점을 물어보세요 (예: 주연 배우 연기 반응은 어때?)"):
            # 사용자 메시지 표시
            st.session_state["chat_history"].append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
            
            # AI 응답 생성
            with st.spinner("AI가 답변을 생성 중입니다..."):
                context_comments = st.session_state.get("chat_context_comments", "")
                
                sys_prompt_followup = (
                    "역할: 너는 앞서 분석한 댓글 데이터를 바탕으로 사용자의 구체적인 질문에 답하는 조사관이다.\n"
                    "규칙:\n"
                    "1. 질문과 직접 관련된 댓글만 필터링하여 근거로 제시하라.\n"
                    "2. 뇌피셜(추측)을 자제하고 데이터에 기반해 답하라.\n"
                    "3. 댓글 인용 시 욕설은 마스킹 처리하라.\n"
                )
                
                # 이전 대화 맥락 포함 (최근 2턴)
                conversation_context = ""
                for m in st.session_state["chat_history"][-4:]:
                    conversation_context += f"[{m['role']}]: {m['content']}\n"

                payload_followup = (
                    f"댓글 데이터:\n{context_comments[:150000]}\n\n"
                    f"이전 대화:\n{conversation_context}\n\n"
                    f"사용자 질문: {prompt}"
                )
                
                ai_reply = call_gemini_integrated(sys_prompt_followup, payload_followup)
                
                st.session_state["chat_history"].append({"role": "assistant", "content": ai_reply})
                st.rerun()

else:
    # 데이터가 없을 때 안내 문구
    st.info("ℹ️ AI 대화 기능은 상단에서 [분석 시작]을 완료한 후에 사용할 수 있습니다.")
# endregion