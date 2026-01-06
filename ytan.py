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
import html as _html
from pathlib import Path
from streamlit.components.v1 import html as st_html

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from datetime import datetime, timedelta

import pymongo
from pymongo import MongoClient
import certifi

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

# region [1. 설정 및 상수 (Config & Constants)]
# ==========================================
st.set_page_config(
    page_title="YT(PGC) Data Tracker", 
    page_icon="📊",
    layout="wide", 
    initial_sidebar_state="collapsed" 
)
# endregion


# region [1-1. 입장게이트 (보안 인증)]
# ==========================================
def _rerun():
    """스트림릿 버전 호환 리런 함수"""
    if hasattr(st, "rerun"): st.rerun()
    else: st.experimental_rerun()

def get_cookie_manager():
    return stx.CookieManager(key="yt_auth_cookie_manager")

def _hash_password(password: str) -> str:
    return hashlib.sha256(str(password).encode()).hexdigest()

def check_password_with_cookie() -> bool:
    cookie_manager = get_cookie_manager()
    secret_pwd = st.secrets.get("DASHBOARD_PASSWORD")
    if not secret_pwd:
        if "general" in st.secrets: secret_pwd = st.secrets["general"].get("DASHBOARD_PASSWORD")
            
    if not secret_pwd:
        st.error("🔒 설정 오류: Secrets에 'DASHBOARD_PASSWORD'가 설정되지 않았습니다.")
        st.stop()
        
    hashed_secret = _hash_password(str(secret_pwd))
    cookies = cookie_manager.get_all()
    COOKIE_NAME = "yt_dashboard_auth"
    current_token = cookies.get(COOKIE_NAME)
    
    is_cookie_valid = (current_token == hashed_secret)
    is_session_valid = st.session_state.get("auth_success", False)
    
    if is_cookie_valid or is_session_valid:
        if is_cookie_valid and not is_session_valid:
            st.session_state["auth_success"] = True
        return True

    st.markdown("#### 🔒 Access Restricted")
    st.caption("관계자 외 접근이 제한된 페이지입니다.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        input_pwd = st.text_input("Password", type="password", key="login_pw_input")
        login_btn = st.button("Login", type="primary", use_container_width=True)

    if login_btn:
        if _hash_password(input_pwd) == hashed_secret:
            expires = datetime.now() + timedelta(days=1)
            cookie_manager.set(COOKIE_NAME, hashed_secret, expires_at=expires)
            st.session_state["auth_success"] = True
            st.success("✅ 인증 성공")
            time.sleep(0.5)
            _rerun()
        else:
            st.error("❌ 비밀번호가 일치하지 않습니다.")
    return False

if not check_password_with_cookie(): st.stop()
# endregion


# region [1-2. 배포 환경 설정 (Secrets 복원)]
# ==========================================
if "tokens" in st.secrets:
    for file_name, content in st.secrets["tokens"].items():
        if not file_name.endswith(".json"): file_name += ".json"
        if not os.path.exists(file_name):
            with open(file_name, "w", encoding='utf-8') as f: f.write(content)
# endregion


# region [1-3. 디자인 및 상수]
# ==========================================
custom_css = """
    <style>
        header[data-testid="stHeader"] { background: transparent; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stDecoration"] {display: none;}
        .block-container { padding-top: 1rem; padding-bottom: 3rem; }
        .stApp { background-color: #f8f9fa; }
        div[data-testid="stMetric"] { background-color: white; padding: 15px; border-radius: 10px; border: 1px solid #eee; box-shadow: 0 2px 4px rgba(0,0,0,0.02); text-align: center; }
        div[data-testid="stMetricLabel"] { font-size: 0.9rem; color: #6c757d; }
        div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; color: #2d3436; }
        [data-testid="stForm"] { background-color: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #e0e0e0; padding: 20px; }
        h1, h2, h3, h4 { color: #2d3436; font-weight: 700; }
        .stDataFrame { border: 1px solid #f0f0f0; border-radius: 8px; }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

MAX_WORKERS = 7
SCOPES = ['https://www.googleapis.com/auth/yt-analytics.readonly', 'https://www.googleapis.com/auth/youtube.readonly']
DEFAULT_LIMIT_DATE = "2024-01-01"

ISO_MAPPING = {
    'KR': 'KOR', 'US': 'USA', 'JP': 'JPN', 'VN': 'VNM', 'TH': 'THA', 
    'ID': 'IDN', 'TW': 'TWN', 'PH': 'PHL', 'MY': 'MYS', 'IN': 'IND',
    'BR': 'BRA', 'MX': 'MEX', 'RU': 'RUS', 'GB': 'GBR', 'DE': 'DEU',
    'FR': 'FRA', 'CA': 'CAN', 'AU': 'AUS', 'HK': 'HKG', 'SG': 'SGP'
}

def render_md_allow_br(text: str) -> str:
    raw = (text or "").strip()
    raw = re.sub(r"^\s*```[a-zA-Z]*\s*", "", raw)
    raw = re.sub(r"\s*```\s*$", "", raw)
    escaped = html.escape(raw)
    escaped = re.sub(r"&lt;br\s*/?&gt;", "<br>", escaped, flags=re.IGNORECASE)
    return escaped
# endregion


# region [2. 유틸리티 함수 (Utilities)]
# ==========================================
def normalize_text(text):
    if not text: return ""
    return re.sub(r'[^a-zA-Z0-9가-힣]', '', text).lower()

def format_korean_number(num):
    if num == 0: return "0회"
    s = ""
    if num >= 100000000:
        eok = num // 100000000; rem = num % 100000000
        s += f"{int(eok)}억 "
        num = rem
    if num >= 10000:
        man = num // 10000; rem = num % 10000
        s += f"{int(man)}만 "
        num = rem
    if num > 0: s += f"{int(num)}"
    return s.strip() + "회"

TRAFFIC_MAP = {
    'YT_SEARCH': '유튜브 검색', 'RELATED_VIDEO': '추천 동영상',
    'BROWSE_FEATURES': '탐색 기능', 'EXT_URL': '외부 링크',
    'NO_LINK_OTHER': '기타', 'PLAYLIST': '재생목록',
    'VIDEO_CARD': '카드/최종화면', 'NOTIFICATION': '알림'
}
def map_traffic_source(key): return TRAFFIC_MAP.get(key, key)

def parse_utc_to_kst_date(utc_str):
    try:
        dt_utc = datetime.strptime(utc_str, "%Y-%m-%dT%H:%M:%SZ")
        return (dt_utc + timedelta(hours=9)).date()
    except: return None

def parse_duration_to_minutes(duration_str):
    if not duration_str: return 0.0
    pattern = re.compile(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?')
    match = pattern.match(duration_str)
    if not match: return 0.0
    h, m, s = match.groups()
    total_sec = (int(h or 0) * 3600) + (int(m or 0) * 60) + (int(s or 0))
    return round(total_sec / 60, 1)

# ==========================================
# [변경] MongoDB 연결 및 데이터 처리 함수
# ==========================================
@st.cache_resource
def init_mongo():
    try:
        if "mongo" not in st.secrets: return None
        uri = st.secrets["mongo"]["uri"]
        # certifi: SSL 인증서 오류 방지용
        return MongoClient(uri, tlsCAFile=certifi.where())
    except Exception as e:
        print(f"MongoDB Init Error: {e}")
        return None

def save_to_mongodb(file_name, content_list):
    try:
        client = init_mongo()
        if not client: return False, "DB 연결 실패"
        
        db = client.get_database("yt_dashboard")
        col_videos = db.get_collection("videos")
        col_meta = db.get_collection("metadata")
        
        # 1. 기존 데이터 삭제 (Clean Update)
        col_videos.delete_many({"source_file": file_name})
        
        # 2. 새 데이터 삽입
        if content_list:
            docs = []
            for item in content_list:
                doc = item.copy()
                doc['source_file'] = file_name
                docs.append(doc)
            col_videos.insert_many(docs)
            
        # 3. 메타데이터(업데이트 시간) 갱신
        col_meta.update_one(
            {"_id": file_name},
            {"$set": {"updated_at": datetime.now(), "count": len(content_list)}},
            upsert=True
        )
        
        # 캐시 초기화
        load_from_mongodb.clear()
        get_last_update_time.clear()
        
        return True, f"MongoDB Saved ({len(content_list)} items)"

    except Exception as e:
        return False, str(e)

@st.cache_data(ttl=3600, show_spinner=False)
def load_from_mongodb(file_name):
    try:
        client = init_mongo()
        if not client: return []
        
        db = client.get_database("yt_dashboard")
        # source_file이 일치하는 문서 조회 (_id 제외)
        return list(db.get_collection("videos").find({"source_file": file_name}, {"_id": 0, "source_file": 0}))

    except Exception as e:
        print(f"Load Error: {e}")
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_last_update_time(file_name):
    try:
        client = init_mongo()
        if not client: return None
        
        db = client.get_database("yt_dashboard")
        doc = db.get_collection("metadata").find_one({"_id": file_name})
        
        if doc and 'updated_at' in doc:
            ts = doc['updated_at']
            # KST 변환
            dt_kst = ts + timedelta(hours=9)
            return dt_kst.strftime("%Y-%m-%d %H:%M")
        return None
    except: return None
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
            
        table_rows.append({"연령": clean_age, "성별": "남" if gender=='male' else "여", "조회수": int(count), "비율": pct})

    male_vals = [male_data[l] for l in display_labels]
    female_vals = [female_data[l] for l in display_labels]
    male_vals_neg = [-v for v in male_vals] 

    fig = go.Figure()
    fig.add_trace(go.Bar(y=display_labels, x=male_vals_neg, name='남성', orientation='h', marker=dict(color='#5684D5'), text=[f"{v:.1f}%" if v>0 else "" for v in male_vals], textposition='auto'))
    fig.add_trace(go.Bar(y=display_labels, x=female_vals, name='여성', orientation='h', marker=dict(color='#FF7675'), text=[f"{v:.1f}%" if v>0 else "" for v in female_vals], textposition='auto'))
    
    max_val = max(max(male_vals) if male_vals else 0, max(female_vals) if female_vals else 0)
    rng = max_val * 1.2 if max_val > 0 else 10

    fig.update_layout(barmode='overlay', xaxis=dict(tickvals=[-rng, 0, rng], ticktext=[f"{rng:.0f}%", "0%", f"{rng:.0f}%"], range=[-rng, rng]), margin=dict(l=10, r=10, t=30, b=10), height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    
    df = pd.DataFrame(table_rows)
    if not df.empty:
        df['연령'] = pd.Categorical(df['연령'], categories=display_labels, ordered=True)
        df = df.sort_values(['연령', '성별'])

    return fig, df, f"👥 성별/연령 (남 {total_male:.1f}% vs 여 {total_female:.1f}%)"

def get_traffic_chart(traffic_dict):
    if not traffic_dict: return None
    sorted_t = sorted(traffic_dict.items(), key=lambda x: x[1], reverse=True)
    labels = []; values = []
    for k, v in sorted_t[:5]: labels.append(map_traffic_source(k)); values.append(v)
    if len(sorted_t) > 5: labels.append("기타"); values.append(sum(v for k,v in sorted_t[5:]))
    if not values: return None
    fig = px.pie(names=labels, values=values, hole=0.5, color_discrete_sequence=['#00b894', '#00cec9', '#55efc4', '#81ecec'])
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(margin=dict(l=20, r=20, t=0, b=20), height=300, showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
    return fig

def get_keyword_bar_chart(keyword_dict):
    if not keyword_dict: return None
    sorted_k = sorted(keyword_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    if not sorted_k: return None
    words = [k[0] for k in sorted_k][::-1]; counts = [k[1] for k in sorted_k][::-1]
    fig = go.Figure(go.Bar(x=counts, y=words, orientation='h', marker=dict(color='#fdcb6e'), text=[f"{int(v):,}" for v in counts], textposition='auto'))
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300, xaxis=dict(visible=False), paper_bgcolor='rgba(0,0,0,0)')
    return fig

def get_channel_share_chart(ch_details, highlight_channel=None):
    if not ch_details: return None
    sorted_ch = sorted(ch_details, key=lambda x: x['total_views'], reverse=True)
    labels = []; values = []
    for ch in sorted_ch[:5]: labels.append(ch['channel_name']); values.append(ch['total_views'])
    if len(sorted_ch) > 5: labels.append("그 외 채널"); values.append(sum(ch['total_views'] for ch in sorted_ch[5:]))
    if sum(values) == 0: return None
    colors = ['#6c5ce7' if l == highlight_channel else '#dfe6e9' for l in labels] if highlight_channel else px.colors.qualitative.Plotly
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5, marker=dict(colors=colors))])
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(margin=dict(l=20, r=20, t=0, b=20), height=300, showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
    return fig

def get_engagement_chart(ch_details, highlight_channel=None):
    if not ch_details: return None
    sorted_ch = sorted(ch_details, key=lambda x: x['total_views'], reverse=True)[:10]
    names = [c['channel_name'] for c in sorted_ch]
    l_r = [(c['total_likes']/c['total_views']*100) if c['total_views']>0 else 0 for c in sorted_ch]
    s_r = [(c['total_shares']/c['total_views']*100) if c['total_views']>0 else 0 for c in sorted_ch]
    if not names: return None
    lc = ['#ff7675' if n == highlight_channel else '#dfe6e9' for n in names] if highlight_channel else '#ff7675'
    sc = ['#74b9ff' if n == highlight_channel else '#dfe6e9' for n in names] if highlight_channel else '#74b9ff'
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=l_r, name='좋아요(%)', marker_color=lc))
    fig.add_trace(go.Bar(x=names, y=s_r, name='공유(%)', marker_color=sc))
    fig.update_layout(barmode='group', margin=dict(l=20, r=20, t=10, b=40), height=300, legend=dict(orientation="h", y=1.02), paper_bgcolor='rgba(0,0,0,0)')
    return fig

def get_country_map(country_stats):
    if not country_stats: return None
    data = []
    for k, v in country_stats.items():
        iso3 = ISO_MAPPING.get(k, k)
        c_name = {'KR':'대한민국','US':'미국','JP':'일본','VN':'베트남'}.get(k, k)
        data.append({'iso_alpha': iso3, 'views': v, 'country': c_name, 'fmt': format_korean_number(v)})
    if not data: return None
    df_map = pd.DataFrame(data)
    fig = go.Figure(go.Choropleth(locations=df_map['iso_alpha'], z=df_map['views'], text=df_map['country'], customdata=df_map[['country','fmt']], colorscale='Teal', hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>"))
    fig.update_geos(showcountries=True, countrycolor="#E0E0E0", projection_type='natural earth', fitbounds="locations", bgcolor='rgba(0,0,0,0)')
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=400, dragmode='pan', paper_bgcolor='rgba(0,0,0,0)')
    return fig

def get_daily_trend_chart(daily_stats, recent_gap=0):
    if not daily_stats: return None
    sorted_dates = sorted(daily_stats.keys())
    daily_views = [daily_stats[d] for d in sorted_dates]
    cum_views = []; cur = 0
    for v in daily_views: cur += v; cum_views.append(cur)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sorted_dates, y=cum_views, mode='lines+markers', name='누적 조회수', line=dict(color='#6c5ce7', width=3)))
    
    if recent_gap > 0 and sorted_dates:
        last_d = sorted_dates[-1]; last_v = cum_views[-1]
        today_s = datetime.today().strftime("%Y-%m-%d")
        if last_d != today_s:
            fig.add_trace(go.Scatter(x=[last_d, today_s], y=[last_v, last_v+recent_gap], mode='lines+markers', name='실시간(추정)', line=dict(color='#ff7675', dash='dot')))
            
    dtick_val = "D1" if len(sorted_dates) <= 31 else None
    fig.update_layout(title="📈 누적 조회수 성장 추이", margin=dict(l=20,r=20,t=40,b=20), height=350, xaxis=dict(dtick=dtick_val), hovermode="x unified", paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.02))
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
    # [내부함수] MongoDB 로깅
    def log_to_db(level, msg, detail=None):
        try:
            client = init_mongo()
            if client:
                db = client.get_database("yt_dashboard")
                db.get_collection("system_logs").insert_one({
                    'level': level,
                    'msg': msg,
                    'detail': str(detail),
                    'source': 'process_sync_channel',
                    'timestamp': datetime.now()
                })
        except: pass

    if status_box is None:
        class DummyBox:
            def success(self, m): pass
            def error(self, m): print(f"[Error] {m}")
            def warning(self, m): pass
            def info(self, m): pass
            def markdown(self, m): pass
            def caption(self, m): pass
        status_box = DummyBox()

    file_label = os.path.basename(token_file).replace("token_", "").replace(".json", "")
    creds = get_creds_from_file(token_file)
    if not creds: 
        status_box.error(f"❌ [{file_label}] 토큰 오류")
        return None
        
    try:
        youtube = googleapiclient.discovery.build('youtube', 'v3', credentials=creds)
        ch_res = youtube.channels().list(part='snippet,contentDetails', mine=True).execute()
        if not ch_res['items']: return None
        ch_info = ch_res['items'][0]; ch_name = ch_info['snippet']['title']
        uploads_id = ch_info['contentDetails']['relatedPlaylists']['uploads']
        
        cache_name = f"cache_{os.path.basename(token_file)}"
        
        if force_rescan:
            cached_videos = []
            cached_ids = set()
            # [디버깅] 실제 적용된 날짜를 눈으로 확인시켜줌 (범인 검거용)
            status_box.info(f"🔥 [{ch_name}] 전체 재수집 시작 (Limit: {limit_date})")
        else:
            cached_videos = load_from_mongodb(cache_name)
            cached_ids = {v['id'] for v in cached_videos}
            status_box.info(f"🔄 [{ch_name}] DB 확인 ({len(cached_videos)}개)...")
        
        new_videos = []; next_pg = None; stop = False
        consecutive_cached_count = 0
        SAFE_BUFFER = 50 
        
        while not stop:
            req = youtube.playlistItems().list(part='snippet', playlistId=uploads_id, maxResults=50, pageToken=next_pg)
            res = req.execute()
            
            items = res.get('items', [])
            if not items:
                # 아이템이 없으면 바로 멈추지 말고 페이지 토큰이라도 있는지 확인 (빈 페이지 방지)
                if not res.get('nextPageToken'):
                    stop = True
                else:
                    next_pg = res.get('nextPageToken')
                    time.sleep(0.1) # 과속 방지
                    continue
            
            for item in items:
                vid = item['snippet']['resourceId']['videoId']
                p_at = item['snippet']['publishedAt']
                
                # 날짜 제한 체크 (String 비교)
                if p_at < limit_date: 
                    stop = True; break
                
                if not force_rescan and vid in cached_ids:
                    consecutive_cached_count += 1
                    if consecutive_cached_count >= SAFE_BUFFER:
                        status_box.caption(f"✋ 안전지대 도달 (연속 {SAFE_BUFFER}개 중복). 수집 종료.")
                        stop = True
                        break
                    continue 
                else:
                    consecutive_cached_count = 0
                    new_videos.append({
                        'id': vid, 
                        'title': item['snippet']['title'], 
                        'date': p_at, 
                        'description': item['snippet']['description']
                    })
            
            if len(new_videos) > 0 and len(new_videos)%50==0:
                status_box.markdown(f"🏃 **[{ch_name}]** +{len(new_videos)}")
            
            next_pg = res.get('nextPageToken')
            if not next_pg: stop = True
            
            # [핵심 수정] 과속 방지 턱 (API 누락 방지)
            time.sleep(0.3)
        
        if force_rescan:
            final_list = new_videos
        else:
            final_list = new_videos + cached_videos
            
        # [핵심 수정] 최종 저장 전 중복 제거 (Clean Data)
        # ID가 같은 녀석이 있으면 하나만 남김
        final_list = list({v['id']:v for v in final_list}.values())
        
        is_ok, msg = save_to_mongodb(cache_name, final_list)
        
        if is_ok:
            if new_videos:
                status_box.success(f"🔥 **[{ch_name}] +{len(new_videos)} 업데이트** (총 {len(final_list)}개)")
                log_to_db('success', f"[{ch_name}] 업데이트 완료", f"추가: {len(new_videos)}")
            else:
                status_box.success(f"✅ **[{ch_name}] 최신 유지** (총 {len(final_list)}개)")
        else:
            status_box.error(f"DB 저장 실패: {msg}")
            log_to_db('error', f"[{ch_name}] 저장 실패", msg)
        
        return {'creds': creds, 'name': ch_name, 'videos': final_list}
        
    except Exception as e:
        status_box.error(f"에러: {e}")
        log_to_db('fatal', f"[{ch_name}] 로직 에러", str(e))
        return {'error': str(e)}

def process_analysis_channel(channel_data, keyword, vid_start, vid_end, anl_start, anl_end):
    creds = channel_data['creds']; videos = channel_data['videos']
    norm_kw = normalize_text(keyword)
    
    # [1] 필터링 (Keyword & Date)
    temp_target_ids = [] 
    id_map = {}; date_map = {}
    
    for v in videos:
        if norm_kw in normalize_text(v['title']) or norm_kw in normalize_text(v.get('description','')):
            v_dt = parse_utc_to_kst_date(v['date'])
            if v_dt and (vid_start <= v_dt <= vid_end):
                temp_target_ids.append(v['id'])
                id_map[v['id']] = v['title']
                date_map[v['id']] = v['date']
    
    target_ids = list(dict.fromkeys(temp_target_ids))
    if not target_ids: return None
    
    # [2] API 준비
    yt_anl = googleapiclient.discovery.build('youtubeAnalytics', 'v2', credentials=creds)
    youtube = googleapiclient.discovery.build('youtube', 'v3', credentials=creds)
    
    tot_v=0; tot_l=0; tot_s=0; over_1m=0
    demo=defaultdict(float); traffic=defaultdict(float); country=defaultdict(float); daily=defaultdict(float); kws=defaultdict(float)
    w_avg_sum=0; v_for_avg=0; top_vids=[]
    
    today = datetime.now().date()
    anl_start_date = datetime.strptime(str(anl_start), "%Y-%m-%d").date() if isinstance(anl_start, str) else anl_start
    anl_end_date = datetime.strptime(str(anl_end), "%Y-%m-%d").date() if isinstance(anl_end, str) else anl_end
    use_hybrid = anl_end_date >= (today - timedelta(days=2))
    
    # ===== [3] 배치 처리 함수 (병렬 최적화 적용) =====
    def fetch_batch_data(batch_ids):
        vid_str = ",".join(batch_ids)
        
        # 각 API 요청을 정의 (실행은 하지 않음)
        def _get_main_metrics():
            r = yt_anl.reports().query(ids='channel==MINE', startDate=anl_start, endDate=anl_end, metrics='views,likes,averageViewPercentage', dimensions='video', filters=f'video=={vid_str}').execute()
            return {row[0]: {'v': row[1], 'l': row[2], 'p': row[3]} for row in r.get('rows', [])}

        def _get_shares():
            r = yt_anl.reports().query(ids='channel==MINE', startDate=anl_start, endDate=anl_end, metrics='shares', filters=f'video=={vid_str}').execute()
            return r.get('rows', [])

        def _get_demo():
            r = yt_anl.reports().query(ids='channel==MINE', startDate=anl_start, endDate=anl_end, metrics='viewerPercentage', dimensions='ageGroup,gender', filters=f'video=={vid_str}').execute()
            return r.get('rows', [])

        def _get_traffic():
            r = yt_anl.reports().query(ids='channel==MINE', startDate=anl_start, endDate=anl_end, metrics='views', dimensions='insightTrafficSourceType', filters=f'video=={vid_str}').execute()
            return r.get('rows', [])

        def _get_keywords():
            try:
                r = yt_anl.reports().query(ids='channel==MINE', startDate=anl_start, endDate=anl_end, metrics='views', dimensions='insightTrafficSourceDetail', filters=f'video=={vid_str};insightTrafficSourceType==YT_SEARCH', maxResults=15, sort='-views').execute()
                return r.get('rows', [])
            except: return []

        def _get_countries():
            r = yt_anl.reports().query(ids='channel==MINE', startDate=anl_start, endDate=anl_end, metrics='views', dimensions='country', filters=f'video=={vid_str}', maxResults=50).execute()
            return r.get('rows', [])

        def _get_daily():
            r = yt_anl.reports().query(ids='channel==MINE', startDate=anl_start, endDate=anl_end, metrics='views', dimensions='day', filters=f'video=={vid_str}', sort='day').execute()
            return r.get('rows', [])

        def _get_realtime():
            r = youtube.videos().list(part='statistics,contentDetails', id=vid_str).execute()
            return {item['id']: item for item in r.get('items', [])}

        # ThreadPoolExecutor를 사용하여 8개의 요청을 동시에 실행
        with ThreadPoolExecutor(max_workers=8) as executor:
            f_main = executor.submit(_get_main_metrics)
            f_share = executor.submit(_get_shares)
            f_demo = executor.submit(_get_demo)
            f_traf = executor.submit(_get_traffic)
            f_keyw = executor.submit(_get_keywords)
            f_ctry = executor.submit(_get_countries)
            f_daily = executor.submit(_get_daily)
            f_real = executor.submit(_get_realtime)

            # 결과 수집 (여기서 대기 발생, 하지만 병렬이므로 가장 느린 요청 시간만큼만 소요)
            return (
                f_main.result(), f_share.result(), f_demo.result(), 
                f_traf.result(), f_keyw.result(), f_ctry.result(), 
                f_daily.result(), f_real.result()
            )

    # [4] 메인 루프
    for i in range(0, len(target_ids), 50):
        batch = target_ids[i:i+50]
        
        try:
            results = fetch_batch_data(batch)
            process_queue = [(batch, results)]
        except:
            # 배치 실패 시 개별 시도 (안전장치)
            process_queue = []
            for single_id in batch:
                try:
                    res_single = fetch_batch_data([single_id])
                    process_queue.append(([single_id], res_single))
                except: pass

        # [5] 데이터 집계
        for batch_ids, (l_anl, l_s, l_demo, l_tr, l_kws, l_ctr, l_day, l_rt) in process_queue:
            if l_s: tot_s += l_s[0][0]
            
            b_v = sum(x['v'] for x in l_anl.values())
            for r in l_demo: demo[f"{r[0]}_{r[1]}"] += b_v*(r[2]/100)
            for r in l_tr: traffic[r[0]] += r[1]
            for r in l_kws: 
                if r[0]!='GOOGLE_SEARCH': kws[r[0]]+=r[1]
            for r in l_ctr: country[r[0]]+=r[1]
            for r in l_day: daily[r[0]]+=r[1]
            
            for vid_id in batch_ids:
                v_up = parse_utc_to_kst_date(date_map[vid_id])
                a_data = l_anl.get(vid_id, {'v':0,'l':0,'p':0})
                rt_item = l_rt.get(vid_id, {})
                rt_stat = rt_item.get('statistics', {})
                rt_v = int(rt_stat.get('viewCount',0)); rt_l = int(rt_stat.get('likeCount',0))
                
                fin_v = a_data['v']; fin_l = a_data['l']
                if use_hybrid:
                    if v_up >= anl_start_date: fin_v=rt_v; fin_l=rt_l
                    else:
                        if fin_v == 0 and rt_v > 0: fin_v=rt_v; fin_l=rt_l
                
                tot_v += fin_v; tot_l += fin_l
                if fin_v>0 and a_data['p']>0:
                    w_avg_sum += (fin_v*a_data['p']); v_for_avg += fin_v
                
                if fin_v >= 1000000: over_1m += 1
                
                if fin_v > 0:
                    dur = parse_duration_to_minutes(rt_item.get('contentDetails',{}).get('duration'))
                    top_vids.append({
                        'id': vid_id, 'title': id_map.get(vid_id,'?'),
                        'views': rt_v, 'period_views': fin_v, 'period_likes': fin_l,
                        'avg_pct': a_data['p'], 'duration_min': dur
                    })
    
    top_vids.sort(key=lambda x: x['period_views'], reverse=True)
    
    return {
        'channel_name': channel_data['name'], 
        'video_count': len(target_ids),
        'total_views': tot_v, 'total_likes': tot_l, 'total_shares': tot_s,
        'avg_view_pct': (w_avg_sum/v_for_avg) if v_for_avg>0 else 0,
        'demo_counts': demo, 'traffic_counts': traffic, 'country_counts': country,
        'daily_stats': daily, 'keywords_counts': kws, 'top_video_stats': top_vids,
        'over_1m_count': over_1m
    }

# [스케줄러] - log_to_mongo 의존성 제거
def job_auto_update_data():
    print(f"⏰ [Auto] 시작: {datetime.now()}")
    token_files = glob.glob("token_*.json")
    if not token_files: return
    
    try:
        cnt = 0
        for tf in token_files:
            res = process_sync_channel(tf, DEFAULT_LIMIT_DATE, None, False)
            if res and 'error' not in res: cnt+=1
        
        load_from_mongodb.clear()
        get_last_update_time.clear()
        
        # 로그 저장 (직접 DB 호출)
        try:
            client = init_mongo()
            if client:
                client.get_database("yt_dashboard").get_collection("system_logs").insert_one({
                    'level': 'info', 'msg': "스케줄러 완료",
                    'detail': f"성공: {cnt}/{len(token_files)}",
                    'source': 'scheduler',
                    'timestamp': datetime.now()
                })
        except: pass
        
    except Exception as e:
        # 에러 로그 (직접 DB 호출)
        try:
            client = init_mongo()
            if client:
                client.get_database("yt_dashboard").get_collection("system_logs").insert_one({
                    'level': 'fatal', 'msg': "스케줄러 오류",
                    'detail': str(e),
                    'source': 'scheduler',
                    'timestamp': datetime.now()
                })
        except: pass

@st.cache_resource
def init_scheduler():
    s = BackgroundScheduler()
    s.add_job(job_auto_update_data, CronTrigger(hour=9, minute=0, timezone=pytz.timezone('Asia/Seoul')))
    s.start()

init_scheduler()
# endregion


# region [5. 메인 UI 및 실행 로직 (Main UI & Execution)]
# ==========================================
st.title("📊 YT(PGC) Data Tracker")

with st.sidebar:
    st.header("🎛️ 데이터 관리")
    if 'admin_auth' not in st.session_state: st.session_state['admin_auth'] = False
    
    if not st.session_state['admin_auth']:
        if st.text_input("관리자 비밀번호", type="password") == st.secrets.get("admin",{}).get("password",""):
            st.session_state['admin_auth'] = True; st.rerun()
            
    if st.session_state['admin_auth']:
        token_files = glob.glob("token_*.json")
        st.markdown("---")
        
        if token_files:
            last_ts = get_last_update_time(f"cache_{os.path.basename(token_files[0])}")
            if last_ts: st.info(f"🕒 DB 최신화: {last_ts}")
            
        if st.button("🔄 최신 영상 업데이트 (수동)", type="primary", use_container_width=True):
            st.session_state['channels_data'] = []
            ph = {tf: st.empty() for tf in token_files}
            ready = []
            ctx = get_script_run_ctx()
            def worker(tf, sb): 
                add_script_run_ctx(ctx=ctx)
                return process_sync_channel(tf, DEFAULT_LIMIT_DATE, sb, False)
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                fs = {ex.submit(worker, tf, ph[tf]): tf for tf in token_files}
                for f in as_completed(fs):
                    r = f.result()
                    if r and 'name' in r: ready.append(r)
            
            if ready:
                st.success("업데이트 완료!")
                load_from_mongodb.clear()
                get_last_update_time.clear()
                time.sleep(1)
                st.rerun()

        st.markdown("---")
        with st.expander("⚠️ DB 초기화 및 전체 재수집"):
            if 'admin_unlocked' not in st.session_state: st.session_state['admin_unlocked'] = False
            if not st.session_state['admin_unlocked']:
                if st.text_input("2차 비밀번호", type="password") == "dima1234":
                    st.session_state['admin_unlocked'] = True; st.rerun()
            
            if st.session_state['admin_unlocked']:
                ld = st.date_input("마지노선", value=pd.to_datetime(DEFAULT_LIMIT_DATE))
                if st.button("🔥 전체 데이터 덮어쓰기", type="secondary"):
                    st.session_state['channels_data'] = []
                    ph = {tf: st.empty() for tf in token_files}
                    ready = []
                    ctx = get_script_run_ctx()
                    def deep_worker(tf, sb):
                        add_script_run_ctx(ctx=ctx)
                        return process_sync_channel(tf, ld.strftime("%Y-%m-%d"), sb, True)
                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                        fs = {ex.submit(deep_worker, tf, ph[tf]): tf for tf in token_files}
                        for f in as_completed(fs):
                            r = f.result()
                            if r and 'name' in r: ready.append(r)
                    if ready:
                        st.success("완료!")
                        load_from_mongodb.clear()
                        get_last_update_time.clear()
                        time.sleep(1)
                        st.rerun()

# [메인 로직]
if 'channels_data' not in st.session_state or not st.session_state['channels_data']:
    token_files = glob.glob("token_*.json")
    temp = []
    for tf in token_files:
        c_name = f"cache_{os.path.basename(tf)}"
        vids = load_from_mongodb(c_name)
        if vids:
            creds = get_creds_from_file(tf)
            if creds: temp.append({'creds': creds, 'name': os.path.basename(tf).replace("token_","").replace(".json",""), 'videos': vids})
    
    if temp: st.session_state['channels_data'] = temp
    else: st.info("👋 데이터 준비 중... (DB가 비어있다면 관리자 메뉴에서 업데이트하세요)")

if 'channels_data' in st.session_state and st.session_state['channels_data']:
    data = st.session_state['channels_data']
    tv = sum(len(c['videos']) for c in data)
    st.markdown(f"<div style='background:white;padding:10px;border-radius:8px;border:1px solid #eee;margin-bottom:20px'>✅ <b>채널:</b> {len(data)}개 | 📁 <b>영상:</b> {tv:,}개</div>", unsafe_allow_html=True)
    
    with st.form("anl_form"):
        st.subheader("🔍 통합 분석")
        c1,c2,c3 = st.columns([2,1,1])
        kw = c1.text_input("분석 IP", placeholder="예: 눈물의 여왕")
        vd = c2.date_input("업로드 기간", value=(datetime.today().replace(day=1), datetime.today()))
        ad = c3.date_input("조회 기간", value=(datetime.today().replace(day=1), datetime.today()))
        if st.form_submit_button("분석 시작", type="primary", use_container_width=True):
            if not kw.strip(): st.error("키워드 입력 필요")
            else:
                v_s = vd[0]; v_e = vd[1] if len(vd)>1 else vd[0]
                a_s = ad[0]; a_e = ad[1] if len(ad)>1 else ad[0]
                st.session_state['anl_kw'] = kw
                
                res = []
                bar = st.progress(0, "분석 중...")
                ctx = get_script_run_ctx()
                def w(cd, k, vs, ve, ast, aet):
                    add_script_run_ctx(ctx=ctx)
                    return process_analysis_channel(cd, k, vs, ve, ast, aet)
                
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                    fs = {ex.submit(w, c, kw, v_s, v_e, a_s, a_e): c for c in data}
                    dn = 0
                    for f in as_completed(fs):
                        dn+=1; bar.progress(dn/len(data), f"채널 {dn}/{len(data)}")
                        r = f.result()
                        if r: res.append(r)
                bar.empty()
                st.session_state['anl_res'] = res
                if not res: st.warning("결과 없음 (기간/키워드 확인)")

    if 'anl_res' in st.session_state and st.session_state['anl_res']:
        raw = st.session_state['anl_res']
        st.divider()
        st.markdown(f"### 📊 결과: <span style='color:#2980b9'>{st.session_state['anl_kw']}</span>", unsafe_allow_html=True)
        
        opts = ["전체 합산"] + sorted([d['channel_name'] for d in raw])
        sel = st.selectbox("채널 선택", opts)
        tgt = raw if sel=="전체 합산" else [d for d in raw if d['channel_name']==sel]
        
        fin_v=0; fin_l=0; fin_s=0; fin_1m=0; fin_cnt=0
        stt=defaultdict(float); trf=defaultdict(float); ctr=defaultdict(float); day=defaultdict(float); kws=defaultdict(float)
        w_avg=0; v_avg=0; top_v=[]
        
        for d in tgt:
            fin_v+=d['total_views']; fin_l+=d['total_likes']; fin_s+=d['total_shares']
            fin_1m+=d['over_1m_count']; fin_cnt+=d['video_count']
            if d['avg_view_pct']>0: w_avg+=(d['avg_view_pct']*d['total_views']); v_avg+=d['total_views']
            for k,v in d['demo_counts'].items(): stt[k]+=v
            for k,v in d['traffic_counts'].items(): trf[k]+=v
            for k,v in d['country_counts'].items(): ctr[k]+=v
            for k,v in d['daily_stats'].items(): day[k]+=v
            for k,v in d['keywords_counts'].items(): kws[k]+=v
            top_v.extend(d['top_video_stats'])
            
        avg_p = (w_avg/v_avg) if v_avg>0 else 0
        gap = fin_v - sum(day.values())
        
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("조회수", f"{int(fin_v):,}"); m2.metric("영상수", f"{fin_cnt:,}"); m3.metric("100만+", f"{fin_1m:,}")
        m4.metric("지속률", f"{avg_p:.1f}%"); m5.metric("좋아요", f"{int(fin_l):,}"); m6.metric("공유", f"{int(fin_s):,}")
        st.write("")
        
        f_d, df_d, _ = get_pyramid_chart_and_df(stt, fin_v)
        if f_d:
            c1,c2=st.columns([1.6,1])
            with c1: 
                with st.container(border=True):
                    st.markdown("##### 👥 성별/연령 분포")
                    st.plotly_chart(f_d, use_container_width=True)
            with c2: 
                with st.container(border=True):
                    st.markdown("##### 📋 상세 데이터")
                    df_d_disp = df_d.copy()
                    df_d_disp['조회수'] = df_d_disp['조회수'].apply(lambda x: f"{x:,}")
                    df_d_disp['비율'] = df_d_disp['비율'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(df_d_disp, use_container_width=True, hide_index=True, height=300)
        st.write("")
            
        f_t = get_daily_trend_chart(day, gap)
        if f_t: 
            with st.container(border=True):
                st.markdown("##### 📈 일별 조회수 추이")
                st.plotly_chart(f_t, use_container_width=True)
        st.write("")
        
        st.markdown("##### 🥇 인기 영상 Top 100")
        with st.container(border=True):
            if top_v:
                dedup = {v['id']:v for v in top_v}.values()
                top100 = sorted(dedup, key=lambda x:x['period_views'], reverse=True)[:100]
                df = pd.DataFrame(top100)
                
                df['link'] = df['id'].apply(lambda x: f"https://youtu.be/{x}")
                
                df_s = df[['title','period_views','avg_pct','period_likes','link']].copy()
                df_s.columns=['제목','조회수','지속률','좋아요','바로가기']
                
                df_s['조회수'] = df_s['조회수'].apply(lambda x: f"{x:,}")
                df_s['좋아요'] = df_s['좋아요'].apply(lambda x: f"{x:,}")
                
                st.data_editor(
                    df_s, 
                    column_config={
                        "바로가기": st.column_config.LinkColumn(display_text="Watch 🎬"), 
                        "지속률": st.column_config.NumberColumn(format="%.1f%%")
                    }, 
                    hide_index=True, 
                    use_container_width=True
                )
            else: st.caption("데이터가 없습니다.")
        st.write("")
        
        # [수정: 시각화 추가] 채널별 점유율 & 글로벌 지도 병렬 배치
        f_share = get_channel_share_chart(tgt, highlight_channel=(sel if sel!="전체 합산" else None))
        f_map = get_country_map(ctr)
        
        c_share, c_map = st.columns(2)
        with c_share:
             if f_share:
                with st.container(border=True):
                    st.markdown("##### 🏆 채널별 조회수 점유율")
                    st.plotly_chart(f_share, use_container_width=True)
        with c_map:
            if f_map:
                with st.container(border=True):
                    st.markdown("##### 🌍 글로벌 조회수 분포")
                    st.plotly_chart(f_map, use_container_width=True)
        st.write("")

        # [박스 유지] 유입경로 & 검색어
        r2_1, r2_2 = st.columns(2)
        f_tr = get_traffic_chart(trf); f_kw = get_keyword_bar_chart(kws)
        with r2_1: 
            if f_tr: 
                with st.container(border=True):
                    st.markdown("##### 🚦 유입 경로 Top 5")
                    st.plotly_chart(f_tr, use_container_width=True)
        with r2_2: 
            if f_kw: 
                with st.container(border=True):
                    st.markdown("##### 🔍 Top 10 검색어 (SEO)")
                    st.plotly_chart(f_kw, use_container_width=True)
# endregion