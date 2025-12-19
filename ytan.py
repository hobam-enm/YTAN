import streamlit as st
import os
import glob
import json
import time
import re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import google.oauth2.credentials
import googleapiclient.discovery
import google.auth.transport.requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from datetime import datetime, timedelta

# region [1. 설정 및 상수 (Config & Constants)]
# ==========================================
# 기본 페이지 설정 및 디자인, 상수 정의
# ==========================================
st.set_page_config(
    page_title="Drama YouTube Insight", 
    page_icon="📊",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# UI 디자인 CSS
custom_css = """
    <style>
        /* 1. 헤더 투명화 및 불필요 요소 숨김 (사이드바 버튼 유지) */
        header[data-testid="stHeader"] { background: transparent; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stDecoration"] {display: none;}
        
        /* 2. 메인 컨텐츠 여백 조정 */
        .block-container { padding-top: 1rem; padding-bottom: 3rem; }

        /* 3. 앱 배경 설정 */
        .stApp { background-color: #f8f9fa; }

        /* 4. 카드 및 메트릭 스타일 */
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
# endregion


# region [2. 유틸리티 함수 (Utilities)]
# ==========================================
# 텍스트 정제, 숫자 포맷팅, 날짜 변환 등 헬퍼 함수
# ==========================================
def normalize_text(text):
    if not text: return ""
    return re.sub(r'[^a-zA-Z0-9가-힣]', '', text).lower()

def format_korean_number(num):
    """숫자를 '1억 2345만 6789회' 형태로 변환"""
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

# [신규] 영상 길이 파싱 함수 (PT1H2M10S -> 분 단위 변환)
def parse_duration_to_minutes(duration_str):
    if not duration_str: return 0.0
    pattern = re.compile(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?')
    match = pattern.match(duration_str)
    if not match: return 0.0
    h, m, s = match.groups()
    total_sec = (int(h or 0) * 3600) + (int(m or 0) * 60) + (int(s or 0))
    return round(total_sec / 60, 1)
# endregion


# region [3. 시각화 함수 (Visualization)]
# ==========================================
# Plotly를 이용한 차트 생성 함수들
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
    daily_stats: Analytics API 일별 조회수
    recent_gap: Data API(실시간) 총합 - Analytics 총합
    """
    # [1] Analytics 데이터(daily_stats)가 없으면, 리얼타임 데이터가 있어도 차트 생성 X
    if not daily_stats: return None
    
    dates = sorted(daily_stats.keys())
    views = [daily_stats[d] for d in dates]
    
    fig = go.Figure()
    
    # [2] 확정된 Analytics 데이터 (실선 - 보라색)
    fig.add_trace(go.Scatter(
        x=dates, y=views, mode='lines+markers', name='확정 조회수',
        line=dict(color='#6c5ce7', width=3), marker=dict(size=6)
    ))
    
    # [3] 실시간 데이터 연결 (있을 경우만)
    if recent_gap > 0 and dates:
        last_date_str = dates[-1]
        last_val = views[-1]
        
        # 날짜 충돌 방지: 리얼타임 포인트는 무조건 마지막 Analytics 날짜보다 미래여야 함
        # 오늘 날짜를 구하되, 마지막 Analytics 날짜와 같거나 작으면 하루 뒤로 설정
        today_dt = datetime.today()
        last_anl_dt = datetime.strptime(last_date_str, "%Y-%m-%d")
        
        if today_dt.date() <= last_anl_dt.date():
            target_dt = last_anl_dt + timedelta(days=1)
        else:
            target_dt = today_dt
            
        target_date_str = target_dt.strftime("%Y-%m-%d")
        
        # 점선 그래프 추가 (마지막 확정일 ~ 타겟일)
        fig.add_trace(go.Scatter(
            x=[last_date_str, target_date_str],
            y=[last_val, recent_gap],
            mode='lines+markers',
            name='실시간(추정)',
            line=dict(color='#ff7675', width=3, dash='dot'), # 붉은 점선
            marker=dict(size=8, symbol='star')
        ))
        
        fig.add_annotation(
            x=target_date_str, y=recent_gap,
            text="Realtime (Est.)", showarrow=True, arrowhead=1,
            yshift=10, font=dict(color="#d63031", size=10)
        )

    # [4] Y축 포맷 설정 (콤마)
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=350, 
        xaxis=dict(title="날짜", tickformat="%Y-%m-%d"),
        yaxis=dict(title="조회수", tickformat=","), # #,### 포맷
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
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
# Google API 인증, 데이터 동기화, 분석 로직
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
    # ... [이전 코드와 동일] ...
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
        cache_file = f"cache_nov_{token_file}"
        cached_videos = []
        
        cached_ids = set()
        
        if not force_rescan and os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f: cached_videos = json.load(f)
            cached_ids = {v['id'] for v in cached_videos}
            status_box.info(f"🔄 [{ch_name}] 확인 중...")
        else: status_box.info(f"⏳ [{ch_name}] 스캔 시작")
        
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
        final_list = new_videos + cached_videos if not force_rescan else new_videos
        if new_videos or force_rescan:
            with open(cache_file, 'w', encoding='utf-8') as f: json.dump(final_list, f, ensure_ascii=False, indent=2)
            status_box.success(f"✅ **[{ch_name}]** 완료 (+{len(new_videos)})")
        else: status_box.success(f"✅ **[{ch_name}]** 최신")
        return {'creds': creds, 'name': ch_name, 'videos': final_list}
    except Exception as e:
        status_box.error(f"❌ 에러: {str(e)}")
        return {'error': str(e)}

def process_analysis_channel(channel_data, keyword, vid_start, vid_end, anl_start, anl_end):
    creds = channel_data['creds']; videos = channel_data['videos']
    norm_keyword = normalize_text(keyword)
    target_ids = []
    id_map = {} 
    video_date_map = {} # 영상별 업로드 날짜 저장용
    
    # 1. 대상 영상 필터링
    for v in videos:
        t_match = norm_keyword in normalize_text(v['title'])
        d_match = norm_keyword in normalize_text(v.get('description', ''))
        if not (t_match or d_match): continue
        
        v_dt_kst = parse_utc_to_kst_date(v['date']) # KST Date 객체
        if v_dt_kst and (vid_start <= v_dt_kst <= vid_end): 
            target_ids.append(v['id'])
            id_map[v['id']] = v['title']
            video_date_map[v['id']] = v['date'] # UTC 문자열 그대로 저장 ("2025-01-01T00:00:00Z")
            
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
    
    # 48시간 기준점 설정 (UTC 기준)
    now_utc = datetime.utcnow()
    threshold_dt = now_utc - timedelta(hours=48)
    
    batch_size = 50 
    for i in range(0, len(target_ids), batch_size):
        batch_ids = target_ids[i : i + batch_size]
        vid_str = ",".join(batch_ids)
        
        # [A] Analytics 데이터 수집
        # 배치 전체 합산 데이터 (공유, 인구통계 등 개별 매핑 어려운 것들)
        anl_views_map = {} # 영상별 Analytics 조회수 저장
        anl_likes_map = {} # 영상별 Analytics 좋아요 저장
        anl_retention_map = {} # 영상별 지속률 저장
        
        try:
            # 1. 전체 합산용 (공유 등)
            r_b = yt_anl.reports().query(ids='channel==MINE', startDate=anl_start, endDate=anl_end, metrics='shares', filters=f'video=={vid_str}').execute()
            if 'rows' in r_b and r_b['rows']:
                total_shares += r_b['rows'][0][0] # 공유는 Data API에 없으므로 Analytics 전적으로 신뢰

            # 2. 영상별 상세 데이터 (조회수, 좋아요, 지속률) -> 48시간 로직 적용을 위해 'dimensions=video'로 쪼개서 받음
            r_v = yt_anl.reports().query(ids='channel==MINE', startDate=anl_start, endDate=anl_end, metrics='views,likes,averageViewPercentage', dimensions='video', filters=f'video=={vid_str}').execute()
            if 'rows' in r_v and r_v['rows']:
                for r in r_v['rows']:
                    # r[0]: video_id, r[1]: views, r[2]: likes, r[3]: avg_pct
                    anl_views_map[r[0]] = r[1]
                    anl_likes_map[r[0]] = r[2]
                    anl_retention_map[r[0]] = r[3]

            # 3. 기타 차트용 데이터 (이건 합산치 그대로 사용)
            # 인구통계
            r_d = yt_anl.reports().query(ids='channel==MINE', startDate=anl_start, endDate=anl_end, metrics='viewerPercentage', dimensions='ageGroup,gender', filters=f'video=={vid_str}').execute()
            # 배치 전체 뷰수 구하기 (가중치 계산용)
            batch_total_view_anl = sum(anl_views_map.values())
            if 'rows' in r_d and r_d['rows']:
                for r in r_d['rows']: demo[f"{r[0]}_{r[1]}"] += batch_total_view_anl * (r[2] / 100)

            # 유입경로
            r_t = yt_anl.reports().query(ids='channel==MINE', startDate=anl_start, endDate=anl_end, metrics='views', dimensions='insightTrafficSourceType', filters=f'video=={vid_str}').execute()
            if 'rows' in r_t and r_t['rows']:
                for r in r_t['rows']: traffic[r[0]] += r[1]
            
            # 검색어 (Top 15)
            try:
                r_k = yt_anl.reports().query(ids='channel==MINE', startDate=anl_start, endDate=anl_end, metrics='views', dimensions='insightTrafficSourceDetail', filters=f'video=={vid_str};insightTrafficSourceType==YT_SEARCH', maxResults=15, sort='-views').execute()
                if 'rows' in r_k and r_k['rows']:
                    for r in r_k['rows']:
                        if r[0] != 'GOOGLE_SEARCH': keywords_count[r[0]] += r[1]
            except: pass

            # 국가
            r_c = yt_anl.reports().query(ids='channel==MINE', startDate=anl_start, endDate=anl_end, metrics='views', dimensions='country', filters=f'video=={vid_str}', maxResults=50).execute()
            if 'rows' in r_c and r_c['rows']:
                for r in r_c['rows']: country[r[0]] += r[1]

            # 일별 추이
            r_day = yt_anl.reports().query(ids='channel==MINE', startDate=anl_start, endDate=anl_end, metrics='views', dimensions='day', filters=f'video=={vid_str}', sort='day').execute()
            if 'rows' in r_day and r_day['rows']:
                for r in r_day['rows']: daily[r[0]] += r[1]
        
        except: pass

        # [B] Data API (실시간) 데이터 수집 및 [C] 하이브리드 합산
        try:
            rt_res = youtube.videos().list(part='statistics,contentDetails', id=vid_str).execute()
            
            rt_stats_map = {}
            rt_content_map = {}
            if 'items' in rt_res:
                for item in rt_res['items']:
                    rt_stats_map[item['id']] = item['statistics']
                    rt_content_map[item['id']] = item['contentDetails']

            # === [핵심 로직] 영상별 날짜 비교하여 합산 ===
            for vid_id in batch_ids:
                # 1. 영상 날짜 확인
                v_date_str = video_date_map.get(vid_id)
                if not v_date_str: continue
                
                v_upload_dt = datetime.strptime(v_date_str, "%Y-%m-%dT%H:%M:%SZ")
                is_recent = v_upload_dt > threshold_dt # 48시간 이내 업로드 여부

                # 2. 데이터 가져오기
                # Analytics 데이터
                a_v = anl_views_map.get(vid_id, 0)
                a_l = anl_likes_map.get(vid_id, 0)
                a_pct = anl_retention_map.get(vid_id, 0)
                
                # Data API 데이터 (실시간)
                stats = rt_stats_map.get(vid_id, {})
                rt_v = int(stats.get('viewCount', 0))
                rt_l = int(stats.get('likeCount', 0))
                
                # 3. 결정 로직 (User Request)
                final_v = 0
                final_l = 0
                
                if is_recent:
                    # [Case 1] 최신 영상 (<48h) -> Data API (실시간 누적값) 채택
                    final_v = rt_v
                    final_l = rt_l
                else:
                    # [Case 2] 오래된 영상 (>48h) -> Analytics API (기간 내 데이터) 채택
                    final_v = a_v
                    final_l = a_l
                
                # 4. 총계 합산
                total_views += final_v
                total_likes += final_l
                
                # 가중 평균 (지속률은 Analytics에만 있음. 최신 영상은 지속률 0일 확률 높음)
                if final_v > 0 and a_pct > 0:
                    w_avg_sum += (final_v * a_pct)
                    v_for_avg += final_v
                
                # 100만 카운트 (이건 항상 실시간 기준이 정확함 - 명예의 전당 느낌)
                if rt_v >= 1000000: over_1m_count += 1

                # 5. Top 리스트용 데이터 (리스트에는 항상 '현재 상태'를 보여주는게 좋음 -> Data API 사용)
                # 단, '기간 내 조회수'를 보여주고 싶다면 final_v를 써야 함.
                # 보통 리스트는 "이 영상의 현재 스펙"을 보는 용도이므로 rt_v(실시간) 유지
                if rt_v > 0:
                    top_video_stats.append({
                        'id': vid_id,
                        'title': id_map.get(vid_id, 'Unknown'),
                        'views': rt_v,       # 리스트 표시용: 실시간 조회수
                        'likes': rt_l,       # 리스트 표시용: 실시간 좋아요
                        'period_views': final_v, # (옵션) 기간 내 조회수
                        'avg_pct': a_pct if a_pct > 0 else None,
                        'duration_min': parse_duration_to_minutes(rt_content_map.get(vid_id, {}).get('duration'))
                    })

        except Exception as e:
            print(f"Error processing batch: {e}")
            pass

        time.sleep(0.05)

    if not top_video_stats and total_views == 0: return None
    
    top_video_stats.sort(key=lambda x: x['views'], reverse=True)

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
# 사이드바, 메인 대시보드 UI, 실행 컨트롤
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
        cf = f"cache_nov_{tf}"
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
            vs_str, ve_str = v_start, v_end
            as_str = a_start.strftime("%Y-%m-%d"); ae_str = a_end.strftime("%Y-%m-%d")
            
            prog_bar = st.progress(0, text="데이터 분석 중...")
            
            # 여기서 계산은 일단 채널별로 다 수행해서 list에 담음
            ch_details_results = []
            
            ctx = get_script_run_ctx()
            def worker(cd, kw, vs, ve, ast, aet):
                add_script_run_ctx(ctx=ctx)
                return process_analysis_channel(cd, kw, vs, ve, ast, aet)

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = {ex.submit(worker, ch, keyword, vs_str, ve_str, as_str, ae_str): ch for ch in data}
                done = 0
                for f in as_completed(futures):
                    done += 1
                    prog_bar.progress(done/len(data), text=f"채널 분석 중... ({done}/{len(data)})")
                    res = f.result()
                    if res: ch_details_results.append(res)

            prog_bar.empty()
            
            # 원본 데이터 세션 저장
            st.session_state['analysis_raw_results'] = ch_details_results
            st.session_state['analysis_keyword'] = keyword

    # --- 결과 렌더링 세션 (여기서 집계 및 선택 필터링 수행) ---
    if 'analysis_raw_results' in st.session_state and st.session_state['analysis_raw_results']:
        raw_data = st.session_state['analysis_raw_results']
        current_kw = st.session_state['analysis_keyword']
        
        st.divider()
        st.markdown(f"### 📊 분석 리포트: <span style='color:#2980b9;'>{current_kw}</span>", unsafe_allow_html=True)
        
        # [2. 수정] 채널 선택기 (작게 상단 배치)
        ch_names = sorted([d['channel_name'] for d in raw_data])
        sel_options = ["전체 채널 합산"] + ch_names
        
        c_sel_col, _ = st.columns([1, 2])
        with c_sel_col:
            selected_ch = st.selectbox("분석 대상 채널 선택", sel_options, label_visibility="collapsed")
        
        # --- 선택에 따른 실시간 집계 (Aggregation) ---
        if selected_ch == "전체 채널 합산":
            target_data = raw_data
        else:
            target_data = [d for d in raw_data if d['channel_name'] == selected_ch]
            
        # 집계 변수 초기화
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
            
            # 평균 지속률 가중치용
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
        
        # [3. 수정] 일별 추이용 Gap 계산 (Hybrid Total - Analytics Sum)
        anl_total_daily = sum(final_daily.values())
        recent_gap = final_views - anl_total_daily
        
        # --- UI 그리기 ---
        safe_anl_date = datetime.now() - timedelta(days=3)
        safe_str = safe_anl_date.strftime("%Y-%m-%d")
        
        if final_views > 0 or len(final_top_videos) > 0:
            st.caption(f"ℹ️ **데이터 기준**: 인구통계/경로 등은 **~{safe_str}** 확정치, **총 조회수/좋아요 및 리스트**는 **실시간(Realtime)** 데이터입니다.")

            # [섹션 0] 핵심 지표
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("총 조회수", f"{int(final_views):,}")
            m2.metric("분석 영상", f"{final_vid_count:,}개")
            m3.metric("100만+ 영상", f"{final_over_1m:,}개")
            m4.metric("평균 시청지속률", f"{final_avg_pct:.1f}%")
            m5.metric("총 좋아요", f"{int(final_likes):,}")
            m6.metric("총 공유", f"{int(final_shares):,}")
            st.write("")

            # [섹션 1] 성별/연령 (있을 때만)
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

            # [1. 수정] 일별 조회수 추이 (전체 1행)
            # Analytics 데이터가 없어도 Gap(Data API)이 있으면 차트를 그리기 위해 조건 완화
            fig_trend = get_daily_trend_chart(final_daily, recent_gap)
            if fig_trend:
                st.markdown("##### 📈 일별 조회수 추이 (Analytics + Realtime Gap)")
                with st.container(border=True):
                    st.plotly_chart(fig_trend, use_container_width=True)
                st.write("")

            # [1. 수정] 인기 영상 리스트 (전체 1행)
            st.markdown("##### 🥇 인기 영상 TOP 100 (실시간 기준)")
            with st.container(border=True):
                if final_top_videos:
                    unique_vids_map = {v['id']: v for v in final_top_videos}
                    deduped_vids = list(unique_vids_map.values())
                    top_vids = sorted(deduped_vids, key=lambda x: x['views'], reverse=True)[:100]
                    
                    df_top = pd.DataFrame(top_vids)
                    df_top['link'] = df_top['id'].apply(lambda x: f"https://youtu.be/{x}")
                    df_show = df_top[['title', 'views', 'avg_pct', 'likes', 'link']].copy()
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

            # [섹션 2] 유입/검색어
            fig_traffic = get_traffic_chart(final_traffic)
            fig_keywords = get_keyword_bar_chart(final_keywords)
            
            # [4. 수정] 둘 중 하나라도 있어야 행 생성
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
            
            # [섹션 3] 점유율/반응
            # 점유율은 '전체' 보기일 때만 의미가 있으므로, '전체 채널 합산'일 때만 표시하거나,
            # 특정 채널 선택 시 전체 대비 점유율을 보여주는 로직 필요.
            # 여기서는 편의상 점유율 차트는 전체 보기 모드에서만, 반응 차트는 항상 표시
            
            show_share = (selected_ch == "전체 채널 합산") and (len(raw_data) > 1)
            fig_share = get_channel_share_chart(raw_data, highlight_channel=None) if show_share else None
            # 특정 채널 선택시엔 그 채널을 하이라이트해서 보여줄 수도 있음
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

            # [섹션 4] 지도
            fig_map = get_country_map(final_country)
            if fig_map:
                st.markdown("##### 🌍 글로벌 조회수 분포")
                with st.container(border=True):
                    st.plotly_chart(fig_map, use_container_width=True)
                st.write("")

            # [섹션 5] 효율성 (Scatter)
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
