

# Put this near the top of your script (after imports)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
from streamlit.components.v1 import html
import base64
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os
os.environ["MAPBOX_API_KEY"] = "pk.eyJ1Ijoic3RyZWFtbGl0IiwiYSI6ImNqd2d4enVmNTAwNGY0M3A1cGhzZzI4emgifQ.Cf1bOmWqkJpXyYh0SgYz_g"

#remove streamlit header and footer


# --- Hide all Streamlit UI and Cloud branding ---
hide_streamlit_branding = """
    <style>
    /* Hide Streamlit main header and footer */
    #MainMenu {visibility: hidden !important;}
    header {visibility: hidden !important;}
    footer {visibility: hidden !important;}

    /* Hide possible Streamlit Cloud floating buttons */
    [data-testid="stStatusWidget"] {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    [data-testid="stToolbar"] {display: none !important;}
    [data-testid="stDecorationContainer"] {display: none !important;}
    .stAppDeployButton {display: none !important;}
    button[data-testid="manage-app-button"] {display: none !important;}
    div[class*="_link_"] {display: none !important;}
    div[title="Manage app"] {display: none !important;}
    div[data-testid="stActionButton"] {display: none !important;}
    
    /* 👇 Key trick: globally hide any Streamlit Cloud bottom-right floating button */
    [class*="st-emotion-cache"] button[title*="Manage"], 
    [class*="st-emotion-cache"] button[title*="View"],
    [class*="st-emotion-cache"] a[href*="streamlit.app"],
    [class*="st-emotion-cache"] svg[xmlns*="http"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }

    /* Hide Streamlit Cloud overlay container completely */
    div[style*="position: fixed"][style*="right: 0px"][style*="bottom: 0px"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
    }
    </style>

    <script>
    // In case Streamlit Cloud injects after render — try removing again
    const hideCloudButton = () => {
        const elems = document.querySelectorAll('button[title*="Manage"], button[title*="View"], a[href*="streamlit.app"], div[class*="_link_"]');
        elems.forEach(el => el.style.display = "none");
    };
    setInterval(hideCloudButton, 1500);
    </script>
"""
st.markdown(hide_streamlit_branding, unsafe_allow_html=True)


#remove top padding
st.set_page_config(layout="wide")
st.markdown("""
    <style>
        div.block-container { padding-top: 2rem !important; }
        section[data-testid="stTabs"] { margin-top: 30px !important; }
    </style>
""", unsafe_allow_html=True)


# ------------------- GLOBAL STYLES (sticky tabs, tight spacing) -------------------
st.markdown(
    """
    <style>
    /* reduce top padding */
    .block-container { padding-top: 0.75rem; }

    /* make tab bar sticky at top */
    section[data-testid="stTabs"] {
        position: -webkit-sticky;
        position: sticky;
        top: 56px;  /* below Streamlit header */
        z-index: 9999;
        background: rgba(255,255,255,0.98);
        padding-top: 6px;
        padding-bottom: 6px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }

    /* style subtabs similarly if present */
    div[data-testid="stTabbedContent"] > div:first-child {
        margin-top: 0;
    }

    /* smaller vertical gaps overall for dashboard feel */
    .streamlit-expanderHeader, .stMarkdown { margin-top: 0.25rem; margin-bottom: 0.25rem; }

    /* center align numeric cells in pandas/st.dataframe (styler-based fallback) */
    .dataframe td, .dataframe th { text-align: left; } /* default left; we'll use pandas Styler for specific centering */
    </style>
    """,
    unsafe_allow_html=True,
)


st.set_page_config(page_title="Highway Dashboard V15 - ui", layout="wide")

#global variables
# --- Step 5: Rename for uniform column naming ---
rename_map = {
    "Created At": "Created_At",
    "GPS लोकेशन": "GPS_Location",
    "फोटो खींचे": "Photo",
    "चेकिंग बिंदु नाम": "Checkpoint",
    "रूट चयन करे": "Route",
    "ड्यूटी SI का नियुक्ति थाना": "Agent",
    "सब ठीक है?": "Status",
    "समस्या का चयन करे": "Issue",
    "अन्य समस्या का विवरण दे": "Issue_Details",
}

# ======================================================
# 🔹 Columns we want to display throughout the dashboard
# ======================================================
display_columns = [
    "Date", "Time", "Agent", "Status", "Issue", "Police_Station",
    "Circle", "Block", "ULD", "SDM", "Route",
    "Latitude", "Longitude"
]


@st.cache_data(ttl=13000)
def load_data():
    
    # --- Load data from Google Sheet ---
    sheet_url = "https://docs.google.com/spreadsheets/d/1Dd0tXuDNoYInzOYZrnrDLDnJxbNAwMaNkfJr4ZQ9IEo/export?format=csv&gid=845685650"
    try:
        df = pd.read_csv(sheet_url)
    except Exception as e:
        st.error(f"❌ Failed to load data from Google Sheet: {e}")
        return pd.DataFrame()

    df.columns = df.columns.map(str).str.strip()

    # --- Step 1: Flatten duplicate columns safely ---
    if df.columns.duplicated().any():
        for col in df.columns[df.columns.duplicated()].unique():
            dup_cols = df.loc[:, df.columns == col]
            df[col] = dup_cols.bfill(axis=1).iloc[:, 0]
            df.drop(columns=dup_cols.columns[1:], inplace=True)

    # --- Step 2: Merge both 'घटना स्थल का थाना चयन करें' columns ---
    same_cols = [c for c in df.columns if "घटना स्थल का थाना चयन करें" in c]
    if len(same_cols) >= 2:
        df["Incident_Station"] = df[same_cols].bfill(axis=1).iloc[:, 0]
    elif len(same_cols) == 1:
        df["Incident_Station"] = df[same_cols[0]]
    else:
        df["Incident_Station"] = None

    # --- Step 3: Function to safely merge any pair ---
    def combine_cols(df, col_a, col_b, new_name):
        if col_a in df.columns and col_b in df.columns:
            df[new_name] = df[col_a].combine_first(df[col_b])
        elif col_a in df.columns:
            df[new_name] = df[col_a]
        elif col_b in df.columns:
            df[new_name] = df[col_b]
        else:
            df[new_name] = None

    # --- Step 4: Merge paired route columns ---
    combine_cols(df, "घटना का थाना", "PS_name", "Police_Station")
    combine_cols(df, "पुलिस क्षेत्राधिकारी", "circle", "Circle")
    combine_cols(df, "ब्लॉक का नाम", "Block_name", "Block")
    combine_cols(df, "नगर स्थानीय निकाय का नाम", "urban_local_body", "ULD")
    combine_cols(df, "SDM", "SDM", "SDM")

    
    df.rename(columns=rename_map, inplace=True)

    # --- Step 6: Parse datetime robustly ---
    if "Created_At" in df.columns:
        def safe_parse_datetime(x):
            try:
                return pd.to_datetime(x, errors="coerce", utc=False)
            except Exception:
                return pd.NaT
        df["Created_At"] = df["Created_At"].apply(safe_parse_datetime)
        df["Date"] = df["Created_At"].dt.date
        df["Hour"] = df["Created_At"].dt.hour

    # --- Step 7: Extract Latitude / Longitude ---
    if "GPS_Location" in df.columns:
        def extract_lat_lon(gps_value):
            if pd.isna(gps_value):
                return None, None
            gps_value = (
                str(gps_value)
                .replace("(", "")
                .replace(")", "")
                .replace("[", "")
                .replace("]", "")
                .replace("Lat:", "")
                .replace("Lon:", "")
                .replace("lat:", "")
                .replace("lon:", "")
            )
            parts = gps_value.replace(",", " ").split()
            nums = []
            for p in parts:
                try:
                    nums.append(float(p))
                except ValueError:
                    continue
            if len(nums) >= 2:
                return nums[0], nums[1]
            return None, None

        df[["Latitude", "Longitude"]] = df["GPS_Location"].apply(
            lambda x: pd.Series(extract_lat_lon(x))
        )
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    else:
        df["Latitude"], df["Longitude"] = None, None

    # --- Step 8: Ensure 'Incident_Station' is 1D and text ---
    if "Incident_Station" in df.columns:
        if isinstance(df["Incident_Station"], pd.DataFrame):
            df["Incident_Station"] = df["Incident_Station"].iloc[:, 0]
        df["Incident_Station"] = df["Incident_Station"].apply(
            lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) > 0 else x
        )
        df["Incident_Station"] = df["Incident_Station"].astype(str).fillna("Unknown")

    # --- Step 9: Normalize categorical text columns ---
    for col in ["Status", "Issue"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("Unknown")

    # --- Step 10: Hardcode Issues when Status == "नहीं" ---
    mask = df["Status"].str.strip() == "नहीं"
    if mask.any():
        df.loc[mask, "Issue"] = df.loc[mask, "Issue"].apply(
            lambda x: x
            if x in ["सड़क पर पार्किंग", "सड़क दुर्घटना", "छुट्टा पशु", "अन्य"]
            else "अन्य"
        )

    return df

# Load data
df = load_data()
    

# ------------------- HELPERS -------------------
#allign text centrally in the data tables
def df_center_numeric(df, numeric_only=True):
    """Return a Pandas Styler where numeric columns are center aligned for display in st.dataframe."""
    styled = df.copy()
    if numeric_only:
        num_cols = styled.select_dtypes(include=[np.number]).columns.tolist()
    else:
        num_cols = styled.columns.tolist()
    sty = styled.style.set_properties(**{col: 'text-align: center;' for col in num_cols}, subset=num_cols)
    return sty



@st.cache_data(show_spinner=False)
def compute_issue_clusters(df, grid_m=500, min_points=5, status_filter="नहीं"):
    """Grid-based clusters; returns cluster_summary and filtered df used to compute clusters."""
    df_filtered = df.copy()
    if status_filter is not None:
        df_filtered = df_filtered[df_filtered["Status"].astype(str).str.strip() == status_filter]

    df_filtered = df_filtered.dropna(subset=["Latitude", "Longitude"]).copy()
    if df_filtered.empty:
        return pd.DataFrame(), df_filtered

    deg = grid_m / 111_000.0
    # create grid-based cluster id
    df_filtered["cluster_id"] = ((df_filtered["Latitude"] / deg).astype(int).astype(str)
                                 + "_" + (df_filtered["Longitude"] / deg).astype(int).astype(str))

    # summary per cluster and Issue
    cluster_summary = (df_filtered.groupby(["cluster_id", "Issue"], as_index=False)
                       .agg(Latitude=("Latitude", "mean"),
                            Longitude=("Longitude", "mean"),
                            Count=("Issue", "count")))
    cluster_summary = cluster_summary[cluster_summary["Count"] >= min_points].reset_index(drop=True)
    return cluster_summary, df_filtered

def folium_map_from_clusters(cluster_summary, df_points, show_photos=True):
    """Build a folium map with cluster markers; popup shows small table + photo if available."""
    mid_lat = cluster_summary["Latitude"].mean()
    mid_lon = cluster_summary["Longitude"].mean()
    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=10, tiles="OpenStreetMap")
    cluster = MarkerCluster().add_to(m)

    for _, row in cluster_summary.iterrows():
        cid = row["cluster_id"]
        count = int(row["Count"])
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])
        # gather sample rows from df_points belonging to cluster (show upto 5 rows)
        points = df_points[df_points["cluster_id"] == cid].head(5)
        # build popup html
        html_parts = [f"<b>Cluster:</b> {cid}<br><b>Reports:</b> {count}<hr>"]
        for i, r in points.iterrows():
            agent = r.get("Agent", "")
            station = r.get("Incident_Station", "")
            issue = r.get("Issue", "")
            photo = r.get("Photo", "")
            html_parts.append(f"<b>{agent}</b> — {issue} ({station})<br>")
            if show_photos and isinstance(photo, str) and photo.strip():
                # if photo looks like data URI or URL show; else skip
                if photo.startswith("http") or photo.startswith("data:"):
                    html_parts.append(f"<img src='{photo}' width='160' /><br>")
        popup = folium.Popup(folium.IFrame("<br>".join(html_parts), width=260, height=200), max_width=300)
        folium.CircleMarker(location=[lat, lon],
                            radius=4 + np.log1p(count) * 3,
                            color='crimson',
                            fill=True,
                            fill_opacity=0.8,
                            popup=popup).add_to(cluster)
    return m

# ------------------- SAMPLE USAGE: Tabs -------------------
tabs = st.tabs(["Daily Highway Patrol", "Issues", "Map", "Overview"])
#CSS tabs
# ------------------- Enhanced Tabs UI -------------------
# Make tabs full-width and sticky when scrolling
st.markdown(
    """
    <style>
    /* === Make tab bar full width === */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: space-between;
        background-color: #f8f9fa;
        border-bottom: 2px solid #dcdcdc;
        width: 100%;
        padding: 0.3rem 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* === Style individual tabs === */
    .stTabs [data-baseweb="tab"] {
        flex-grow: 1;
        text-align: center;
        font-weight: 600;
        color: #2c3e50;
        border-radius: 6px;
        padding: 12px 0px;
        margin: 0 4px;
        transition: all 0.2s ease-in-out;
    }

    /* === Hover effect === */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e8f0fe;
        color: #1a73e8;
    }

    /* === Active tab styling === */
    .stTabs [aria-selected="true"] {
        background-color: #1a73e8 !important;
        color: white !important;
        font-weight: 700 !important;
        border-bottom: 3px solid #155ab6;
    }

    /* === Sticky tab bar === */
    .stTabs [data-baseweb="tab-list"] {
        position: sticky;
        top: 0;
        z-index: 999;
    }

    /* === Page spacing to avoid overlap === */
    .block-container {
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------- TAB: Overview -------------------
# ====================================================
# 📍 Overview Tab — Full Patrol Map with Cluster Controls
# ====================================================

with tabs[0]:
    st.header("Daily Highway Patrol Data and Clusters")

    
    # 2️⃣ Filter for recent patrol window (8 PM yesterday → now)
    # Current IST time
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    # Remove timezone info (make tz-naive)
    now = now_ist.replace(tzinfo=None)
    yesterday_8pm = (now - timedelta(days=1)).replace(hour=20, minute=0, second=0, microsecond=0)

    # Ensure dataframe is tz-naive
    df["Created_At"] = pd.to_datetime(df["Created_At"], errors="coerce")

    # ✅ Compare safely
    df_patrol = df[
        (pd.to_datetime(df["Created_At"]) >= yesterday_8pm)
        & (pd.to_datetime(df["Created_At"]) <= now)
    ].copy()

    st.write(f"📅 Showing records from **{yesterday_8pm.strftime('%d %b %Y %I:%M %p')}** → **{now.strftime('%d %b %Y %I:%M %p')}**")
    st.write(f"Total patrol records: {len(df_patrol)}")

    # 3️⃣ Split data by Status
    df_no = df_patrol[df_patrol["Status"].astype(str).str.strip() == "नहीं"].copy()
    df_yes = df_patrol[df_patrol["Status"].astype(str).str.strip() == "हाँ"].copy()

    
    # 5️⃣ Color coding by Issue & Status
    issue_colors = {
        "हाँ": "blue",
        "नहीं": "red",
    }

    
    # 6️⃣ Combine clustered + unclustered data for plotting
    all_points = pd.concat([
        df_no.assign(Status="नहीं"),
        df_yes.assign(Status="हाँ")
    ], ignore_index=True)

    # Map center
    center_lat = all_points["Latitude"].mean()
    center_lon = all_points["Longitude"].mean()

    # 7️⃣ Build the Plotly map
    fig = px.scatter_mapbox(
        all_points,
        lat="Latitude",
        lon="Longitude",
        color="Status",
        hover_name="Agent",
        hover_data={
            "Issue": True,
            "Incident_Station": True,
            "BDO": True,
            "Circle": True,
            "Status": True,
            "Created_At": True
        },
        color_discrete_map=issue_colors,
        size_max=12,
        zoom=9,
        height=650,
    )
    # ✅ Increase point marker size
    fig.update_traces(
        marker=dict(
            sizeref=0.003,        # Smaller value = bigger circles
            sizemode="area",      # Scale by area (not radius)
            sizemin=10,           # Minimum size in pixels
            opacity=0.65,         # Slight transparency for overlaps
            symbol="circle",
            size = 20       # Ensures circular markers
        )
    )

    
    # Map styling
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r":0,"t":0,"l":0,"b":0},
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=14)
        ),
    )

    # 8️⃣ Render map in bordered frame
    st.markdown("---")
    border_html = f"""
    <div style="border:4px solid #007ACC;border-radius:12px;overflow:hidden;margin-bottom:15px;">
        {fig.to_html(include_plotlyjs='cdn', full_html=False, config={'displayModeBar': False})}
    </div>
    """
    html(border_html, height=720)


    
    st.markdown("---")

    # ============================================================
    # ⏰ Hourly Submission Chart (Stacked: हाँ as base + Issues)
    # ============================================================

    st.subheader("⏰ Hourly Submissions – Combined Patrol Overview")

    # --- Radio selector for data mode ---
    # Create two columns: left for selector, right for chart
    left_col, right_col = st.columns([1, 6])  # adjust ratio as needed

    with left_col:
        
        status_mode_hour = st.radio(
            "Choose Mode",
            options=[
                "total submissions",
                "total submissions (stacked)",
                "total issues reported",
                "total issues reported (stacked)",
            ],
            index=1,
            horizontal=False,  # 🔹 vertical layout
            key="hourly_status_mode_patrolling_stacked_total",
        )
        
    with right_col:
    
        # --- Prepare base data ---
        df_patrol["Created_At"] = pd.to_datetime(df_patrol["Created_At"], errors="coerce")
        df_patrol["Hour"] = df_patrol["Created_At"].dt.hour

        # Filter subsets
        df_no = df_patrol[df_patrol["Status"].astype(str).str.strip() == "नहीं"].copy()
        df_yes = df_patrol[df_patrol["Status"].astype(str).str.strip() == "हाँ"].copy()


        # Safety check
        if "Issue" not in df_no.columns:
            st.error("❌ 'Issue' column missing in data.")
            st.stop()

        # --- Mode 1️⃣: “नहीं only” (still split by Issue) ---
        if status_mode_hour == "total issues reported (stacked)":
            hourly_issue = (
                df_no.groupby(["Hour", "Issue"])
                .size()
                .reset_index(name="Count")
                .sort_values(["Hour", "Count"], ascending=[True, False])
            )
            order = ["छुट्टा पशु", "सड़क पर पार्किंग", "दुर्घटना", "अन्य"]
            
            fig_hour = px.bar(
                hourly_issue,
                x="Hour",
                y="Count",
                color="Issue",
                title="Hourly Reports (Status='नहीं' by Issue)",
                barmode="stack",  # 🔹 stacked bars (within नहीं)
                category_orders={"Issue": order},   # <<< enforce order
                color_discrete_map={
                    "छुट्टा पशु": "red",
                    "सड़क पर पार्किंग": "green",
                    "अन्य": "orange",
                    "दुर्घटना": "yellow",
                },
            )

        elif status_mode_hour == "total issues reported":
            plot_df = df_no
            plot_df["Created_At"] = pd.to_datetime(plot_df["Created_At"])
            plot_df["Hour"] = plot_df["Created_At"].dt.hour
            hourly = (plot_df.groupby("Hour").size().reset_index(name="Count"))
            fig_hour = px.bar(hourly, x="Hour", y="Count", title="Reports by Hour new")
            
        elif status_mode_hour == "total submissions":
            plot_df = df_patrol

            plot_df["Created_At"] = pd.to_datetime(plot_df["Created_At"])
            plot_df["Hour"] = plot_df["Created_At"].dt.hour
            hourly = (plot_df.groupby("Hour").size().reset_index(name="Count"))
            fig_hour = px.bar(hourly, x="Hour", y="Count", title="Reports by Hour")
            
        # --- Mode 2️⃣: “Both statuses” (हाँ as base, Issues stacked on top) ---
        else:
            # Add a unified label field
            df_no["Status_Label"] = df_no["Issue"].astype(str)
            df_yes["Status_Label"] = "हाँ"

            df_combined = pd.concat([df_yes, df_no], ignore_index=True)

            # Group by hour + label
            hourly_combined = (
                df_combined.groupby(["Hour", "Status_Label"])
                .size()
                .reset_index(name="Count")
                .sort_values(["Hour", "Count"], ascending=[True, False])
            )
            order = ["हाँ", "छुट्टा पशु", "सड़क पर पार्किंग", "दुर्घटना", "अन्य"]
            fig_hour = px.bar(
                hourly_combined,
                x="Hour",
                y="Count",
                color="Status_Label",
                title="Hourly Patrol Submissions (हाँ base + Issues stacked)",
                barmode="stack",  # 🔹 stack हाँ at bottom, issues on top
                category_orders={"Status_Label": order},   # <<< enforce order
                color_discrete_map={
                    "हाँ": "blue",
                    "छुट्टा पशु": "red",
                    "सड़क पर पार्किंग": "green",
                    "दुर्घटना": "yellow",
                    "अन्य" : "orange"
                },
            )

        # --- Common layout styling ---
        fig_hour.update_layout(
            xaxis_title="Hour of the Day",
            yaxis_title="Total Patrol Submissions",
            bargap=0.15,
            hovermode="x unified",
            height=450,
            margin=dict(t=50, b=40, l=40, r=40),
            legend=dict(
                title="Status / Issue Type",
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
            ),
        )

        # ✅ NEW: Use `config` dict ONLY — no deprecated kwargs
        plotly_config = {
            "scrollZoom": True,
            "displayModeBar": True,  # optional; set False to hide toolbar
            "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
            "toImageButtonOptions": {"format": "png", "filename": "hourly_chart"},
            "showTips": False,
        }

        # ✅ Clean rendering — no deprecated args, no warnings
        st.plotly_chart(fig_hour, use_container_width=True, config=plotly_config)



    
    # 9️⃣ Full patrol data table (only selected columns)
    st.markdown("---")
    
    st.markdown("### 📋 Patrol Data (Relevant Columns)")

    if "display_columns" in globals():
        final_cols = [c for c in display_columns if c in df_patrol.columns]
        df_display = df_patrol[final_cols].copy()
    else:
        st.warning("⚠️ display_columns not defined globally, showing all columns.")
        df_display = df_patrol

    
    # Show centered, formatted table
    #st.dataframe(df_center_numeric(df_display), use_container_width=True)
    
    
    st.dataframe(
        df_center_numeric(df_display),
        width='content',
        #height=400,
    )


    # Download filtered data
    csv = df_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Patrol Data (Selected Columns)",
        csv,
        "patrol_data_filtered.csv",
        "text/csv"
    )



# ------------------- TAB: Issues (Tab 1 subtabs) -------------------
with tabs[1]:
    sub_tabs = st.tabs(["ड्यूटी थाना -wise", "घटना का थाना wise", "BDO-wise",
                        "नगर निकाय wise", "SDM-wise", "पुलिस Circle-wise",
                        "Daily Issue-wise", "Issue & Hour of Reporting"])
    # Example for sub_tab 6 (Daily Issue-wise) and 7 (Issue & Hour) with requested filters:
    # Sub-tab 6: date-wise plot with option to choose status filter ('नहीं' only or both)

    #def plot_issue_chart(group_cols, title, data=None)
    def plot_issue_chart(group_cols, title, data=None):
        # ✅ Use provided data or fallback to global df
        if data is None:
            data = df

        # ✅ Safety: Verify required columns
        missing = [c for c in group_cols if c not in data.columns]
        if missing:
            st.warning(f"⚠️ Missing columns for {title}: {missing}")
            return

        # ✅ Group the data
        grouped = (
            data.groupby(group_cols)
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
        )

        if grouped.empty:
            st.warning(f"No valid data available for {title}.")
            return

        # ✅ Create interactive bar chart
        # ✅ Create interactive bar chart
        fig = px.bar(
            grouped,
            x=group_cols[0],
            y="Count",
            color=group_cols[1] if len(group_cols) > 1 else None,
            title=title,
            barmode="group",
        )

        # ✅ Modern Plotly layout (no deprecation warnings)
        fig.update_layout(
            xaxis_title=group_cols[0],
            yaxis_title="Count",
            showlegend=True,
            xaxis_tickangle=-45,
            height=500,
            margin=dict(t=50, b=40, l=40, r=40),
            hovermode="x unified",
            bargap=0.15,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
            ),
        )

        # ✅ Use `config` dict ONLY (modern standard)
        plotly_config = {
            "scrollZoom": True,
            "displayModeBar": True,   # optional; set False to hide toolbar
            "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
            "toImageButtonOptions": {"format": "png", "filename": "hourly_chart"},
            "showTips": False,
            "responsive": True,
        }

        # ✅ Modern Streamlit render call (replaces deprecated width='stretch')
        st.plotly_chart(fig, use_container_width=True, config=plotly_config)

        # =====================================================
        # 📊 Pivoted Table for Cleaner Display
        # =====================================================
        st.markdown("---")

        st.markdown("### 📋 Pivoted Grouped Summary (Agent vs Issue)")

        if len(group_cols) == 2:
            pivot_df = (
                grouped.pivot(index=group_cols[0], columns=group_cols[1], values="Count")
                .fillna(0)
                .astype(int)
                .reset_index()
            )
            st.dataframe(pivot_df, width='stretch')

            # 💾 Download Button
            csv_data = pivot_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Pivoted Data (CSV)",
                data=csv_data,
                file_name=f"pivoted_summary_{title.replace(' ', '_')}.csv",
                mime="text/csv",
            )

            st.info(f"✅ Displayed {len(pivot_df)} {group_cols[0]} across {len(pivot_df.columns)-1} issues.")
        else:
            # Fallback for single-column grouping
            st.dataframe(grouped, width='stretch')
            csv_data = grouped.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Grouped Data (CSV)",
                data=csv_data,
                file_name=f"grouped_summary_{title.replace(' ', '_')}.csv",
                mime="text/csv",
            )




    with sub_tabs[0]:
        st.subheader("Duty thana and Issue-wise Analysis")

        # ✅ Filter only 'Not OK' submissions
        not_ok_df = df[df["Status"].str.strip() == "नहीं"].copy()

        # ✅ Keep only rows where 'Owners' (Agent) column has valid values
        if "Agent" in not_ok_df.columns:
            not_ok_df = not_ok_df[
                not_ok_df["Agent"].notna() &
                (not_ok_df["Agent"].astype(str).str.strip() != "") &
                (~not_ok_df["Agent"].astype(str).isin(["nan", "None", "Unknown", "Unspecified"]))
            ]

        # ✅ Clean up Issue values for display
        if "Issue" in not_ok_df.columns:
            not_ok_df["Issue"] = (
                not_ok_df["Issue"]
                .replace(["nan", "None", "Unknown", "NaT"], pd.NA)
                .fillna("Unspecified")
                .astype(str)
            )

        # ✅ Plot only if there’s valid agent data
        if not_ok_df.empty:
            st.warning("No valid 'Agent' data found for Not OK submissions.")
        else:
            plot_issue_chart(
                ["Agent", "Issue"],
                "Agent and Issue-wise Counts",
                data=not_ok_df
            )


    with sub_tabs[1]:
        st.subheader("Incident Station (जहाँ घटना हुई है ) and Issue-wise Analysis")

        # ✅ Keep only "Not OK" submissions
        not_ok_df = df[df["Status"].str.strip() == "नहीं"].copy()

        # ✅ Keep only rows with valid Incident_Station values
        if "Incident_Station" in not_ok_df.columns:
            not_ok_df = not_ok_df[
                not_ok_df["Incident_Station"].notna() &
                (not_ok_df["Incident_Station"].astype(str).str.strip() != "") &
                (~not_ok_df["Incident_Station"].astype(str).isin(["nan", "None", "Unknown", "Unspecified"]))
            ]

        # ✅ Clean up Issue column for chart labels
        if "Issue" in not_ok_df.columns:
            not_ok_df["Issue"] = (
                not_ok_df["Issue"]
                .replace(["nan", "None", "Unknown", "NaT"], pd.NA)
                .fillna("Unspecified")
                .astype(str)
            )

        # ✅ Plot only if data exists
        if not_ok_df.empty:
            st.warning("No valid 'Incident Station' data found for Not OK submissions.")
        else:
            plot_issue_chart(
                ["Incident_Station", "Issue"],
                "Incident Station and Issue-wise Counts",
                data=not_ok_df
            )


    with sub_tabs[2]:
        st.subheader("BDO and Issue-wise Analysis")
        plot_issue_chart(["Block", "Issue"], "BDO and Issue-wise Counts")


    with sub_tabs[3]:
        st.subheader("संबंधित नगर निकाय  and Issue-wise Analysis")
        plot_issue_chart(["ULD", "Issue"], "ULD and Issue-wise Counts")


    with sub_tabs[4]:
        st.subheader("SDM and Issue-wise Analysis")
        plot_issue_chart(["SDM", "Issue"], "SDM and Issue-wise Counts")


    with sub_tabs[5]:
        st.subheader("पुलिस Circle and Issue-wise Analysis")
        plot_issue_chart(["Circle", "Issue"], "Circle and Issue-wise Counts")


    with sub_tabs[6]:
        st.subheader("📅 Daily Issue-wise (Date-wise Analysis)")

        # ============================================================
        # ⏰ Daily Submission Chart (Stacked: हाँ as base + Issues)
        # ============================================================

        st.subheader("Daily Submissions – Combined Patrol Overview")

        # --- Layout: selector (left) and chart (right) ---
        left_col, right_col = st.columns([1.2, 6])

        with left_col:
            status_mode_date = st.radio(
                "📊 Select Data Mode",
                options=[
                    "total submissions",
                    "total submissions (stacked)",
                    "total issues reported",
                    "total issues reported (stacked)",
                ],
                index=1,
                horizontal=False,
                key="daily_status_mode_patrolling_stacked_total",
            )

        with right_col:
            # --- Prepare Data ---
            df_patrol = df.copy()
            df_patrol["Created_At"] = pd.to_datetime(df_patrol["Created_At"], errors="coerce")
            df_patrol["Date"] = df_patrol["Created_At"].dt.date

            df_no = df_patrol[df_patrol["Status"].astype(str).str.strip() == "नहीं"].copy()
            df_yes = df_patrol[df_patrol["Status"].astype(str).str.strip() == "हाँ"].copy()

            # Safety check for Issue column
            if "Issue" not in df_no.columns:
                st.error("❌ 'Issue' column missing in data.")
                st.stop()

            # --- Define color and order globally ---
            color_map = {
                "हाँ": "blue",
                "छुट्टा पशु": "red",
                "सड़क पर पार्किंग": "green",
                "दुर्घटना": "yellow",
                "अन्य": "orange",
            }
            order = ["हाँ", "छुट्टा पशु", "सड़क पर पार्किंग", "दुर्घटना", "अन्य"]

            # --- Handle different view modes ---
            if status_mode_date == "total issues reported (stacked)":
                daily = (
                    df_no.groupby(["Date", "Issue"])
                    .size()
                    .reset_index(name="Count")
                    .sort_values(["Date", "Count"], ascending=[True, False])
                )
                fig_daily = px.bar(
                    daily,
                    x="Date",
                    y="Count",
                    color="Issue",
                    barmode="stack",
                    category_orders={"Issue": order[1:]},  # exclude "हाँ"
                    color_discrete_map=color_map,
                    title="Daily Reports by Issue (Status='नहीं')",
                )

            elif status_mode_date == "total issues reported":
                daily = (
                    df_no.groupby("Date")
                    .size()
                    .reset_index(name="Count")
                    .sort_values("Date")
                )
                fig_daily = px.bar(
                    daily,
                    x="Date",
                    y="Count",
                    title="Total Issues Reported per Day (Status='नहीं')",
                    color_discrete_sequence=["red"],
                )

            elif status_mode_date == "total submissions":
                daily = (
                    df_patrol.groupby("Date")
                    .size()
                    .reset_index(name="Count")
                    .sort_values("Date")
                )
                fig_daily = px.bar(
                    daily,
                    x="Date",
                    y="Count",
                    title="Total Submissions per Day",
                    color_discrete_sequence=["blue"],
                )

            else:  # total submissions (stacked)
                df_no["Status_Label"] = df_no["Issue"].astype(str).replace({"": "अन्य", "nan": "अन्य"})
                df_yes["Status_Label"] = "हाँ"

                df_combined = pd.concat([df_yes, df_no], ignore_index=True)

                daily = (
                    df_combined.groupby(["Date", "Status_Label"])
                    .size()
                    .reset_index(name="Count")
                    .sort_values(["Date", "Count"], ascending=[True, False])
                )

                fig_daily = px.bar(
                    daily,
                    x="Date",
                    y="Count",
                    color="Status_Label",
                    barmode="stack",
                    category_orders={"Status_Label": order},
                    color_discrete_map=color_map,
                    title="Daily Patrol Submissions by Status / Issue",
                )

            # --- Modern layout ---
            fig_daily.update_layout(
                xaxis_title="Date",
                yaxis_title="Total Patrol Submissions",
                bargap=0.15,
                hovermode="x unified",
                height=450,
                margin=dict(t=50, b=40, l=40, r=40),
                legend=dict(
                    title="Status / Issue Type",
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                    font=dict(size=12),
                ),
            )

            # --- Plot config ---
            plotly_config = {
                "scrollZoom": True,
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
                "toImageButtonOptions": {"format": "png", "filename": "daily_patrol_chart"},
                "showTips": False,
                "responsive": True,
            }

            # --- Render chart ---
            st.plotly_chart(fig_daily, use_container_width=True, config=plotly_config)

        # ============================================================
        # 📊 Data Table (Pivoted Daily Data)
        # ============================================================

        st.markdown("---")
        st.markdown("### 📋 Daily Data (Pivoted Summary)")

        if not daily.empty:
            # ✅ Ensure Status_Label exists
            if "Status_Label" not in daily.columns:
                daily["Status_Label"] = daily.get("Issue", "अन्य")

            # ✅ Pivot data
            pivot_daily = (
                daily
                .pivot(index="Date", columns="Status_Label", values="Count")
                .fillna(0)
                .astype(int)
                .reset_index()
            )

            # ✅ Maintain consistent column order
            ordered_cols = ["Date", "हाँ", "छुट्टा पशु", "सड़क पर पार्किंग", "दुर्घटना", "अन्य"]
            pivot_daily = pivot_daily[[c for c in ordered_cols if c in pivot_daily.columns]]

            # ✅ Center-align numbers
            def df_center_numeric(df):
                styled = df.style.set_properties(**{'text-align': 'center'})
                styled.set_table_styles([
                    dict(selector='th', props=[('text-align', 'center')])
                ])
                return styled

            st.dataframe(df_center_numeric(pivot_daily), use_container_width=True)

            # ✅ Download button
            csv = pivot_daily.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Pivoted Daily Data",
                csv,
                "daily_pivoted.csv",
                "text/csv",
            )

        else:
            st.warning("⚠️ No daily data available to display.")

    # Sub-tab 7: Hourly plot
    with sub_tabs[7]:
        # ============================================================
        # ⏰ Hourly Submission Chart (Stacked: हाँ as base + Issues)
        # ============================================================
        st.subheader("Hourly Submissions – Combined Patrol Overview")

        # --- Layout: left (selector) + right (chart) ---
        left_col, right_col = st.columns([1.2, 6])

        with left_col:
            status_mode_hour = st.radio(
                "📊 Choose Mode",
                options=[
                    "total submissions",
                    "total submissions (stacked)",
                    "total issues reported",
                    "total issues reported (stacked)",
                ],
                index=1,
                horizontal=False,
                key="hourly_status_mode_patrolling_stacked_total_",
            )

        with right_col:
            # --- Prepare Data ---
            df_patrol = df.copy()
            df_patrol["Created_At"] = pd.to_datetime(df_patrol["Created_At"], errors="coerce")
            df_patrol["Hour"] = df_patrol["Created_At"].dt.hour

            # --- Filter subsets ---
            df_no = df_patrol[df_patrol["Status"].astype(str).str.strip() == "नहीं"].copy()
            df_yes = df_patrol[df_patrol["Status"].astype(str).str.strip() == "हाँ"].copy()

            # --- Safety check ---
            if "Issue" not in df_no.columns:
                st.error("❌ 'Issue' column missing in data.")
                st.stop()

            # --- Define color and order globally ---
            color_map = {
                "हाँ": "blue",
                "छुट्टा पशु": "red",
                "सड़क पर पार्किंग": "green",
                "दुर्घटना": "yellow",
                "अन्य": "orange",
            }
            order = ["हाँ", "छुट्टा पशु", "सड़क पर पार्किंग", "दुर्घटना", "अन्य"]

            # --- Mode 1️⃣: total issues reported (stacked) ---
            if status_mode_hour == "total issues reported (stacked)":
                hourly = (
                    df_no.groupby(["Hour", "Issue"])
                    .size()
                    .reset_index(name="Count")
                    .sort_values(["Hour", "Count"], ascending=[True, False])
                )
                fig_hour = px.bar(
                    hourly,
                    x="Hour",
                    y="Count",
                    color="Issue",
                    title="Hourly Reports (Status='नहीं' by Issue)",
                    barmode="stack",
                    category_orders={"Issue": order[1:]},
                    color_discrete_map=color_map,
                )

            # --- Mode 2️⃣: total issues reported ---
            elif status_mode_hour == "total issues reported":
                hourly = (
                    df_no.groupby("Hour")
                    .size()
                    .reset_index(name="Count")
                    .sort_values("Hour")
                )
                fig_hour = px.bar(
                    hourly,
                    x="Hour",
                    y="Count",
                    title="Hourly Reports (Status='नहीं')",
                    color_discrete_sequence=["red"],
                )

            # --- Mode 3️⃣: total submissions ---
            elif status_mode_hour == "total submissions":
                hourly = (
                    df_patrol.groupby("Hour")
                    .size()
                    .reset_index(name="Count")
                    .sort_values("Hour")
                )
                fig_hour = px.bar(
                    hourly,
                    x="Hour",
                    y="Count",
                    title="Total Submissions per Hour",
                    color_discrete_sequence=["blue"],
                )

            # --- Mode 4️⃣: total submissions (stacked) ---
            else:
                df_no["Status_Label"] = df_no["Issue"].astype(str).replace({"": "अन्य", "nan": "अन्य"})
                df_yes["Status_Label"] = "हाँ"

                df_combined = pd.concat([df_yes, df_no], ignore_index=True)

                hourly = (
                    df_combined.groupby(["Hour", "Status_Label"])
                    .size()
                    .reset_index(name="Count")
                    .sort_values(["Hour", "Count"], ascending=[True, False])
                )

                fig_hour = px.bar(
                    hourly,
                    x="Hour",
                    y="Count",
                    color="Status_Label",
                    barmode="stack",
                    category_orders={"Status_Label": order},
                    color_discrete_map=color_map,
                    title="Hourly Patrol Submissions (हाँ base + Issues stacked)",
                )

            # --- Layout ---
            fig_hour.update_layout(
                xaxis_title="Hour of the Day",
                yaxis_title="Total Patrol Submissions",
                bargap=0.15,
                hovermode="x unified",
                height=450,
                margin=dict(t=50, b=40, l=40, r=40),
                legend=dict(
                    title="Status / Issue Type",
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                    font=dict(size=12),
                ),
            )

            # --- Modern Plotly config ---
            plotly_config = {
                "scrollZoom": True,
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
                "toImageButtonOptions": {"format": "png", "filename": "hourly_patrol_chart"},
                "showTips": False,
                "responsive": True,
            }

            # --- Render chart safely ---
            st.plotly_chart(fig_hour, use_container_width=True, config=plotly_config)

        # ============================================================
        # 📊 Data Table (Pivoted Hourly Data)
        # ============================================================
        st.markdown("---")
        st.markdown("### 📋 Hourly Data (Pivoted Summary)")

        if not hourly.empty:
            # ✅ Ensure Status_Label exists
            if "Status_Label" not in hourly.columns:
                hourly["Status_Label"] = hourly.get("Issue", "अन्य")

            # ✅ Pivot data
            pivot_hourly = (
                hourly
                .pivot(index="Hour", columns="Status_Label", values="Count")
                .fillna(0)
                .astype(int)
                .reset_index()
            )

            # ✅ Enforce column order
            ordered_cols = ["Hour", "हाँ", "छुट्टा पशु", "सड़क पर पार्किंग", "दुर्घटना", "अन्य"]
            pivot_hourly = pivot_hourly[[c for c in ordered_cols if c in pivot_hourly.columns]]

            # ✅ Center-align numbers
            def df_center_numeric(df):
                styled = df.style.set_properties(**{'text-align': 'center'})
                styled.set_table_styles([
                    dict(selector='th', props=[('text-align', 'center')])
                ])
                return styled

            st.dataframe(df_center_numeric(pivot_hourly), use_container_width=True)

            # ✅ Download button
            csv = pivot_hourly.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Pivoted Hourly Data",
                csv,
                "hourly_pivoted.csv",
                "text/csv",
            )

        else:
            st.warning("⚠️ No hourly data available to display.")


        

# ------------------- TAB: Map -------------------
with tabs[2]:
    st.header("Hotspot Map (Issue-wise clusters)")

    # controls for cluster size and min points (as requested)
    col1, col2 = st.columns([1,1])
    grid_m = col1.slider("Cluster grid size (m)", 100, 2000, 500, 50)
    min_pts = col2.slider("Min points per cluster", 2, 10, 5, 1)

    cluster_summary, df_map_filtered = compute_issue_clusters(df, grid_m=grid_m, min_points=min_pts, status_filter="नहीं")
    if cluster_summary.empty:
        st.warning("No clusters with these settings.")
    else:
        st.write("Found clusters:", len(cluster_summary))
        # Plot quick Plotly map (fast) with color by Issue and sizes by Count
        issue_color_map = {"छुट्टा पशु":"red", "सड़क पर पार्किंग":"blue", "दुर्घटना":"green"}
        cluster_summary["MarkerSize"] = np.interp(cluster_summary["Count"], [cluster_summary["Count"].min(), cluster_summary["Count"].max()], [8,50])
        fig_map = px.scatter_mapbox(cluster_summary, lat="Latitude", lon="Longitude",
                                    color="Issue", size="MarkerSize",
                                    color_discrete_map=issue_color_map,
                                    hover_data=["cluster_id","Count"],
                                    zoom=9, height=650, mapbox_style="open-street-map")
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        # Render with border by converting to HTML (ensures border wraps)
        border_html = f"""
        <div style="border:4px solid #007ACC;border-radius:10px;overflow:hidden;padding:6px;background:#fff;">
            {fig_map.to_html(include_plotlyjs='cdn', full_html=False)}
        </div>
        """
        html(border_html, height=720)

        # Below the map: show the raw rows for the top N selected cluster (no click-handling here)
        st.markdown("---")
        st.markdown("### Cluster summary (pivoted Agent x Issue) — center aligned numbers")
        pivot = (df_map_filtered[df_map_filtered["cluster_id"].isin(cluster_summary["cluster_id"])]
                 .groupby(["Agent","Issue"]).size().reset_index(name="Count")
                 .pivot(index="Agent", columns="Issue", values="Count").fillna(0).astype(int).reset_index())
        st.dataframe(df_center_numeric(pivot), width='stretch')
        csv = pivot.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download pivoted", csv, "pivoted.csv", "text/csv")

# ---------------- Overview Tab ---------------- #
with tabs[3]:
    st.title("Overview Dashboard")

    daily_counts = df.groupby(["Date", "Status"]).size().reset_index(name="Count")
    
    #new bar
    # --- Create the Plotly bar chart ---
    fig = px.bar(
        daily_counts,
        x="Date",
        y="Count",
        color="Status",
        barmode="group",
        title="Daily Submissions (including Not OK)"
    )

    # --- Safe and modern layout styling ---
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Total Submissions",
        height=500,
        bargap=0.15,
        hovermode="x unified",
        margin=dict(t=50, b=40, l=40, r=40),
        legend=dict(
            title="Status",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
    )

    # --- ✅ Modern Plotly config (no deprecated keyword args) ---
    plotly_config = {
        "scrollZoom": True,
        "displayModeBar": True,   # set to False if you want to hide toolbar
        "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
        "toImageButtonOptions": {"format": "png", "filename": "daily_submissions"},
        "showTips": False,
        "responsive": True,
    }

    # --- ✅ Safe Streamlit rendering (replaces width='stretch') ---
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)


    col1, col2, col3 = st.columns(3)
    col1.metric("Total Submissions", len(df))
    col2.metric("Not OK Submissions", df[df["Status"].astype(str).str.strip() == "नहीं"].shape[0])
    col3.metric("Unique Agents", df["Agent"].nunique() )

# -----------------------------------------------
# 📊 GOOGLE SHEET DATA DOWNLOAD SECTION
# -----------------------------------------------
import io
st.markdown("---")
st.markdown("##### 📁 Download Full Google Sheet Data")

# Assume your full dataframe is df_raw or df
data_to_download = df_raw.copy() if "df_raw" in locals() else df.copy()

# Convert to Excel in memory
to_excel = io.BytesIO()
data_to_download.to_excel(to_excel, index=False, sheet_name="Full_Data")
to_excel.seek(0)

# Center-aligned button using columns
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    st.download_button(
        label="⬇️ Download Full Data as Excel",
        data=to_excel,
        file_name="cyber_dashboard_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# --- Center-aligned download button ---
# Create 3 equal columns and place the button in the center one
with c3:
    # The actual button
    if st.button("Fetch / Refresh Data"):
        st.session_state.sheet_url = sheet_input.strip()
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

# Footer
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("<div class='footer'>© Cyber Cell Shahjahanpur · Highway Patrol Dashboard</div>", unsafe_allow_html=True)
st.markdown("---")

# ------------------- END -------------------
