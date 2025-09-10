import streamlit as st
import pandas as pd
import pydeck as pdk
import branca.colormap as cm
from typing import List
import numpy as np
import h3

_latlng_to_h3 = getattr(h3, "latlng_to_cell", None) or getattr(h3, "geo_to_h3")

@st.cache_data(ttl="2d")
def get_h3point_df(resolution: int, row_count: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate random lon/lat points on the globe and return distinct H3 cells.
    Mirrors the original SQL which sampled uniformly over lon/lat.
    """
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-180.0, 180.0, size=row_count)
    lat = rng.uniform(-90.0, 90.0, size=row_count)

    # Compute H3 index for each point
    h3_indices = [
        _latlng_to_h3(float(la), float(lo), int(resolution))
        for la, lo in zip(lat, lon)
    ]

    # Distinct cells, as in the original SELECT DISTINCT
    uniq = pd.unique(pd.Series(h3_indices, name="H3"))
    return pd.DataFrame({"H3": uniq})

@st.cache_data(ttl="2d")
def get_coverage_layer(df: pd.DataFrame, line_color: List[int]) -> pdk.Layer:
    return pdk.Layer(
        "H3HexagonLayer",
        df,
        get_hexagon="H3",
        stroked=True,
        filled=False,
        auto_highlight=True,
        elevation_scale=45,
        pickable=True,
        extruded=False,
        get_line_color=line_color,
        line_width_min_pixels=1,
    )

# UI
min_v_1, max_v_1, v_1, z_1, lon_1, lat_1 = (
    0, 2, 0, 1, 0.9982847947205775, 2.9819747220001886,
)

col1, col2 = st.columns([70, 30])
with col1:
    h3_resolut_1 = st.slider("H3 resolution", min_value=min_v_1, max_value=max_v_1, value=v_1)

with col2:
    levels_option = st.selectbox("Levels", ("One", "Two", "Three"))

# Base layer
df = get_h3point_df(h3_resolut_1, 100_000)
layer_coverage_1 = get_coverage_layer(df, [36, 191, 242])
visible_layers_coverage_1 = [layer_coverage_1]

# Additional levels (higher resolution => more/smaller cells)
if levels_option == "Two":
    df_coverage_level_1 = get_h3point_df(h3_resolut_1 + 1, 100_000)
    layer_coverage_1_level_1 = get_coverage_layer(df_coverage_level_1, [217, 102, 255])
    visible_layers_coverage_1 = [layer_coverage_1, layer_coverage_1_level_1]

if levels_option == "Three":
    df_coverage_level_1 = get_h3point_df(h3_resolut_1 + 1, 100_000)
    layer_coverage_1_level_1 = get_coverage_layer(df_coverage_level_1, [217, 102, 255])

    df_coverage_level_2 = get_h3point_df(h3_resolut_1 + 2, 1_000_000)
    layer_coverage_1_level2 = get_coverage_layer(df_coverage_level_2, [18, 100, 129])

    visible_layers_coverage_1 = [
        layer_coverage_1,
        layer_coverage_1_level_1,
        layer_coverage_1_level2,
    ]

st.pydeck_chart(
    pdk.Deck(
        map_provider='carto',
        map_style='light',
        initial_view_state=pdk.ViewState(
            latitude=lat_1, longitude=lon_1, zoom=z_1, height=400
        ),
        tooltip={"html": "<b>ID:</b> {H3}", "style": {"color": "white"}},
        layers=visible_layers_coverage_1,
    )
)
