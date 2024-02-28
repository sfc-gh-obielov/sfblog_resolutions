import streamlit as st
import pandas as pd
import pydeck as pdk
import branca.colormap as cm
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col
from typing import List
import json
from PIL import Image

session = Session.builder.configs(st.secrets["geodemo"]).create()

@st.cache_resource
def get_h3point_df(resolution: float, row_count: int) -> pd.DataFrame:
    return session.sql(
        f"select distinct h3_point_to_cell_string(ST_POINT(UNIFORM( -180 , 180 , random()), UNIFORM( -90 , 90 , random())), {resolution}) as h3 from table(generator(rowCount => {row_count}))"
    ).to_pandas()


@st.cache_resource
def get_coverage_layer(df: pd.DataFrame, line_color: List) -> pdk.Layer:
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

min_v_1, max_v_1, v_1, z_1, lon_1, lat_1 = ( 0, 2, 0, 1, 0.9982847947205775, 2.9819747220001886,)
col1, col2 = st.columns([70, 30])
with col1:
    h3_resolut_1 = st.slider(
        "H3 resolution", min_value=min_v_1, max_value=max_v_1, value=v_1)

with col2:
    levels_option = st.selectbox("Levels", ("One", "Two", "Three"))

df = get_h3point_df(h3_resolut_1, 100000)
layer_coverage_1 = get_coverage_layer(df, [36, 191, 242])

visible_layers_coverage_1 = [layer_coverage_1]

if levels_option == "Two":
    df_coverage_level_1 = get_h3point_df(h3_resolut_1 + 1, 100000)
    layer_coverage_1_level_1 = get_coverage_layer(df_coverage_level_1, [217, 102, 255])
    visible_layers_coverage_1 = [layer_coverage_1, layer_coverage_1_level_1]

if levels_option == "Three":
    df_coverage_level_1 = get_h3point_df(h3_resolut_1 + 1, 100000)
    layer_coverage_1_level_1 = get_coverage_layer(df_coverage_level_1, [217, 102, 255])

    df_coverage_level_2 = get_h3point_df(h3_resolut_1+2, 1000000)

    layer_coverage_1_level2 = get_coverage_layer(df_coverage_level_2, [18, 100, 129])
    visible_layers_coverage_1 = [
        layer_coverage_1,
        layer_coverage_1_level_1,
        layer_coverage_1_level2, ]

st.pydeck_chart(
    pdk.Deck(map_provider='carto', 
        map_style='light',
        initial_view_state=pdk.ViewState(
            latitude=lat_1, longitude=lon_1, zoom=z_1, height=400
        ),
        tooltip={"html": "<b>ID:</b> {H3}", "style": {"color": "white"}},
        layers=visible_layers_coverage_1,
    )
)