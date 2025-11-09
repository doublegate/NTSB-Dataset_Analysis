"""Reusable chart components for Streamlit dashboard.

This module provides functions for creating consistent Plotly visualizations.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Optional, List


def create_line_chart(
    df: pd.DataFrame, x: str, y: str, title: str, color: Optional[str] = None, **kwargs
) -> go.Figure:
    """Create a Plotly line chart.

    Args:
        df: DataFrame with data
        x: Column name for x-axis
        y: Column name for y-axis
        title: Chart title
        color: Optional column for color coding
        **kwargs: Additional Plotly arguments

    Returns:
        Plotly Figure object
    """
    fig = px.line(df, x=x, y=y, title=title, color=color, **kwargs)
    fig.update_layout(
        height=400, hovermode="x unified", showlegend=True if color else False
    )
    fig.update_traces(mode="lines+markers")
    return fig


def create_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: Optional[str] = None,
    orientation: str = "v",
    **kwargs,
) -> go.Figure:
    """Create a Plotly bar chart.

    Args:
        df: DataFrame with data
        x: Column name for x-axis
        y: Column name for y-axis
        title: Chart title
        color: Optional column for color coding
        orientation: 'v' for vertical, 'h' for horizontal
        **kwargs: Additional Plotly arguments

    Returns:
        Plotly Figure object
    """
    fig = px.bar(
        df, x=x, y=y, title=title, color=color, orientation=orientation, **kwargs
    )
    fig.update_layout(height=400)
    return fig


def create_scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: Optional[str] = None,
    size: Optional[str] = None,
    **kwargs,
) -> go.Figure:
    """Create a Plotly scatter plot.

    Args:
        df: DataFrame with data
        x: Column name for x-axis
        y: Column name for y-axis
        title: Chart title
        color: Optional column for color coding
        size: Optional column for marker size
        **kwargs: Additional Plotly arguments

    Returns:
        Plotly Figure object
    """
    fig = px.scatter(df, x=x, y=y, title=title, color=color, size=size, **kwargs)
    fig.update_layout(height=400)
    return fig


def create_pie_chart(
    df: pd.DataFrame, values: str, names: str, title: str, **kwargs
) -> go.Figure:
    """Create a Plotly pie chart.

    Args:
        df: DataFrame with data
        values: Column name for slice values
        names: Column name for slice labels
        title: Chart title
        **kwargs: Additional Plotly arguments

    Returns:
        Plotly Figure object
    """
    fig = px.pie(df, values=values, names=names, title=title, **kwargs)
    fig.update_layout(height=400)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def create_choropleth_map(
    df: pd.DataFrame,
    locations: str,
    color: str,
    title: str,
    color_continuous_scale: str = "Reds",
    **kwargs,
) -> go.Figure:
    """Create a US state choropleth map.

    Args:
        df: DataFrame with data
        locations: Column name with state codes
        color: Column name for color scale
        title: Chart title
        color_continuous_scale: Plotly color scale
        **kwargs: Additional Plotly arguments

    Returns:
        Plotly Figure object
    """
    fig = px.choropleth(
        df,
        locations=locations,
        locationmode="USA-states",
        color=color,
        scope="usa",
        title=title,
        color_continuous_scale=color_continuous_scale,
        **kwargs,
    )
    fig.update_layout(height=500)
    return fig


def create_heatmap(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    title: str,
    color_scale: str = "Reds",
    **kwargs,
) -> go.Figure:
    """Create a Plotly heatmap.

    Args:
        df: DataFrame with data
        x: Column name for x-axis
        y: Column name for y-axis
        z: Column name for cell values
        title: Chart title
        color_scale: Plotly color scale
        **kwargs: Additional Plotly arguments

    Returns:
        Plotly Figure object
    """
    # Pivot data for heatmap
    pivot_df = df.pivot(index=y, columns=x, values=z)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale=color_scale,
            **kwargs,
        )
    )
    fig.update_layout(title=title, height=400)
    return fig


def create_treemap(
    df: pd.DataFrame,
    path: List[str],
    values: str,
    title: str,
    color: Optional[str] = None,
    **kwargs,
) -> go.Figure:
    """Create a Plotly treemap.

    Args:
        df: DataFrame with data
        path: List of column names for hierarchy
        values: Column name for rectangle sizes
        title: Chart title
        color: Optional column for color coding
        **kwargs: Additional Plotly arguments

    Returns:
        Plotly Figure object
    """
    fig = px.treemap(df, path=path, values=values, title=title, color=color, **kwargs)
    fig.update_layout(height=500)
    return fig


def create_histogram(
    df: pd.DataFrame, x: str, title: str, nbins: Optional[int] = None, **kwargs
) -> go.Figure:
    """Create a Plotly histogram.

    Args:
        df: DataFrame with data
        x: Column name for histogram
        title: Chart title
        nbins: Number of bins
        **kwargs: Additional Plotly arguments

    Returns:
        Plotly Figure object
    """
    fig = px.histogram(df, x=x, title=title, nbins=nbins, **kwargs)
    fig.update_layout(height=400)
    return fig


def create_box_plot(
    df: pd.DataFrame,
    x: Optional[str],
    y: str,
    title: str,
    color: Optional[str] = None,
    **kwargs,
) -> go.Figure:
    """Create a Plotly box plot.

    Args:
        df: DataFrame with data
        x: Column name for categories (optional)
        y: Column name for values
        title: Chart title
        color: Optional column for color coding
        **kwargs: Additional Plotly arguments

    Returns:
        Plotly Figure object
    """
    fig = px.box(df, x=x, y=y, title=title, color=color, **kwargs)
    fig.update_layout(height=400)
    return fig


def create_line_with_confidence(
    df: pd.DataFrame,
    x: str,
    y: str,
    y_upper: str,
    y_lower: str,
    title: str,
    line_name: str = "Actual",
    **kwargs,
) -> go.Figure:
    """Create line chart with confidence interval.

    Args:
        df: DataFrame with data
        x: Column name for x-axis
        y: Column name for y values
        y_upper: Column name for upper bound
        y_lower: Column name for lower bound
        title: Chart title
        line_name: Name for main line
        **kwargs: Additional arguments

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Add main line
    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df[y],
            mode="lines+markers",
            name=line_name,
            line=dict(color="blue"),
        )
    )

    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=df[x].tolist() + df[x].tolist()[::-1],
            y=df[y_upper].tolist() + df[y_lower].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(0,100,200,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=True,
            name="95% Confidence Interval",
        )
    )

    fig.update_layout(title=title, height=400, hovermode="x unified")

    return fig
