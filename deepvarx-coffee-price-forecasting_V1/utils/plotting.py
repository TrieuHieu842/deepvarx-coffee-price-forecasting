"""Các hàm vẽ biểu đồ Plotly dùng chung cho nhiều trang."""

import plotly.express as px
import plotly.graph_objects as go
from statsmodels.graphics.tsaplots import acf, pacf


def line_chart(df, x, y, color=None, title=""):
    fig = px.line(df, x=x, y=y, color=color, title=title)
    fig.update_layout(height=420, margin=dict(t=50, b=30))
    return fig


def correlation_heatmap(corr_df, title="Ma trận tương quan"):
    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title=title,
        aspect="auto",
    )
    fig.update_layout(height=500, margin=dict(t=50, b=30))
    return fig


def acf_pacf_chart(series, nlags=30):
    s = series.dropna()
    nlags = min(nlags, len(s) // 2 - 1)
    acf_vals = acf(s, nlags=nlags)
    pacf_vals = pacf(s, nlags=nlags)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name="ACF"))
    fig.update_layout(title="Autocorrelation Function (ACF)", height=350, margin=dict(t=50, b=30))

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, name="PACF", marker_color="orange"))
    fig2.update_layout(title="Partial Autocorrelation Function (PACF)", height=350, margin=dict(t=50, b=30))

    return fig, fig2


def loss_curve_chart(train_losses, val_losses=None, title="Loss theo epoch"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_losses, mode="lines", name="Train loss"))
    if val_losses is not None:
        fig.add_trace(go.Scatter(y=val_losses, mode="lines", name="Validation loss"))
    fig.update_layout(title=title, xaxis_title="Epoch", yaxis_title="Loss", height=400)
    return fig


def forecast_vs_actual_chart(dates, actual, predicted, title="Dự báo vs Thực tế"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual, mode="lines", name="Thực tế", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=dates, y=predicted, mode="lines", name="Dự báo", line=dict(color="#ff7f0e", dash="dash")))
    fig.update_layout(title=title, height=450, margin=dict(t=50, b=30))
    return fig


def metric_comparison_bar(metrics_df, metric_col, title=""):
    fig = px.bar(metrics_df, x="Mô hình", y=metric_col, color="Mô hình", title=title, text_auto=".3f")
    fig.update_layout(height=400, margin=dict(t=50, b=30), showlegend=False)
    return fig
