import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
import pymc as pm
from PIL import Image
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly


st.set_page_config(layout="wide")


@st.cache(allow_output_mutation=True)
def read_raw_data():
    df = pd.read_csv("peyton_manning.csv")
    return df


# Read data
df_raw = read_raw_data()
df = df_raw.copy()

# Plot layout
layout = {
    "width": 800,
    "height": 400,
    "paper_bgcolor": "#FFFFFF",
    "plot_bgcolor": "#FFFFFF",
    "autosize": True,
    "margin": {"l": 40, "r": 10, "t": 20, "b": 20},
}

st.header("Deep-Dive into Prophet")

# Rebuild trend model
with st.expander("🎓"):
    st.subheader("Trend model")

    def get_trend_plot(trend, growth, offset, s):
        fig = make_subplots(
            rows=3,
            cols=1,
            vertical_spacing=0.2,
            subplot_titles=("Trend g(t)", "Growth", "Offset"),
            shared_xaxes=False,
        )

        trend_style = dict(color="#437FB6", dash="solid")
        other_sytle = dict(color="#7FAACD", dash="solid")
        reference_style = dict(color="#FF7E79", width=2, dash="dashdot")

        fig.add_trace(go.Scatter(x=t, y=trend, name="Trend", line=trend_style), 1, 1)
        fig.add_trace(go.Scatter(x=t, y=growth, name="growth", line=other_sytle), 2, 1)
        fig.add_trace(go.Scatter(x=t, y=offset, name="offset", line=other_sytle), 3, 1)

        for s_i in s:
            fig.add_vline(x=s_i, line=reference_style)

        fig.for_each_xaxis(
            lambda x: x.update(
                showgrid=True,
                gridcolor="lightgrey",
                nticks=10,
                griddash="dot",
                zeroline=False,
                linecolor="lightgrey",
            )
        )
        fig.for_each_yaxis(
            lambda y: y.update(
                showgrid=True,
                gridcolor="lightgrey",
                griddash="dot",
                zeroline=False,
                linecolor="lightgrey",
            )
        )

        fig.update_layout(layout)

        return fig

    trend_0, trend_1, trend_2, trend_3 = st.columns([3, 1, 1, 4])

    with trend_0:
        trend_image = Image.open("trend_2.png")
        st.image(trend_image, use_column_width=True)

    with trend_1:
        st.markdown("**Initial growth and offset**")
        k = st.slider("k", min_value=0, max_value=5, value=1, step=1)  # base growth
        m = st.slider("m", min_value=0, max_value=5, value=1, step=1)  # base offset

    with trend_2:
        st.markdown("**Delta of growth**")
        slider_names = ["d1", "d2", "d3", "d4"]
        d1 = st.slider("d1", min_value=-10, max_value=10, value=0, step=1)
        d2 = st.slider("d2", min_value=-10, max_value=10, value=0, step=1)
        d3 = st.slider("d3", min_value=-10, max_value=10, value=0, step=1)
        d4 = st.slider("d4", min_value=-10, max_value=10, value=0, step=1)

    with trend_3:
        st.markdown("**Combination of different parts**")
        n_changepoints = 4  # Number of changepoints S
        t = np.arange(1000)  # time steps
        s = np.linspace(0, max(t), n_changepoints + 2)[1:-1]  # array of change points

        A = (t[:, None] > s) * 1

        delta = np.array([d1, d2, d3, d4])  # growth rate adjustments

        growth = (k + A @ delta) * t
        gamma = -s * delta
        offset = m + A @ gamma
        trend = growth + offset
        fig = get_trend_plot(trend, growth, offset, s)
        st.plotly_chart(fig, theme=None, use_container_width=True)

# Rebuild seasonal model
with st.expander("🎓🎓"):
    st.subheader("Seasonal model")

    seasonal_1_1, seasonal_1_2, seasonal_1_3 = st.columns([2, 1, 4])

    with seasonal_1_1:
        seasonal_image = Image.open("seasonal_2.png")
        st.image(seasonal_image, use_column_width=True)

    with seasonal_1_2:
        st.markdown("**Scaling sin & cos**")
        a1 = st.slider("a1", min_value=0, max_value=5, value=0, step=1)
        a2 = st.slider("a2", min_value=0, max_value=5, value=0, step=1)
        b1 = st.slider("b1", min_value=0, max_value=5, value=0, step=1)
        b2 = st.slider("b2", min_value=0, max_value=5, value=0, step=1)

    periodicity = 25
    a = [a1, a2]
    b = [b1, b2]

    def get_fourier_plot1(a, b, periodicity):
        comb_style = dict(color="#437FB6", dash="solid")
        sin_style = dict(color="#7FAACD", dash="dot")
        cos_style = dict(color="#7FAACD", dash="dashdot")

        order = 2
        x = np.linspace(0, 100, 100)

        cos_n = np.array(
            [
                a[i] * np.cos((2 * np.pi * (i + 1) * x) / periodicity)
                for i in range(0, order)
            ]
        )

        sin_n = np.array(
            [
                b[i] * np.sin((2 * np.pi * (i + 1) * x) / periodicity)
                for i in range(0, order)
            ]
        )

        y = (cos_n + sin_n).sum(axis=0)

        fig = make_subplots(
            rows=2,
            cols=1,
            vertical_spacing=0.2,
            subplot_titles=("Combined sin & cos terms", "Individual sin & cos terms"),
            shared_xaxes=False,
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name="y",
                line=comb_style,
            ),
            1,
            1,
        )

        for i in range(0, order):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=cos_n[i],
                    name=f"a_{i+1}_cos",
                    line=cos_style,
                ),
                2,
                1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=sin_n[i],
                    name=f"b_{i+1}_sin",
                    line=sin_style,
                ),
                2,
                1,
            )

        fig.update_layout(layout)

        fig.for_each_xaxis(
            lambda x: x.update(
                showgrid=True,
                gridcolor="lightgrey",
                nticks=20,
                griddash="dot",
                zeroline=False,
                linecolor="lightgrey",
            )
        )
        fig.for_each_yaxis(
            lambda y: y.update(
                showgrid=True,
                gridcolor="lightgrey",
                griddash="dot",
                zeroline=False,
                linecolor="lightgrey",
            )
        )

        return fig

    with seasonal_1_3:
        fig_fourier_1 = get_fourier_plot1(a, b, periodicity)
        st.plotly_chart(fig_fourier_1, theme=None, use_container_width=True)

# Rebuild seasonal combination
with st.expander("🎓🎓🎓"):
    st.subheader("Multiple seasonalities")

    seasonal_2_1, seasonal_2_2, seasonal_2_3 = st.columns([1, 1, 4])

    with seasonal_2_1:
        st.markdown("**Order of weekly seasonality**")
        order_weekly = st.slider(
            "order_weekly", min_value=1, max_value=10, value=1, step=1
        )
    with seasonal_2_2:
        st.markdown("**Order of yearly seasonality**")
        order_yearly = st.slider(
            "order_yearly", min_value=1, max_value=10, value=1, step=1
        )

    def get_fourier_plot2(order_weekly, order_yearly):
        np.random.seed(42)

        beta_weekly = np.random.normal(size=2 * order_weekly)
        beta_yearly = np.random.normal(size=2 * order_yearly) + 2

        x = np.arange(3 * 365)

        periodicity_weekly = 7
        periodicity_yearly = 365

        def fourier_series_vectorized(t, p, n):
            # 2 pi n / p
            x = 2 * np.pi * np.arange(1, n + 1) / p
            # 2 pi n / p * t
            x = x * t[:, None]
            x = np.concatenate((np.cos(x), np.sin(x)), axis=1)
            return x

        y_weekly = (
            fourier_series_vectorized(x, periodicity_weekly, order_weekly) @ beta_weekly
        )

        y_yearly = (
            fourier_series_vectorized(x, periodicity_yearly, order_yearly) @ beta_yearly
        )

        y = y_weekly + y_yearly

        comb_style = dict(color="#437FB6", dash="solid")
        yearly_style = dict(color="#7FAACD", dash="solid")
        weekly_style = dict(color="#7FAACD", dash="solid")

        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.1)

        fig.add_trace(
            go.Scatter(x=x, y=y, name="total_seasonality", line=comb_style), 1, 1
        )
        fig.add_trace(
            go.Scatter(x=x, y=y_weekly, name="weekly", line=yearly_style), 2, 1
        )
        fig.add_trace(
            go.Scatter(x=x, y=y_yearly, name="yearly", line=weekly_style), 2, 1
        )

        fig.for_each_xaxis(
            lambda x: x.update(
                showgrid=True,
                gridcolor="lightgrey",
                nticks=20,
                griddash="dot",
                zeroline=False,
                linecolor="lightgrey",
            )
        )
        fig.for_each_yaxis(
            lambda y: y.update(
                showgrid=True,
                gridcolor="lightgrey",
                griddash="dot",
                zeroline=False,
                linecolor="lightgrey",
            )
        )

        fig.update_layout(layout)

        return fig

    with seasonal_2_3:
        fig_fourier_2 = get_fourier_plot2(order_weekly, order_yearly)

        st.plotly_chart(fig_fourier_2, theme=None, use_container_width=True)

# Fitting the parameters
with st.expander("🎓🎓🎓🎓"):
    st.subheader("Fitting the model - MAP")

    (
        fitting_map_1,
        fitting_map_2,
        fitting_map_3,
    ) = st.columns([1, 2, 2])

    with fitting_map_1:
        st.subheader("Parameters")
        with st.form("fit_model"):
            add_trend = st.checkbox("add_trend", value=True)

            add_yearly_seasonality = st.checkbox("add_yearly", value=False)
            add_weekly_seasonality = st.checkbox("add_weekly", value=False)

            set_changepoint_prior_scale = st.number_input(
                "changepoint_prior_scale", value=0.05, format="%.4f"
            )

            set_yearly_seasonality_prior_scale = st.number_input(
                "yearly_seasonality_prior_scale", value=10.00, format="%.4f"
            )

            set_weekly_seasonality_prior_scale = st.number_input(
                "weekly_seasonality_prior_scale", value=10.00, format="%.4f"
            )

            fit = st.form_submit_button("fit")

        if fit:
            model = pm.Model()

    with fitting_map_2:
        st.subheader("Forecast")

        if fit:
            # Python
            m = Prophet(
                n_changepoints=25,
                weekly_seasonality=False,
                yearly_seasonality=False,
                changepoint_prior_scale=set_changepoint_prior_scale,
            )
            if add_yearly_seasonality:
                m.add_seasonality(
                    name="yearly",
                    period=365.25,
                    fourier_order=10,
                    prior_scale=set_yearly_seasonality_prior_scale,
                )

            if add_weekly_seasonality:
                m.add_seasonality(
                    name="weekly",
                    period=7,
                    fourier_order=3,
                    prior_scale=set_weekly_seasonality_prior_scale,
                )

            m.fit(df)

            # Python
            future = m.make_future_dataframe(periods=0)
            forecast = m.predict(future)

            fig_forecast = plot_plotly(m, forecast, uncertainty=False)
            fig_forecast.update_layout(
                dict(
                    autosize=True,
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    margin=dict(l=40, r=10, t=20, b=20),
                )
            )

            fig_forecast.data[0]["marker"]["size"] = 2.5
            fig_forecast.data[0]["marker"]["opacity"] = 0.5
            fig_forecast.data[0]["marker"]["color"] = "grey"

            st.plotly_chart(fig_forecast, theme=None, use_container_width=True)

            fig_components = plot_components_plotly(m, forecast, uncertainty=False)
            fig_components.update_layout(
                dict(
                    autosize=True,
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    margin=dict(l=40, r=10, t=20, b=20),
                )
            )

            if add_yearly_seasonality | add_weekly_seasonality:
                fitting_map_3.subheader("Components")
                fitting_map_3.plotly_chart(
                    fig_components, theme=None, use_container_width=True
                )
