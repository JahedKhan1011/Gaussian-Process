# Gaussian-Process
import numpy as np
import plotly.graph_objects as go
from ipywidgets import interact, widgets
def update_layout_of_graph(fig: go.Figure, title: str = 'Plot') -> go.Figure:
    fig.update_layout(
        width=800,
        height=600,
        autosize=False,
        plot_bgcolor='white',  # Set the plot background color to white
        paper_bgcolor='white',  # Set the paper background color to white
        title=title,
    )
    fig.update_layout(
        xaxis_title='input values',
        yaxis_title='output values',
        legend=dict(yanchor="top", y=0.9, xanchor="right", x=0.95),
        title={'x': 0.5, 'xanchor': 'center'}
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
    return fig

def line_scatter (
    visible: bool = True,
    x_lines: np.array = np.array([]),
    y_lines: np.array = np.array([]),
    name_line: str = 'Predicted function',
    showlegend: bool = True,
) -> go.Scatter:
    # Adding the lines
    return go.Scatter(
        visible=visible,
        line=dict(color="blue", width=2),
        x=x_lines,
        y=y_lines,
        name=name_line,
        showlegend= showlegend
    )

def dot_scatter(
    visible: bool = True,
    x_dots: np.array = np.array([]),
    y_dots: np.array = np.array([]),
    name_dots: str = 'Observed points',
    showlegend: bool = True
) -> go.Scatter:
    # Adding the dots
    return go.Scatter(
        x=x_dots,
        visible=visible,
        y=y_dots,
        mode="markers",
        name=name_dots,
        marker=dict(color='red', size=8),
        showlegend=showlegend
    )

def uncertainty_area_scatter(
        visible: bool = True,
        x_lines: np.array = np.array([]),
        y_upper: np.array = np.array([]),
        y_lower: np.array = np.array([]),
        name: str = "mean plus/minus standard deviation",
) -> go.Scatter:

    return go.Scatter(
        visible=visible,
        x=np.concatenate((x_lines, x_lines[::-1])),  # x, then x reversed
        # upper, then lower reversed
        y=np.concatenate((y_upper, y_lower[::-1])),
        fill='toself',
        fillcolor='rgba(189,195,199,0.5)',
        line=dict(color='rgba(200,200,200,0)'),
        hoverinfo="skip",
        showlegend=True,
        name= name,
    )


class SquaredExponentialKernel:
    def __init__(self, length=1, sigma_f=1):
        self.length = length
        self.sigma_f = sigma_f


    def __call__(self, argument_1, argument_2):
        return float(self.sigma_f *
                     np.exp(-np.linalg.norm(argument_1 - argument_2)**2 /
                            (2 * self.length**2)))
x_lines = np.arange(-10, 10, 0.1)
kernel = SquaredExponentialKernel(length=1, sigma_f=1)

fig0 = go.FigureWidget(data=[
    line_scatter(
        x_lines=x_lines,
        y_lines=np.array([kernel(x, 0) for x in x_lines]),
    )
])

fig0 = update_layout_of_graph(fig0, title='Squared exponential kernel')

@interact(length=(0.1, 3, 0.1), argument_2=(-10, 10, 0.1))
def update(length=1, argument_2=0):
    with fig0.batch_update():
        kernel = SquaredExponentialKernel(length=length, sigma_f=1)
        fig0.data[0].y = np.array([kernel(x, argument_2) for x in x_lines])

fig0.show()
