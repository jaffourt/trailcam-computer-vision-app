import dash
import dash_daq as daq
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State

from faunafinder import FaunaFinder
from faunafinder.webcam import Webcam

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1(children="FaunaFinder Dash App"),
    dbc.Row(html.Img(id='image',
                     src="assets/none.png",
                     style={'height': '10%', 'width': '90%'})),
    html.Div(id='slider-output-container'),
    html.Button(id='start-button', n_clicks=0, children='Start', disabled=True),
    html.Button(id='stop-button', n_clicks=0, children='Stop'),
    html.Div(children="Update interval (in seconds)"),
    dbc.Row([
        dbc.Col(daq.Slider(
            id='interval-slider',
            min=10,
            max=300,
            value=10,
            marks={i: str(i) for i in range(10, 310, 50)},
            step=10
        )),
    ]),
    dcc.Interval(
        id='interval-component',
        interval=1000,  # in milliseconds
        n_intervals=0,
        disabled=True,
    ),
    html.H4(id="webcam-header", children="No webcam configured."),
    html.Div(
        [
            dcc.Dropdown(
                id='dropdown',
                options=[],
                value=None
            )
        ],
        style={'width': '50%'}
    ),
    html.Button(id='check-webcam-button', n_clicks=0, children='Check Webcams'),
    html.Button(id='configure-webcam-button', n_clicks=0, children='Load Webcam'),
])


@app.callback(
    dash.dependencies.Output('slider-output-container', 'children'),
    [dash.dependencies.Input('interval-slider', 'value'),
     dash.dependencies.Input('interval-component', 'disabled')])
def update_interval_text(value, disabled):
    if disabled:
        return f"Press start to start updating"
    return f'Updating every {value} seconds'


@app.callback(Output('interval-component', 'interval'),
              [Input('start-button', 'n_clicks')],
              [State('interval-slider', 'value')])
def set_interval(n_clicks, interval):
    # enable the interval component, set interval update from slider value
    return interval * 1000


@app.callback(Output('interval-component', 'disabled'),
              [Input('start-button', 'n_clicks'),
               Input('stop-button', 'n_clicks')])
def enable_interval(start_clicks, stop_clicks):
    if start_clicks > stop_clicks:
        return False
    return True


@app.callback(Output('image', 'src'),
              [Input('interval-component', 'n_intervals')],
              prevent_initial_call=True)
def update(n):
    return finder()


@app.callback(
    Output('dropdown', 'options'),
    [Input('check-webcam-button', 'n_clicks')],
    prevent_initial_call=True)
def update_dropdown(n_clicks):
    return [{'label': name, 'value': i} for i, name in Webcam.get_webcams().items()]


@app.callback(Output('webcam-header', 'children'), Output('start-button', 'disabled'),
              [Input('configure-webcam-button', 'n_clicks'), Input('dropdown', 'value')],
              prevent_initial_call=True)
def create_webcam_instance(n_clicks, value):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'configure-webcam-button' in changed_id:
        if value is not None:
            finder.create_webcam_instance(value)
            return "Webcam selected.", False
    else:
        # don't update
        raise Exception


if __name__ == '__main__':
    finder = FaunaFinder(model_path="trained_models/finetuned_vg_166_2023-01-05 21:49:42.657696.pt",
                         label_json_path="data/class_labels.json", error_threshold=0.1)
    app.run_server()
