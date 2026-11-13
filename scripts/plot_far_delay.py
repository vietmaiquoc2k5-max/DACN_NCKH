"""Plot measured FAR/delay rows, keeping absent measurements visibly blocked."""
import argparse
import csv
import math
from pathlib import Path
import plotly.graph_objects as go


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Output already exists')
    with args.csv.open(encoding='utf-8',newline='') as stream: rows=list(csv.DictReader(stream))
    valid=[]
    for row in rows:
        try:
            far,delay=float(row['far']),float(row['delay'])
            if math.isfinite(far) and math.isfinite(delay) and min(far,delay)>=0:
                valid.append((far,delay,row.get('config_hash','')))
        except (ValueError,TypeError,KeyError):continue
    fig=go.Figure()
    if valid:
        fig.add_trace(go.Scatter(x=[r[1] for r in valid],y=[r[0] for r in valid],text=[r[2] for r in valid],mode='markers'))
        fig.update_layout(xaxis_title='Detection delay (s)',yaxis_title='FAR/hour')
    else:fig.add_annotation(text='BLOCKED: no measured FAR/Delay pairs; no metrics fabricated',showarrow=False)
    fig.write_html(str(args.output),include_plotlyjs=True)
    print('MEASURED_POINTS='+str(len(valid)))


if __name__=='__main__':main()
