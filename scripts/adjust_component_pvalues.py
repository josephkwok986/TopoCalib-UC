"""Holm adjustment over the four component comparisons in Appendix A."""
import argparse
import json
from pathlib import Path


def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = [0.] * len(values)
    maximum = 0.
    for rank, i in enumerate(order):
        if not 0 <= values[i] <= 1:
            raise ValueError('p-values must be in [0, 1]')
        maximum = max(maximum, min(1., (len(values)-rank)*values[i]))
        adjusted[i] = maximum
    return adjusted


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,nargs=4,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if len(set(a.input)) != 4:
        p.error('Four distinct comparison results are required')
    rows=[json.loads(path.read_text()) for path in a.input]
    values=[row['randomization_p'] for row in rows]
    results=[dict(input=str(path),**row,holm_adjusted_p=value) for path,row,value in zip(a.input,rows,holm(values))]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(results,indent=2)+'\n')


if __name__ == '__main__':
    main()
