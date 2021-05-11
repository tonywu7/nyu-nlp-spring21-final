from pathlib import Path
from textwrap import indent

if __name__ == '__main__':
    with open('summary.txt', 'w+') as f:
        ps = sorted(p for p in Path('.').iterdir() if p.suffix == '.txt')
        for p in ps:
            with open(p) as f2:
                f.write(p.with_suffix('').name + '\n\n' + indent(f2.read(), '  ') + '\n')
