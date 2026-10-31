import argparse
import json
from pathlib import Path
from .prepare import prepare

def main(argv=None):
    parser = argparse.ArgumentParser(description='Visual Search local retrieval demo')
    parser.add_argument('command', choices=['prepare','demo','query'])
    parser.add_argument('--text')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if args.command=='prepare':
        print(json.dumps(prepare(root/'artifacts'), indent=2))
    else:
        from .demo import demo, query
        print(json.dumps(demo(root) if args.command=='demo' else query(root,args.text),indent=2))
    return 0
