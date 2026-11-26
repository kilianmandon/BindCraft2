import sys

from bindcraft.cli import main


if __name__=='__main__':
    args = [
    'design', "dev/test_config.json", 
    '--core', 'benchmark' ,
    '--set', 'project_folder=dev/runs/test_run',
    ]
    sys.exit(main(args))