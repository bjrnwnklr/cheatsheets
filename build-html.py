from pathlib import Path
import subprocess


if __name__ == '__main__':

    # definitions: root dir, suffix to search for, paths to output files and css file
    p = Path('.')
    md_suffix = '*.md'
    output_dir = p / 'html'
    css_file = p / 'css' / 'pandoc.css'

    # define pandoc command
    # pandoc_tmpl = 'pandoc {} --output {} --css {} --from markdown --to hmtl --standalone --highlight-style kate --toc --toc-depth 2'
    
    # get all md files
    files = list(p.glob(md_suffix))

    # go through each file and run the pandoc command
    for f in files:
        # pandoc_cmd = pandoc_tmpl.format(f.name, output_dir / f.with_suffix('.html'), css_file)
        pandoc_cmd = [
            'pandoc',
            f.name,
            '--output',
            str(output_dir / f.with_suffix('.html')),
            '--css',
            '../css/pandoc.css',
            '--from',
            'markdown',
            '--to',
            'html',
            '--standalone',
            '--highlight-style',
            'kate',
            '--toc',
            '--toc-depth',
            '2',
            '--mathjax'
        ]
        # print(pandoc_cmd)
        pandoc_output = subprocess.run(pandoc_cmd, capture_output=True)
        print(pandoc_output)