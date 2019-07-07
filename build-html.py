from pathlib import Path


if __name__ == '__main__':

    p = Path('.')
    ext = '*.md'
    output_dir = p / 'html'
    css_file = p / 'css' / 'pandoc.css'

    # define pandoc command
    pandoc_tmpl = 'pandoc {} --output {} --css {} --from markdown --to hmtl --standalone --highlight-style kate --toc --toc-depth 2'
    
    # get all md files
    files = list(p.glob(ext))

    for f in files:
        pandoc_cmd = pandoc_tmpl.format(f.name, f.stem + '.html', css_file)
        print(pandoc_cmd)