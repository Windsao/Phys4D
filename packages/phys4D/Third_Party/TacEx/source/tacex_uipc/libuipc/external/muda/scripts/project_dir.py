import pathlib as pl

def project_dir():

    this_file = pl.Path(__file__).resolve()
    return this_file.parent.parent