# -*- coding: utf-8 -*-

from libdalton import fileio
from libdalton import analyze

if __name__=='__main__':
    input_file_name = fileio.validate_input(__file__)
    
    analysis = analyze.Analysis(input_file_name)
    
    analysis.run()