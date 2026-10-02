# encoding: utf-8

import astropy.io.fits as fits
import numpy as np
import datetime 


def nearest_file(file_list: list, start_date: str): 
    """
    given a list of files (file_list) and date string (start_date), 
    return the index of the nearest file in time 
    """
    time_strings = [file_list['files'][i]['date-obs'] for i in range(len(file_list['files']))]
    times = [datetime.datetime.fromisoformat(t) for t in time_strings]
    reference_time = datetime.datetime.fromisoformat(start_date)
    closest_index = min(
        range(len(times)), 
        key=lambda i: abs(times[i] - reference_time)
    )
    return closest_index 
