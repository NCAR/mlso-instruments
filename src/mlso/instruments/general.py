# encoding: utf-8

import datetime

def nearest_file(file_list: list[dict], start_date: str):
    """
    Determines the index of the nearest file in a provided list to the given start date. 

    Parameters 
    ----------
    file_list: list of dictionaries 
        Generated using MLSO api's client functionality 
    start_date: string 
        Time in which you want the closest file, in isoformat "yyyy-mm-ddThh:mm:ss" 
   
    Returns 
    -------
    closest_index: integer 
        Index in file_list that has observation time closest to start_date 
    """
    time_strings = [file_list['files'][i]['date-obs'] for i in range(len(file_list['files']))]
    times = [datetime.datetime.fromisoformat(t) for t in time_strings]
    reference_time = datetime.datetime.fromisoformat(start_date)
    closest_index = min(
        range(len(times)),
        key=lambda i: abs(times[i] - reference_time)
    )
    return closest_index
