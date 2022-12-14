date_drive_to_seq_mapping = {
    # Date         # Drive  # Seq  #Start seq # End seq
    "2011_10_03_drive_0027" : "00", #000000 004540
    "2011_10_03_drive_0042" : "01", #000000 001100
    "2011_10_03_drive_0034" : "02", #000000 004660
    "2011_09_26_drive_0067" : "03", #000000 000800
    "2011_09_30_drive_0016" : "04", #000000 000270
    "2011_09_30_drive_0018" : "05", #000000 002760
    "2011_09_30_drive_0020" : "06", #000000 001100
    "2011_09_30_drive_0027" : "07", #000000 001100
    "2011_09_30_drive_0028" : "08", #001100 005170
    "2011_09_30_drive_0033" : "09", #000000 001590
    "2011_09_30_drive_0034" : "10"  #000000  001200
}

seq_to_date_mapping = {
    # Date         # Drive  # Seq  #Start seq # End seq
    "00": "2011_10_03", #000000 004540
    "01": "2011_10_03", #000000 001100
    "02": "2011_10_03", #000000 004660
    "03": "2011_09_26", #000000 000800
    "04": "2011_09_30", #000000 000270
    "05": "2011_09_30", #000000 002760
    "06": "2011_09_30", #000000 001100
    "07": "2011_09_30", #000000 001100
    "08": "2011_09_30", #001100 005170
    "09": "2011_09_30", #000000 001590
    "10": "2011_09_30"  #000000  001200
}

seq_to_drive_mapping = {
    # Date         # Drive  # Seq  #Start seq # End seq
    "00": "0027", #000000 004540
    "01": "0042", #000000 001100
    "02": "0034", #000000 004660
    "03": "0067", #000000 000800
    "04": "0016", #000000 000270
    "05": "0018", #000000 002760
    "06": "0020", #000000 001100
    "07": "0027", #000000 001100
    "08": "0028", #001100 005170
    "09": "0033", #000000 001590
    "10": "0034"  #000000  001200
}

seq_to_start_idx = {
    "00": 0,
    "01": 0,
    "02": 0,
    "03": 0,
    "04": 0,
    "05": 0,
    "06": 0,
    "07": 0,
    "08": 1100,
    "09": 0,
    "10": 0
}

seq_to_end_idx = {
    "00": 4540,
    "01": 1100,
    "02": 4660,
    "03": 800,
    "04": 270,
    "05": 2760,
    "06": 1100,
    "07": 1100,
    "08": 5170,
    "09": 1590,
    "10": 1200
}