Data structure as follows:
on_seq, off_seq, label with header.

Enconding the CSV file before:
`python Csv2pkl.py sample.csv`
This will output an `<file>_sci3_dim7.pkl` file.

to predict:
`python MFH_script.py sample_sci3_dim7.pkl`
this will output prediction file: `CRISPR-MFH_<file>.csv`

NOTE: tensorflow==2.15 should be installed!