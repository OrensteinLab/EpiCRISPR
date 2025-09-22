import pandas as pd
import pysam
import os
import argparse
from multiprocessing import Pool, cpu_count

def reverse_complement(seq):
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]

def process_chromosome(args):
    chrom, group_df, fasta_folder, flank_size, cols = args
    chrom_col, start_col, end_col, strand_col = cols
    fasta_path = os.path.join(fasta_folder, f"{chrom}.fa")

    group_df = group_df.copy()
    group_df['upstream'] = None
    group_df['downstream'] = None

    if not os.path.exists(fasta_path):
        print(f"WARNING: FASTA file not found for {chrom} — skipping")
        return group_df
    else: print(f'Found fasta file for chrom: {chrom}\n:{fasta_path}')

    fasta = pysam.FastaFile(fasta_path)

    for i, row in group_df.iterrows():
        start = int(row[start_col])
        end = int(row[end_col])
        strand = row[strand_col]

        try:
            if strand == "+":
                upstream = fasta.fetch(chrom, max(0, start - flank_size), start).upper()
                downstream = fasta.fetch(chrom, end, end + flank_size).upper()
            elif strand == "-":
                upstream = reverse_complement(fasta.fetch(chrom, end, end + flank_size)).upper()
                downstream = reverse_complement(fasta.fetch(chrom, max(0, start - flank_size), start)).upper()
            else:
                upstream = downstream = None
        except Exception as e:
            print(f"Error fetching {chrom}:{start}-{end} on strand {strand}: {e}")
            upstream = downstream = None

        group_df.at[i, 'upstream'] = upstream
        group_df.at[i, 'downstream'] = downstream

    fasta.close()
    return group_df

def get_flank_sequences(
    df,
    fasta_folder,
    chrom_col='chrom',
    start_col='chromStart',
    end_col='chromEnd',
    strand_col='strand',
    flank_size=20,
    n_threads=None,
    use_multiprocessing=True
):
    """
    Extracts upstream and downstream DNA sequences from per-chromosome FASTA files
    for genomic intervals in a DataFrame.

    This function supports strand-aware extraction and parallel processing. It adds
    two new columns to the DataFrame: 'upstream' and 'downstream', each containing
    the flanking sequences (in uppercase) relative to the strand.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing genomic coordinates and a unique 'Index' column.
    fasta_folder : str
        Path to the directory containing per-chromosome FASTA files (e.g., chr1.fa, chrX.fa).
    chrom_col : str, optional
        Column name for chromosome (default: 'chrom').
    start_col : str, optional
        Column name for the start coordinate (default: 'chromStart').
    end_col : str, optional
        Column name for the end coordinate (default: 'chromEnd').
    strand_col : str, optional
        Column name for strand information ('+' or '-') (default: 'strand').
    flank_size : int, optional
        Number of base pairs to extract upstream and downstream of the interval (default: 20).
    n_threads : int or None, optional
        Number of parallel processes to use (default: all but one core if multiprocessing is enabled).
    use_multiprocessing : bool, optional
        Whether to parallelize processing across chromosomes (default: True).

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with two additional columns:
        - 'upstream': 20 bp upstream of the interval (strand-aware)
        - 'downstream': 20 bp downstream of the interval (strand-aware)
        Row order and original 'Index' values are preserved.
    """

    required = {chrom_col, start_col, end_col, strand_col, 'Index'}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain: {required}")

    df = df.set_index("Index", drop=False)
    grouped = df.groupby(chrom_col)
    args_list = [
        (chrom, group_df, fasta_folder, flank_size, (chrom_col, start_col, end_col, strand_col))
        for chrom, group_df in grouped
    ]

    if use_multiprocessing:
        if n_threads is None:
            n_threads = max(1, cpu_count() - 1)
        print(f"Processing {len(args_list)} chromosomes with {n_threads} threads...")
        with Pool(n_threads) as pool:
            result_parts = pool.map(process_chromosome, args_list)
    else:
        print(f"Processing {len(args_list)} chromosomes serially...")
        result_parts = [process_chromosome(args) for args in args_list]

    result_df = pd.concat(result_parts).set_index("Index", drop=False)
    result_df = result_df.loc[df.index]
    return result_df

def main():
    parser = argparse.ArgumentParser(description="Extract upstream/downstream sequences from chromosome FASTAs.")

    parser.add_argument("input_csv", help="Path to input CSV with Index and coordinate columns")
    parser.add_argument("fasta_folder", help="Path to folder containing chr*.fa files")
    parser.add_argument("--chrom_col", default="chrom", help="Column name for chromosome")
    parser.add_argument("--start_col", default="chromStart", help="Column name for start coordinate")
    parser.add_argument("--end_col", default="chromEnd", help="Column name for end coordinate")
    parser.add_argument("--strand_col", default="strand", help="Column name for strand")
    parser.add_argument("--multiprocess", action="store_true", help="Enable multiprocessing")
    parser.add_argument("--flank_size", type=int, default=20, help="Number of bases upstream/downstream to extract")
    parser.add_argument("--output_csv", default="output_with_flanks.csv", help="Output file name")

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    # Ensure types are correct
    df[args.start_col] = df[args.start_col].astype(int)
    df[args.end_col] = df[args.end_col].astype(int)

    if 'Index' not in df.columns:
        df['Index'] = df.index

    df = get_flank_sequences(
        df,
        fasta_folder=args.fasta_folder,
        chrom_col=args.chrom_col,
        start_col=args.start_col,
        end_col=args.end_col,
        strand_col=args.strand_col,
        flank_size=args.flank_size,
        use_multiprocessing=args.multiprocess
    )

    df.to_csv(args.output_csv, index=False)
    print(f"Done. Output written to {args.output_csv}")

if __name__ == "__main__":
    main()
