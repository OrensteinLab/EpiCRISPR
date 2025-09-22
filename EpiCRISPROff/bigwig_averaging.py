# BigWig methylation #
import pyBigWig
import sys
import numpy as np
import os
import argparse
import re
def average_bigwig_signals(bw_files, output_path, output_name):
    """
    Averages per-base signal values from multiple BigWig files and writes the result
    to a new BigWig file. Input files can have different chromosome sets and lengths.

    Parameters
    ----------
    bw_files : list of str
        List of input BigWig file paths to average.
    output_path : str
        Path to directory where the output BigWig will be saved.
    output_name : str
        Name of the output BigWig file (e.g., "average.bw").

    Returns
    -------
    output_file : str
        Full path to the created output BigWig file.
    """
    bws = [pyBigWig.open(f) for f in bw_files]

    # Union of all chromosomes and their maximum lengths
    all_chroms = {}
    for bw in bws:
        for chrom, length in bw.chroms().items():
            all_chroms[chrom] = max(all_chroms.get(chrom, 0), length)

    # Create output file
    if not re.search(r'\.(bw|bigwig)$', output_name, re.IGNORECASE):
        output_name += ".bw"
    output_file = os.path.join(output_path, output_name)
    out = pyBigWig.open(output_file, "w")
    out.addHeader(list(all_chroms.items()))

    # Iterate over chromosomes
    for chrom, length in all_chroms.items():
        all_vals = []

        for bw in bws:
            if chrom in bw.chroms():
                vals = np.array(bw.values(chrom, 0, length))
                vals = np.nan_to_num(vals, nan=0.0)
            else:
                vals = np.zeros(length)
            all_vals.append(vals)

        # Average
        stacked = np.stack(all_vals, axis=0)
        nonzero_mask = np.any(stacked != 0, axis=0)
        mean_vals = np.mean(stacked, axis=0)

        # Compress to intervals
        starts, ends, scores = [], [], []
        i = 0
        while i < length:
            if nonzero_mask[i]:
                start = i
                score = mean_vals[i]
                while i < length and nonzero_mask[i] and mean_vals[i] == score:
                    i += 1
                starts.append(start)
                ends.append(i)
                scores.append(score)
            else:
                i += 1

        if starts:
            out.addEntries([chrom]*len(starts), starts, ends=ends, values=scores)

    # Close all files
    for bw in bws:
        bw.close()
    out.close()
    return output_file
def main():
    parser = argparse.ArgumentParser(
        description="Average per-base signal from multiple BigWig files."
    )
    parser.add_argument(
        "bw_files", nargs="+", help="List of BigWig files to average"
    )
    parser.add_argument(
        "--output_path", required=True, help="Directory to save the output BigWig file"
    )
    parser.add_argument(
        "--output_name", required=True, help="Name of the output BigWig file (e.g., average.bw)"
    )

    args = parser.parse_args()

    # Get list of BigWig files
    bw_files = [
        f for f in args.bw_files
        if os.path.isfile(f) and re.search(r'\.(bw|bigwig)$', f, re.IGNORECASE)
    ]

    if not bw_files:
        print("No valid BigWig (.bw or .bigWig) files provided.", file=sys.stderr)
        sys.exit(1)


    print(f"Found {len(bw_files)} BigWig files. Averaging...")
    output_file = average_bigwig_signals(bw_files, args.output_path, args.output_name)
    print(f"Averaged BigWig written to: {output_file}")

if __name__ == "__main__":
    main()