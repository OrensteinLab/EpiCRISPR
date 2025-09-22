# Adding Flanking Sequence to Off-Target Data

This tool extracts **strand-aware upstream and downstream flanking sequences** for genomic intervals (e.g. CRISPR off-target sites) using per-chromosome reference FASTA files.

## Input Requirements

- A CSV file containing off-target genomic coordinates.
  - Must include the following columns:
    - `Index` — unique row identifier
    - Chromosome (e.g., `chrom`)
    - Start position (e.g., `chromStart`)
    - End position (e.g., `chromEnd`)
    - Strand (`+` or `-`)
- A folder containing per-chromosome FASTA files:
  - Files named `chr1.fa`, `chr2.fa`, ..., `chrX.fa`, `chrY.fa`, `chrM.fa`
  - Each `.fa` file must be **indexed** with samtools:

    ```bash
    samtools faidx chr1.fa
    ```

## Usage

```bash
python extract_flanks.py input.csv /path/to/genome_fasta \
  --chrom_col chrom \
  --start_col chromStart \
  --end_col chromEnd \
  --strand_col strand \
  --flank_size 20 \
  --output_csv output_with_flanks.csv
```

### Notes:
- If your CSV uses default column names (`chrom`, `chromStart`, `chromEnd`, `strand`, and `Index`), you can omit the column name arguments.
- To enable parallel processing, add the `--multiprocess` flag:

  ```bash
  --multiprocess
  ```

- If you **omit `--multiprocess`**, the script will run serially by default.

## Example

```bash
python extract_flanks.py off_targets.csv ./hg38_chr_fa \
  --multiprocess \
  --output_csv off_targets_with_flanks.csv
```

## Output

The script outputs a CSV file with two new columns:
- `upstream` — 20 bp upstream sequence (strand-aware)
- `downstream` — 20 bp downstream sequence (strand-aware)

## Genome FASTA Folder Structure

```
/path/to/genome_fasta/
├── chr1.fa
├── chr1.fa.fai
├── chr2.fa
├── chr2.fa.fai
├── ...
├── chrX.fa
├── chrY.fa
```

Make sure each `.fa` is indexed using:

```bash
samtools faidx chrN.fa
```

---

Created for high-throughput flanking sequence annotation of CRISPR off-target datasets.
