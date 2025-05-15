import polars as pl

if __name__ == "__main__":
    # Read DataFrames
    old_df = pl.read_csv("Organ.csv")
    df = pl.read_csv("Organ2.csv")

    # Get common columns
    common_cols = [
        "isBourdon",
        "flueDepth",
        "frequency",
        "cutUpHeight",
        "diameterToe",
        "acousticIntensity",
    ]

    # Convert types to be consistent across all DataFrames
    # Cast all numeric columns to float (to avoid Int64/Float64 mismatches)
    old_df = old_df.select([pl.col(col).cast(pl.Float64) for col in old_df.columns])

    max_df = df.select(
        [pl.col(col).cast(pl.Float64) for col in common_cols]
        + [
            pl.col(f"maxPartial{i}").alias(f"partial{i}").cast(pl.Float64)
            for i in range(1, 9)
        ]
    )

    min_df = df.select(
        [pl.col(col).cast(pl.Float64) for col in common_cols]
        + [
            pl.col(f"minPartial{i}").alias(f"partial{i}").cast(pl.Float64)
            for i in range(1, 9)
        ]
    )

    # Concatenate DataFrames
    result_df = pl.concat([old_df, max_df, min_df], how="vertical")

    # Write output
    output_file = "allOrgan.csv"
    result_df.write_csv(output_file)

    print(f"Processing complete. Output saved to {output_file}")
    print(
        f"Original rows: {len(old_df)} + {len(df)} (duplicated), Output rows: {len(result_df)}"
    )
