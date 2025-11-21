import yaml
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys


KBET_LOOKUP_DF = pd.read_csv("kbet_lookup_table.csv")

def parse_yaml_data(file_path: Path) -> list:
    """Loads and flattens data from a single YAML file."""
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        return []
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {file_path}: {e}", file=sys.stderr)
        return []

    flat_data = []
    for entry in data:
        # Keep only the portion after the slash
        dataset = entry['dataset_id'].split('/')[-1] 
        method = entry['method_id']
        
        # Ensure metric_ids and metric_values exist and are lists of the same length
        if 'metric_ids' in entry and 'metric_values' in entry and \
           len(entry['metric_ids']) == len(entry['metric_values']):
            
            # Combine metric_ids and metric_values into (id, value) pairs
            for metric_id, metric_value in zip(entry['metric_ids'], entry['metric_values']):
                flat_data.append({
                    'Dataset ID': dataset,
                    'Metric ID': metric_id,
                    'Method ID': method,
                    'Metric Value': metric_value
                })
        else:
            print(f"Warning: Skipping malformed entry in {file_path}: {entry}", file=sys.stderr)

    return flat_data

def parse_yaml_data(file_path: Path) -> list:
    """Loads and flattens data from a single YAML file."""
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        return []
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {file_path}: {e}", file=sys.stderr)
        return []

    flat_data = []
    for entry in data:
        # Keep only the portion after the slash
        dataset = entry['dataset_id'].split('/')[-1] 
        method = entry['method_id']
        
        # Ensure metric_ids and metric_values exist and are lists of the same length
        if 'metric_ids' in entry and 'metric_values' in entry and \
           len(entry['metric_ids']) == len(entry['metric_values']):
            
            # Combine metric_ids and metric_values into (id, value) pairs
            for metric_id, metric_value in zip(entry['metric_ids'], entry['metric_values']):
                flat_data.append({
                    'Dataset ID': dataset,
                    'Metric ID': metric_id,
                    'Method ID': method,
                    'Metric Value': metric_value
                })
        else:
            print(f"Warning: Skipping malformed entry in {file_path}: {entry}", file=sys.stderr)

    return flat_data

def process_all_data(all_flat_data: list, lookup_df: pd.DataFrame) -> pd.DataFrame:
    """Processes all aggregated flat data, handles lookups, pivots, and finalizes."""
    
    if not all_flat_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_flat_data)
    
    # Replace the string 'nan' with a numerical NaN for proper handling
    df['Metric Value'] = df['Metric Value'].replace('nan', np.nan)
    
    # 3. Handle 'kbet' -1 lookups
    kbet_mask = (df['Metric ID'] == 'kbet') & (df['Metric Value'] == -1)
    kbet_to_lookup = df[kbet_mask].copy()

    if not kbet_to_lookup.empty:
        # Merge the -1 entries with the lookup table
        merged_kbet = pd.merge(
            kbet_to_lookup[['Dataset ID', 'Method ID', 'Metric Value']],
            lookup_df,
            left_on=['Dataset ID', 'Method ID'],
            right_on=['dataset', 'method'],
            how='left'
        )

        # Identify values found in the lookup table
        found_mask = merged_kbet['kbet_value'].notna()
        
        # Update the original DataFrame with the looked-up kbet values
        df.loc[kbet_to_lookup.index[found_mask], 'Metric Value'] = merged_kbet.loc[found_mask, 'kbet_value'].values
        
        # Print warning for -1 entries not found in the lookup table
        not_found_count = (~found_mask).sum()
        if not_found_count > 0:
            # Get the details of the missing lookups for a better warning
            missing_lookups = merged_kbet[~found_mask][['Dataset ID', 'Method ID']].drop_duplicates()
            print(f"Warning: {not_found_count} 'kbet' entries with value -1 were not found in the lookup table and will remain -1.")
            print("Missing (Dataset ID, Method ID) combinations:")
            print(missing_lookups.to_string(index=False), file=sys.stderr)


    # Convert all metric values to numeric (coerce will turn unhandled -1s into NaN temporarily)
    df['Metric Value'] = pd.to_numeric(df['Metric Value'], errors='coerce')
    
    # Replace the 'NaN's resulting from coercion of -1s back to the number -1
    # This ensures that non-looked-up -1s are treated as a numeric value -1
    df.loc[kbet_mask & ~found_mask, 'Metric Value'] = -1.0 
    
    # 4. Pivot the DataFrame
    pivot_df = df.pivot_table(
        index=['Metric ID', 'Dataset ID'], 
        columns='Method ID', 
        values='Metric Value', 
        aggfunc='first'
    ).reset_index()

    # 5. Replace any missing values (NaN) with zeros
    pivot_df = pivot_df.fillna(0)

    # 6. Reorder columns
    method_columns = [col for col in pivot_df.columns if col not in ['Metric ID', 'Dataset ID']]
    final_columns = ['Metric ID', 'Dataset ID'] + sorted(method_columns) # Sort method columns for consistency
    final_pivot_df = pivot_df[final_columns]
    
    return final_pivot_df

def main():
    """Main function to handle command-line arguments and orchestrate the process."""
    parser = argparse.ArgumentParser(
        description="Process a list of YAML result files into a pivoted CSV format."
    )
    parser.add_argument(
        'files', 
        metavar='FILE', 
        type=str, 
        nargs='+', 
        help='A list of YAML file paths to process.'
    )
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        default='results_output.csv', 
        help='The name of the output CSV file.'
    )
    
    args = parser.parse_args()

    all_flat_data = []
    print(f"Processing {len(args.files)} YAML file(s)...")

    # 1. Load and flatten data from all specified files
    for file_path_str in args.files:
        file_path = Path(file_path_str)
        if file_path.exists():
            print(f"  -> Loading {file_path_str}")
            all_flat_data.extend(parse_yaml_data(file_path))
        else:
            print(f"Skipping: File not found at '{file_path_str}'", file=sys.stderr)
    
    # 2. Process all aggregated data
    final_df = process_all_data(all_flat_data, KBET_LOOKUP_DF)

    # 3. Save the output
    if not final_df.empty:
        final_df.to_csv(args.output, index=False)
        print(f"\nSuccessfully processed data and saved to **{args.output}**")
        print("\n---")
    else:
        print("\nNo valid data was loaded or processed. Output file not created.")
    
    # Clean up dummy files after execution (optional, for testing only)
    Path('file1.yaml').unlink(missing_ok=True)
    Path('file2.yaml').unlink(missing_ok=True)


if __name__ == '__main__':
    # To run this script from your terminal, you would execute:
    # python your_script_name.py file1.yaml file2.yaml -o combined_results.csv
    main()