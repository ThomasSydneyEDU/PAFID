import csv
import re
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
HUMAN_DATA_DIR = SCRIPT_DIR / "human_data"
DEFAULT_OUTPUT = SCRIPT_DIR / "extracted_survey_data.csv"
STIMULI_REF = SCRIPT_DIR / "food_survey_reference.csv"

def main():
    parser = argparse.ArgumentParser(
        description="Extract per-food responses from Qualtrics exports and attach image file names "
                    "via a stimuli reference CSV. If no input CSVs are given, auto-discovers "
                    "*_surveydata.csv in ./human_data/."
    )
    parser.add_argument(
        "input_csvs",
        nargs="*",
        help="Path(s) to the Qualtrics export CSV(s) to process. "
             "If omitted, all *_surveydata.csv files in ./human_data/ are used.",
    )
    parser.add_argument(
        "--out",
        dest="output_csv",
        default=str(DEFAULT_OUTPUT),
        help=f"Path to write the combined extracted output CSV (default: {DEFAULT_OUTPUT.name}).",
    )

    args = parser.parse_args()

    # Auto-discover input CSVs in human_data/ if none were provided on the CLI
    if not args.input_csvs:
        if not HUMAN_DATA_DIR.is_dir():
            sys.exit(
                f"[ERROR] No input CSVs given and human_data folder not found: {HUMAN_DATA_DIR}"
            )
        discovered = sorted(HUMAN_DATA_DIR.glob("*_surveydata.csv"))
        if not discovered:
            sys.exit(
                f"[ERROR] No input CSVs given and no *_surveydata.csv files found in {HUMAN_DATA_DIR}"
            )
        args.input_csvs = [str(p) for p in discovered]
        print(f"[INFO] Auto-discovered {len(args.input_csvs)} input CSV(s) in {HUMAN_DATA_DIR.name}/:")
        for p in args.input_csvs:
            print(f"        - {Path(p).name}")

    input_files = args.input_csvs
    output_file = args.output_csv
    stimuli_ref_file = str(STIMULI_REF)

    def load_stimuli_info(csv_path: str):
        """Load stimuli info from reference CSV.
        
        Returns:
            name_map: dict mapping ImageID -> ImageName
            easy_set: set of ImageIDs that are marked as 'isEasy' == 1
        """
        name_map = {}
        easy_set = set()
        
        if not os.path.exists(csv_path):
            print(f"[WARN] Stimuli reference file not found: {csv_path}. Proceeding without image names.")
            return name_map, easy_set
            
        try:
            # utf-8-sig handles BOM that sometimes appears in CSV exports
            with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader) # Skip header
                except StopIteration:
                    return name_map, easy_set

                for row in reader:
                    if len(row) >= 2:
                        name = row[0].strip()
                        ref = row[1].strip()
                        if ref:
                            name_map[ref] = name
                            
                        # Check for isEasy column (index 2)
                        if len(row) >= 3:
                            is_easy = row[2].strip()
                            if is_easy == '1':
                                easy_set.add(ref)
        except Exception as e:
            print(f"[WARN] Could not load stimuli reference file '{csv_path}': {e}. Proceeding without image names.")
        return name_map, easy_set

    # Mapping from Question Code to Output Column Name
    QUESTION_MAPPING = {
        'Q1_1': 'CalorieDensity',
        'Q2_1': 'Healthiness',
        'Q3_1': 'Appeal',
        'Q4': 'Familiarity',
        'Q17': 'FoodName',
        'Q24_1': 'Sweet',
        'Q24_2': 'Salty',
        'Q24_3': 'Sour',
        'Q24_4': 'Bitter',
        'Q24_5': 'Umami',
        'Q24_6': 'Fatty',
        'Q24_7': 'Spicy'
    }

    # Output columns: participant, food, image ID, image name, source, quality check, and all relevant question columns
    fieldnames = ['DataSource', 'ParticipantNumber', 'Time', 'QualityCheck', 'FoodItem', 'ImageID', 'ImageName', 'IsEasy'] + list(QUESTION_MAPPING.values())

    try:
        # Load the stimuli info
        image_name_map, easy_set = load_stimuli_info(stimuli_ref_file)
        if image_name_map:
            print(f"Loaded {len(image_name_map)} image names and {len(easy_set)} 'easy' items from '{stimuli_ref_file}'.")

        all_extracted_rows = []
        participant_quality = {} # {ResponseId: {'easy_seen': 0, 'easy_passed': 0}}
        participant_stats = {}   # {ResponseId: {'source': '', 'images': set(), 'time': ''}}
        
        # Mapping for anonymization
        participant_id_map = {}
        next_participant_id = 1

        # PASS 1: Read all files, extract rows, and calculate quality metrics
        for input_file in input_files:
            print(f"Processing file: {input_file}...")
            
            # Determine Source Label from filename (e.g., "MTURK_surveydata.csv" -> "MTURK")
            base = os.path.basename(input_file)
            source_label = base.split('_')[0].upper()
            
            if not os.path.exists(input_file):
                print(f"  [ERROR] File not found: {input_file}. Skipping.")
                continue

            try:
                with open(input_file, 'r', encoding='utf-8-sig') as infile:
                    reader = csv.reader(infile)
                    
                    try:
                        headers = next(reader)
                        descriptions = next(reader)
                        import_ids = next(reader) # Skip import IDs row
                    except StopIteration:
                        print(f"  [WARN] File {input_file} seems empty or malformed. Skipping.")
                        continue

                    try:
                        response_id_idx = headers.index('ResponseId')
                    except ValueError:
                        print(f"  [ERROR] 'ResponseId' column not found in {input_file}. Skipping.")
                        continue

                    # Try to find StartDate and EndDate columns
                    try:
                        start_date_idx = headers.index('StartDate')
                        end_date_idx = headers.index('EndDate')
                    except ValueError:
                        start_date_idx = -1
                        end_date_idx = -1
                        print(f"  [WARN] 'StartDate' or 'EndDate' column not found in {input_file}. Time calculation will be skipped.")

                    # Map headers to food items
                    food_map = {}
                    header_pattern = re.compile(r'^(\d+)_(Q\w+(?:_\d+)?)$')
                    image_id_pattern = re.compile(r'-\s*(IM_[A-Za-z0-9]+)\s*-')

                    for i, header in enumerate(headers):
                        match = header_pattern.match(header)
                        if match:
                            food_id = match.group(1)
                            question_code = match.group(2)

                            if question_code in QUESTION_MAPPING:
                                attribute_name = QUESTION_MAPPING[question_code]
                                
                                if food_id not in food_map:
                                    food_map[food_id] = {}
                                
                                food_map[food_id][attribute_name] = i

                                if 'ImageID' not in food_map[food_id]:
                                    desc = descriptions[i]
                                    img_match = image_id_pattern.search(desc)
                                    if img_match:
                                        food_map[food_id]['ImageID'] = img_match.group(1)

                    print(f"  Found {len(food_map)} unique food items in {input_file}.")

                    file_row_count = 0
                    for row in reader:
                        if len(row) <= response_id_idx:
                            continue
                            
                        response_id = row[response_id_idx]
                        if not response_id:
                            continue

                        # Calculate Time Duration
                        duration_str = ""
                        if start_date_idx != -1 and end_date_idx != -1 and len(row) > max(start_date_idx, end_date_idx):
                            s_str = row[start_date_idx]
                            e_str = row[end_date_idx]
                            if s_str and e_str:
                                try:
                                    # Expected format: DD/MM/YYYY HH:MM
                                    s_dt = datetime.strptime(s_str, "%d/%m/%Y %H:%M")
                                    e_dt = datetime.strptime(e_str, "%d/%m/%Y %H:%M")
                                    diff = e_dt - s_dt
                                    duration_str = str(diff)
                                except ValueError:
                                    # Try alternative format just in case (e.g. YYYY-MM-DD HH:MM:SS)
                                    try:
                                        s_dt = datetime.strptime(s_str, "%Y-%m-%d %H:%M:%S")
                                        e_dt = datetime.strptime(e_str, "%Y-%m-%d %H:%M:%S")
                                        diff = e_dt - s_dt
                                        duration_str = str(diff)
                                    except ValueError:
                                        pass # Could not parse date

                        # Initialize stats for this participant if new
                        if response_id not in participant_stats:
                            participant_stats[response_id] = {'source': source_label, 'images': set(), 'time': duration_str}
                            participant_quality[response_id] = {'easy_seen': 0, 'easy_passed': 0}
                        
                        # Assign Participant Number
                        if response_id not in participant_id_map:
                            participant_id_map[response_id] = next_participant_id
                            next_participant_id += 1

                        for food_id, attributes in food_map.items():
                            image_id = attributes.get('ImageID', '')
                            image_name = image_name_map.get(image_id, '')

                            out_row = {
                                'DataSource': source_label,
                                'ResponseId': response_id, # Kept for internal logic, not written
                                'ParticipantNumber': participant_id_map[response_id],
                                'Time': duration_str,
                                'FoodItem': food_id,
                                'ImageID': image_id,
                                'ImageName': image_name,
                                'IsEasy': 1 if image_id in easy_set else 0,
                                # QualityCheck will be filled in Pass 2
                            }
                            
                            has_data = False
                            familiarity_str = ''
                            
                            for attr_name in QUESTION_MAPPING.values():
                                col_idx = attributes.get(attr_name)
                                if col_idx is not None and col_idx < len(row):
                                    val = row[col_idx]
                                    out_row[attr_name] = val
                                    if val: 
                                        has_data = True
                                    if attr_name == 'Familiarity':
                                        familiarity_str = val
                                else:
                                    out_row[attr_name] = ''
                            
                            if has_data:
                                all_extracted_rows.append(out_row)
                                
                                # Update stats
                                if image_name:
                                    participant_stats[response_id]['images'].add(image_name)

                                # Update quality metrics
                                if image_id in easy_set:
                                    participant_quality[response_id]['easy_seen'] += 1
                                    try:
                                        match = re.match(r'^\s*(\d+)', familiarity_str)
                                        if match:
                                            score = int(match.group(1))
                                            if score > 2:
                                                participant_quality[response_id]['easy_passed'] += 1
                                    except Exception:
                                        pass
                        
                        file_row_count += 1
                        if file_row_count % 100 == 0:
                            print(f"  Processed {file_row_count} participants in current file...")
                    
                    print(f"  Finished {input_file}. Extracted rows from this file.")
            
            except Exception as e:
                print(f"  [ERROR] Error processing file {input_file}: {e}")
                import traceback
                traceback.print_exc()

        # PASS 2: Calculate final percentages and write to file
        print(f"\nWriting {len(all_extracted_rows)} rows to {output_file}...")
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()

            for row in all_extracted_rows:
                rid = row['ResponseId']
                
                # Calculate quality %
                seen = participant_quality[rid]['easy_seen']
                passed = participant_quality[rid]['easy_passed']
                pct = (passed / seen * 100) if seen > 0 else 0.0
                
                row['QualityCheck'] = f"{pct:.1f}"
                writer.writerow(row)

        print(f"Extraction complete. Total rows written: {len(all_extracted_rows)}.")
        
        # --- Summary Generation ---
        print("\n--- Summary ---")
        print(f"Total Unique Participants: {len(participant_stats)}")
        print("Ratings per Participant & Quality Check (Familiarity > 2 on 'Easy' items):")
        
        for rid, stats in participant_stats.items():
            src = stats['source']
            images = stats['images']
            time_val = stats.get('time', 'N/A')
            seen = participant_quality[rid]['easy_seen']
            passed = participant_quality[rid]['easy_passed']
            pct = (passed / seen * 100) if seen > 0 else 0.0
            
            print(f"  - {src}: {rid}: {len(images)} items rated | Time: {time_val} | Quality Check: {pct:.1f}% ({passed}/{seen} easy items passed)")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
