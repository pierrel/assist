#!/usr/bin/env python3
"""
Summarize test results from JUnit XML files in edd/history directory.
Outputs a CSV with test results across multiple runs.
"""
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET


def parse_timestamp_from_filename(filename):
    """Extract timestamp from filename like 'results-20260204-2300.xml'"""
    try:
        parts = filename.replace('.xml', '').split('-')
        if len(parts) >= 3:
            date_str = parts[1]  # YYYYMMDD
            time_str = parts[2]  # HHMM
            dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M")
            return dt
    except (ValueError, IndexError):
        return None


def extract_error_type(error_message):
    """Extract just the error type from an error message"""
    # Split on colon to get the error type
    if ':' in error_message:
        error_type = error_message.split(':')[0].strip()
        return error_type
    else:
        # If no colon, return the whole message (might be just error type)
        return error_message.strip()


def parse_xml_file(filepath):
    """Parse a JUnit XML file and extract test results"""
    tree = ET.parse(filepath)
    root = tree.getroot()

    results = []

    # Find all testcase elements
    for testcase in root.findall('.//testcase'):
        classname = testcase.get('classname', '')
        name = testcase.get('name', '')

        # Construct full test name
        if classname:
            full_name = f"{classname}.{name}"
        else:
            full_name = name

        # Check if test passed or failed
        error = testcase.find('error')
        failure = testcase.find('failure')

        if error is not None:
            # Extract error message and get just the error type
            error_msg = error.get('message', 'Error occurred')
            result = extract_error_type(error_msg)
        elif failure is not None:
            # Extract failure message and get just the error type
            failure_msg = failure.get('message', 'Test failed')
            result = extract_error_type(failure_msg)
        else:
            # Test passed
            result = 'PASS'

        results.append({
            'test_name': full_name,
            'result': result
        })

    return results


def main():
    # Find the edd/history directory
    script_dir = Path(__file__).parent.parent
    history_dir = script_dir / 'edd' / 'history'

    if not history_dir.exists():
        print(f"Error: Directory {history_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Get all XML files sorted by timestamp
    xml_files = []
    for filepath in history_dir.glob('results-*.xml'):
        timestamp = parse_timestamp_from_filename(filepath.name)
        if timestamp:
            xml_files.append((timestamp, filepath))

    xml_files.sort()  # Sort by timestamp

    # Take the last 10 (or fewer)
    xml_files = xml_files[-10:]

    if not xml_files:
        print("Error: No XML files found in edd/history", file=sys.stderr)
        sys.exit(1)

    # Parse all files and collect results
    test_results = defaultdict(dict)  # test_name -> {timestamp -> result}
    timestamps = []

    for timestamp, filepath in xml_files:
        timestamps.append(timestamp)
        results = parse_xml_file(filepath)

        for result in results:
            test_name = result['test_name']
            test_results[test_name][timestamp] = result['result']

    # Calculate failure rates
    test_failure_rates = {}
    for test_name, runs in test_results.items():
        total_runs = len(runs)
        failures = sum(1 for result in runs.values() if result != 'PASS')
        failure_rate = (failures / total_runs * 100) if total_runs > 0 else 0
        test_failure_rates[test_name] = failure_rate

    # Write CSV
    output_file = script_dir / 'test_summary.csv'

    with open(output_file, 'w', newline='') as csvfile:
        # Create header
        header = ['Test Name']
        for ts in timestamps:
            header.append(ts.strftime('%Y-%m-%d %H:%M'))
        header.append('Failure Rate (%)')

        writer = csv.writer(csvfile)
        writer.writerow(header)

        # Write each test
        for test_name in sorted(test_results.keys()):
            row = [test_name]

            # Add result for each timestamp
            for ts in timestamps:
                result = test_results[test_name].get(ts, '')
                row.append(result)

            # Add failure rate
            row.append(f"{test_failure_rates[test_name]:.1f}")

            writer.writerow(row)

    print(f"Summary written to {output_file}")
    print(f"Analyzed {len(xml_files)} test runs")
    print(f"Found {len(test_results)} unique tests")


if __name__ == '__main__':
    main()
