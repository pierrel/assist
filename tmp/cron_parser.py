#!/usr/bin/env python3

from datetime import datetime
from croniter import croniter
from croniter import CronError


def validate_cron_expression(cron_expr):
    """
    Validates the cron expression format.
    
    Args:
        cron_expr (str): The cron expression to validate.
        
    Returns:
        bool: True if the cron expression is valid, False otherwise.
    """
    if not cron_expr or len(cron_expr.split()) != 5:
        return False
    
    # Check for basic syntax (e.g., no empty fields, no extra characters)
    fields = cron_expr.split()
    for field in fields:
        if not field:
            return False
        # Basic check for common cron patterns
        if not any(
            char in field for char in ['*', '/', '-', ',']
        ):
            # Allow simple numbers or ranges
            if not (field.isdigit() or ('-' in field and field.replace('-', '').isdigit())):
                return False
    
    return True


def get_next_cron_time(cron_expr):
    """
    Calculates the next execution time for a given cron expression.
    
    Args:
        cron_expr (str): The cron expression to parse.
        
    Returns:
        datetime: The next execution time as a datetime object.
        str: Error message if the cron expression is invalid.
    """
    if not validate_cron_expression(cron_expr):
        return "Invalid cron expression: '{}'".format(cron_expr)
    
    try:
        cron = croniter(cron_expr, datetime.now())
        next_time = cron.get_next_datetime()
        return next_time
    except CronError as e:
        return "Error parsing cron expression: {}".format(str(e))


if __name__ == "__main__":
    # Example usage
    print("Testing cron expression parser...")
    
    # Valid cron expressions
    cron_expressions = [
        "* * * * *",  # Every minute
        "0 9 * * *",  # Every day at 9 AM
        "*/5 * * * *",  # Every 5 minutes
        "0 0 * * 0",  # Every Sunday at midnight
    ]
    
    for expr in cron_expressions:
        next_time = get_next_cron_time(expr)
        if isinstance(next_time, datetime):
            print(f"Cron expression: {expr}")
            print(f"Next execution time: {next_time}")
            print("-" * 50)
        else:
            print(f"Cron expression: {expr}")
            print(f"Error: {next_time}")
            print("-" * 50)
    
    # Invalid cron expression
    invalid_expr = "0 9 * *"  # Missing field
    next_time = get_next_cron_time(invalid_expr)
    print(f"Cron expression: {invalid_expr}")
    print(f"Error: {next_time}")