from langchain.agents import ExceptionTool
from langchain.runnables import Runnable

# Define your main runnable function

def my_runnable(input):
    # Simulate some processing
    if input == "error":
        raise ValueError("Simulated error!")
    return "Processed successfully!"

# Create an ExceptionTool instance

exception_tool = ExceptionTool(
    fallbacks=[Runnable.from_function(my_runnable)],
    exceptions_to_handle=(BaseException,)
)

# Call the tool
try:
    result = exception_tool.run("error")
    print(result)
except BaseException as e:
    print(f"An error occurred: {e}")
