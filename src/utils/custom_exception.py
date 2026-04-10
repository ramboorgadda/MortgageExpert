import sys

class CustomException(Exception):
    def __init__(self, message: str, error_detail: Exception = None):
        self.error_message = self.get_detailed_error_message(message, error_detail)
        super().__init__(self.error_message)
        

    def __str__(self):
        return f"CustomException (Error Detail: {self.error_message}): {super().__str__()}"
    @staticmethod
    def get_detailed_error_message(message: str,error_detail: Exception = None):
        exc_tb = getattr(error_detail, "__traceback__", None)
        if exc_tb is None:
            _, _, exc_tb = sys.exc_info()
        if exc_tb is None:
            return message
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = message
        return f"Error occurred in file: {file_name} at line: {line_number} with message: {error_message}"