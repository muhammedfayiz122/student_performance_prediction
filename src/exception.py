import sys

def error_message_detail(error,  error_detail:sys):
    """
    """
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename 
    #print those error details (file_name, line number, error_mssg)
    error_message = f"Error occured in {file_name} at line number {exc_tb.tb_lineno} with error message {str(error)}"
    return error

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

        def __str__(self):
            return self.error_message