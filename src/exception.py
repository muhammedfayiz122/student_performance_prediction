import sys

def error_message_detail(error,  error_detail:sys):
    """
    It display where error occurs
    """
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename 
    #print those error details (file_name, line number, error_mssg)
    error_message = (
        f"\n\n\nError occured in : {file_name} "
        f"\nat line number : {exc_tb.tb_lineno} "
        f"\nwith error message : {str(error)}"
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message