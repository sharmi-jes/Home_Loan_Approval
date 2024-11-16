import sys
import os
from src.logger import logging

# define a own exception function(to understand the where error occured)
def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occures in python script should be the file name[{0}] and file no [{1}] and error is [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    )

    return error_message

# create a class customexception
class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        # inherit the error_message from the parent
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message
    

if __name__=="__main__":
    try:
        a=1/0
    except Exception as e:
        logging.info("divided by zero")
        raise CustomException(e,sys)