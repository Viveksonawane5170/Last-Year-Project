"""
Custom exception handling for the whole project.
Every component (ingestion, transformation, trainer) wraps its risky code
in try/except and raises CustomException(e, sys) so that we always know
WHICH file and WHICH line an error came from — not just the raw traceback.
"""

import sys


def error_message_detail(error, error_detail: sys):
    """
    Builds a readable error message including file name and line number.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    error_message = (
        f"Error occurred in python script [{file_name}] "
        f"at line number [{line_number}] "
        f"error message: [{str(error)}]"
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message


# ---------------------------------------------------------------
# Usage pattern (use this in every component file):
#
# from src.exception import CustomException
# from src.logger import logging
# import sys
#
# try:
#     risky_code()
# except Exception as e:
#     logging.error("Something went wrong during X")
#     raise CustomException(e, sys)
# ---------------------------------------------------------------
