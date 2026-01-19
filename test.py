import sys, os
from custom_excpetion import SentimentException
from logger import get_logger


logger = get_logger(__name__)

def divide(a, b):

    try:
        if b == 0:
            raise ZeroDivisionError("Division by Zero is not allowed")
        res = a / b

        logger.info(f"Division is Succesfull : {a} / {b} = {res}")

        return res
    except ZeroDivisionError as e:
        logger.error(f"Failed to divide {a} by {b}", exc_info=True)
        raise SentimentException(
            error_message= f"Cannot do this division: {str(e)}",
            error_detail= sys.exc_info()
        ) from e
    except Exception as e:
        logger.exception("Error occurred")
        raise SentimentException(
            error_message= f"Something wrong during calculation",
            error_detail= sys.exc_info()
        ) from e


if __name__ == "__main__":

    logger.info("Application started")

    try:
        print("50 / 4",divide(10,2))

        print("6 / 0", divide(6,0))
    except SentimentException as e:
        logger.error("It is due to error: {} ".format(e))
    finally:
        logger.info("Application end")