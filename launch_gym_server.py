#!/usr/bin/env python
"""
Pytorch-cpp-rl OpenAI gym server main script.
"""
import logging

from gym_server.server import Server
from gym_server.zmq_client import ZmqClient
import pdp

def main():
    """
    Host the server.
    """
    # If anything is logged during imports, it messes up our logging so we
    # reset the logging module here
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG,
                        format=('%(asctime)s %(funcName)s '
                                '[%(levelname)s]: %(message)s'),
                        datefmt='%Y%m%d %H:%M:%S')
    logging.info("Initializing gym server")

    zmq_client = ZmqClient(10201)
    logging.info("Connecting to client")
    zmq_client.send("Connection established")
    logging.info("Connected")
    server = Server(zmq_client)

    try:
        server.serve()
    except Exception as e:
        # Option A: Just log and re-raise:
        import traceback
        logging.error("An exception occurred:\n%s", traceback.format_exc())
        raise

        # Option B: Or log and exit gracefully:
        logging.error("Error in server.serve(): %s", e)
        exit(1)



if __name__ == '__main__':
    main()
