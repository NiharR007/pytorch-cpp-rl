"""
Pytorch-cpp-rl OpenAI gym server ZMQ client.
"""
import zmq
import msgpack
import msgpack_numpy as m  # Import msgpack_numpy
m.patch()  # Patch msgpack to handle numpy types
import logging
import numpy as np

class ZmqClient:
    """
    Provides a ZeroMQ interface for communicating with client.
    """

    def __init__(self, port: int):
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.bind(f"tcp://*:{port}")
        logging.info("ZmqClient initialized and bound to port %d", port)

    def receive(self) -> bytes:
        """
        Gets a message from the client.
        Blocks until a message is received.
        """
        message = self.socket.recv()
        logging.debug("Received raw message: %s", message)
        try:
            response = msgpack.unpackb(message, raw=False)
            logging.debug("Unpacked message: %s", response)
        except msgpack.exceptions.ExtraData as e:
            logging.error("ExtraData exception during unpacking: %s", e)
            response = message
        except Exception as e:
            logging.error("Unexpected exception during unpacking: %s", e)
            response = message
        return response

    def send(self, message: object):
        """
        Sends a message to the client.
        """
        logging.debug("Preparing to send data: %s", message)

        if isinstance(message, str):
            logging.debug("Sending string message: %s", message)
            self.socket.send_string(message)
        elif isinstance(message, bytes):
            logging.debug("Sending byte message: %s", message)
            self.socket.send(message)
        else:
            try:
                # With msgpack_numpy patched, we no longer need a custom encoder.
                packed_message = msgpack.packb(message, use_bin_type=True)
                logging.debug("Packed message: %s", packed_message)
                self.socket.send(packed_message)
                logging.info("Sent packed message successfully.")
            except Exception as e:
                logging.error("Error during serialization: %s", e)
                raise
