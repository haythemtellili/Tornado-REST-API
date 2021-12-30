import ssl
import json
import logging
import time
from io import BytesIO
from datetime import datetime
import numpy as np
import cv2

from PIL import Image, UnidentifiedImageError
from tornado.ioloop import IOLoop
from tornado.options import define, options
from tornado.web import Application, RequestHandler
from tornado.httpclient import AsyncHTTPClient, HTTPError

# file with functions related to model initialization and prediction
from utils import initialize_model, predict_one_sample

ssl._create_default_https_context = ssl._create_unverified_context
logging.basicConfig(
    level=logging.INFO,
    filename="model_service.log",
    filemode="a",
    format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
)
logger = logging.getLogger(__name__)

http_client = AsyncHTTPClient(defaults=dict(request_timeout=1000, connect_timeout=1000))

define("port", default=8001, help="run on the given port")
define(
    "detection_model", default="Mask_rcnn_model_v2.pt", help="model to segment objects"
)
define("iou_threshold", default=0.2, help="IOU threshold")
define("mask_thresh", default=0.2, help="Mask threshold")
define(
    "classification_confidence_threshold",
    default=0.7,
    help="classification confidence threshold",
)


def make_app(args):
    """
    Function initializing tornado application and its request handlers.

    initialize_model() - is called once on service start and initializes model and additional arguments needed for
                         inference. Returns a dictionary which is supplied as keyword arguments to initialize().
    """
    handlers = [
        (r"/predict/bytes", PredictHandler, initialize_model(args)),
        (r"/predict/url", PredictHandler, initialize_model(args)),
    ]

    return Application(handlers)


class PredictHandler(RequestHandler):
    def initialize(self, **model_with_arguments):
        """
        Tornado's hook for handler’s initialization. Called for each request.

        :param initialized_model: dictionary with initialized model and other arguments

        :return: None
        """
        self.model_with_arguments = model_with_arguments

        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/octet-stream",
            "Authorization": "Basic rBGyxNZiD/SRpKjvHidGBgHbD8GoHccamX4tdbxR5LQPShnFq6sy/KlqkMWtDTXtUt1I9fW6gq6jB7Pq8jC4Dg==",
        }

    def get(self):
        logger.info(f"\n\get request: {datetime.now()}")
        self.write("get call \n")
        self.write("Hi from Docker, support defect")

    async def post(self):
        logger.info(f"\n\nPost request: {datetime.now()}")

        # read request data
        request_data = self.request.body
        # get predictions
        predictions = await self.get_predictions(request_data)
        logger.info(predictions)
        self.write(json.dumps(predictions))

    async def get_predictions(self, request_data):
        logger.info(f"\n\nPost request: {datetime.now()}")

        tic_first = time.perf_counter()
        # getting image from url
        # image = await self.async_load_image_from_url(request_data['Image'])
        # getting image from bytes
        image = self.load_image_from_bytes(request_data)
        preds = predict_one_sample(image, **self.model_with_arguments)
        tic_second = time.perf_counter()
        req_handle_time = tic_second - tic_first
        logger.info(f"Request handle time: {req_handle_time} second(s).")

        response_data = {
            "id": "1",
            "project": "Mastec",
            "iteration": "1",
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "predictions": preds["predictions"],
        }

        return response_data

    @staticmethod
    def load_image_from_bytes(img_bytes):
        """
                Reads an image that is represented as bytes.
                INPUT:
            img_bytes: (bytes) image represented as bytes (e.g., r.content where r is a request)
        RETURNS:
            image: (numpy.ndarray) Numpy array (H,W,C) H: height, W: width, C: channels (=3)
                """
        with BytesIO(img_bytes) as f:
            contents = f.read()
        logger.info("Hey!")
        logger.info(type(contents))

        image = np.asarray(bytearray(contents), dtype="uint8")
        logger.info(type(image))
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image

    @staticmethod
    async def async_load_image_from_url(url, http_client=http_client):
        """
        :param url: (string)
        :return image: (PIL.Image)

        Reads an image from a URL
        """
        try:
            requested_image = await http_client.fetch(url)
            logger.info(
                f"Get image by URL request time: {requested_image.request_time} second(s)."
            )
            image = Image.open(requested_image.buffer).convert("RGB")
            return image
        except HTTPError as e:
            logger.info(f"Exception occured with url: {url}")
            return e
        except UnidentifiedImageError as e:
            logger.info(f"Exception occured with url: {url}")
            return e
        except ConnectionResetError as e:
            logger.info(f"Exception occured with url: {url}")
            return e
        except Exception as e:
            return e


if __name__ == "__main__":
    options.parse_command_line()
    print(f"Application is running on port: {options.port}")

    app = make_app(options)
    app.listen(options.port)
    ioloop_instance = IOLoop.instance()
    ioloop_instance.start()
