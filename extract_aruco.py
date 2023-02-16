import argparse
import cv2 as cv
from cv2 import aruco
import json
import numpy as np
from typing import Dict
from collections import defaultdict


class ArucoExtractor:
    def __init__(self, video_path: str):
        self.video_path: str = video_path
        self.extracted_markers: Dict[str, Dict[str, np.ndarray]] = defaultdict(defaultdict)
        self.aruco_dictionary: aruco.Dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters: aruco.DetectorParameters = aruco.DetectorParameters()

    def extract_markers(self) -> Dict[str, Dict[str, np.ndarray]]:
        video: cv.VideoCapture = cv.VideoCapture(self.video_path)
        current_frame = 0
        ret, frame = video.read()
        while ret:
            gray: cv.Mat = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            corners, ids, _ = aruco.detectMarkers(
                gray, self.aruco_dictionary, parameters=self.parameters)

            if len(corners) > 0:
                for idx, id in enumerate(ids):
                    self.extracted_markers[current_frame][id.item(
                    )] = corners[idx][0].tolist()

            current_frame += 1

            ret, frame = video.read()

        return self.extracted_markers


def save_markers(markers: Dict[str, Dict[str, np.ndarray]], output_path: str) -> None:

    with open(output_path, 'w') as out:
        json.dump(markers, out)


if __name__ == "__main__":
    arg_parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog='ArucoExtractor',
        description="""
            Goes through video specified by --input argument, extract aruco markers
            and saves those information to a json file specified by --output parameter"""
    )

    arg_parser.add_argument("-i", "--input-file", help="Input video to be processed", required=True, dest='input')
    arg_parser.add_argument("-o", "--output-file", help="File where results should be stored", required=True, dest='output')
    args = arg_parser.parse_args()

    extractor: ArucoExtractor = ArucoExtractor(args.input)
    markers: Dict[str, Dict[str, np.ndarray]] = extractor.extract_markers()

    save_markers(markers, args.output)
