# This file is metrics of recall using LocalizationPipeline
import numpy as np
from geopy.distance import geodesic
from tqdm import tqdm

from aero_vloc.primitives.uav_seq import UAVSeq
from aero_vloc.retrieval_system import RetrievalSystem
from aero_vloc.localization_pipeline import LocalizationPipeline

RECALL_LIST = [1, 2, 5, 10, 20]


# Calculates the distance between two geocoordinates in meters
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates the distance between two geocoordinates in meters
    """
    return geodesic((lat1, lon1), (lat2, lon2)).meters



def retrieval_recall(
    uav_seq: UAVSeq,
    retrieval_system: RetrievalSystem,
    vpr_k_closest: int,
    feature_matcher_k_closest: int | None,
) -> np.ndarray:
    """
    The metric finds the number of correctly matched frames based on retrieval results

    :param uav_seq: Sequence of UAV images
    :param retrieval_system: Instance of RetrievalSystem class
    :param vpr_k_closest: Determines how many best images are to be obtained with the VPR system
    :param feature_matcher_k_closest: Determines how many best images are to be obtained with the feature matcher
    If it is None, then the feature matcher turns off

    :return: Array of Recall values for all N < vpr_k_closest,
             or for all N < feature_matcher_k_closest if it is not None
    """
    if feature_matcher_k_closest is not None:
        recalls = np.zeros(feature_matcher_k_closest)
    else:
        recalls = np.zeros(vpr_k_closest)
    for idx, uav_image in enumerate(tqdm(uav_seq, desc="Calculating of recall")):
        predictions, _, _ = retrieval_system(
            uav_image, vpr_k_closest, feature_matcher_k_closest
        )
        for i, prediction in enumerate(predictions):
            map_tile = retrieval_system.sat_map[prediction]
            
            if (
                map_tile.top_left_lat
                > uav_image.gt_latitude
                > map_tile.bottom_right_lat
            ) and (
                map_tile.top_left_lon
                < uav_image.gt_longitude
                < map_tile.bottom_right_lon
            ):
                recalls[i:] += 1
                break

    retrieval_system.end_of_query_seq()
    recalls = recalls / len(uav_seq.uav_images)
    return recalls



def reference_recall(
    uav_seq: UAVSeq,
    localization_pipeline: LocalizationPipeline,
    k_closest: int,
    threshold: int,
) -> float:
    """
    The metric finds the number of correctly matched frames based on georeference error

    :param uav_seq: Sequence of UAV images
    :param localization_pipeline: Instance of LocalizationPipeline class
    :param k_closest: Specifies how many predictions for each query the global localization should make.
    If this value is greater than 1, the best match will be chosen with local matcher
    :param threshold: The distance between query and reference geocoordinates,
    below which the frame will be considered correctly matched

    :return: Recall value
    """
    recall_value = 0
    localization_results = localization_pipeline(uav_seq, k_closest)
    for loc_res, uav_image in zip(localization_results, uav_seq):
        if loc_res is not None:
            lat, lon = loc_res
            error = calculate_distance(
                lat, lon, uav_image.gt_latitude, uav_image.gt_longitude
            )
            if error < threshold:
                recall_value += 1
    return recall_value / len(uav_seq.uav_images)
