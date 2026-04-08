from typing import Optional, Tuple

from aero_vloc.utils.homography_estimator import HomographyEstimator
from aero_vloc.primitives.uav_seq import UAVSeq
from aero_vloc.retrieval_system import RetrievalSystem


class LocalizationPipeline:
    """
    Allows to create a localizator based on the retrieval system and homography estimator.
    """

    def __init__(
        self,
        retrieval_system: RetrievalSystem,
        homography_estimator: HomographyEstimator,
    ):
        self.retrieval_system = retrieval_system
        self.homography_estimator = homography_estimator

    def __call__(
        self,
        query_seq: UAVSeq,
        k_closest: int,
    ) -> list[Optional[Tuple[float, float]]]:
        """
        Calculates UAV locations using the retrieval system and homography estimator.

        :param query_seq: The sequence of images for which locations should be calculated
        :param k_closest: Specifies how many predictions for each query the global localization should make.
        :return: List of geocoordinates. Also, the values can be None if the location could not be determined
        """
        localization_results = []
        for query_image in query_seq:
            (
                res_prediction,
                matched_kpts_query,
                matched_kpts_reference,
            ) = self.retrieval_system(
                query_image, k_closest, feature_matcher_k_closest=1
            )

            res_prediction = res_prediction[0]
            matched_kpts_query = matched_kpts_query[0]
            matched_kpts_reference = matched_kpts_reference[0]

            chosen_sat_image = self.retrieval_system.sat_map[res_prediction]
            estimator_result = self.homography_estimator(
                matched_kpts_query,
                matched_kpts_reference,
                query_image,
                self.retrieval_system.feature_matcher.resize,
            )
            if estimator_result is None:
                localization_results.append(None)
                continue
            (
                latitude,
                longitude,
            ) = self.retrieval_system.sat_map.geo_referencer.get_lat_lon(
                chosen_sat_image,
                estimator_result,
                self.retrieval_system.feature_matcher.resize,
            )
            localization_results.append((latitude, longitude))
        self.retrieval_system.end_of_query_seq()
        return localization_results