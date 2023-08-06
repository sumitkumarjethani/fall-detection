import numpy as np
from typing import List, Optional
from abc import ABC, abstractmethod
from ..object_detection.yolo import ObjectDetectionSample

# from ultralytics import YOLO
# from ultralytics.engine.results import Results as YoloResults


class ClassificationRule(ABC):
    @abstractmethod
    def check(self, objs: List[ObjectDetectionSample]) -> bool:
        pass


class PersonIsAlone(ClassificationRule):
    """
    Checks whether the given prediction is
    compose by one and only one person
    """

    def check(self, objs: List[ObjectDetectionSample]) -> bool:
        persons_bb = np.array([o.xyxy for o in objs if o.class_name == "person"])
        return persons_bb.shape[0] == 1


class PersonNotOnFurniture(ClassificationRule):
    _furniture_class_names = ["couch", "bed"]
    """
    Checks whether there is a person and a couch
    in the objs detected and checks if the person is
    on it
    """

    def _is_person_on_furniture(self, person_bb, furnitures_bb):
        person_x_min, person_y_min, person_x_max, person_y_max = (
            person_bb[0],
            person_bb[1],
            person_bb[2],
            person_bb[3],
        )
        for furniture_bb in furnitures_bb:
            furniture_x_min, furniture_y_min, furniture_x_max, furniture_y_max = (
                furniture_bb[0],
                furniture_bb[1],
                furniture_bb[2],
                furniture_bb[3],
            )
            # Comprobar si el bounding box de la persona está completamente fuera del bounding box del mueble
            if (
                furniture_x_min <= person_x_min
                and person_x_max <= furniture_x_max
                and furniture_y_min <= person_y_min
                and person_y_max <= furniture_y_max
            ):
                return True  # La persona está en un mueble
        return False  # La persona no está en ningún mueble

    def check(self, objs: List[ObjectDetectionSample]) -> bool:
        furnitures_bb = np.array(
            [o.xyxy for o in objs if o.class_name in self._furniture_class_names]
        )
        persons_bb = np.array([o.xyxy for o in objs if o.class_name == "person"])
        if persons_bb.shape[0] == 1 and furnitures_bb.shape[0] > 0:
            return not self._is_person_on_furniture(persons_bb[0], furnitures_bb)
        return True


class PersonNotHorizontal(ClassificationRule):
    def _is_person_bbox_horizontal(self, person_bb):
        xmin, ymin, xmax, ymax = person_bb[0], person_bb[1], person_bb[2], person_bb[3]
        dx = int(xmax) - int(xmin)
        dy = int(ymax) - int(ymin)
        difference = dy - dx
        if difference < 0:
            return True
        return False

    def check(self, objs: List[ObjectDetectionSample]) -> bool:
        persons_bb = np.array([o.xyxy for o in objs if o.class_name == "person"])
        return not self._is_person_bbox_horizontal(persons_bb[0])


class RulesChecker:
    """
    Runs the check method on a list of rules and returns
    an ratio of successful rules over total rules
    """

    def __init__(
        self,
        rules: List[ClassificationRule],
        weights: Optional[List[float]] = None,
        threshold=0.5,
    ):
        self._rules = rules
        self._weights = (
            weights if weights != None else [1 / len(rules) for _ in range(len(rules))]
        )
        self._threshold = threshold

    def check(self, objs: List[ObjectDetectionSample]) -> bool:
        return self.__call__(objs) >= self._threshold

    def __call__(self, objs: List[ObjectDetectionSample]) -> float:
        return np.average(
            [int(r.check(objs)) for r in self._rules], weights=self._weights
        )
