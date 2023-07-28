class StateDetector(object):
    """Shows if the current state of the system of a given class name given the pose classification."""

    def __init__(self, class_name, enter_threshold=6, exit_threshold=4):
        # class name which we want to detect ()
        self._class_name = class_name

        # If pose passes given threshold, then we enter the pose.
        self._enter_threshold = enter_threshold
        self._exit_threshold = exit_threshold

        # Either we are in given pose or not.
        self._pose_entered = False

    def __call__(self, pose_classification):
        """
        Takes the pose classification (smoothed if possible) and returns
        the if providad class name state is detected

        We use two thresholds. First you need to go above the higher one to enter
        the pose, and then you need to go below the lower one to exit it. Difference
        between the thresholds makes it stable to prediction jittering (which will
        cause wrong counts in case of having only one threshold).

        Args:
          pose_classification: Pose classification dictionary on current frame.
            Sample:
              {
                'class_name_1': 8.3,
                'class_name_2': 1.7,
              }

        Returns:
          current state (0 or 1) given the frame classification
        """
        # Get pose confidence.
        pose_confidence = 0.0
        if self._class_name in pose_classification:
            pose_confidence = pose_classification[self._class_name]

        # On the very first frame or if we were out of the pose, just check if we
        # entered it on this frame and update the state.
        if not self._pose_entered:
            self._pose_entered = pose_confidence > self._enter_threshold
            return int(self._pose_entered)

        # If we were in the pose and are exiting it, then increase the counter and
        # update the state.
        if pose_confidence < self._exit_threshold:
            self._pose_entered = False

        return int(self._pose_entered)
