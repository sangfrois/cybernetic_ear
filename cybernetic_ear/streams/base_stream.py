from abc import ABC, abstractmethod

class BaseStream(ABC):
    """
    Abstract base class for a feature extraction stream.
    Ensures that all stream processors have a consistent interface.
    """

    @abstractmethod
    def process_chunk(self, chunk, feature_bus):
        """
        Processes a single chunk of audio data and updates the feature bus.
        This method must be implemented by all subclasses.

        Args:
            chunk (np.ndarray): A numpy array containing the audio data for the chunk.
            feature_bus (FeatureBus): The shared feature bus to update with new values.
        """
        pass

    def __call__(self, chunk, feature_bus):
        """
        Allows the object to be called like a function, which is a convenient
        way to handle the processing.
        """
        self.process_chunk(chunk, feature_bus)
