import tensorflow as tf

# timeframe = [15, 60]
# input_length = 100
# output_length = 3
# ma_lengths = [1, 3, 5, 13]

class Oracle:
    def __init__(self, timeframe, prediction_goals: list):
        self.predictors = {}

        # Load predictors for each goal
        for goal in prediction_goals:
            self.predictors[goal] = tf.keras.models.load_model(f'/predictors/{timeframe}T_{goal}_predictor.h5')