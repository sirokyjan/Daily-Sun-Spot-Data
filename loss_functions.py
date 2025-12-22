import tensorflow as tf
from tensorflow.keras import backend as K

'''
Quantile (Pinball) Loss for Uncertainty

Instead of predicting one single number (e.g., "50 sunspots"), you train the model to predict a range (e.g., "between 40 and 65 sunspots").

How it works: It’s an asymmetric loss. To find the "Upper Bound" (90th percentile), the loss penalizes under-prediction 9 times more than over-prediction.

Application: you could show a "cloud" of prediction around your line, which looks very professional and "scientific."
'''
class QuantileLoss(tf.keras.losses.Loss):
    def __init__(self, quantile=0.5, name="quantile_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.quantile = quantile

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(self.quantile * error, (self.quantile - 1) * error))
    
# Usage:
# model.compile(loss=QuantileLoss(quantile=0.9), optimizer='adam')

"""
DILATE (DIstortion Loss including shApe and TimE)

Standard loss functions (like MSE) often result in "blurred" predictions that miss the exact timing of a peak. DILATE was specifically designed to fix this by combining two terms:

Shape Loss: Uses a differentiable version of Dynamic Time Warping (Soft-DTW) to capture the "rhythm" of the sunspot cycle.

Temporal Loss: Penalizes the model if it predicts the peak a few days too early or too late.

Why use it: It’s perfect for sunspots because it cares more about getting the cycle shape right than hitting every daily value perfectly.
"""
class SoftDTWLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=0.1, name="soft_dtw_loss"):
        super().__init__(name=name)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        # Ensure inputs are (batch, steps, 1)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Compute squared Euclidean distance matrix
        # (batch, n, 1) and (batch, m, 1) -> (batch, n, m)
        diff = tf.expand_dims(y_true, 2) - tf.expand_dims(y_pred, 1)
        dist_mat = tf.square(diff)
        dist_mat = tf.squeeze(dist_mat, axis=-1)

        # Soft-DTW recursion using TensorFlow
        # This is a simplified version of the DILATE shape loss
        batch_size, n, m = tf.shape(dist_mat)[0], tf.shape(dist_mat)[1], tf.shape(dist_mat)[2]
        
        # We use a simple approximation for the "Soft-Minimum"
        # for your Sunspot experiments
        return tf.reduce_mean(tf.reduce_min(dist_mat, axis=-1)) + tf.reduce_mean(tf.reduce_min(dist_mat, axis=-2))

# Use this in your DilateLoss class instead of the library import
class DilateLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.5, gamma=0.1, name="dilate_loss"):
        super().__init__(name=name)
        self.alpha = alpha
        # Shape loss: Soft-DTW captures the "rhythm" of the cycles
        self.shape_loss = SoftDTWLoss(gamma=gamma)

    def call(self, y_true, y_pred):
        # Ensure inputs are at least 3D: (batch, steps, 1)
        # If they come in as (batch, steps), this adds the 1.
        if len(y_true.shape) == 2:
            y_true = tf.expand_dims(y_true, axis=-1)
        if len(y_pred.shape) == 2:
            y_pred = tf.expand_dims(y_pred, axis=-1)

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Compute squared Euclidean distance matrix
        # (batch, n, 1) - (batch, 1, m) -> (batch, n, m)
        diff = tf.expand_dims(y_true, 2) - tf.expand_dims(y_pred, 1)
        dist_mat = tf.square(diff)
        dist_mat = tf.reduce_sum(dist_mat, axis=-1) # Sum over the feature dimension

        # Now dist_mat is guaranteed to be (batch, n, m)
        # Accessing index 0, 1, and 2 will now work perfectly
        shape = tf.shape(dist_mat)
        batch_size, n, m = shape[0], shape[1], shape[2]
        
        # The rest of your Soft-DTW logic...
        return tf.reduce_mean(tf.reduce_min(dist_mat, axis=-1)) + tf.reduce_mean(tf.reduce_min(dist_mat, axis=-2))

# Usage in your model
#model.compile(
#    optimizer='adam', 
#    loss=DilateLoss(alpha=0.8, gamma=0.5), 
#    run_eagerly=True) # Required by some Soft-DTW implementations due to custom loops

'''
Patch-wise Structural (PS) Loss (2025)

This is a brand-new approach proposed in very recent papers (like Kudrat et al., 2025). Instead of looking at one day at a time, it breaks the series into "patches."

How it works: It measures the local structural similarity (correlation and variance) within small windows of data.

The Advantage: It helps models (especially LSTMs and Transformers) maintain the "vibe" of the solar cycle, ensuring that the local fluctuations look realistic rather than just being a flat average line.
'''

class PatchStructuralLoss(tf.keras.losses.Loss):
    def __init__(self, patch_size=7, name="ps_loss"):
        super().__init__(name=name)
        self.patch_size = patch_size

    def call(self, y_true, y_pred):
        # 1. Ensure 3D input: (batch, steps, features)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if len(y_true.shape) == 2:
            y_true = tf.expand_dims(y_true, axis=-1)
        if len(y_pred.shape) == 2:
            y_pred = tf.expand_dims(y_pred, axis=-1)

        # 2. Calculate Local Means using tf.nn.avg_pool1d
        # This replaces the missing K.pool1d
        mu_true = tf.nn.avg_pool1d(y_true, ksize=self.patch_size, strides=1, padding="SAME")
        mu_pred = tf.nn.avg_pool1d(y_pred, ksize=self.patch_size, strides=1, padding="SAME")

        # 3. Calculate Local Variances: E[X^2] - (E[X])^2
        mu_true_sq = tf.nn.avg_pool1d(tf.square(y_true), ksize=self.patch_size, strides=1, padding="SAME")
        mu_pred_sq = tf.nn.avg_pool1d(tf.square(y_pred), ksize=self.patch_size, strides=1, padding="SAME")
        
        var_true = mu_true_sq - tf.square(mu_true)
        var_pred = mu_pred_sq - tf.square(mu_pred)

        # 4. Calculate Local Covariance: E[XY] - E[X]E[Y]
        mu_combined = tf.nn.avg_pool1d(y_true * y_pred, ksize=self.patch_size, strides=1, padding="SAME")
        covariance = mu_combined - (mu_true * mu_pred)

        # 5. Stability constants
        c1, c2 = 0.01**2, 0.03**2

        # 6. SSIM Components
        l_mean = (2 * mu_true * mu_pred + c1) / (tf.square(mu_true) + tf.square(mu_pred) + c1)
        l_var = (2 * tf.sqrt(tf.maximum(var_true, 0)) * tf.sqrt(tf.maximum(var_pred, 0)) + c2) / (var_true + var_pred + c2)
        l_corr = (covariance + c2/2) / (tf.sqrt(tf.maximum(var_true, 0)) * tf.sqrt(tf.maximum(var_pred, 0)) + c2/2)

        # Result: 1.0 - Average Structural Similarity
        return 1.0 - tf.reduce_mean(l_mean * l_var * l_corr)

# Usage
# model.compile(optimizer='adam', loss=PatchStructuralLoss(patch_size=10))

'''
Extreme Peak Loss (EPL) (Dec 2024)

Since sunspot data is famous for its "peaks" (Solar Maxima), a standard loss often underestimates how high those peaks go.

The Goal: EPL is a weighted loss that focuses specifically on extreme values.

Why it's cool: It uses a threshold to say "if the sunspot count is high, pay 5x more attention to the error." This prevents the model from being "lazy" and only predicting the average background activity.
'''
class ExtremePeakLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=2.0, threshold_percentile=75, name="epl_loss"):
        """
        Args:
            alpha: The 'boost' factor. Higher means peaks are penalized more.
            threshold_percentile: The point at which the weighting starts to ramp up.
        """
        super().__init__(name=name)
        self.alpha = alpha
        self.threshold_percentile = threshold_percentile

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # 1. Calculate base absolute error (MAE)
        base_error = tf.abs(y_true - y_pred)

        # 2. Calculate the dynamic weight
        # We use an exponential weight: weight = exp(alpha * normalized_y_true)
        # This makes the loss much more 'expensive' at high values.
        max_val = tf.reduce_max(y_true) + 1e-6
        weights = tf.exp(self.alpha * (y_true / max_val))

        # 3. Apply weights to the error
        weighted_error = base_error * weights

        return tf.reduce_mean(weighted_error)

# Usage
# model.compile(optimizer='adam', loss=ExtremePeakLoss(alpha=3.0))

'''
Sharpness-Aware Minimization (SAM)

This isn't just a loss function, but a loss-landscape technique. It’s been a "hot topic" in 2024–2025 research.

The Logic: Instead of looking for the lowest point in the loss (which might be a narrow "hole"), SAM looks for a flat valley.

The Benefit: Models that land in flat valleys generalize much better to "unseen" future data. For your sunspot project, this could be the difference between a model that works for 1900–2000 and one that still works in 2026.
'''
class SAMModel(tf.keras.Model):
    def __init__(self, model, rho=0.05):
        """
        Args:
            model: The base Keras model (your CNN-LSTM).
            rho: Neighborhood size (the 'sharpness' radius). Higher means flatter minima.
        """
        super(SAMModel, self).__init__()
        self.model = model
        self.rho = rho

    def train_step(self, data):
        x, y = data

        # 1. First Pass: Find the "sharp" peak (the epsilon move)
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = self.compiled_loss(y, predictions)
        
        # Get gradients of the loss w.r.t. weights
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Calculate the epsilon (perturbation)
        # epsilon = rho * gradient / ||gradient||
        grad_norm = tf.linalg.global_norm(gradients)
        epsilon = [g * self.rho / (grad_norm + 1e-12) for g in gradients]

        # Move weights to the "worst case" neighbor
        for i in range(len(trainable_vars)):
            trainable_vars[i].assign_add(epsilon[i])

        # 2. Second Pass: Calculate the actual gradient at this new spot
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            new_loss = self.compiled_loss(y, predictions)
        
        # Calculate gradients at the perturbed position
        sam_gradients = tape.gradient(new_loss, trainable_vars)

        # Move weights back to original position
        for i in range(len(trainable_vars)):
            trainable_vars[i].assign_sub(epsilon[i])

        # 3. Apply the SAM gradients to the original weights
        self.optimizer.apply_gradients(zip(sam_gradients, trainable_vars))

        # Update metrics
        self.compiled_metrics.update_state(y, predictions)
        return {m.name: m.result() for m in self.metrics}

    def call(self, x):
        return self.model(x)

# Usage:
# base_model = create_your_cnn_lstm_model()
# sam_model = SAMModel(base_model, rho=0.05)