# U-net Model Summary (OSCD dataset)

**Step-1:** This step defines the configuration for model training and patch generation. It sets the number of training epochs to 30 and segments the dataset into overlapping 64×64 tiles with a 32-pixel overlap. It also filters out tiles with less than 3% pixel change to ensure the model focuses on learning significant changes. This setup is particularly useful in change detection tasks, often applied in satellite or medical imaging.

**Step-2:** This block imports all essential libraries for data handling, image processing, training visualization, evaluation, and file management. Tools like NumPy, PIL, and scikit-image handle image data, while sklearn and tqdm support evaluation and feedback. These imports set the stage for seamless training, visualization, and debugging.

**Step-3:** This step defines the dataset directories for the OSCD dataset, which contains before-and-after satellite images and corresponding change labels. It extracts subfolder names representing different cities used in training and testing. This allows the model to learn generalized patterns of change across geographically diverse regions, which is essential for robust change detection.

**Step-4:** This step defines key functions to preprocess satellite image pairs and labels by tiling them into smaller patches with optional overlap. It filters out patches with minimal change using a pixel-wise threshold, reducing class imbalance and enhancing training focus. The final function, dataset_from_folder, loads and processes all cities’ data into three aligned numpy arrays — ready to be fed into the U-Net model

**Step-5:** This step generates the training and testing datasets by processing image-label triplets into overlapping and optionally filtered tiles. The training set includes only patches with ≥3% change, improving data efficiency, while the test set includes all data for unbiased evaluation. Output shapes confirm dataset readiness for model input.( Verifies that the train/test datasets have been correctly assembled. Ensures shape alignment among image pairs and labels — all must match exactly (e.g., (N, 64, 64, 3) for RGB images and (N, 64, 64, 1) for binary masks).

**Step-6:** This step prepares training and validation datasets using a pixel-wise differencing technique, converting bi-temporal satellite image pairs into change-focused inputs. It ensures compatibility with TensorFlow/Keras and the segmentation_models library. While simplistic, differencing provides a strong signal for change detection and serves as a practical preprocessing baseline.

**Step -7:** This step builds a U-Net segmentation model using a ResNet34 encoder pretrained on ImageNet, processes the input data to match backbone expectations, and compiles the model with a hybrid loss and robust evaluation metrics. This architecture is powerful for change detection, combining deep feature learning with pixel-level precision.

**Step-8:** This step trains the U-Net model on the prepared dataset (x_train, y_train) for a specified number of epochs. The model is validated on the validation set (x_val, y_val) after each epoch. After training, the model is saved to disk for later use, ensuring that you can avoid retraining the model from scratch in the future.

**Step-9:** AUC-ROC Curve: Calculates the Area Under the ROC Curve (AUC) to assess the model's discriminatory ability, and visualizes it with a ROC curve.
Precision, Recall, Accuracy: Evaluates how well the model is performing in binary classification (change/no change) using several important metrics (precision, recall, accuracy, F1 score), and optionally saves the results to a text file.
These evaluation methods provide a comprehensive analysis of the model’s performance, helping you to assess its suitability for the task of change detection.

**Step-10:** 
1.	Model Prediction: The model makes predictions based on the pixel-wise differences between paired images, which helps in detecting changes between them.
2.	Evaluation on Test Set:
o	AUC-ROC: The AUC score is calculated to measure the model's ability to discriminate between the change and no-change classes.
o	Precision, Recall, Accuracy, F1 Score: These metrics assess the model's performance in terms of both correct change detection and avoidance of false positives/negatives.
3.	Evaluation on Train Set: Similar evaluations are performed on the training set to understand how well the model is generalizing.
The output includes both visual and numeric performance metrics. This is essential for understanding how well the model can detect changes in satellite imagery and assess its strengths/weaknesses.	
