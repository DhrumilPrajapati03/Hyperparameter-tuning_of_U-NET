# U-net Model Summary (OSCD dataset)

**Step-1:** This step defines the configuration for model training and patch generation. It sets the number of training epochs to 30 and segments the dataset into overlapping 64×64 tiles with a 32-pixel overlap. It also filters out tiles with less than 3% pixel change to ensure the model focuses on learning significant changes. This setup is particularly useful in change detection tasks, often applied in satellite or medical imaging.

**Step-2:** This block imports all essential libraries for data handling, image processing, training visualization, evaluation, and file management. Tools like NumPy, PIL, and scikit-image handle image data, while sklearn and tqdm support evaluation and feedback. These imports set the stage for seamless training, visualization, and debugging.

**Step-3:** This step defines the dataset directories for the OSCD dataset, which contains before-and-after satellite images and corresponding change labels. It extracts subfolder names representing different cities used in training and testing. This allows the model to learn generalized patterns of change across geographically diverse regions, which is essential for robust change detection.

**Step-4:** This step defines key functions to preprocess satellite image pairs and labels by tiling them into smaller patches with optional overlap. It filters out patches with minimal change using a pixel-wise threshold, reducing class imbalance and enhancing training focus. The final function, dataset_from_folder, loads and processes all cities’ data into three aligned numpy arrays — ready to be fed into the U-Net model
