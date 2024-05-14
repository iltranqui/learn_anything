## Segmentation Tasks

A segmentation task in computer vision involves partitioning an image into multiple segments or regions based on certain criteria. These tasks are fundamental in understanding and interpreting visual data. Here's a breakdown of the different types of segmentation and their key differences:

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/2b278f9a-157d-4bc6-976e-47fadea501cf/c40b955f-1d69-41ed-bc4a-1f1a888cbed0/Untitled.png)

1. **Instance Segmentation**:
    - Definition: Instance segmentation involves identifying and delineating individual objects within an image. It not only segments different objects but also distinguishes between separate instances of the same object.
    - Key Differences:
        - Provides pixel-level classification for each object instance in an image.
        - Requires distinguishing between multiple objects of the same class (e.g., differentiating between two cars in an image).
        - Often used in applications where precise object detection and delineation are necessary, such as autonomous driving, medical imaging, and robotics.
    
    !https://miro.medium.com/v2/resize:fit:828/format:webp/1*rgliupBanbeMYW7xXYH5Lw.jpeg
    
2. **Semantic Segmentation**:
    - Definition: Semantic segmentation involves categorizing each pixel in an image into a specific class or category, without differentiation between individual object instances.
    - Key Differences:
        - Assigns a class label to each pixel, representing the category of the object it belongs to.
        - Does not differentiate between instances of the same class; all pixels of the same class are treated equally.
        - Widely used in tasks where understanding the overall scene semantics is important, such as scene understanding, image retrieval, and video surveillance.
    - Inputs and Inputs
        - Image [batch, 3, height, input]
        - Mask [batch, 1, height, input] → uint8 for id for each class
        - id2label → the number assoicated with each label
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/2b278f9a-157d-4bc6-976e-47fadea501cf/6e153be7-e486-4d3e-9c39-d5c217fd9fa2/Untitled.png)
    
3.  **Panoptic Segmentation**:
    - Definition: Panoptic segmentation unifies instance segmentation and semantic segmentation tasks by providing a comprehensive understanding of the scene, where all pixels are labeled with a category label and a unique instance ID.
    - Key Differences:
        - Combines both instance and semantic segmentation into a single framework.
        - Aims to label every pixel in the image with both a semantic class (what object the pixel belongs to) and an instance ID (which specific instance of that object it is).
        - Addresses the limitations of traditional segmentation tasks by providing a more complete understanding of the scene.
        - Useful in various applications requiring detailed scene understanding, such as augmented reality, autonomous navigation, and video analysis.

These segmentation tasks play crucial roles in computer vision applications, enabling machines to understand and interpret visual information with increasing accuracy and efficiency.

- Inputs and Outputs
    - Image [batch, 3, height, input]
    - Mask [batch, 1, height, input] → uint8 for id for each class
    - id2label → the number assoicated with each label


### Semantic Segmentation Pipeline Design (Input/Output)
Requirements for the Pipeline in order that future datasets are ready to be used in that specific format 

* Input: 
    1. Images/Pictures: in RGB [3, height, width] format *.jpg in float32
    2. Masks: in format [1, heigth, width] format *.png in uint8
        * Masks are supposed to be without the ground element -> future interations will make sure that the the option of adding a background is also present
    3. id2josn: For the Segmentation Tasks, there is a necessity to know what each single label in the mask is refering to. 

* Output: 
    1. Predicted Mask: [1, heigth, width] format tensor.float32
        * Predict also the transformat to a png file for every single element
        * Export the Model in ONNX  
