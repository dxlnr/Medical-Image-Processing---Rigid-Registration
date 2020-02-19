# Medical-Image-Processing---Rigid-Registration

# Background
Two different images of a patient, taken at two different times, usually have significant movement between them. This is because the patient is in different poses, because of internal movements, e.g., breathing, and because of other physical changes that occurred in the time that passed between the scans. Registering the two images is needed to compare the content, e.g., track the differences, or to evaluate the efficacy of a treatment.

Rigid registration consists of computing the translation and rotation that aligns two images assuming that the anatomical structures of interest retain their shape. Rigid registration of 2D images requires computing three parameters: two translations and one rotation, while 3D scans requires computing six parameters: three translations and three rotations.

Rigid registration algorithms can be categorized into two groups: geometry and intensity based. In geometric-based algorithms, the features to be matched, e.g. points are first identified in each image and then paired. The sum of the square distances between the points is then minimized to find the rigid transformation. In intensity-based algorithms, a similarity measure is defined between the images. The transformation that maximizes the similarity is the desired one. The rigid registration algorithms are iterative – in each step, a transformation is generated and tested to see if it reduces the sum of the squared distances between the points or increases the similarity between the images.


# Task Description

Registration by features

The goal of this task is to perform an automatic registration between two ophthalmology 2D images using features. The images are Fundus Auto Flourescence images (FAF) of patients that suffer age related macular degeneration (AMD), a condition of retinal atrophy that causes vision loss.

Detect the changes

The goal of this task is to detect the changes of the lesions between two registered ophthalmology 2D images as you can see below. On the left we can see a baseline image with the changes in the red and on the right, we can see the follow-up image with the changes in red.

# Results

Check out the result.pdf