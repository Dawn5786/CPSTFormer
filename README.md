This code is for the work "Complex Dual-Tree Pyramid Scattering Transformer"

For detailed environment configuration requirements, please look at the requirement.txt file.

Detailed data preprocessing, hyperparameter selection, and training Settings are as follows：

1. Image Classification Task

CIFAR Dataset 

Data Preprocessing: CIFAR-10 is a small image classification data set containing 10 categories，each category consists of 50,000 training and 10,000 testing images. The color image with a resolution of $32×32$. CIFAR-100 is a classification data set with 100 categories but fewer samples (500 for training and 100 for testing) with the same resolution. The input with the resolution of $32×32$ in the initial pyramid branch is padded to $48×48$, and then divided into parallel inputs of size $[48×48, 24×24, 12×12]$.
Hyperparameter Settings: Parallel scattering input branches index $b$ is set to $[2,3,4]$, the scale scattering parameter $q$ is configured as $[1,2,2]$, the scattering dimensions are set to $[21,129,129]$, and the scattering learning rate (lr\_scattering) is set to $0.1$.
Training: Training starts from scratch with a learning rate of $0.1$, a maximum learning rate of $6e^{-2}$, and 300 total epochs. Weight decay and momentum are set to $5e^{-4}$ and $0.9$, respectively.

Tiny-ImageNet Dataset

Data Preprocessing: Tiny-Imagenet consists of 100k training images, 10k validation images, and 10k test images belong to 200 categories with a resolution of $64×64$ pixels. The $64×64$ input in the initial pyramid branch is padded to $80×80$ and then divided into parallel inputs of size $[80×80, 40×40, 20×20]$. 
Hyperparameter Settings: Parallel scattering input branches index $b$ is set to $[3,4,5]$, the global scale scattering parameter $q$ is set to $2$, the scattering dimensions are uniformly set to $[129,129,129]$, and lr\_scattering is $0.1$.
Training: Network is randomly initialized and trained with a cosine decay learning rate scheduler. The initial learning rate is $5e^{-4}$, the weight decay is $5e^{-2}$, and the warm-up decay is $10$ epochs. The total epochs are set to 300. 

COVIDx CRX-3 Dataset
Data Preprocessing: COVIDx CXR-3 is a medical chest X-ray detection data set containing chest X-ray images of COVID-19 patients from a multinational cohort with a resolution of $1024×1024$. There are 29986 training set images and 400 testing set images. Input images in the initial branch are reshaped to $384×384$ and divided into parallel inputs of sizes $[96×96, 48×48, 24×24]$ with a pyramid sequence of patch sizes $(16, 8, 4)$.
Hyperparameter Settings: We take more levels of scattering components for aggregation, and $q=3$ is applied to branches $b=[4,5]$, while $q=2$ is retained for $b=3$.
Training: The networks are trained for $400$ epochs using a maximum learning rate of $1e^{-2}$. The learning rate scheduler is set to OneCycleLR. The weight decay is set to $5e^{-4}$. The momentum is set to $0.9$.  

2. Object Tracking Task
   
TrackingNet Dataset

Data Preprocessing: TrackingNet is a large-scale tracking dataset including 30k videos withmore than 14 million 780 annotations. We evaluate our CPST-Tracker on 511 official test sequences. To maintain spatial consistency of scattering features, we pad the template input to the same size as the search region $224×224$. 
Hyperparameter Settings: The number of hidden channels per stage is configured as $[48, 192, 192, 384]$, corresponding layer counts set to $[2, 2, 4, 2]$.
Training: The CPST-Tracker is trained on LaSOT, TrackingNet, GOT-10k, and COCO 2017 training splits. Optimization is performed using AdamW with a learning rate of $5e^{-4}$, weight decay of $1e^{-4}$, and a backbone-specific learning rate of $5e^{-5}$. The total epochs are set to 300. DropPath is applied with a rate of $0.1$.
