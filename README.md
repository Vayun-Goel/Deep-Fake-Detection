# WINTER PROJECT IIT MANDI
# DEEP FAKE DETECTION

This project involves experimentation and implement of the task of deep fake detection. I refer to 2 research papers for this task :- 
1) paper 1 involves creating a pipeline of multiple resnet models trained for different objectives. It is constructed such that there are 3 levels, the first level deals with the binary classification between real and fake images, the second level deals with if the image is classified as fake, classifying it as either diffusion model constructed or GAN constructed image, and the last layer deals with classifying which specific model the image comes from given it is from diffusion family or GAN family.

The authors consider this method more viable than just having a 1 level classification of all the models along with a real class as each stage can potentially learn its specified tasks better than combining everything and giving that task to only 1 resnet model.

below i will share some of my findings trying to implement the paper and resporting necessary conclusions which explain to us why we need to transition to an approach considered in paper 2.

paper link : https://arxiv.org/abs/2303.00608

The relevent files are present in the paper 1 folder.
1.1) The first file is the level 1 file which highlights the task of real vs fake analysis

setting - used only dalle images (1000 in number) against real images (1000 in number) to try and find dependance on i. diffusion model to diffusion model and ii. diffusion model to GAN
The model RESNET-18 is used to and trained for 5 epochs and the results are evaluated on the fake images of ["glide_100_10","guided","idm_100"] diffusion models first and then ["biggan","stargan","stylegan"] GAN models next.

Findings and Results :- 

i. The results are disappointing as we find that the rest of the diffusion models have accuracies of around ~ [50,26,36] clearing stating that training on 1 particular diffusion model does gaurentee an overall positive results for the rest of the models
ii. The results are again disappointing as we find that the GAN models have accuracies < 50%, stating that training diffusion model does gaurentee an overall positive results for GANs. 

However these results are exaggerated due to overfitting of the model on dalle images a re - evalaution must be done with a more diverse dataset with other model data along wiht early stopping and more regularization.

These results tell us, as suggested in the paper, that the model latches on to the unique fingerprints left by the generatation model and these signatures are unique to a model and 1 model's data cannot be used as a universal detector for all possible fake images or for generators to come in future.

1.2) The second file is the level 2 file which is tasked with Diffusion vs GANs task

setting - The data used for diffusion models consist of data of - dalle, glide_50_27, guided, idm_200 and for GANs data of - biggan,stargan and which face is real is used to make a more diverse data set this time.
RESNET-18 is again used and trained for 4 epochs and evaluated on first for diffusion model data of - ["glide_100_10","idm_100_cfg"] and then GAN data of - ["gaugan","stylegan"]

Findings and Results :- 

i. for the diffusion models, the results are great with accuracies over 96% however for GANs the results are dissapointing with less than 35%. This shows that diffusion models are a higher dependence on each other and the signatures left by diffusion models though fainter than GANs are more similar and widespread whereas for GANs the interdependence is less and thus to classfy then properly we must include similar models, defeating the purpose of making a universal model for deep fake detection.

1.3) The third and fourth files deal with the level 3 task which is the exact model prediction

setting - data for all the models are divided into train and test sets and used for training, RESNET-18 is again used for both the files and trained for 4 epochs

1.3.1) file 3 (diffusion models) :-

Findings and Results :-

The diffusing models seperately did not show great results as diffusion models from the same subfamily eg - the glide family, got confused among one another but if the subfamilies are combined they show great results over 90%

1.3.2) 

Findings and Results :-

The GAN models show good results apart from the biggan varient with over 90% accuracies 

(NOTE: 1. The exact accuracies and details can be found in the files
       2. The final pipeline file is not shown as its not possible to load all the models simulateously on the free colab however, it is easy to piece together all the models into a pipeline as suggested in the paper to give the final model )

OVERALL RESULTS: These results depict that we do require a solution which does not depend on unique model characteristics as different models leave different fingerprints as is the case in families as well as subfamiles, Thus, we need to explore solutions which eliminates these unique traits left by the image generation models and focus on more of the style part of the images generated to make a futuristic ideal fake detector, thus we transition to the second paper discussing potential solutions to these problems

