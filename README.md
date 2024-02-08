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

2) paper 2 is considered next as a potential fix to the limitations of findings depicted in the previous experimentation. The underlying issue was since the neural net was trained on only dalle images which 1 type of a diffusion model, the neural net had become heavily biased to the characteristic features (frequency spikes) of specific dalle images and thus performs poorly on not only different family gans but shows undesirable results even on the same family diffusion models.

The proposed approach in the paper revolves around creating an embedding first of the image using a pre-trained vision transformer and then using a classification technique to classify the embeddings as real or fake. The idea is to not feed the images directly as that leads to the neural net to latch onto the specific model characteristics and ignore the other details as the style of the image,etc. By sending the images through the vision transformer, we expect the image to loose such details and expect only the characteristic features inherent to all fake images remain in the embeddings and for that same reason, we do not train the VIT instead we use the pre-trained one. We hope that apon classification on the embeddings which seemingly retain only the features of the image can help generalize better to other models well which we prove in our experimentations.

paper link : https://arxiv.org/abs/2302.10174

The relevent files are present in the paper 2 folder.
The experimentation is done in the following manner :

step 1) The CLIP VIT L-14 vision transformer was used to convert the input images to embeddings. The embeddings were of 768 dimensions.
step 2) Using autoencoders, the dimensionality of the high dimesional vector embeddings were reduced to 64 dimensional vector. (they were also reduced to 32 dimensional ones to check the difference in the performance in the later experiments)
step 3) The 64 dimensional vectors were used to then train a random forest classifier for the task of real vs fake classification.

1000 images of dalle and 1000 real images were used for training the random forest classifier

Results and findings : apon training, we test on different diffusion and gan models just like we did in the previous experiments and we observe much better results. 

diffusion models :- a)glide_50_27 => 92% 
                    b)guided      => 82%
                    c)ldm_200     => 97%

GANs models :-      a)biggan           => 98%
                    b)gaugan           => 100%
                    c)stargan          => 95%
                    d)stylegan         => 61%
                    e)whichfaceisreal  => 62%

We find that despite being trained only on dalle images, the pipeline does well to classify other model images as fake which was a huge limitation in the previous experimentation, however we find that there is a dip in the performance in stylegan and whichfaceisreal. The work needs more refinement as to why the dip occurs in those gans but performs well on all other models, The suspected reason is the train data is not diverse enough to include different types of real images of different objects and materials and since it lacks face images, we suspect that is the reason is has a dip in those images.

further experimentation and future experimentation : we then condense the vectors to 32 dimension vector to check the dip in the performance and we notice similar results with 1-2% change thus condensing the vector to an even smaller vector does not harm in computation compromised scenarios. In future we plan to test on other VITs such as B-16 model, etc.

Conclusion : Thus, we prove that in order to make a SOTA deepfake detector, which is not biased towards any particular model, we must somehow remove the model signatures left in the generation and train the classifier based on only the inherent features of the fake images. In the future as more powerful generators arise with fainter signatures, such universal techniques become crucial to make a robust deepfake detector.

