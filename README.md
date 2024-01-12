Weed Classificationfor Robotic Agriculture![ref1]

000 ![](Aspose.Words.a020766f-28f1-4d78-bcbe-a3b42ab14857.002.png) 001

002 Weed Classificationfor Robotic Agriculture

003

004 ![](Aspose.Words.a020766f-28f1-4d78-bcbe-a3b42ab14857.003.png) 005

006 Metehan Sarikaya1

007

008

009 Abstract ![](Aspose.Words.a020766f-28f1-4d78-bcbe-a3b42ab14857.004.jpeg)

010 This text discusses the critical role of fertilizers in 011 crop production but highlights the risks posed by 012 chemicalfertilizerscontainingharmfulheavymet- 013 als. It briefly touches on the benefits of organic 014 fertilizers as a solution, noting the difficulty for 015 somefarmersinadoptingthisapproach. Themain 016 focus shifts to a proposed research topic: ”Weed 017 Classificationfor Robotic Agriculture using Ma- 018 chine Learning.” This project aims to create a 019 machine-learning model to identify and classify 020 weeds in agricultural fields, aiding in targeted 021 weed control by agricultural robots. The objec- Figure 1. Enter Caption

022 tive is to reduce pesticide use, promote organic

023 farming, and support sustainable agriculture prac-

024 tices. The text emphasizes the potential of such a

025

026 modeldecreasedin real-timepesticideweedapplicationidentification,and the promotileadingonto weedorganiccontrol,farmingreducingpractices.the useThisofispesticidesparticularlyandrelepromotingvant in 027

028 offarming,eco-friendlywhichfaarmingvoids synthetictechniquesinputslikeinorgfaanicvor theuseconteof harmfulxt of sustainablechemicalsagriculture,is a key objectiwhereve.minimizing the 029 of natural pest control and nutrient optimization

030 methods. The machine learning model developed in this project will 031 be trained to recognize different types of weeds and classify 032 them appropriately. This will enable the agricultural robots

033 1. Introduction to identify the weeds in real-time and carry out targeted 034 weed control. As a result, the amount of pesticide used

35  Fertilizers play a crucial role in enhancing crop production can be significantly reduced, leading to more sustainable
35  and ensuring bountiful harvests. However, overreliance on farming practices. Organic farming relies on ecologically
35  chemical fertilizers poses risks to both human health and sound pest control methods and primarily utilizes biological
35  the environment. These substances often contain harmful fertilizers derived from animal waste and nitrogen-fixing
35  heavy metals like lead, mercury, cadmium, and uranium, cover crops. This approach consciously avoids synthetic
35  which can inflictdamage on vital organs such as the kidneys, inputs like pesticides, fertilizers, and hormones, opting in-
35  liver, and lungs. Moreover, these heavy metals are linked to stead for techniques such as crop rotation, organic waste
35  various other health hazards. These damages not only affect utilization, farm manure, rock additives, and crop residues
35  today but can also have irreversible consequences on nature to protect plants and optimize nutrient utilization. Organic
35  in the future. One of the main solutions to all this harm is farming stands as a sustainable and eco-friendly alternative
35  the usage of organic fertilizer but it could be hard for some to conventional agricultural practices, garnering increasing
35  of the farmers. We will not dive into details since we are popularity on a global scale.
35  not experts in terms of farming.

048 1.1. Dataset Spotlight

49  TheRoboticproposedAgricultureresearchusingtopicMachineis “WLearning”.eed ClassificationThe primaryfor
49  aim of this project is to develop a machine-learning model Ata mosaicthe coreof of17,509our projectimagesliescapturingthe ‘DeepWthe essenceeeds’ dataset,of nine
49  that can accurately identify and classify different types of distinct weed classes. Each class, from the tenacious Chinee
49  weeds in agricultural fields. The output from this model will apple to the adaptable Siam weed, offers a unique glimpse
49  be used to guide agricultural robots to carry out targeted into the challenges of weed classification.

164

2
Weed Classificationfor Robotic Agriculture![ref1]

055 ![](Aspose.Words.a020766f-28f1-4d78-bcbe-a3b42ab14857.005.jpeg)056 057 058 059 060 061 062 063 

064 065 066 067

068 Figure 2. Enter Caption

069

070

071 Class distribution of DeepWeeds Dataset 072

73  Unveiling the class distribution reveals compelling patterns
73  — some weeds stand out prominently, while others play a
73  subtler role. Tackling these imbalances becomes our com-
73  pass for targeted model training.

077

78  Theuniform.distribSomeutionweedof weedtypesclasseswield morewithininfluence,our datasetswhileis oth-not
78  ers are less common. Recognizing these imbalances guides
78  our strategy, ensuring our model is finely tuned to distin-
78  guish both influentialand lesser-known weed types during
78  training. A pivotal addition to the DeepWeeds dataset is
78  the ‘Negative’ class, representing unwanted seeds or weeds
78  in agriculture. Identifying and classifying these instances
78  is paramount for effective weed management. This unique
78  dimension aligns seamlessly with our broader mission of
78  promoting sustainable agriculture by minimizing the impact
78  of undesirable elements.

089

090

91  2. Related Work
91  In recent years, several studies have been conducted on
91  the application of machine learning techniques for weed
91  detection in agriculture.

095

96  AhmedHushamAl-Badri’sstudyonweeddetectionthrough
96  machine learning techniques offers an in-depth exploration
96  of this field. This work extensively reviews various machine-
96  learning methods used for weed detection, emphasizing the
96  challenges faced within this domain. Moreover, it delves
96  into potential techniques that could shape the future of weed
96  detection methodologies (Al-Badri, Year).
96  Kun Hu’s research on deep learning techniques for weed
96  recognition in large-scale grain production systems serves
96  as a valuable resource in this field. The paper presents an
96  encompassing overview of deep-learning methodologies
96  applied specifically to weed recognition within expansive
96  grain production systems. It extensively discusses the theo- 109

retical underpinnings of deep learning, exploring relevant building blocks and architectures crucial for effective weed detection (Hu, Year).

“Weed Detection Using Deep Learning: A Systematic Liter- ature Review” provides a systematic review of the literature on the use of deep learning for weed detection. It highlights the effectiveness of deep learning techniques for weed de- tection and provides a valuable reference for future research in this field. By studying and building upon the findings of these papers, the proposed project aims to contribute to the fieldof robotic agriculture through the development of a robustweedclassificationmodel. Thesuccessofthisproject will not only enhance the efficiency of weed control in agri- culture but also contribute to the promotion of sustainable farming practices.

3\. The Approach

We will use three different transfer learning models in ad- dition to reference paper models, which are Inception-v3 and ResNet-50. After experimenting, we will compare our results with the reference paper in the Experimental Results part, but I will give information about our approach to solve the explained problem.

1. MobileNetV2

MobileNetV2 is a type of convolutional neural network ar- chitecture designed specificallyfor mobile and edge devices with limited computational resources. It’s an improvement over the original MobileNet architecture, aiming to be more efficientin speed, size, and accuracy. Developed by Google, MobileNetV2 employs several techniques to achieve its ef- ficiency, such as depthwise separable convolutions, linear bottlenecks, andinvertedresiduals. Thesetechniquesreduce the number of parameters and computations required while preserving or even enhancing the network’s performance. MobileNetV2 is often used in various computer vision ap- plications on mobile devices, including image classification, object detection, and image segmentation, due to its ability to balance between accuracy and computational efficiency.

2. YOLO (You Only Look Once)

We want to start this part by solving a little confusion about

how we used Yolo as a classification model. Until the YOLOv8, the YOLO model backbone used the object detec- tion and segmentation model, but since its eighth version, it exhibited enhanced capabilities beyond mere object detec- tion — it is now capable of image classification.

The YOLOv8 classificationmodel is developed to accom- plish real-time detection of 1000 predetermined classes within images. Unlike other tasks that rely on pre-training models using datasets such as COCO or ImageNet, the dis-

110 ![](Aspose.Words.a020766f-28f1-4d78-bcbe-a3b42ab14857.006.jpeg)111 112 113 114 115 116 

117 118 119 120 121

122

123 Figure 3. Enter Caption

124

125

126  tinctivefeatureofimageclassificationliesinitsfocusoncat-
126  egorizing entire images into predefinedlabels, as opposed to
126  delineating bounding boxes around identifiedclasses within
126  the image. This attribute proves beneficial when the pri-
126  mary objective is discerning the class to which an image
126  belongs, without necessitating precise object localization or
126  delineation of their specificshapes.

133

134  Notably, YOLOv8 comprises various models of differing
134  sizes, a topic that warrants exploration in our forthcoming
134  blog entry. Specifically, the next post will delve into a com-
134  parativeanalysisbetweentheclassicYOLOarchitectureand
134  the architecture tailored for classification purposes. This
134  exploration will elucidate the disparities and advancements
134  intrinsic to each model, shedding light on their respective
134  strengths and functionalities.
134  Specifically, the model denoted as YOLOv8N-CLS.pt fol-
134  lows a parallel structure with the YOLOv8 architecture up
134  to its 9th layer, which is denoted as SPPF in the original
134  framework. However, in YOLOv8N-CLS.pt, a modification
134  has been introduced at this layer, deviating from the SPPF
134  utilization.

148

149  Thisexcerptillustratestheterminallayersofthenanomodel,
149  representing its fundamental and streamlined architecture.
149  The primary function of this particular layer arrangement is
149  geared toward the task of classification. The utilization of
149  AdaptiveAvgPool2dcontributestodimensionalityreduction,
149  while the subsequent single fully connected layer facilitates
149  the classificationprocess.

156

157  3.3. Experiment 1 With MobileNetV2
157  We conducted three core experiments; our firstexploration
157  involved the implementation of MobileNetV2. We used
157  some data augmentation methods in one of the experiments,
157  so we want to begin with it. We did a rescale shearing trans-
157  formation and horizontal flip. we used rescale to scale the
157  pixel values of the images to be between 0 and 1. It’s a nor-

malization step that helps in faster convergence during train- ing. We used shearrange to specify the range within which random shearing transformations will be applied to the im- ages. Shearing transforms the image by pushing one side of the image in a particular direction, creating a sort of ’tilt’. We used horizontalflipto randomly flipimages horizontally. This augmentation is useful for tasks where horizontal flip- ping doesn’t change the interpretation of the image (like in most cases). We commented out (validationsplit, zoom- range, widthshiftrange, heightshiftrange). In refining the architecture, we augmented the model with supplementary layers, notably integrating globalAveragePooling2D. This addition was instrumental in reducing dimensionality, al- lowing for a more streamlined and effective classification process. Additionally, to facilitate classification,we intro- duced a dense layer with softmax activation.

3\.4. Experiment 2 With YOLOX-CLS.pt

We did two experiments on Yolo and got better accuracy than expected. We can say that the implementation of Yolo was much easier than the Keras models. We used nano and large models. The difference between large and Xlarge models is evident in their layers and parameters. An Xlarge model boasts nearly 38 times the parameters of a nano model.

We did our first experiment with xlarge model as we said this model has 183 layers and 56153369 parameters. We did data augmentation using Albumentations. It provides a wide rangeoftechniquestoaugmentandmanipulateimages,such as rotation, flipping,scaling, changes in brightness, contrast adjustments, and more. These augmentations help enhance the variety and diversity of the training data for machine learning models, ultimately improving their robustness and generalization to new, unseen data. You can see the used methods below;

1- RandomResizedCrop(p=1.0, height=256, width=256, scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), inter- polation=1)

2- HorizontalFlip(p=0.5), ColorJitter(p=0.5, bright- ness=[0.6, 1.4], contrast=[0.6, 1.4], satura- tion=[0.30000000000000004, 1.7], hue=[-0.015, 0.015])

3- Normalize(p=1.0, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max-pixel-value=255.0)

4- ToTensorV2(always-apply=True, p=1.0, transpose- mask=False)

We employed the SGD optimizer with a learning rate of 0.01 for our model. Our dataset comprises 15,956 images in the training set and 4,850 in the validation split. The experiment was conducted over 50 epochs.

SGD optimizer is short of Stochastic Gradient Descent. It’s

164

5
Weed Classificationfor Robotic Agriculture![ref1]

165  a well-known optimization algorithm used in machine learn-
165  ing and deep learning to minimize the loss function. Essen-
165  tially, it tweaks the parameters of a model in small steps be
165  aware that t if you make this step as much large as dataset,
165  it would be traditional gradient descent, after splitting steps
165  adjust them in the direction that minimizes the loss by using
165  gradients computed on smaller subsets of the data, which
165  makes it faster than traditional gradient descent methods 173

174 3.5. Experiment 3 With YOLON-CLS.pt 175

176  On nano, the model has 99 layers and 1449817 parameters.
176  This model is small in terms of size compared to xlarge. We
176  used albumentations to do data augmentation. You can see
176  the used methods below with the parameters;
176  1- RandomResizedCrop (p=1.0, height=64, width=64,
176  scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), inter-
176  polation=1)

183

184  2- HorizontalFlip(p=0.5)
184  3- ColorJitter(p=0.5, brightness=[0.6, 1.4], contrast=[0.6,
184  1.4], saturation=[0.30000000000000004, 1.7], hue=[-0.015,
184  0.015])

188

189  4- Normalize(p=1.0, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0,
189  1.0), max-pixel-value=255.0)
189  5- ToTensorV2(always-apply=True, p=1.0, transpose-
189  mask=False)

193

194  We employed the AdamW optimizer with a learning rate of
194  0.000714 for our model. AdamW is an improved version of
194  the Adam optimization algorithm that handles weight decay
194  in a more effective way, potentially enhancing a model’s
194  ability to generalize by controlling large weight values dur-
194  ing training. W stands for weight decay. Here we mean
194  adding a regularization term to the loss function. We are
194  doing this to prevent overfitting. And as expected from reg-
194  ularization this modificationcan help better generalization
194  and performance in some cases, especially when training
194  deep neural networks. Our dataset comprises 15,956 images
194  in the training set and 4,850 in the validation split. The
194  experiment was conducted over 50 epochs.

207

208 4. Experimental Results

209

210  In this part, we will demonstrate our results from the ex-
210  plained experiment on The Approach part. To evaluate our
210  findings,we used a confusion matrix and demonstrated re-
210  sults class by class. Also, as numeric metrics, we used
210  the overall accuracy of all classes, precision, recall, and
210  F1 score. Before beginning this part, you can get short
210  information about these metrics below;
210  Confusion Matrix is a matrix that shows how accurate our
210  prediction is. You can easily calculate true positive, false

positive, and true negative rates. It helps us understand where our model is making mistakes.

Accuracy is the correctness of the model’s compared to actual labels. It’s calculated as the number of correct predic- tions divided by the total number of predictions. Precision is how much our prediction is as precise as expected you can calculate it easily by dividing correct prediction by total positive prediction. Recall measures how much we are able to identify actual positives. Precision and Recall are useful when datasets have class imbalance and because of that, we used these two metrics to evaluate our result. Some- times Precision and Recall could give biased and deficient information and could cause a bias-variance trade-off. We considered this and used the F1 score too. The F1 score is the harmonic mean of precision and recall.

We got terrible results on our first experiment if we com- pared it with reference paper and other models. As you can see on the plots below, while train accuracy is increasing, validation, which is still less than the reference paper result, is constant. This result could be considered an overfitting problem. We can apply some techniques we already used early stopping and augmentation. Hyperparameter optimiza- tion could be the next step but we will leave it as future work and focus on Yolo.

![](Aspose.Words.a020766f-28f1-4d78-bcbe-a3b42ab14857.007.png)

Figure 4. MobileNetV2 Results

Here you are seeing visualization of losses and accuracies of train and validation. You can see that validation results have terrible results and are nearly steady. As we said we got the best result from yolo models. We used 10 epoch to train model and we used early stopping you can see that there is not need to continue training after 8 epoch because there is not increase of accuracy.

4\.1. Results of Yolo Models

We achieved an accuracy of almost 99% for top-1 predic- tions and a perfect 100% for top-5 predictions. Top-1 accu- racy reflectsthe model’s precision in identifying an object’s class with the highest confidence,while top-5 accuracy as- sesses if the correct class is among the top 5 predictions. Our confusion matrix in the graphs illustrates this perfor- mance. We encountered no false positives but experienced

219

7
Weed Classificationfor Robotic Agriculture![ref1]

220 ![](Aspose.Words.a020766f-28f1-4d78-bcbe-a3b42ab14857.008.png)221 222 ![](Aspose.Words.a020766f-28f1-4d78-bcbe-a3b42ab14857.009.jpeg)223 224 225 226 227 228 229 230 231 232 233 

234 Figure 5. Loss of Nano Model 235 Figure 7. Confusion Matrix of Nano Model ![](Aspose.Words.a020766f-28f1-4d78-bcbe-a3b42ab14857.010.jpeg)236

237

238

239

240 ![](Aspose.Words.a020766f-28f1-4d78-bcbe-a3b42ab14857.011.png)

241 

242 

243 

244 

245 

246 

247 

248 

249 Figure 6. Accuracy of Nano Model 

250 

251 

252  oneportantor twtoonotemisclassificationsthat the negatiinvetheclassnegcomprisesative class.aIt’broads im- 
252  spectrum of weeds. Although we intended to address the Figure 8. YoloV8XL Loss vs Epoch
252  class imbalance, after given the promising results and time
252  constraints, we will not need to consider this anymore.

256

257  We achieved an accuracy of almost 92% for top-1 predic-
257  tions and a perfect 99% for top-5 predictions. you can see ![](Aspose.Words.a020766f-28f1-4d78-bcbe-a3b42ab14857.012.png)
257  the nano model confusion matrix in the graphs illustrates 
257  this performance. As you can see, there are more false 
257  classificationsthan large model. 
257  Classificationresults showcasing Precision, Recall, and F1 
257  Score are displayed in the table above for the Nano Model, 
257  while the XLarge Model’s outcomes are presented below. 
257  As expected, the XLarge Model yielded better results com- 
257  pared to the Nano Model, owing to its larger parameter size, 
257  which is 34 times greater than that of the Nano Model. 268 

269 

270 

271 Figure 9. YoloV8XL accuracy vs epoch 272

273

274

9
Weed Classificationfor Robotic Agriculture![ref1]

275 ![](Aspose.Words.a020766f-28f1-4d78-bcbe-a3b42ab14857.013.png)276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322

Figure 10. Confusion Matrix of Xlarge Model

Table 1. ClassificationMetrices for Yolo Nano and XLarge modesl on Species



|DATA SET|PRECISION|RECALL|F1|
| - | - | - | - |
|CHINEE APPLE LANTANA PARKINSONIA PARTHENIUM PRICKLY ACACIA RUBBER VINE SIAM WEED SNAKE WEED OTHER|<p>0\.844 0.905 0.972 0.904 0.942 0.920</p><p>0\.939 0.865 0,953</p>|0\.790 0,871 0,972 0.875 0.99 0.99 1.00 0.98 0.957|0\.816 0.887 0.948 0.889 0.916 0.937 0.915 0.872 0.954|
|CHINEE APPLE LANTANA PARKINSONIA PARTHENIUM PRICKLY ACACIA RUBBER VINE SIAM WEED SNAKE WEED OTHER|0\.993 0.905 0.996 1 0.993 0.992 0.9998 0.989 0.996|0\.983 0,996 1 0.992 0.993 0.992 1.00 0.982 0.996|0\.987 0.989 0.997 0.995 0.993 0.992 0.999 0.985 0.996|

5. Conclusions

In summary, we introduced a new approach to reference papers and problems and got better results from existing solutions. We can say that the best model YoloV8XL-clf is production-ready and expecting future investment to use in the Australian Farming Industry. In terms of our profession, we can say that. In the future, we are considering working on different regions with different weed types. We believe that We believe that after reaching a certain level our project will pave the way for the discovery of new plant species in the future and can have various impacts ranging from

323 agriculture to medicine. 324

325  5.1. Data Availability
325  The DeepWeeds dataset and source code for this work
325  are publicly available through the corresponding authors’
325  GitHub repository:[ LINK]( https://github.com/AlexOlsen/DeepWeed)
6. Citations and References References

   Al-Badri, A. H. Classificationof weed using machine learn-

ing techniques: a review—challenges, current and future potential techniques. Journal of Plant Diseases and Pro- tection, 129, 2022.

Hu, K. Deep learning techniques for in-crop weed recog-

nition in large-scale grain production systems: a review. Precision Agriculture, 2023.

Murad, N. Y. Weed detection using deep learning: A sys-

tematic literature review. 23, 2023.

Olsen, A. Deepweeds: A multiclass weed species image dataset for deep learning. ScientificReports, 9, 2019.
329

11

[ref1]: Aspose.Words.a020766f-28f1-4d78-bcbe-a3b42ab14857.001.png
