# Transfer Learning

We used the ticket found on Imagenet and used it on  <a href="https://www.cs.toronto.edu/~kriz/cifar.html">CIPHAR-100</a>.

## CIFAR-100

<p align="justify"> This dataset contains 100 classes containing 600 images each. The images are 32x32 colour images. There are 500 training images and 100 testing images per class. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs). </p>

#### Superclasses	(Classes)
- aquatic mammals	(beaver, dolphin, otter, seal, whale)
- fish (aquarium fish, flatfish, ray, shark, trout)
- flowers	(orchids, poppies, roses, sunflowers, tulips)
- food containers	(bottles, bowls, cans, cups, plates)
- fruit and vegetables (apples, mushrooms, oranges, pears, sweet peppers)
- household electrical devices (clock, computer keyboard, lamp, telephone, television)
- household furniture (bed, chair, couch, table, wardrobe)
- insects (bee, beetle, butterfly, caterpillar, cockroach)
- large carnivores (bear, leopard, lion, tiger, wolf)
- large man-made outdoor things	(bridge, castle, house, road, skyscraper)
- large natural outdoor scenes (cloud, forest, mountain, plain, sea)
- large omnivores and herbivores (camel, cattle, chimpanzee, elephant, kangaroo)
- medium-sized mammals (fox, porcupine, possum, raccoon, skunk)
- non-insect invertebrates (crab, lobster, snail, spider, worm)
- people (baby, boy, girl, man, woman)
- reptiles (crocodile, dinosaur, lizard, snake, turtle)
- small mammals (hamster, mouse, rabbit, shrew, squirrel)
- trees (maple, oak, palm, pine, willow)
- vehicles 1 (bicycle, bus, motorcycle, pickup truck, train)
- vehicles 2 (lawn-mower, rocket, streetcar, tank, tractor)

## Transfer Learning Results

<p align="justify"> After running our model we get a top 5 accuracy of 81% and 72% for the top 1 accuracy. In comparison, in the paper <a href="https://arxiv.org/pdf/1908.01314v4.pdf">Towards Eﬀective Low-bitwidth Convolutional Neural Networks</a> by Bohan Zhuang, Chunhua Shen, Mingkui Tan, Lingqiao Liu, Ian Reid, we can see that AlexNet developped yields respectively 88% and 65% for top 5 accuracy and top 1 accuracy. </p>
