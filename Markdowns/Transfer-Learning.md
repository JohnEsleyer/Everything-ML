### What is Transfer Learning?
Transfer learning is about leveraging feature representations from a pre-trained model, so you don't have to train a new model from scratch.

The pre-trained models are usually trained on massive datasets that are standard benchmark in the computer vision frontier. The weights obtained from the models can be reused in other computer vision tasks.

These models can be used directly in making predictions on new tasks or integerated into the process of training a new model. Including the pre-trained models in a new model leads to lower training time and lower generalization error.

Transfer learning is particularly very useful when you have a small training dataset. In this case, you can, for example, use the weights from the pre-trained models to initialize the weights of the new model. As you will see later, transfer learning can also be applied to natural langauge processing problems.

### What is the difference between transfer learning and fine-tuning?
Fine-tuning is an optional step in transfer learning. Fine-tuning will usually improve the performance of the model. However, since you have to retrain the entire model, you'll likely overfit.

### Why use transfer learning?
Assuming you have 100 images of cats and 100 dogs and want to build a model to classify the images. How would you train a model using this small dataset? You can train your model from scratch, but it will most likely overfit horribly. Enter transfer learning. Generally speaking, there are two bigs reasons why you want to use transfer learning:
- **traning models with high accuracy requires a lot of data**. For example, the ImageNet dataset contains over 1 million images. In the real world, you are unlikely to have such a large 
dataset.
- assuming that you has that kind of dataset, you might still not have the resource required to train a model on such a large dataset. Hence transfer learning makes a lot of sense if you don't have the compute resources needed to train models on huge datasets.
- Even if you had the compute resources at your disposal, you will have to wait for days or weeks to train such a model. Therefore using a pre-trained model will save you precious time.

### When not to use Transfer Learning?
Transfer learning will not work when the high-level features learned by the bottom layers are not sufficient to differentiate the classes in your problem. For example, a pre-trained model may be a very good at identifying a door but not whether a door is closed or open. In this case, you can use the low-level features (of the pre-trained network) instead of high-level features. In this case, yuo will have to retrain more layers of the model or use the features from earlier layers.

When datasets are not similar, features transfer poorly. 

You might find yourself in a situation where you consider the removal of some layers from the pre-trained model. Transfer learning is unlikely

### How to implement transfer learning?
Usually, the first step is to instantiate the base model using one of the architectures such as ResNet or Xception. You can also optionally download the pre-trained weights. If you don’t download the weights, you will have to use the architecture to train your model from scratch. Recall that the base model will usually have more units in the final output layer than you require. When creating the base model, you, therefore, have to remove the final output layer. Later on, you will add a final output layer that is compatible with your problem. 

**Freeze layers so they don't change during training**

Freezing the layers from the pre-trained model is vital. This is because you don’t want the weights in those layers to be re-initialized. If they are, then you will lose all the learning that has already taken place. This will be no different from training the model from scratch. 

**Add new trainable layers**
The next step is to add new trainable layers that will turn old features into predictions on the new dataset. This important because the pre-trained model is loaded without the final output layer.

**Train the new layers on the dataset**
Remember that the pre-trained model's final output will most likely be different from the output that you want for your model. For example, pre-trained models trained on the ImageNet dataset will output 1000 classes. However, your model might just have two classes. In this case, you have to train the model with a new output layer in place.

Therefore, you will add some new dense layers as you please, but most importantly, a final dense layer with units corresponding to the number of outputs expected by your model.

**Improve the model via fine-tuning**
Once you have done the previous step, you will have a model that can make predictions on your dataset. Optionally, you can improve its performance through fine-tuning. Fine-tuning is done by unfreezing the base model or part of it and training the entire model again on the whole dataset at a very low learning rate. The low learning rate will increase the performance of the model on the new dataset while preventing overfitting. 

The learning rate has to be low because the model is quite large while the dataset is small. This is a recipe for overfitting, hence the low learning rate. Recompile the model once you have made these changes so that they can take effect. This is because the behavior of a model is frozen whenever you call the compile function. That means that you have to call the compile function again whenever you want to change the model’s behavior. The next step will be to train the model again while monitoring it via callbacks to ensure it does not overfit. 