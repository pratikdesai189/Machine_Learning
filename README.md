For CNN

Step - 1
To Save the model file use below command.

`model.save("my_model_1new.keras")`

Step - 2 
Load the saved model

`loaded_model = tf.keras.models.load_model("my_model_1new.keras")`

Step - 3
Evaluate the loaded model

`loaded_model.evaluate(np.expand_dims(X_test, axis=-1), y_test)`
