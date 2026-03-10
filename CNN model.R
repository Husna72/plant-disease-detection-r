install.packages("keras")
install.packages("tensorflow")
install.packages("tidyverse")
install.packages("caret")
install.packages("ggplot2")

library(keras)
install_keras() 

library(tensorflow)
library(tidyverse)
library(caret)
library(ggplot2)

data_dir <- "C:/Users/husna/OneDrive/Documents/R Projects for GitHub/Plant Village Project/color"

img_height <- 224   
img_width  <- 224   
batch_size <- 32    

train_datagen <- image_data_generator(
  rescale            = 1/255,      
  validation_split   = 0.2,        
  rotation_range     = 20,         
  zoom_range         = 0.2,        
  horizontal_flip    = TRUE,       
  width_shift_range  = 0.2,        
  height_shift_range = 0.2         
)

train_generator <- flow_images_from_directory(
  directory    = data_dir,
  generator    = train_datagen,
  target_size  = c(img_height, img_width),
  batch_size   = batch_size,
  class_mode   = "categorical",  
  subset       = "training",
  seed         = 42
)

val_datagen <- image_data_generator(
  rescale          = 1/255,
  validation_split = 0.2
)
val_generator <- flow_images_from_directory(
  directory   = data_dir,
  generator   = val_datagen,
  target_size = c(img_height, img_width),
  batch_size  = batch_size,
  class_mode  = "categorical",
  subset      = "validation",
  seed        = 42
)

cat("Training samples:  ", train_generator$samples, "\n")
cat("Validation samples:", val_generator$samples, "\n")
cat("Number of classes: ", length(train_generator$class_indices), "\n")

class_names <- names(train_generator$class_indices)
saveRDS(class_names, "class_names.rds")

num_classes <- length(train_generator$class_indices)
cat("Training with", num_classes, "classes\n")

model_scratch <- keras_model_sequential(name = "plant_cnn_scratch") %>%
 
  layer_conv_2d(filters=32, kernel_size=c(3,3), activation="relu",
                input_shape=c(224, 224, 3)) %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  
  layer_conv_2d(filters=64, kernel_size=c(3,3), activation="relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  
  layer_conv_2d(filters=128, kernel_size=c(3,3), activation="relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  
  layer_dropout(rate=0.3) %>%
 
  layer_flatten() %>%
  
  layer_dense(units=256, activation="relu") %>%
  layer_dropout(rate=0.4) %>%
  layer_dense(units=num_classes, activation="softmax")

model_scratch %>% compile(
  optimizer = optimizer_adam(learning_rate=0.001),
  loss      = "categorical_crossentropy",  
  metrics   = c("accuracy")
)

summary(model_scratch)

  history_scratch <- model_scratch %>% fit(
    train_generator,
    epochs          = 10,             
    validation_data = val_generator,
    callbacks = list(
      callback_early_stopping(patience=5, restore_best_weights=TRUE),
      callback_model_checkpoint("best_scratch_model.h5", save_best_only=TRUE)
    )
  )

plot(history_scratch)