library(keras)

if (file.exists("best_mobilenet_model.h5")) {
  cat("MobileNet already trained — delete .h5 file to retrain.\n")
  stop()
}

data_dir <- "C:/Users/husna/OneDrive/Documents/R Projects for GitHub/Plant Village Project/color"

train_generator <- flow_images_from_directory(
  directory  = data_dir,
  generator  = image_data_generator(
    rescale=1/255, validation_split=0.2,
    rotation_range=20, zoom_range=0.2,
    horizontal_flip=TRUE
  ),
  target_size = c(224, 224), batch_size=32,
  class_mode  = "categorical", subset="training", seed=42
)

val_generator <- flow_images_from_directory(
  directory   = data_dir,
  generator   = image_data_generator(rescale=1/255, validation_split=0.2),
  target_size = c(224, 224), batch_size=32,
  class_mode  = "categorical", subset="validation", seed=42
)

num_classes <- length(train_generator$class_indices)
cat("Classes found:", num_classes, "\n")

cat("Loading MobileNet pretrained weights...\n")
base_model <- application_mobilenet(
  weights     = "imagenet",
  include_top = FALSE,
  input_shape = c(224, 224, 3)
)
freeze_weights(base_model)

predictions <- base_model$output %>%
  layer_global_average_pooling_2d() %>%
  layer_dense(units=128, activation="relu") %>%
  layer_dropout(rate=0.5) %>%
  layer_dense(units=num_classes, activation="softmax")

model_mobilenet <- keras_model(
  inputs  = base_model$input,
  outputs = predictions
)

cat("\nPhase 1: Training classification head only (fast)...\n")
model_mobilenet %>% compile(
  optimizer = optimizer_adam(learning_rate=0.001),
  loss      = "categorical_crossentropy",
  metrics   = c("accuracy")
)

history_p1 <- model_mobilenet %>% fit(
  train_generator,
  epochs          = 5,
  validation_data = val_generator,
  callbacks = list(
    callback_early_stopping(patience=3, restore_best_weights=TRUE),
    callback_model_checkpoint("best_mobilenet_model.h5", save_best_only=TRUE)
  )
)

cat(sprintf("\nPhase 1 best val accuracy: %.2f%%\n",
            max(history_p1$metrics$val_accuracy) * 100))

cat("\nPhase 2: Fine-tuning MobileNet layers (slower)...\n")
unfreeze_weights(base_model, from="conv_pw_11_relu")

model_mobilenet %>% compile(
  optimizer = optimizer_adam(learning_rate=0.0001),  # 10x smaller!
  loss      = "categorical_crossentropy",
  metrics   = c("accuracy")
)

history_p2 <- model_mobilenet %>% fit(
  train_generator,
  epochs          = 10,
  validation_data = val_generator,
  callbacks = list(
    callback_early_stopping(patience=5, restore_best_weights=TRUE),
    callback_model_checkpoint("best_mobilenet_model.h5", save_best_only=TRUE)
  )
)

scores <- model_mobilenet %>% evaluate(val_generator, verbose=1)
cat(sprintf("\nMobileNet Final Accuracy: %.2f%%\n", scores[[2]] * 100))

plot(history_p2)
