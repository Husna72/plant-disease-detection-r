library(keras)
library(tidyverse)
library(caret)

model      <- load_model_hdf5("best_mobilenet_model.h5")
class_names <- readRDS("class_names.rds")

val_generator <- flow_images_from_directory(
  directory   = "C:/Users/husna/OneDrive/Documents/R Projects for GitHub/Plant Village Project/color",
  generator   = image_data_generator(rescale=1/255, validation_split=0.2),
  target_size = c(224, 224), batch_size=32,
  class_mode  = "categorical", subset="validation", seed=42
)

# Get predictions
val_generator$reset()
preds_raw    <- model %>% predict(val_generator)
pred_classes <- apply(preds_raw, 1, which.max) - 1
true_classes <- val_generator$classes

pred_labels <- factor(class_names[pred_classes + 1], levels=class_names)
true_labels <- factor(class_names[true_classes + 1], levels=class_names)

# Confusion matrix heatmap
conf_matrix <- table(Predicted=pred_labels, Actual=true_labels)
conf_df     <- as.data.frame(conf_matrix)

shorten <- function(x) gsub(".*___", "", x)
conf_df$Predicted <- shorten(conf_df$Predicted)
conf_df$Actual    <- shorten(conf_df$Actual)

ggplot(conf_df, aes(x=Predicted, y=Actual, fill=Freq)) +
  geom_tile(color="white", linewidth=0.3) +
  scale_fill_gradient(low="white", high="#16a34a") +
  theme_minimal(base_size=7) +
  theme(axis.text.x=element_text(angle=45, hjust=1),
        plot.title=element_text(size=12, face="bold")) +
  labs(title="Confusion Matrix — MobileNet Plant Disease Model",
       subtitle="MobileNet Transfer Learning | 96.49% Accuracy",
       x="Predicted", y="Actual")

ggsave("confusion_matrix.png", width=14, height=12, dpi=150)
cat("Confusion matrix saved!\n")