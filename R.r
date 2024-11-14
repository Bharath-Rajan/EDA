### 6. Design EDA on various variable and row filters in R for cleaning data. 


# R code for filtering and plotting
data <- data.frame(
  ID = c(1, 2, 3, 4, 5),
  Score = c(90, 85, 70, NA, 95)
)

# Filtering rows with no NA
clean_data <- na.omit(data)

# Plotting
library(ggplot2)
ggplot(clean_data, aes(x = factor(ID), y = Score)) +
  geom_bar(stat = "identity", fill = "lightblue") +
  labs(title = "Scores of Students", x = "Student ID", y = "Score")



### 17. Develop various variable and row filters in R for cleaning data. 


# R code for filtering
data <- data.frame(
  ID = c(1, 2, 3, 4),
  Score = c(80, NA, 70, 90)
)

# Filtering rows with no NA
clean_data <- na.omit(data)

# Visualization
boxplot(clean_data$Score,
        main = "Boxplot of Scores",
        ylab = "Scores",
        col = "lightgreen")


### 19. Explore a dataset and apply row filters in R for cleaning data. Apply Scatter plot features in R on sample datasets and visualize.


# R code for filtering and plotting
data <- data.frame(
  ID = c(1, 2, 3, 4),
  Age = c(20, 25, NA, 30),
  Height = c(160, 170, 180, 175)
)

# Filtering rows with no NA
clean_data <- na.omit(data)

# Scatter plot
plot(clean_data$Age, clean_data$Height,
     main = "Age vs Height",
     xlab = "Age",
     ylab = "Height (cm)",
     col = "blue", pch = 19)

### 21. Build different variable and row filters in R for data cleaning. 


# R code for filtering and plotting
data <- data.frame(
  ID = c(1, 2, 3, 4, 5),
  Score = c(80, NA, 70, 90, 85)
)

# Filtering rows with no NA
clean_data <- na.omit(data)

# Plotting
boxplot(clean_data$Score,
        main = "Boxplot of Scores",
        ylab = "Scores",
        col = "lightgreen")
