if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")


################################
# Create edx set, validation set
################################

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

edx <- edx %>% mutate(year=year(as.POSIXct(timestamp,origin="1970-01-01")))
validation <- validation %>% mutate(year=year(as.POSIXct(timestamp,origin="1970-01-01")))

###################################
# Split Edx in train and test set
###################################

# Test Set set will be 20% of Edx data
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y =edx$rating, times = 1,
                                  p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in validation set are also in edx set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(temp,removed,test_index)

######################################
# RMSE function to evaluate prediction
######################################

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

################################

mu <- mean(edx$rating)


# compute the rmse just by using the mean
rmse_results <- data_frame(method = "Just the average",
                           RMSE_Test = RMSE(test_set$rating, mu),
                           RMSE_Valid = RMSE(validation$rating, mu))



# Include a Movie effect in the model
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

movie_avgs_v <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

predicted_ratings_v <- mu + validation %>% 
  left_join(movie_avgs_v, by='movieId') %>%
  .$b_i

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",
                                     RMSE_Test = RMSE(predicted_ratings, test_set$rating),
                                     RMSE_Valid = RMSE(predicted_ratings_v, validation$rating)))



# Include a User effect in the model
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

user_avgs_v <- edx %>% 
  left_join(movie_avgs_v, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings_v <- validation %>% 
  left_join(movie_avgs_v, by='movieId') %>%
  left_join(user_avgs_v, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred


rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE_Test = RMSE(predicted_ratings, test_set$rating),
                                     RMSE_Valid = RMSE(predicted_ratings_v, validation$rating)))



# Regularize the movie effect of the model with a fix parameter
lambda <- 3
movie_reg_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda)) 

predicted_ratings <- test_set %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  .$pred

movie_reg_avgs_v <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda)) 

predicted_ratings_v <- validation %>% 
  left_join(movie_reg_avgs_v, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  .$pred

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Semi-regularized Movie Effect Model",  
                                     RMSE_Test = RMSE(predicted_ratings, test_set$rating),
                                     RMSE_Valid = RMSE(predicted_ratings_v, validation$rating)))

rm(predicted_ratings,predicted_ratings_v,lambda)

 # Regularize the Movie and Effect model with the optimal parameter on test_set
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

lambda <- lambdas[which.min(rmses)]

movie_reg_avgs_v <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda)) 

user_reg_avgs_v <- edx %>% 
  left_join(movie_reg_avgs_v, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

predicted_ratings_v <- validation %>% 
  left_join(movie_reg_avgs_v, by='movieId') %>%
  left_join(user_reg_avgs_v, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred


rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User Effect Model",  
                                     RMSE_Test = min(rmses),
                                     RMSE_Valid = RMSE(predicted_ratings_v, validation$rating)))


# Include a regularized genre effect
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  b_g <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

lambda <- lambdas[which.min(rmses)]

movie_reg_avgs_v <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda)) 

user_reg_avgs_v <- edx %>% 
  left_join(movie_reg_avgs_v, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

genres_reg_avgs_v <- edx %>% 
  left_join(movie_reg_avgs_v, by='movieId') %>%
  left_join(user_reg_avgs_v, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+lambda))

predicted_ratings_v <- validation %>% 
  left_join(movie_reg_avgs_v, by='movieId') %>%
  left_join(user_reg_avgs_v, by='userId') %>%
  left_join(genres_reg_avgs_v, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  .$pred


rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User + Genres Effect Model",  
                                     RMSE_Test = min(rmses),
                                     RMSE_Valid = RMSE(predicted_ratings_v, validation$rating)))


rmse_results %>% knitr::kable()


rm(rmses,lambdas,lambda,predicted_ratings,predicted_ratings_v,
   movie_avgs,user_avgs,movie_avgs_v,user_avgs_v,
   movie_reg_avgs,user_reg_avgs,genres_reg_avgs,
   movie_reg_avgs_v,user_reg_avgs_v,genres_reg_avgs_v)


########################
# SGD algorithm
########################

factors <- 40 # number of latent factors in p and q
lrate <- 0.01 # learning rate for SGD algorithm
reg <- 0.02 # regularization term for SGD algorithm
iter <- 10 # number of SGD iterations


user <- as.list(sort(unique(train_set$userId)))
movie <- as.list(sort(unique(train_set$movieId)))
nu <- length(user)
nm <- length(movie)

userdf <- data.frame(userId=as.numeric(user)) %>%
  mutate(u=which(user==userId))

moviedf <- data.frame(movieId=as.numeric(movie)) %>%
  mutate(m=which(movie==movieId))

train_set <- train_set %>% select(userId, movieId, rating)
train_set <- inner_join(train_set,userdf)
train_set <- inner_join(train_set,moviedf)
test_set <- test_set %>% select(userId, movieId, rating)
test_set <- inner_join(test_set,userdf)
test_set <- inner_join(test_set,moviedf)
valid <- validation %>% select(userId, movieId, rating)
valid <- inner_join(valid,userdf)
valid <- inner_join(valid,moviedf)

nrow <- nrow(train_set)

# initialize user and movie parameters
set.seed(1, sample.kind="Rounding")
p <- replicate(factors,rnorm(nu,0,0.1))
q <- replicate(factors,rnorm(nm,0,0.1))
bu <- rep(0,nu)
bm <- rep(0,nm)

# SGD Loop
for(n in 1:iter){
  cat("\niter :",n, " - time : ", format(Sys.time(),"%H:%M:%S"))
  for(row in 1:nrow){
    u <- train_set[row,"u"]
    m <- train_set[row,"m"]
    err <- train_set[row,"rating"] - ( mu + bu[u] + bm[m] + as.numeric(p[u,]%*%q[m,]))
    bu[u] <- bu[u]+lrate*(err-reg*bu[u])
    bm[m] <- bm[m]+lrate*(err-reg*bm[m])
    p[u,] <- p[u,]+lrate*(err*q[m,]-reg*p[u,])
    q[m,] <- q[m,]+lrate*(err*p[u,]-reg*q[m,])
    rm(u,m,err)
  }
}

# Function to compute our prediction
pred <- function(u,m){
  pred=as.numeric(p[u,]%*%q[m,]+mu+bu[u]+bm[m])
}


prediction <- test %>%
  rowwise() %>% 
  mutate(pred = pred(u,m))

prediction_v <- valid %>%
  rowwise() %>% 
  mutate(pred = pred(u,m))

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="SGD Algorithm",  
                                     RMSE_Test = RMSE(prediction$pred,prediction$rating),
                                     RMSE_Valid = RMSE(prediction_v$pred,prediction_v$rating)))

