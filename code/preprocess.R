library(readr)

setwd('/Users/krishna/MOOC/IndeedML/dataset')
train_df <- data.frame(read_tsv("train.tsv"))
test_df <- data.frame(read_tsv('test.tsv'))

# Remove data with NA



split_tags <- function(tags){
  tags1 <- c('part-time-job',
            'full-time-job',
            'hourly-wage',
            'salary',
            'associate-needed',
            'bs-degree-needed',
            'ms-or-phd-needed',
            "licence-needed",
            '1-year-experience-needed',
            '2-4-years-experience-needed',
            '5-plus-years-experience-needed',
            'supervising-job')
  
  job_tags <- strsplit(tags, " ")
  job_index <- vector(length = 12, mode = "numeric")
  for (jobs in job_tags){
      k <- match(jobs, tags1)
      for (i in k){
        job_index[i] <- 1
      }
  } 
  
  return(job_index)
  }
  

tag_df <- (apply(train_df[, 1], 1,  split_tags))



