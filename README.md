# MovieRec
## Requirements
pip install -r requirements.txt 
## DataBase
Since SQLite's file is too large, 1.27GB after initial all data, PostgreSQL could be a bette choice for higher performance.
1. config your local PostgreSQL in db_config.py and run db_init.py Or using my RDS database.(don't need to init again)
## user_based.py
using cosine for user based collaborative filtering
## user_based_dl.ipynb
just a test
## What to do next (milestone)
1. using different distance function for experience
2. find if there could be more evaluation metrics to compare different model
3. build a Frontend page to show the recommendation outcome

