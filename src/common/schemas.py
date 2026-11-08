# -*- coding: utf-8 -*-
"""
Explicit Schema definitions for MovieLens 25M dataset.
Using StructType avoids expensive schema inference on 25M rows.
"""

from pyspark.sql.types import (
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

# ratings.csv: userId, movieId, rating, timestamp
RATINGS_SCHEMA = StructType(
    [
        StructField("userId", IntegerType(), nullable=False),
        StructField("movieId", IntegerType(), nullable=False),
        StructField("rating", FloatType(), nullable=False),
        StructField("timestamp", LongType(), nullable=False),
    ]
)

# movies.csv: movieId, title, genres
MOVIES_SCHEMA = StructType(
    [
        StructField("movieId", IntegerType(), nullable=False),
        StructField("title", StringType(), nullable=False),
        StructField("genres", StringType(), nullable=True),
    ]
)

# tags.csv: userId, movieId, tag, timestamp
TAGS_SCHEMA = StructType(
    [
        StructField("userId", IntegerType(), nullable=False),
        StructField("movieId", IntegerType(), nullable=False),
        StructField("tag", StringType(), nullable=True),
        StructField("timestamp", LongType(), nullable=False),
    ]
)

# genome-scores.csv: movieId, tagId, relevance
GENOME_SCORES_SCHEMA = StructType(
    [
        StructField("movieId", IntegerType(), nullable=False),
        StructField("tagId", IntegerType(), nullable=False),
        StructField("relevance", FloatType(), nullable=False),
    ]
)

# genome-tags.csv: tagId, tag
GENOME_TAGS_SCHEMA = StructType(
    [
        StructField("tagId", IntegerType(), nullable=False),
        StructField("tag", StringType(), nullable=False),
    ]
)

