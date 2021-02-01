from surprise import KNNWithMeans

# To use item-based cosine similarity
sim_options = {
    "name": "cosine",
    "user_based": False, # Compute similarities btwn items
}

algo = KNNWithMeans(sim_options=sim_options)